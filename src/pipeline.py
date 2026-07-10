from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.candidate_retrieval import SourceAttempt, get_candidate_pool
from src.correction import build_batch_payload, correct_with_gemini
from src.cost_tracker import CostTracker
from src.gemini_vision import extract_raw_prescription
from src.scoring import ScoredCandidate, rank_candidates

LOW_CONFIDENCE_THRESHOLD = 0.6
HIGH_HYBRID_FALLBACK_THRESHOLD = 85.0


def _write_json(path: str, data: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_pipeline(
    image_path: str,
    score_threshold: float = 55.0,
    top_k: int = 3,
    use_semantic: bool = True,
    use_scraped_sources: bool = False,
    scrape_as_fallback: bool = True,
    vision_model: str = "gemini-3.1-flash-lite",
    correction_model: str = "gemini-3.1-flash-lite",
    save_raw_path: Optional[str] = None,
    save_final_path: Optional[str] = None,
    save_cost_path: Optional[str] = None,
) -> Dict:
    """
    Runs the full 8-step pipeline for one image.

    If save_raw_path is given, the Step 1 (pre-correction) extraction is
    written there before any correction happens. If save_final_path is
    given, the final Step 8 output is written there too. If save_cost_path
    is given, a full per-call cost breakdown (CostTracker.to_dict()) is
    written there. The final dict is always returned, and also carries a
    lightweight `_cost_usd` total for quick visibility.
    """
    tracker = CostTracker(image_path=image_path)

    # ---- Step 1: raw Gemini extraction ----------------------------------
    raw = extract_raw_prescription(image_path, model=vision_model, cost_tracker=tracker)
    if save_raw_path:
        _write_json(save_raw_path, copy.deepcopy(raw))

    medications = raw.get("medications", [])

    # ---- Steps 2-4: candidate retrieval + hybrid scoring, per line ------
    candidates_by_line: Dict[int, List[ScoredCandidate]] = {}
    source_debug_by_line: Dict[int, List[SourceAttempt]] = {}
    for med in medications:
        raw_guess = med.get("drug_name")
        line_id = med["line_id"]
        if not raw_guess:
            candidates_by_line[line_id] = []
            continue

        line_debug: List[SourceAttempt] = []
        pool = get_candidate_pool(
            raw_guess,
            use_scraped_sources=use_scraped_sources,
            scrape_as_fallback=scrape_as_fallback,
            debug_log=line_debug,
        )
        source_debug_by_line[line_id] = line_debug
        summary = ", ".join(f"{a.source}:{a.status}({a.candidate_count})" for a in line_debug)
        print(f"  [{raw_guess}] {summary}", file=sys.stderr)
        ranked = rank_candidates(
            raw_guess,
            pool,
            score_threshold=score_threshold,
            top_k=top_k,
            use_semantic=use_semantic,
            cost_tracker=tracker,
        )
        candidates_by_line[line_id] = ranked

    # ---- Step 5: batched payload -----------------------------------------
    payload = build_batch_payload(medications, candidates_by_line)

    # ---- Step 6: one LLM call for correction ------------------------------
    lines_needing_llm = [
        p for p in payload if p["top_3_candidates"] or p["raw_frequency"] or p["raw_duration"]
    ]
    llm_decisions: Dict[int, Dict] = {}
    if lines_needing_llm:
        decisions = correct_with_gemini(
            image_path, lines_needing_llm, model=correction_model, cost_tracker=tracker
        )
        llm_decisions = {d["line_id"]: d for d in decisions}

    # ---- Step 7: merge LLM decision back into each medication ------------
    for med in medications:
        line_id = med["line_id"]
        raw_guess = med.get("drug_name")
        raw_frequency = med.get("frequency")
        raw_duration = med.get("duration")
        ranked = candidates_by_line.get(line_id, [])
        decision = llm_decisions.get(line_id)
        best_candidate = ranked[0] if ranked else None

        llm_pick = (decision or {}).get("drug_name_corrected")
        picked_candidate = next((c for c in ranked if c.name == llm_pick), None)

        if picked_candidate is not None:
            corrected = picked_candidate.name
            llm_confidence = float(decision.get("llm_confidence", 0.0))
            combined_confidence = round(0.6 * llm_confidence + 0.4 * (picked_candidate.hybrid_score / 100), 4)
            reasoning = decision.get("reasoning")
            disagreement = False
        elif best_candidate is not None and best_candidate.hybrid_score >= HIGH_HYBRID_FALLBACK_THRESHOLD:
            corrected = best_candidate.name
            combined_confidence = round((best_candidate.hybrid_score / 100) * 0.7, 4)
            reasoning = (decision or {}).get(
                "reasoning",
                f"LLM found no match; falling back to top-scored candidate (hybrid_score={best_candidate.hybrid_score:.1f}).",
            )
            disagreement = True
        elif llm_pick and llm_pick.strip().upper() != "NONE":
            corrected = llm_pick.strip()
            llm_confidence = float((decision or {}).get("llm_confidence", 0.0))
            combined_confidence = round(llm_confidence * 0.5, 4)  # discount: unverified against any database
            reasoning = (decision or {}).get("reasoning", "LLM free-text read; not matched to any retrieved candidate.")
            disagreement = True
        else:
            corrected = raw_guess  # fallback to raw guess — no method found a trustworthy match
            combined_confidence = 0.0
            reasoning = (decision or {}).get("reasoning", "No matching candidate found.")
            disagreement = True

        needs_review = (not ranked) or disagreement or (combined_confidence < LOW_CONFIDENCE_THRESHOLD)

        frequency = decision.get("frequency_corrected") if decision is not None else raw_frequency
        duration = decision.get("duration_corrected") if decision is not None else raw_duration

        med["drug_name_raw"] = raw_guess
        med["drug_name_corrected"] = corrected
        med["frequency_raw"] = raw_frequency
        med["duration_raw"] = raw_duration
        med["candidates_considered"] = [c.name for c in ranked]
        med["hybrid_scores"] = {c.name: c.hybrid_score for c in ranked}
        med["llm_confidence"] = combined_confidence
        med["llm_reasoning"] = reasoning
        med["needs_review"] = needs_review
        med["source_debug"] = [asdict(a) for a in source_debug_by_line.get(line_id, [])]

        # ---- Step 8: drug_name, frequency, and duration are now all
        med["drug_name"] = corrected
        med["frequency"] = frequency
        med["duration"] = duration

    raw.pop("_metrics", None)
    raw["_cost_usd"] = tracker.total_cost_usd

    if save_cost_path:
        tracker.save(save_cost_path)

    if save_final_path:
        _write_json(save_final_path, raw)

    return raw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def run_pipeline_on_folder(
    folder_path: str,
    out_dir: str = "outputs",
    score_threshold: float = 55.0,
    top_k: int = 3,
    use_semantic: bool = True,
    use_scraped_sources: bool = False,
    scrape_as_fallback: bool = True,
    vision_model: str = "gemini-3.1-flash-lite",
    correction_model: str = "gemini-3.1-flash-lite",
) -> Dict[str, Dict]:
    """
    Runs the full pipeline on every image in folder_path.

    For each image `name.jpg`, writes:
      {out_dir}/raw/name.json     -- Step 1 raw extraction (before correction)
      {out_dir}/final/name.json   -- Step 8 final corrected output

    Returns {filename: final_result_dict}. A failure on one image is logged
    and skipped rather than aborting the whole batch.
    """
    import sys

    from src.cost_tracker import summarize_costs

    folder = Path(folder_path)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {folder_path} (looked for {IMAGE_EXTENSIONS})", file=sys.stderr)

    results: Dict[str, Dict] = {}
    cost_dicts: List[Dict] = []
    for img_path in images:
        stem = img_path.stem
        cost_path = str(Path(out_dir) / "costs" / f"{stem}.json")
        print(f"[{stem}] processing {img_path} ...", file=sys.stderr)
        try:
            result = run_pipeline(
                str(img_path),
                score_threshold=score_threshold,
                top_k=top_k,
                use_semantic=use_semantic,
                use_scraped_sources=use_scraped_sources,
                scrape_as_fallback=scrape_as_fallback,
                vision_model=vision_model,
                correction_model=correction_model,
                save_raw_path=str(Path(out_dir) / "raw" / f"{stem}.json"),
                save_final_path=str(Path(out_dir) / "final" / f"{stem}.json"),
                save_cost_path=cost_path,
            )
            results[stem] = result
            print(
                f"[{stem}] done -> {out_dir}/final/{stem}.json  (${result.get('_cost_usd', 0):.6f})",
                file=sys.stderr,
            )
            with open(cost_path, encoding="utf-8") as f:
                cost_dicts.append(json.load(f))
        except Exception as e:
            print(f"[{stem}] FAILED: {e}", file=sys.stderr)

    if cost_dicts:
        summary = summarize_costs(cost_dicts)
        _write_json(str(Path(out_dir) / "costs" / "_summary.json"), summary)
        print(
            f"\nCost summary: {summary['num_images']} image(s), "
            f"avg ${summary['avg_cost_per_image_usd']:.6f}/image, "
            f"total ${summary['total_cost_usd']:.4f}. "
            f"Projected: {summary['projection_usd']}",
            file=sys.stderr,
        )

    return results


if __name__ == "__main__":
    import json
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/1.jpg"
    result = run_pipeline(path)
    print(json.dumps(result, indent=2))
