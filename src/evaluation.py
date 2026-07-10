from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

FIELD_WEIGHTS = {
    "drug_name": 3.0,
    "strength": 2.0,
    "frequency": 2.0,
    "duration": 2.0,
    "patient_name": 1.0,
    "age": 1.0,
}
FUZZY_MATCH_THRESHOLD = 88.0  # 0-100; tolerates minor formatting diffs ("500mg" vs "500 mg")
NAME_FUZZY_MATCH_THRESHOLD = 80.0  # names are short strings; ratio-based cutoffs are brittle
LOW_CONFIDENCE_THRESHOLD = 0.6

DURATION_UNIT_MAP = {"day": 1, "days": 1, "week": 7, "weeks": 7}


# --------------------------------------------------------------------------
# Low-level field comparison
# --------------------------------------------------------------------------
def _normalize(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    return s or None


def _normalize_duration(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().lower()
    if not s:
        return None

    m = re.match(r"^(\d+)\s*/\s*7$", s)  # "N/7" style = N days
    if m:
        return f"{int(m.group(1))}d"

    m = re.match(r"^(\d+)\s*/\s*52$", s)  # "N/52" style = N weeks (standard Rx shorthand)
    if m:
        return f"{int(m.group(1)) * 7}d"

    m = re.match(r"^(\d+)\s*(day|days|week|weeks)$", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        return f"{n * DURATION_UNIT_MAP[unit]}d"

    return s


def fields_match(gt_value: Optional[str], pred_value: Optional[str], field_name: Optional[str] = None) -> bool:
    if field_name == "duration":
        gt_n, pred_n = _normalize_duration(gt_value), _normalize_duration(pred_value)
    else:
        gt_n, pred_n = _normalize(gt_value), _normalize(pred_value)
    if gt_n is None and pred_n is None:
        return True
    if gt_n is None or pred_n is None:
        return False
    threshold = NAME_FUZZY_MATCH_THRESHOLD if field_name == "patient_name" else FUZZY_MATCH_THRESHOLD
    return gt_n == pred_n or fuzz.ratio(gt_n, pred_n) >= threshold


def classify_field(gt_value: Optional[str], pred_value: Optional[str], field_name: Optional[str] = None) -> str:
    """
    Returns one of: "correct", "hallucinated", "missed", "correct_blank".
    - correct: both present and match
    - correct_blank: both null (correctly declined to guess)
    - hallucinated: predicted a value not supported by ground truth
      (GT null but pred present, OR both present but don't match —
      i.e. the model fabricated/guessed wrong instead of flagging it)
    - missed: GT has a value but pred is null (safe — flagged rather
      than guessed — but still incomplete)
    """
    if field_name == "duration":
        gt_n, pred_n = _normalize_duration(gt_value), _normalize_duration(pred_value)
    else:
        gt_n, pred_n = _normalize(gt_value), _normalize(pred_value)

    if gt_n is None and pred_n is None:
        return "correct_blank"
    if gt_n is None and pred_n is not None:
        return "hallucinated"
    if gt_n is not None and pred_n is None:
        return "missed"

    if field_name == "duration":
        return "correct" if gt_n == pred_n else "hallucinated"

    threshold = NAME_FUZZY_MATCH_THRESHOLD if field_name == "patient_name" else FUZZY_MATCH_THRESHOLD
    return "correct" if fuzz.ratio(gt_n, pred_n) >= threshold else "hallucinated"


# --------------------------------------------------------------------------
# CER / WER
# --------------------------------------------------------------------------
def compute_cer(gt_text: str, pred_text: str) -> float:
    """Character Error Rate = edit distance / len(reference), in [0, inf)."""
    gt_text, pred_text = gt_text or "", pred_text or ""
    if not gt_text:
        return 0.0 if not pred_text else 1.0
    return Levenshtein.distance(gt_text, pred_text) / len(gt_text)


def compute_wer(gt_text: str, pred_text: str) -> float:
    """Word Error Rate = word-level edit distance / word count(reference)."""
    gt_words = (gt_text or "").split()
    pred_words = (pred_text or "").split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    # Map each unique word to a single char so Levenshtein works word-wise.
    vocab = {w: chr(i) for i, w in enumerate(sorted(set(gt_words) | set(pred_words)))}
    gt_seq = "".join(vocab[w] for w in gt_words)
    pred_seq = "".join(vocab[w] for w in pred_words)
    return Levenshtein.distance(gt_seq, pred_seq) / len(gt_words)


# --------------------------------------------------------------------------
# Medication alignment (predicted list may differ in length/order from GT)
# --------------------------------------------------------------------------
def _align_medications(gt_meds: List[Dict], pred_meds: List[Dict]) -> List[Tuple[Optional[Dict], Optional[Dict]]]:
    """
    Greedy alignment by drug_name similarity. Returns a list of (gt_med,
    pred_med) pairs; either side can be None (unmatched -> missed / extra).
    """
    remaining_gt = list(enumerate(gt_meds))
    remaining_pred = list(enumerate(pred_meds))
    pairs: List[Tuple[Optional[Dict], Optional[Dict]]] = []

    scored = []
    for gi, g in remaining_gt:
        for pi, p in remaining_pred:
            sim = fuzz.ratio(_normalize(g.get("drug_name")) or "", _normalize(p.get("drug_name")) or "")
            scored.append((sim, gi, pi))
    scored.sort(reverse=True)

    used_gt, used_pred = set(), set()
    for sim, gi, pi in scored:
        if gi in used_gt or pi in used_pred:
            continue
        if sim < 40:  # too dissimilar to be the "same" line — leave both unmatched
            continue
        pairs.append((gt_meds[gi], pred_meds[pi]))
        used_gt.add(gi)
        used_pred.add(pi)

    for gi, g in remaining_gt:
        if gi not in used_gt:
            pairs.append((g, None))
    for pi, p in remaining_pred:
        if pi not in used_pred:
            pairs.append((None, p))

    return pairs


# --------------------------------------------------------------------------
# Per-image evaluation
# --------------------------------------------------------------------------
def evaluate_prescription(gt: Dict, pred: Dict) -> Dict:
    field_results: List[Dict] = []

    # patient-level fields
    # NOTE: ground truth files use "patient_age" while pipeline output uses
    # "age" for the same field -- map explicitly instead of relying on a
    # shared key name.
    gt_field_map = {"patient_name": "patient_name", "age": "patient_age"}
    for field_name in ("patient_name", "age"):
        gt_val = gt.get(gt_field_map[field_name])
        pred_val = pred.get(field_name)
        field_results.append(
            {
                "field": field_name,
                "line_id": None,
                "classification": classify_field(gt_val, pred_val, field_name=field_name),
                "weight": FIELD_WEIGHTS[field_name],
            }
        )

    # medication fields
    med_pairs = _align_medications(gt.get("medications", []), pred.get("medications", []))
    for gt_med, pred_med in med_pairs:
        for field_name in ("drug_name", "strength", "frequency", "duration"):
            gt_val = (gt_med or {}).get(field_name)
            pred_val = (pred_med or {}).get(field_name)
            field_results.append(
                {
                    "field": field_name,
                    "line_id": (pred_med or {}).get("line_id"),
                    "classification": classify_field(gt_val, pred_val, field_name=field_name),
                    "weight": FIELD_WEIGHTS[field_name],
                }
            )

    total_weight = sum(r["weight"] for r in field_results)
    correct_weight = sum(r["weight"] for r in field_results if r["classification"] in ("correct", "correct_blank"))
    weighted_accuracy = correct_weight / total_weight if total_weight else 1.0

    hallucinated = [r for r in field_results if r["classification"] == "hallucinated"]
    missed = [r for r in field_results if r["classification"] == "missed"]
    hallucination_rate = len(hallucinated) / len(field_results) if field_results else 0.0

    # CER/WER on raw text, if both sides provide it
    cer = wer = None
    gt_text = gt.get("raw_ocr_text") or " ".join(m.get("raw_line", "") for m in gt.get("medications", []) if m.get("raw_line"))
    pred_text = pred.get("raw_ocr_text") or " ".join(m.get("raw_line", "") for m in pred.get("medications", []) if m.get("raw_line"))
    if gt_text:
        cer = round(compute_cer(gt_text, pred_text), 4)
        wer = round(compute_wer(gt_text, pred_text), 4)

    # Calibration: does llm_confidence / needs_review track drug_name correctness
    calibration_points = []
    for gt_med, pred_med in med_pairs:
        if not pred_med:
            continue
        conf = pred_med.get("llm_confidence")
        if conf is None:
            continue
        is_correct = fields_match((gt_med or {}).get("drug_name"), pred_med.get("drug_name"), field_name="drug_name")
        calibration_points.append({"confidence": conf, "correct": is_correct, "needs_review": pred_med.get("needs_review")})

    return {
        "weighted_accuracy": round(weighted_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "hallucinated_fields": hallucinated,
        "missed_fields": missed,
        "field_results": field_results,
        "cer": cer,
        "wer": wer,
        "calibration_points": calibration_points,
    }


# --------------------------------------------------------------------------
# Dataset-level evaluation
# --------------------------------------------------------------------------
def evaluate_dataset(ground_truth_dir: str, predictions_dir: str) -> Dict:
    """
    Matches files by stem: ground_truth_dir/1.json <-> predictions_dir/1.json
    (predictions_dir should point at your outputs/final/ folder).
    """
    gt_dir, pred_dir = Path(ground_truth_dir), Path(predictions_dir)
    # Skip the combined "ground_truth.json" (holds all images keyed by
    # filename, e.g. {"1.jpg": {...}}) — only read the per-image files
    # (1.json, 2.json, ...) which each hold a single record directly.
    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.json")
        if not p.stem.startswith("_") and p.stem != "ground_truth"
    }
    pred_files = {p.stem: p for p in pred_dir.glob("*.json") if not p.stem.startswith("_")}

    common = sorted(set(gt_files) & set(pred_files))
    missing_predictions = sorted(set(gt_files) - set(pred_files))
    missing_ground_truth = sorted(set(pred_files) - set(gt_files))

    per_image: Dict[str, Dict] = {}
    all_calibration_points: List[Dict] = []
    for stem in common:
        gt_raw = json.loads(gt_files[stem].read_text(encoding="utf-8"))
        # Per-image GT files are wrapped as {"1.jpg": {...actual record...}}.
        # Unwrap it if present; otherwise assume the file is already flat.
        if len(gt_raw) == 1 and isinstance(next(iter(gt_raw.values())), dict):
            gt = next(iter(gt_raw.values()))
        else:
            gt = gt_raw
        pred = json.loads(pred_files[stem].read_text(encoding="utf-8"))
        result = evaluate_prescription(gt, pred)
        per_image[stem] = result
        all_calibration_points.extend(result["calibration_points"])

    if per_image:
        avg_accuracy = sum(r["weighted_accuracy"] for r in per_image.values()) / len(per_image)
        avg_hallucination = sum(r["hallucination_rate"] for r in per_image.values()) / len(per_image)
        cers = [r["cer"] for r in per_image.values() if r["cer"] is not None]
        wers = [r["wer"] for r in per_image.values() if r["wer"] is not None]
    else:
        avg_accuracy = avg_hallucination = 0.0
        cers = wers = []

    return {
        "num_images_evaluated": len(common),
        "missing_predictions": missing_predictions,
        "missing_ground_truth": missing_ground_truth,
        "avg_weighted_accuracy": round(avg_accuracy, 4),
        "avg_hallucination_rate": round(avg_hallucination, 4),
        "avg_cer": round(sum(cers) / len(cers), 4) if cers else None,
        "avg_wer": round(sum(wers) / len(wers), 4) if wers else None,
        "calibration": summarize_calibration(all_calibration_points),
        "per_image": per_image,
    }


def summarize_calibration(points: List[Dict]) -> Dict:
    """Buckets predictions by confidence and reports actual accuracy per bucket —
    a well-calibrated system should show accuracy roughly tracking the bucket."""
    if not points:
        return {"buckets": [], "needs_review_precision": None, "needs_review_recall": None}

    buckets_def = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    buckets = []
    for lo, hi in buckets_def:
        in_bucket = [p for p in points if lo <= p["confidence"] < hi]
        if not in_bucket:
            continue
        acc = sum(1 for p in in_bucket if p["correct"]) / len(in_bucket)
        buckets.append({"range": f"{lo:.1f}-{hi:.1f}", "n": len(in_bucket), "accuracy": round(acc, 4)})

    # needs_review as a predictor of "this field is wrong"
    flagged = [p for p in points if p.get("needs_review")]
    wrong = [p for p in points if not p["correct"]]
    flagged_and_wrong = [p for p in points if p.get("needs_review") and not p["correct"]]
    precision = len(flagged_and_wrong) / len(flagged) if flagged else None
    recall = len(flagged_and_wrong) / len(wrong) if wrong else None

    return {
        "buckets": buckets,
        "needs_review_precision": round(precision, 4) if precision is not None else None,
        "needs_review_recall": round(recall, 4) if recall is not None else None,
    }