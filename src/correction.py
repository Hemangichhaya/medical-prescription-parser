from __future__ import annotations

import json
import os
from typing import Dict, List

import google.generativeai as genai

from src.scoring import ScoredCandidate

CORRECTION_PROMPT_TEMPLATE = """You are correcting OCR/handwriting guesses for a
prescription, using the SAME image you (or another pass) transcribed earlier.

For each line below, look at that medication line in the image and:
(a) decide which drug-name candidate (if any) is correct, and
(b) re-read and verify the frequency and duration shown on that line — these
    were only guessed once before and were never double-checked, so treat
    "raw_frequency"/"raw_duration" with the same suspicion as "raw_guess".

Rules — follow exactly:
1. drug_name_corrected: prefer a name from that line's "candidates" list if
   one plausibly matches the handwriting. If NONE of the candidates are
   plausible, but you can confidently read the actual drug name from the
   handwriting/context (even though it isn't in the candidates list),
   output your best transcription of that name instead of "NONE" — do not
   discard a confident, correct-looking read just because retrieval failed
   to find it in a database. Only output "NONE" if you are also genuinely
   unsure what the handwriting says.
2. Use the handwriting shape, letter count, and visible strength/dose on that
   line as evidence — not just which candidate looks like a "common" drug.
3. The "raw_guess" is a noisy OCR transcription and is frequently WRONG —
   it is not a safe default. If "raw_guess" does not look like a real,
   pronounceable drug name (e.g. it has odd letter/digit mixes, or doesn't
   resemble any real medication), do NOT pick it just because it's the
   literal text you see; instead pick whichever candidate best fits the
   visible handwriting shape and letter count, even if it differs
   substantially from "raw_guess". If no candidate fits either, fall back
   to rule 1's free-text best-guess behavior rather than defaulting to
   "NONE".
4. frequency_corrected: re-read the handwriting on this line and output the
   standard abbreviation it actually shows (e.g. OD, BD, TDS, QDS, PRN, HS,
   Nocte, STAT), or the literal text if it's a real frequency not on this
   list. Correct "raw_frequency" if it looks wrong for the handwriting shape.
   Output null only if genuinely illegible or absent from the line.
5. duration_corrected: re-read the handwriting and output the duration
   exactly as shown (e.g. "5 days", "5/7"), correcting "raw_duration" if it
   looks wrong. Output null only if genuinely illegible or absent.
6. llm_confidence is a float 0-1 reflecting how sure you are about the WHOLE
   line (drug name AND frequency AND duration together), based on how well
   they match the visible handwriting.

Lines to correct (JSON):
{payload_json}

Return ONLY valid JSON (no markdown fences, no commentary), a list with one
object per line_id, in this exact shape:
[
  {{
    "line_id": <integer>,
    "drug_name_corrected": "<one of that line's candidates, your best-guess free-text reading if none fit, or 'NONE' if genuinely illegible>",
    "frequency_corrected": "<string or null>",
    "duration_corrected": "<string or null>",
    "llm_confidence": <float 0-1>,
    "reasoning": "<short reason, one sentence>"
  }}
]
"""


def build_batch_payload(
    medications: List[Dict], candidates_by_line: Dict[int, List[ScoredCandidate]]
) -> List[Dict]:
    """Step 5: one JSON list across all medications:
    {line_id, raw_guess, top_3_candidates, raw_frequency, raw_duration}."""
    payload = []
    for med in medications:
        line_id = med["line_id"]
        top3 = candidates_by_line.get(line_id, [])
        payload.append(
            {
                "line_id": line_id,
                "raw_guess": med.get("drug_name"),
                "top_3_candidates": [c.name for c in top3],
                "raw_frequency": med.get("frequency"),
                "raw_duration": med.get("duration"),
            }
        )
    return payload


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def correct_with_gemini(
    image_path: str, payload: List[Dict], model: str = "gemini-3.1-flash-lite", cost_tracker=None
) -> List[Dict]:
    """
    Step 6: single batched LLM call. Sends the prescription image plus the
    full candidate payload and asks Gemini to pick per-line, or NONE.
    Returns a list of {line_id, drug_name_corrected, llm_confidence, reasoning}.
    If cost_tracker (a CostTracker) is passed, records real token usage
    from the response.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (check your .env file).")
    genai.configure(api_key=api_key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    ext = image_path.lower()
    mime = "image/png" if ext.endswith(".png") else "image/jpeg"

    prompt = CORRECTION_PROMPT_TEMPLATE.format(payload_json=json.dumps(payload, indent=2))

    gmodel = genai.GenerativeModel(model)
    response = gmodel.generate_content(
        [{"mime_type": mime, "data": image_bytes}, prompt],
        generation_config={"temperature": 0.0},
    )

    if cost_tracker is not None:
        cost_tracker.record_from_response("correction", model, response)

    return _parse_json(response.text)