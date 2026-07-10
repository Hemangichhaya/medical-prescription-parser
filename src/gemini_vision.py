"""
Step 1 — Raw extraction from the prescription image using Gemini vision.

This is the "first guess" pass. It deliberately transcribes drug names
EXACTLY as handwritten (no auto-correction) so that Step 2-4 (RxNorm /
openFDA / lexical+semantic scoring) has a clean, unbiased raw_guess to
work with. Correction happens later, in correction.py (Step 6).
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict

import google.generativeai as genai

RAW_EXTRACTION_PROMPT = """You are a medical prescription transcription assistant.
Look at this handwritten prescription image carefully and extract ONLY the
following fields. Handwriting may be cursive and hard to read — use context
(common Sri Lankan/South Asian name patterns, typical prescription layout,
standard frequency abbreviations like OD/BD/TDS/QDS/PRN/HS/Nocte) to make
your best judgement rather than defaulting to null on a partial read.

Return ONLY valid JSON (no markdown fences, no commentary) in this exact shape:

{
  "patient_name": "<string or null>",
  "age": "<string or null>",
  "date": "<string or null>",
  "doctor_name": "<string or null>",
  "hospital": "<string or null>",
  "medications": [
    {
      "line_id": "<integer, 0-indexed, one per medication line>",
      "drug_name": "<string, transcribed EXACTLY as handwritten>",
      "strength": "<e.g. '500 mg', or null>",
      "frequency": "<e.g. 'TDS', 'BD', 'OD', 'HS', 'PRN', or null>",
      "duration": "<e.g. '5 days', '5/7', or null>",
      "raw_line": "<the full raw text of this medication line, as written>",
      "confidence": <float 0-1, genuine uncertainty about this line's handwriting>
    }
  ],
  "patient_name_confidence": <float 0-1>,
  "age_confidence": <float 0-1>
}

Rules:
- Scan the ENTIRE image from top to bottom BEFORE answering. List every
  medication line exactly once, even if there are more than 3 — do not stop
  early or truncate to a "typical" number of lines just because the first
  few looked complete. If a line's handwriting is too faint to fully read,
  still include it (with a lower confidence) rather than omitting it.
- The patient name is written near the top of the prescription, typically
  after an abbreviation like "Pt.", "Name:", or directly following the date
  line — NOT the doctor's signature or name at the bottom of the page
  (which is often preceded by "Dr." and followed by credentials like MBBS).
- If you are unsure whether text belongs to the patient name or another
  field (doctor, hospital), prefer null over guessing a value from the
  wrong part of the image.
- Transcribe drug_name EXACTLY as handwritten, letter by letter. Do NOT
  expand, correct, "clean up", or substitute it with a different brand or
  generic name you think it might mean — that correction step happens later
  in the pipeline, with proper drug-database lookups. Your only job here is
  a faithful raw transcription.
- Only use null for patient_name/age/date/doctor_name/hospital if the field
  is truly absent from the image or completely illegible even with context —
  attempt a best-effort transcription for anything partially legible, and
  reflect your uncertainty in the confidence score instead of returning null.
- "x 5/7" style duration means "for 5 days" (the "/7" is a per-week
  denominator) — read the digit before the slash exactly as written.
- Only report a strength if a number is actually visible near the drug name.
- confidence fields must reflect genuine uncertainty, not be a placeholder.
"""


def _parse_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def extract_raw_prescription(
    image_path: str, model: str = "gemini-3.1-flash-lite", cost_tracker=None
) -> Dict:
    """
    Run Gemini vision extraction on a single prescription image.
    Returns the raw (uncorrected) structured JSON plus a `_metrics` block.
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

    gmodel = genai.GenerativeModel(model)

    start = time.perf_counter()
    response = gmodel.generate_content(
        [{"mime_type": mime, "data": image_bytes}, RAW_EXTRACTION_PROMPT],
        generation_config={"temperature": 0.1},
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    if cost_tracker is not None:
        cost_tracker.record_from_response("vision_extraction", model, response)

    result = _parse_json(response.text)

    # Guarantee every medication has a stable line_id even if the model
    # forgot to number them.
    for i, med in enumerate(result.get("medications", [])):
        med.setdefault("line_id", i)

    result["_metrics"] = {
        "provider": "gemini",
        "model": model,
        "stage": "raw_extraction",
        "latency_ms": latency_ms,
        "image_path": image_path,
    }
    return result


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/1.jpg"
    out = extract_raw_prescription(path)
    print(json.dumps(out, indent=2))
