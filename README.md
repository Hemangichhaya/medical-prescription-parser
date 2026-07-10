# Prescription RAG Corrector

Gemini vision extraction → RxNorm/openFDA candidate retrieval → hybrid
lexical+semantic scoring → single batched Gemini correction call.

## What it does (pipeline steps)

1. **Raw extraction** (`src/gemini_vision.py`) — Gemini reads the
   prescription image and transcribes each medication line exactly as
   handwritten, into `drug_name` (raw guess), `strength`, `frequency`,
   `duration`.
2. **Candidate retrieval** (`src/candidate_retrieval.py`) — for each raw
   guess, query RxNorm's `approximateTerm.json` and openFDA's `ndc.json`,
   merge and dedupe into one candidate pool. drugs.com / 1mg scraping is
   supported but **off by default** (see caveat below).
3. **Hybrid scoring** (`src/scoring.py`) — for every candidate:
   `hybrid_score = 0.7 * lexical_score + 0.3 * semantic_score`
   - lexical = max(RapidFuzz `ratio`, `partial_ratio`, `token_sort_ratio`)
   - semantic = cosine similarity between Gemini `text-embedding-004`
     embeddings of the raw guess and the candidate name
4. **Filter + rank** — drop candidates below `--score-threshold`, keep the
   top `--top-k` (default 3) per drug.
5. **Batch payload** (`src/correction.py`) — one JSON list across every
   medication: `{line_id, raw_guess, top_3_candidates, raw_frequency, raw_duration}`.
6. **One LLM call** — the prescription image + the full batched payload go
   to Gemini in a single request. It picks the correct drug-name candidate
   per line by re-reading the handwriting (or returns `"NONE"` — it can
   never invent a name outside the candidate list), and separately
   re-verifies `frequency`/`duration` against the handwriting, since those
   were only guessed once in step 1 and are otherwise never double-checked.
7. **Merge** (`src/pipeline.py`) — each medication gets
   `drug_name_raw`, `drug_name_corrected`, `frequency_raw`, `duration_raw`,
   `candidates_considered`, `hybrid_scores`, `llm_confidence`, `needs_review`.
   The final `drug_name` is chosen by blending the LLM's pick with the RAG
   hybrid (lexical+semantic) score of that candidate — if the LLM says
   `"NONE"` but the hybrid score still found a strong match, that match is
   used instead of falling back to the noisy raw OCR guess (the line stays
   flagged via `needs_review` since the two methods disagreed).
8. **Final output** — `drug_name`, `frequency`, and `duration` are all
   replaced with their re-verified values; `strength` is left untouched
   from step 1.

## Setup

```bash
cd prescription-rag-corrector
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your free Gemini API key from
https://aistudio.google.com/apikey (no credit card required):

```
GEMINI_API_KEY=your_key_here
```

## Run

**Default usage — just point it at a folder of images.** Every step (raw
extraction, RxNorm/openFDA lookup, hybrid scoring, LLM correction, merge)
runs automatically for every image inside, no other flags required:

```bash
python main.py "Handwritten Docs"
```

Run with no arguments at all and it looks for a folder literally named
`Handwritten Docs` in the current directory:

```bash
python main.py
```

For each image `name.jpg` it writes two files:

```
outputs/raw/name.json     <- Step 1 output, before any correction
outputs/final/name.json   <- Step 8 output, the fully corrected result
```

Single-file mode is still available if you only want one image:

```bash
python main.py --image path/to/prescription.jpg
```

Useful flags (all optional — sensible defaults are applied automatically):

| Flag | Default | Purpose |
|---|---|---|
| `--out-dir` | outputs | Where `raw/` and `final/` JSON get written |
| `--score-threshold` | 55.0 | Minimum hybrid score (0-100) to keep a candidate |
| `--top-k` | 3 | Candidates kept per drug before the LLM correction call |
| `--no-semantic` | off | Skip embedding calls — lexical-only scoring (faster, no embedding quota used) |
| `--use-scraped-sources` | off | Also query drugs.com / 1mg (see caveat below) |
| `--vision-model` | gemini-3.1-flash-lite | Model for step 1 |
| `--correction-model` | gemini-3.1-flash-lite | Model for step 6 |

## Output shape

Each medication in the output JSON looks like:

```json
{
  "line_id": 0,
  "drug_name": "Amoxicillin",
  "drug_name_raw": "Amoxicilin",
  "drug_name_corrected": "Amoxicillin",
  "strength": "500 mg",
  "frequency": "TDS",
  "duration": "5 days",
  "raw_line": "Amoxicilin 500mg TDS x5/7",
  "candidates_considered": ["Amoxicillin", "Amoxicillin/Clavulanate", "Amlodipine"],
  "hybrid_scores": {"Amoxicillin": 94.3, "Amoxicillin/Clavulanate": 71.2, "Amlodipine": 58.9},
  "llm_confidence": 0.92,
  "llm_reasoning": "Handwriting matches Amoxicillin; strength 500mg is a standard amoxicillin dose.",
  "needs_review": false
}
```

`needs_review` is `true` whenever there were no candidates above threshold,
the model returned `"NONE"`, or `llm_confidence` came back below 0.6 — treat
those lines as ones a human should double check.

## What happens if a drug isn't in RxNorm or openFDA?

By default (`SCRAPE_AS_FALLBACK = True` in `main.py`), if RxNorm + openFDA
together return zero candidates for a drug, the pipeline automatically
falls back to querying drugs.com/1mg for that drug only — so an unusual or
regional name still gets a shot at a match. `USE_SCRAPED_SOURCES` is a
separate, stronger setting: turning that on queries drugs.com/1mg for
*every* drug, not just the ones RxNorm/openFDA missed.

If even the fallback finds nothing, the pipeline keeps Gemini's raw
handwriting guess as `drug_name_corrected`, sets `needs_review: true`, and
`llm_confidence: 0.0` — so it's clearly flagged for a human to check rather
than silently guessing.

## A note on model names

Google retires Gemini model versions on a rolling basis — `gemini-2.5-flash`
was cut off for new API keys on June 17, 2026. This project defaults to
`gemini-3.1-flash-lite` (the current cheapest GA multimodal model, $0.25/$1.50
per M tokens). If handwriting accuracy needs more than Flash-Lite gives you,
bump `VISION_MODEL`/`CORRECTION_MODEL` at the top of `main.py` to
`gemini-3.5-flash` (pricier, stronger). If you hit a `404 ... no longer
available` error again in the future, it means the model name has been
retired again — check https://ai.google.dev/gemini-api/docs/models for the
current lineup and swap the constant.

## Cost tracking

Every run automatically tracks Gemini API cost per image:

- `outputs/costs/<name>.json` — full per-call breakdown (vision extraction,
  embeddings, correction), using real token counts from `usage_metadata`
  where the API returns them, and a `~4 chars/token` estimate (marked
  `"estimated": true`) for calls that don't (currently just embeddings).
- `outputs/costs/_summary.json` — aggregated after a folder run: average
  cost/image, total cost, and a **projection to 1K / 10K / 100K images**.
- The final output JSON also carries a lightweight `_cost_usd` total per image.

Pricing is hardcoded in `src/cost_tracker.py` from the published Gemini API
rates. Prices change — check https://ai.google.dev/gemini-api/docs/pricing
and update `GEMINI_PRICING` there if your numbers look stale.

## Accuracy evaluation

1. Copy `ground_truth/_template.json` to `ground_truth/<image_stem>.json`
   for each image (e.g. `ground_truth/1.json` for `Handwritten Docs/1.jpg`)
   and hand-label what's actually on the prescription. Leave a field `null`
   if it's genuinely illegible — don't guess; null-vs-null counts as correct.
2. Run the pipeline (`python main.py`) so `outputs/final/*.json` exists.
3. Run:

```bash
python evaluate.py
```

(Folder names are set as constants at the top of `evaluate.py`, same
edit-the-file pattern as `main.py`.) This writes `outputs/eval_report.json`
and prints a summary. It reports:

- **weighted_accuracy** — field-level match rate, weighted higher for
  `drug_name`/`strength`/`frequency`/`duration` than `patient_name`/`age`
- **hallucination_rate** — fraction of fields where the pipeline produced
  a value not supported by ground truth (fabricated or wrong), tracked
  separately from **missed** fields (correctly left blank/flagged instead
  of guessed — the safer failure mode)
- **CER / WER** — character/word error rate on raw transcribed text, if
  your ground truth includes `raw_ocr_text` or medication `raw_line`s
- **calibration** — buckets predictions by `llm_confidence` and reports
  actual accuracy per bucket (a well-calibrated system's buckets should
  roughly track their labels), plus precision/recall of the `needs_review`
  flag as a predictor of an actually-wrong field

Medications are aligned to ground truth by drug-name similarity (not by
list position), since a mis-segmented line shouldn't cause every
subsequent medication in that prescription to look wrong.

## Notes / caveats

- **RxNorm and openFDA** are official, free, no-key-required government
  APIs — safe to rely on.
- **drugs.com / 1mg** candidates come from best-effort HTML scraping of
  their public search pages, not an official API. There's no guaranteed
  contract: page structure can change and break the parser (it fails soft,
  returning no candidates rather than crashing), and scraping may be
  subject to each site's Terms of Service — review those before enabling
  `--use-scraped-sources` for anything beyond casual/personal use. RxNorm +
  openFDA alone already cover the large majority of real-world drug names.
- This tool assists transcription — it is **not** a substitute for
  pharmacist/clinician verification of a prescription before dispensing.
