# Medical Prescription Parser

Gemini vision-language-model extraction → RxNorm/openFDA candidate
retrieval → hybrid lexical+semantic scoring → single batched Gemini
correction call (drug name + strength/frequency/duration).

## Approach

The problem statement allows classical OCR (Tesseract), cloud OCR (Google
Vision, AWS Textract, Azure Form Recognizer), vision-language models
(GPT-4V, Claude, Gemini), or a hybrid. This project uses a **VLM + RAG
hybrid**: Gemini does the actual handwriting reading (VLMs meaningfully
outperform classical/cloud OCR on messy cursive handwriting and irregular
layouts, since they reason over the whole image rather than segmenting
characters), and a retrieval-augmented correction pass grounds Gemini's
drug-name guesses against real drug databases before a final LLM re-check —
so the model isn't just trusting its own first-pass read.

### Why Gemini over GPT-4V / Claude (cost)

All three vision-language models named in the brief are viable for the
handwriting-reading step; Gemini was chosen mainly on cost. Published
per-million-token rates as of July 2026, cheapest current vision-capable
tier from each provider:

| Provider | Model | Input $/M | Output $/M |
|---|---|---|---|
| **Google (used here)** | gemini-3.1-flash-lite | **$0.25** | **$1.50** |
| OpenAI | gpt-4o (the closest current match to "GPT-4V" — GPT-4.1 dropped vision support) | $2.50 | $10.00 |
| Anthropic | claude-haiku-4.5 | $1.00 | $5.00 |

Scaling this project's own measured Step 1 (vision extraction) cost —
$0.00098/image on Gemini — by each provider's rate gives a rough
order-of-magnitude estimate for swapping the vision model:

| Provider | Est. cost/image (Step 1 only) | Est. cost at 100,000 images |
|---|---|---|
| Gemini (measured) | $0.00098 | $98 |
| Claude Haiku 4.5 (estimated) | ~$0.0034 | ~$340 |
| GPT-4o (estimated) | ~$0.0070 | ~$700 |

These GPT-4o/Claude figures are estimates, not measured — they assume
roughly the same input/output token split as the real Gemini run, which
won't be exactly true since each provider tokenizes images and text
differently. Treat them as "same order of magnitude, Gemini clearly
cheapest," not exact figures. See the "Cost" section further down for
this project's actual measured Gemini pricing and full-pipeline projection.

## What it does (pipeline steps)

1. **Raw extraction** (`src/gemini_vision.py`) — Gemini reads the
   prescription image and transcribes each medication line exactly as
   handwritten, into `drug_name` (raw guess), `strength`, `frequency`,
   `duration`.
2. **Candidate retrieval** (`src/candidate_retrieval.py`) — for each raw
   guess, query RxNorm's `approximateTerm.json` and openFDA's `ndc.json`,
   merge and dedupe into one candidate pool. If both come back empty,
   fall back to drugs.com/1mg scraping (see "What happens if a drug isn't
   in RxNorm or openFDA?" below).
3. **Hybrid scoring** (`src/scoring.py`) — for every candidate:
   `hybrid_score = 0.7 * lexical_score + 0.3 * semantic_score`
   - lexical = max(RapidFuzz `ratio`, `partial_ratio`, `token_sort_ratio`)
   - semantic = cosine similarity between Gemini `text-embedding-004`
     embeddings of the raw guess and the candidate name
4. **Filter + rank** — drop candidates below `score_threshold`, keep the
   top `top_k` (default 3) per drug.
5. **Batch payload** (`src/correction.py`) — one JSON list across every
   medication: `{line_id, raw_guess, top_3_candidates, strength, frequency, duration, raw_line}`.
6. **One LLM call** — the prescription image + the full batched payload go
   to Gemini in a single request. For lines with candidates, it picks the
   correct one or returns `"NONE"` — it can never invent a name outside the
   candidate list. For lines with zero candidates, it does a **free
   re-read** of that line instead of being skipped. It also re-checks
   `strength`/`frequency`/`duration` against the image on every line.
7. **Merge** (`src/pipeline.py`) — each medication gets
   `drug_name_raw`, `drug_name_corrected`, `candidates_considered`,
   `hybrid_scores`, `llm_confidence`, `needs_review`, `correction_mode`,
   plus `strength_raw`/`frequency_raw`/`duration_raw`.
8. **Final output** — `drug_name`, `strength`, `frequency`, `duration` are
   all replaced with their corrected values; the original Step 1 values
   are preserved in the `*_raw` fields alongside them.

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
runs automatically for every image inside, no flags required:

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

All settings (output directory, score threshold, top-k, semantic scoring
on/off, scraping on/off, model names) are constants at the top of
`main.py` — edit them there instead of passing terminal flags.

## Output shape

Each medication in the output JSON looks like:

```json
{
  "line_id": 0,
  "drug_name": "Amoxicillin",
  "drug_name_raw": "Amoxicilin",
  "drug_name_corrected": "Amoxicillin",
  "strength": "500 mg",
  "strength_raw": "500 mg",
  "frequency": "TDS",
  "frequency_raw": "BD",
  "duration": "5 days",
  "duration_raw": "5 days",
  "raw_line": "Amoxicilin 500mg TDS x5/7",
  "candidates_considered": ["Amoxicillin", "Amoxicillin/Clavulanate", "Amlodipine"],
  "hybrid_scores": {"Amoxicillin": 94.3, "Amoxicillin/Clavulanate": 71.2, "Amlodipine": 58.9},
  "llm_confidence": 0.92,
  "llm_reasoning": "Handwriting matches Amoxicillin; strength 500mg is a standard amoxicillin dose.",
  "needs_review": false,
  "correction_mode": "candidate_match"
}
```

`needs_review` is `true` whenever there were no candidates above threshold
(i.e. `correction_mode: "free_reread"`), the model returned `"NONE"`, or
`llm_confidence` came back below 0.6 — treat those lines as ones a human
should double check.

## What happens if a drug isn't in RxNorm or openFDA?

By default (`SCRAPE_AS_FALLBACK = True` in `main.py`), if RxNorm + openFDA
together return zero candidates for a drug, the pipeline falls back to
drugs.com/1mg scraping for that drug.

If a line's `candidates_considered` is still empty after all of that, Step 6
switches to a **free re-read mode**: instead of skipping the line entirely,
the LLM correction call looks at that exact line in the image again and
gives its own best transcription, grounded only in what's actually visible —
not picked from a list, but not invented either. This is recorded as
`correction_mode: "free_reread"` on that medication, and always gets
`needs_review: true` regardless of the model's confidence, since nothing
external confirmed the drug actually exists under that name.

## A note on model names

Google retires Gemini model versions on a rolling basis — `gemini-2.5-flash`
was cut off for new API keys on June 17, 2026. This project defaults to
`gemini-3.1-flash-lite` (the current cheapest GA multimodal model). If
handwriting accuracy needs more than Flash-Lite gives you, bump
`VISION_MODEL`/`CORRECTION_MODEL` at the top of `main.py` to
`gemini-3.5-flash` (pricier, stronger). If you hit a `404 ... no longer
available` error again in the future, it means the model name has been
retired again — check https://ai.google.dev/gemini-api/docs/models for the
current lineup and swap the constant.

## Cost

Gemini has a free tier (limited requests/minute and /day) and a paid tier
billed per token. This project's cost tracker always estimates against
**paid-tier rates**, since free-tier quotas make processing any real volume
(1K–100K images) physically impossible — so "cost at scale" is inherently
a paid-tier question.

`gemini-3.1-flash-lite` pricing: **$0.25 / M input tokens, $1.50 / M output
tokens**. Embeddings (`text-embedding-004`) are priced at $0.00 — that
endpoint has historically been free even on the paid tier, but verify at
https://ai.google.dev/gemini-api/docs/pricing before relying on it for a
large bill, since prices change.

**Measured on this project's 6 test images** (`outputs/costs/_summary.json`):

| Metric | Value |
|---|---|
| Avg cost / image | $0.00195 |
| Vision extraction (Step 1) total | $0.005896 |
| Correction (Step 6) total | $0.00580325 |
| **Projected at 1,000 images** | **$1.95** |
| **Projected at 10,000 images** | **$19.50** |
| **Projected at 100,000 images** | **$194.99** |

Every run writes this automatically: `outputs/costs/<name>.json` per image
(full per-call breakdown, real token counts from `usage_metadata` where the
API returns them), and `outputs/costs/_summary.json` after a folder run
(average, total, and the 1K/10K/100K projection above). Pricing is
hardcoded in `src/cost_tracker.py` — update `GEMINI_PRICING` there if the
official rates change.

## Accuracy

Evaluated against 6 hand-labeled ground truth prescriptions
(`outputs/eval_report.json`), using `python evaluate.py`:

| Field | Correct | Hallucinated | Missed | Total |
|---|---|---|---|---|
| `patient_name` (weight 1) | 17% (1/6) | 83% (5/6) | 0 | 6 |
| `age` (weight 1) | 100% (6/6) | 0% (0/6) | 0 | 6 |
| `drug_name` (weight 3) | 72% (13/18) | 22% (4/18) | 1 | 18 |
| `strength` (weight 2) | 72% (13/18) | 11% (2/18) | 3 | 18 |
| `frequency` (weight 2) | 56% (10/18) | 39% (7/18) | 1 | 18 |
| `duration` (weight 2) | 67% (12/18) | 22% (4/18) | 2 | 18 |
| **All fields** | **65%** (55/84) | **26%** (22/84) | **7** | **84** |

| Metric | Value | What it means |
|---|---|---|
| **Weighted accuracy** | **68.2%** | Field-level match rate, weighted 3x for `drug_name`, 2x for `strength`/`frequency`/`duration`, 1x for `patient_name`/`age` |
| **Hallucination rate** | **26.1%** | Fraction of fields where the pipeline produced a value not supported by ground truth (fabricated or wrong) — tracked separately from safely "missed" (left blank/flagged) fields |

**Calibration** (does `llm_confidence` track real correctness on `drug_name`):

Calibration asks a different question than raw accuracy: not "is the model
usually right?" but "when the model *says* it's confident, is it actually
right more often than when it says it's unsure?" A well-calibrated model's
confidence score is trustworthy — you can use it to decide which lines
need a human to double-check. A badly-calibrated model might say "0.9
confident" on lines that are wrong just as often as its "0.3 confident"
lines, which would make the confidence score useless for triage.

To measure this, every `drug_name` prediction is grouped into a confidence
bucket (0.0–0.2, 0.2–0.4, etc.), and for each bucket we check what
fraction of predictions in it were actually correct against ground truth:

| Confidence bucket | n | Accuracy |
|---|---|---|
| 0.4–0.6 | 13 | 69.2% |
| 0.8–1.0 | 4 | 100% |

`n` is how many `drug_name` predictions fell into that bucket. Reading the
table: the 13 lines where the model said it was only 40–60% confident were
right 69.2% of the time, while the 4 lines where it said 80–100% confident
were right 100% of the time. That's the desired pattern — accuracy rises
with confidence, meaning `llm_confidence` is a genuinely useful signal for
deciding which lines to route to a human reviewer, not just a number the
model outputs without it meaning anything.

Higher confidence does correspond to higher accuracy in this sample, though
with only 6 images the bucket sizes are small — treat this as a directional
signal, not a statistically robust calibration curve.

`python evaluate.py` re-runs this against `ground_truth/*.json` vs
`outputs/final/*.json` any time you add more labeled images — see the
"Accuracy evaluation" section below to add your own.

## Accuracy evaluation (how to add more test data)

1. Create one `ground_truth/<image_stem>.json` per image (e.g.
   `ground_truth/1.json` for `Handwritten Docs/1.jpg`) and hand-label
   what's actually on the prescription. Leave a field `null` if it's
   genuinely illegible — don't guess; null-vs-null counts as correct.
2. Run the pipeline (`python main.py`) so `outputs/final/*.json` exists.
3. Run `python evaluate.py` (folder names are constants at the top of that
   file, same edit-the-file pattern as `main.py`). Writes
   `outputs/eval_report.json` and prints a summary.

Medications are aligned to ground truth by drug-name similarity (not by
list position), since a mis-segmented line shouldn't cause every
subsequent medication in that prescription to look wrong. Durations like
`"5/7"` and `"5 days"` are recognized as equivalent via a canonical
day-count normalizer, so format differences don't get penalized as errors.

## Notes / caveats

- **RxNorm and openFDA** are official, free, no-key-required government
  APIs — safe to rely on. They're US-centric, so regional/international
  brand names (common in this dataset) often return nothing from them alone.
- **drugs.com / 1mg** candidates come from best-effort HTML scraping of
  their public search pages, not an official API. There's no guaranteed
  contract: page structure can change and break the parser (it fails soft,
  returning no candidates rather than crashing), and scraping may be
  subject to each site's Terms of Service — review those before enabling
  it for anything beyond casual/personal use.
- This tool assists transcription — it is **not** a substitute for
  pharmacist/clinician verification of a prescription before dispensing.
