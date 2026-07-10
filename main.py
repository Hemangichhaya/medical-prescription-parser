from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv(Path(__file__).resolve().parent / ".env")

from src.pipeline import run_pipeline, run_pipeline_on_folder

IMAGE_FOLDER = "Handwritten Docs"
OUT_DIR = "outputs"
SCORE_THRESHOLD = 40.0
TOP_K = 3
USE_SEMANTIC = True            # False = skip embedding calls, lexical-only scoring
USE_SCRAPED_SOURCES = False    # True = always also query drugs.com / 1mg
SCRAPE_AS_FALLBACK = True      # True = only query drugs.com / 1mg when RxNorm+openFDA found nothing for a drug
VISION_MODEL = "gemini-3.1-flash-lite"      # cheapest current GA multimodal model ($0.25/$1.50 per M tokens)
CORRECTION_MODEL = "gemini-3.1-flash-lite"  # bump to "gemini-3.5-flash" for tougher handwriting if accuracy needs it


def main():
    parser = argparse.ArgumentParser(description="Extract + RAG-correct handwritten prescription images.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=IMAGE_FOLDER,
        help=f"Folder of prescription images to process (default: '{IMAGE_FOLDER}', set at top of main.py)",
    )
    parser.add_argument("--image", default=None, help="Process a single image instead of a folder")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Where raw/ and final/ JSON get written")
    parser.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD, help="Min hybrid score to keep a candidate")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Candidates kept per drug")
    parser.add_argument("--vision-model", default=VISION_MODEL)
    parser.add_argument("--correction-model", default=CORRECTION_MODEL)
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        default=not USE_SEMANTIC,
        help="Disable embedding-based semantic scoring",
    )
    parser.add_argument(
        "--use-scraped-sources",
        action="store_true",
        default=USE_SCRAPED_SOURCES,
        help="Also pull candidates from drugs.com/1mg (best-effort scraping, off by default)",
    )
    args = parser.parse_args()

    common_kwargs = dict(
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        use_semantic=not args.no_semantic,
        use_scraped_sources=args.use_scraped_sources,
        scrape_as_fallback=SCRAPE_AS_FALLBACK,
        vision_model=args.vision_model,
        correction_model=args.correction_model,
    )

    if args.image:
        # ---- Single-file mode --------------------------------------------
        stem = Path(args.image).stem
        result = run_pipeline(
            args.image,
            save_raw_path=str(Path(args.out_dir) / "raw" / f"{stem}.json"),
            save_final_path=str(Path(args.out_dir) / "final" / f"{stem}.json"),
            **common_kwargs,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # ---- Folder / batch mode (default) ------------------------------------
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing every image in: {folder}", file=sys.stderr)
    print(f"Raw extraction  -> {args.out_dir}/raw/<name>.json", file=sys.stderr)
    print(f"Final corrected -> {args.out_dir}/final/<name>.json", file=sys.stderr)

    results = run_pipeline_on_folder(str(folder), out_dir=args.out_dir, **common_kwargs)
    print(f"\nDone. {len(results)} image(s) processed.", file=sys.stderr)


if __name__ == "__main__":
    main()
