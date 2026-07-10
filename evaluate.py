from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation import evaluate_dataset

GROUND_TRUTH_DIR = "ground_truth"
PREDICTIONS_DIR = "outputs/final"
REPORT_OUT = "outputs/eval_report.json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate prescription pipeline output against ground truth.")
    parser.add_argument("--ground-truth", default=GROUND_TRUTH_DIR)
    parser.add_argument("--predictions", default=PREDICTIONS_DIR)
    parser.add_argument("--out", default=REPORT_OUT)
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth)
    pred_dir = Path(args.predictions)
    if not gt_dir.is_dir():
        print(f"Ground truth folder not found: {gt_dir}", file=sys.stderr)
        sys.exit(1)
    if not pred_dir.is_dir():
        print(f"Predictions folder not found: {pred_dir}", file=sys.stderr)
        sys.exit(1)

    report = evaluate_dataset(str(gt_dir), str(pred_dir))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Evaluated {report['num_images_evaluated']} image(s)", file=sys.stderr)
    if report["missing_predictions"]:
        print(f"  Missing predictions for: {report['missing_predictions']}", file=sys.stderr)
    if report["missing_ground_truth"]:
        print(f"  No ground truth for: {report['missing_ground_truth']}", file=sys.stderr)
    print(f"  Avg weighted accuracy : {report['avg_weighted_accuracy']:.2%}", file=sys.stderr)
    print(f"  Avg hallucination rate: {report['avg_hallucination_rate']:.2%}", file=sys.stderr)
    if report["avg_cer"] is not None:
        print(f"  Avg CER: {report['avg_cer']:.4f}   Avg WER: {report['avg_wer']:.4f}", file=sys.stderr)
    print(f"  Full report -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
