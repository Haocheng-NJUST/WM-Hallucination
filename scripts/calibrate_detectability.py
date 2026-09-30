"""Select watermark strength using an empirical 1% FPR threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate matched TPR at an empirical FPR threshold.")
    parser.add_argument("--clean_jsonl", required=True)
    parser.add_argument("--watermarked_jsonl", required=True)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--target_tpr", type=float, default=0.90)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    from supplement.calibration import calibrate_jsonl

    record = calibrate_jsonl(args.clean_jsonl, args.watermarked_jsonl, args.target_fpr, args.target_tpr)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

