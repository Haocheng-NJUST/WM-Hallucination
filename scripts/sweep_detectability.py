"""Calibrate a named strength sweep against one shared clean-score file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the candidate closest to target TPR at empirical FPR.")
    parser.add_argument("--clean_jsonl", required=True, help="JSONL of clean detector scores reused for every candidate.")
    parser.add_argument("--watermarked_jsonl", default=None, help="JSONL containing all candidate configurations and scores.")
    parser.add_argument("--method", default=None, help="Method name to validate, and generation method in candidate mode.")
    parser.add_argument("--candidate_configs", default=None, help="JSON list of CLI option dictionaries for sequential generation.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--input_jsonl", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=150)
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--target_tpr", type=float, default=0.90)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    from supplement.calibration import calibrate_jsonl, load_score_rows

    generated_sweep = args.watermarked_jsonl
    if generated_sweep is None:
        if not all((args.candidate_configs, args.method, args.model_name_or_path, args.input_jsonl, args.output_dir)):
            parser.error("provide --watermarked_jsonl, or provide --candidate_configs, --method, --model_name_or_path, --input_jsonl, and --output_dir")
        candidates = json.loads(Path(args.candidate_configs).read_text(encoding="utf-8"))
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("--candidate_configs must contain a non-empty JSON list")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged = output_dir / "watermarked_strength_sweep.jsonl"
        with merged.open("w", encoding="utf-8") as destination:
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, dict):
                    raise ValueError("Each candidate config must be a JSON object")
                if "calibration_strength_name" not in candidate or "calibration_strength_value" not in candidate:
                    raise ValueError("Each candidate must explicitly set calibration_strength_name and calibration_strength_value")
                candidate_path = output_dir / f"candidate_{index}.jsonl"
                command = [sys.executable, str(PROJECT_ROOT / "scripts" / "generate_paired_outputs.py"),
                           "--model_name_or_path", args.model_name_or_path, "--input_jsonl", args.input_jsonl,
                           "--output_jsonl", str(candidate_path), "--method", args.method,
                           "--max_new_tokens", str(args.max_new_tokens)]
                if args.device:
                    command += ["--device", args.device]
                for key, value in candidate.items():
                    command += [f"--{key}", str(value)]
                subprocess.run(command, check=True)
                for row in load_score_rows(candidate_path):
                    if row.get("variant") == "watermarked":
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        generated_sweep = str(merged)

    record = calibrate_jsonl(args.clean_jsonl, generated_sweep, args.target_fpr, args.target_tpr)
    if args.method is not None and record["method"] != args.method:
        raise ValueError(f"Sweep method is {record['method']!r}, expected {args.method!r}")
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
