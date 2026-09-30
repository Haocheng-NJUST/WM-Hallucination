"""Join structured annotation labels to generated outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join structured annotation labels to paired generation outputs without inferring labels."
    )
    parser.add_argument("--generations_jsonl", required=True, help="Paired generation JSONL.")
    parser.add_argument("--annotations_jsonl", required=True, help="Structured annotation JSONL.")
    parser.add_argument("--output_jsonl", required=True, help="Joined output JSONL.")
    return parser.parse_args()


KEY_FIELDS = ("sample_id", "watermark_method", "iteration", "variant")


def annotation_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    try:
        return tuple(str(row[field]) for field in KEY_FIELDS)
    except KeyError as exc:
        raise ValueError(f"Annotation rows must include {', '.join(KEY_FIELDS)}") from exc


def main() -> None:
    args = parse_args()

    from supplement.data_io import read_jsonl, scan_for_forbidden_text, write_jsonl

    annotations = {}
    for row in read_jsonl(args.annotations_jsonl):
        scan_for_forbidden_text(row)
        key = annotation_key(row)
        if key in annotations:
            raise ValueError(
                "Duplicate annotation for "
                f"sample_id={key[0]} watermark_method={key[1]} iteration={key[2]} variant={key[3]}"
            )
        annotations[key] = {
            k: v for k, v in row.items() if k not in set(KEY_FIELDS)
        }

    joined = []
    missing = []
    for row in read_jsonl(args.generations_jsonl):
        scan_for_forbidden_text(row)
        key = tuple(str(row.get(field)) for field in KEY_FIELDS)
        annotation = annotations.get(key)
        if annotation is None:
            missing.append(key)
            annotation = {}
        joined.append({**row, "annotation": annotation})

    if missing:
        preview = ", ".join(
            f"{sample_id}/{method}/{iteration}/{variant}"
            for sample_id, method, iteration, variant in missing[:5]
        )
        raise ValueError(f"Missing annotations for {len(missing)} generation row(s): {preview}")

    write_jsonl(Path(args.output_jsonl), joined)
    print(f"Wrote {len(joined)} annotated rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
