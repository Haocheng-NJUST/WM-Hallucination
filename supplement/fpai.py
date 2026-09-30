"""User-supplied FPAI lambda calibration via table or callback."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable


def load_fpai_calibrator(table_path: str | None = None, callback_path: str | None = None) -> tuple[Callable[..., float], dict[str, Any]]:
    if bool(table_path) == bool(callback_path):
        raise ValueError("FPAI requires exactly one of --fpai_calibration_table or --fpai_calibrator")
    if callback_path:
        module_name, function_name = callback_path.split(":", 1) if ":" in callback_path else callback_path.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        return function, {"type": "callback", "path": callback_path}
    payload = json.loads(Path(table_path).read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("FPAI calibration table must contain a non-empty entries list")

    def table_calibrator(prefix_detectability: float, relative_position: float, step: int, metadata: dict) -> float:
        for entry in entries:
            if (entry["detectability_min"] <= prefix_detectability <= entry["detectability_max"] and entry["position_min"] <= relative_position <= entry["position_max"]):
                return float(entry["lambda"])
        raise ValueError("No FPAI calibration-table entry matches the current detectability and position")

    return table_calibrator, {"type": "table", "path": str(table_path), "method": payload.get("method")}

