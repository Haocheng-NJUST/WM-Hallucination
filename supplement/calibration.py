"""Empirical matched TPR@FPR calibration from detector score records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def calibration_strength_fields(method: str, config: dict[str, Any]) -> tuple[str, Any]:
    """Return the method parameter used to group a strength sweep."""
    mapping = {
        "KGW": ("delta", "delta"),
        "SWEET": ("delta", "delta"),
        "DiPmark": ("alpha", "alpha"),
        "GumbelSoft": ("tau", "tau"),
        "GumbelMax": ("scale", "scale"),
        "SynthIDStyle": ("strength", "strength"),
        "TextSeal": ("configuration", None),
        "UnbiasedWatermark": ("configuration", None),
        "MorphMark": ("k_exp", "k_exp"),
    }
    name, key = mapping.get(method, ("strength", "strength"))
    if key is None:
        return name, f"{method}:standard"
    return name, config.get(key, "default")


def _score(row: dict[str, Any]) -> float:
    value = row.get("raw_score", row.get("score", row.get("detectability")))
    if isinstance(value, dict):
        value = value.get("score")
    if value is None:
        raise ValueError("Each score row needs raw_score, score, or detectability.score")
    return float(value)


def _strength(row: dict[str, Any]) -> Any:
    if "calibration_strength_name" in row and "calibration_strength_value" in row:
        return (row["calibration_strength_name"], row["calibration_strength_value"])
    if "strength" in row:
        return row["strength"]
    config = row.get("watermark_config") or row.get("config") or {}
    if isinstance(config, dict):
        for key in ("strength", "delta", "scale", "tau", "alpha", "k_exp", "p_0"):
            if key in config:
                return config[key]
    return ("configuration", "unlabeled")


def load_score_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_score"] = _score(row)
                rows.append(row)
    return rows


def empirical_threshold(clean_scores: Iterable[float], target_fpr: float = 0.01) -> float:
    values = np.asarray(list(clean_scores), dtype=float)
    if values.size == 0:
        raise ValueError("At least one clean score is required")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between 0 and 1")
    return float(np.quantile(values, 1.0 - target_fpr, method="higher"))


def calibrate_scores(
    clean_scores: Iterable[float],
    candidates: dict[Any, list[float]],
    method: str,
    target_fpr: float = 0.01,
    target_tpr: float = 0.90,
    score_direction: str = "higher",
) -> dict[str, Any]:
    clean = np.asarray(list(clean_scores), dtype=float)
    if score_direction != "higher":
        raise ValueError("Only score_direction='higher' is currently supported")
    threshold = empirical_threshold(clean, target_fpr)
    measured_fpr = float(np.mean(clean >= threshold))
    best = None
    for strength, values in candidates.items():
        watermarked = np.asarray(values, dtype=float)
        tpr = float(np.mean(watermarked >= threshold)) if watermarked.size else 0.0
        candidate = (abs(tpr - target_tpr), str(strength), strength, tpr, int(watermarked.size))
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("At least one watermarked strength candidate is required")
    return {
        "method": method,
        "target_fpr": float(target_fpr),
        "target_tpr": float(target_tpr),
        "selected_strength": best[2][1] if isinstance(best[2], tuple) else best[2],
        "selected_strength_name": best[2][0] if isinstance(best[2], tuple) else "strength",
        "threshold": threshold,
        "measured_fpr": measured_fpr,
        "measured_tpr": best[3],
        "num_clean": int(clean.size),
        "num_watermarked": best[4],
        "score_direction": score_direction,
    }


def calibrate_jsonl(clean_jsonl: str | Path, watermarked_jsonl: str | Path, target_fpr: float, target_tpr: float) -> dict[str, Any]:
    clean_rows = load_score_rows(clean_jsonl)
    watermarked_rows = load_score_rows(watermarked_jsonl)
    methods = {str(row.get("watermark_method", row.get("method", "unknown"))) for row in watermarked_rows}
    if len(methods) != 1:
        raise ValueError("Watermarked sweep must contain exactly one watermark method")
    candidates: dict[Any, list[float]] = defaultdict(list)
    for row in watermarked_rows:
        candidates[_strength(row)].append(row["_score"])
    return calibrate_scores([row["_score"] for row in clean_rows], candidates, methods.pop(), target_fpr, target_tpr)
