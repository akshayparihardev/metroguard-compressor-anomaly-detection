"""Fit alert thresholds on calib scores and freeze them to disk before test scoring."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np


def format_percentile(percentile: float) -> str:
    """Stable JSON key for a percentile, e.g. 99 -> '99' and 99.5 -> '99.5'."""
    return f"{percentile:g}"


def fit_thresholds(calib_scores: np.ndarray, percentiles: list[float]) -> dict[str, float]:
    """Threshold = percentile of the score on calib windows (never on test)."""
    scores = np.asarray(calib_scores, dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("calib scores must be non-empty and finite")
    return {format_percentile(p): float(np.percentile(scores, p)) for p in percentiles}


def load_frozen_thresholds(thresholds_path: Path) -> dict:
    """Read thresholds.json, or an empty mapping when nothing is frozen yet."""
    if not thresholds_path.is_file():
        return {}
    return json.loads(thresholds_path.read_text(encoding="utf-8"))


def freeze_thresholds(
    thresholds_path: Path,
    model_name: str,
    score_name: str,
    thresholds: dict[str, float],
    n_calib_windows: int,
) -> datetime:
    """Write one model/score entry into thresholds.json with the time it was frozen."""
    frozen = load_frozen_thresholds(thresholds_path)
    frozen_at = datetime.now().replace(microsecond=0)
    frozen.setdefault(model_name, {})[score_name] = {
        "frozen_at": frozen_at.isoformat(),
        "n_calib_windows": n_calib_windows,
        "thresholds": thresholds,
    }
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8")
    return frozen_at


def alert_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    """A window raises an alert when its score is strictly above the threshold."""
    return np.asarray(scores) > threshold
