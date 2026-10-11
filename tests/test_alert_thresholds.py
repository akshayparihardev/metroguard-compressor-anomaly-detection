import numpy as np
import pytest

from metroguard.alert_thresholds import (
    alert_mask,
    fit_thresholds,
    freeze_thresholds,
    load_frozen_thresholds,
)


def test_threshold_is_percentile_of_calib_scores():
    calib = np.arange(1, 101, dtype=np.float32)
    thresholds = fit_thresholds(calib, [99, 99.5])
    assert thresholds["99"] == pytest.approx(np.percentile(calib, 99))
    assert thresholds["99.5"] > thresholds["99"]


def test_threshold_ignores_everything_except_calib_scores():
    calib = np.linspace(0.0, 1.0, 1000)
    before = fit_thresholds(calib, [99])
    np.random.default_rng(0).normal(size=10_000)
    assert fit_thresholds(calib.copy(), [99]) == before


def test_invalid_calib_scores_are_rejected():
    with pytest.raises(ValueError):
        fit_thresholds(np.array([]), [99])
    with pytest.raises(ValueError):
        fit_thresholds(np.array([1.0, np.nan]), [99])


def test_alert_requires_score_strictly_above_threshold():
    assert alert_mask(np.array([1.0, 2.0, 3.0]), 2.0).tolist() == [False, False, True]


def test_freezing_merges_models_and_records_time(tmp_path):
    path = tmp_path / "thresholds.json"
    freeze_thresholds(path, "zscore_baseline", "max_abs_z", {"99": 4.2}, 7100)
    freeze_thresholds(path, "rbm_k1", "free_energy", {"99": 1.5}, 7100)
    frozen = load_frozen_thresholds(path)
    assert set(frozen) == {"zscore_baseline", "rbm_k1"}
    entry = frozen["zscore_baseline"]["max_abs_z"]
    assert entry["thresholds"] == {"99": 4.2}
    assert entry["n_calib_windows"] == 7100
    assert "frozen_at" in entry
