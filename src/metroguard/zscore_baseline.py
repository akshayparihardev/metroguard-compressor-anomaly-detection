"""Z-score anomaly baseline on analog window means."""

import numpy as np
import pandas as pd

MEAN_ABS_Z = "mean_abs_z"
MAX_ABS_Z = "max_abs_z"
SCORE_NAMES = (MAX_ABS_Z, MEAN_ABS_Z)
STD_TOLERANCE = 1e-8


def analog_mean_columns(table: pd.DataFrame) -> list[str]:
    """The analog window-mean columns used by the baseline."""
    return [c for c in table.columns if c.endswith("_mean")]


def fit_zscore_reference(fit_means: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per feature from fit windows; zero-spread features get std 1."""
    std = fit_means.std(ddof=0)
    return pd.DataFrame(
        {"mean": fit_means.mean(), "std": std.where(std >= STD_TOLERANCE, 1.0)}
    ).astype(np.float64)


def zscore_scores(means: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Per window: max and mean over features of |z|, using the fit mean and std."""
    z = (means[reference.index] - reference["mean"]) / reference["std"]
    absolute = z.abs()
    return pd.DataFrame(
        {MAX_ABS_Z: absolute.max(axis=1), MEAN_ABS_Z: absolute.mean(axis=1)}
    ).astype(np.float32)
