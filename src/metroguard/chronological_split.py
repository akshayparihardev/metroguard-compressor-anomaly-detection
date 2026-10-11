"""Chronological fit / fit_validation / calib / test split of time windows."""

import numpy as np
import pandas as pd

FIT_TRAIN = "fit_train"
FIT_VALIDATION = "fit_validation"
CALIB = "calib"
TEST = "test"
SPLIT_NAMES = (FIT_TRAIN, FIT_VALIDATION, CALIB, TEST)
FIT_SPLITS = (FIT_TRAIN, FIT_VALIDATION)
SUPPORTED_TRAIN_PERIOD = "first_calendar_month"


def training_period_end(first_timestamp: pd.Timestamp, train_period: str) -> pd.Timestamp:
    """Exclusive end of the training period (start of the month after the first month)."""
    if train_period != SUPPORTED_TRAIN_PERIOD:
        raise ValueError(f"Unsupported train_period {train_period!r}; use {SUPPORTED_TRAIN_PERIOD}")
    return first_timestamp.normalize().replace(day=1) + pd.offsets.MonthBegin(1)


def assign_split(
    window_starts: pd.Series,
    usable: np.ndarray,
    train_end: pd.Timestamp,
    calib_fraction: float,
    fit_validation_fraction: float,
) -> pd.Series:
    """Label every window by time; usable training windows decide the fraction boundaries."""
    in_train_usable = (window_starts < train_end).to_numpy() & usable
    ordered_usable = window_starts[in_train_usable].sort_values().reset_index(drop=True)
    n_train = len(ordered_usable)
    n_calib = round(calib_fraction * n_train)
    n_fit = n_train - n_calib
    n_validation = round(fit_validation_fraction * n_fit)
    if n_fit - n_validation < 1 or n_calib < 1 or n_validation < 1:
        raise ValueError(f"Too few usable training windows ({n_train}) for the requested split")
    calib_start = ordered_usable.iloc[n_fit]
    validation_start = ordered_usable.iloc[n_fit - n_validation]
    labels = np.select(
        [
            window_starts >= train_end,
            window_starts >= calib_start,
            window_starts >= validation_start,
        ],
        [TEST, CALIB, FIT_VALIDATION],
        default=FIT_TRAIN,
    )
    return pd.Series(pd.Categorical(labels, categories=SPLIT_NAMES), index=window_starts.index)


def split_start_time(window_starts: pd.Series, split: pd.Series, name: str) -> pd.Timestamp:
    """First window start belonging to a split."""
    return window_starts[(split == name).to_numpy()].min()
