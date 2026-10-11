"""Windowing, standardisation, discretisation and binarisation of sensor readings."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
STD_SOURCE_COLUMNS = ("Motor_current", "TP2", "TP3")
MEAN_SUFFIX = "_mean"
STD_SUFFIX = "_std"
ACTIVE_FRACTION_SUFFIX = "_active_fraction"
ACTIVE_BINARY_SUFFIX = "_active"
BINARY_THRESHOLD = 0.5
CONSTANT_STD_TOLERANCE = 1e-8


@dataclass(frozen=True)
class FeatureStandardiser:
    """Per-feature mean and scale fitted on fit windows; constant features get scale 1."""

    columns: list[str]
    mean: list[float]
    scale: list[float]
    constant_columns: list[str]


def analog_columns(readings: pd.DataFrame) -> list[str]:
    """Analog sensors: every non-timestamp, non-boolean column of the readings cache."""
    return [c for c in readings.columns if c != TIMESTAMP_COLUMN and readings[c].dtype != np.bool_]


def digital_columns(readings: pd.DataFrame) -> list[str]:
    """Digital sensors: the boolean columns of the readings cache."""
    return [c for c in readings.columns if readings[c].dtype == np.bool_]


def expected_readings_per_window(timestamps: pd.Series, window_seconds: int) -> float:
    """Readings a full window holds, from the observed cadence (median positive delta)."""
    deltas = timestamps.diff().dropna().dt.total_seconds()
    return window_seconds / float(deltas[deltas > 0].median())


def minimum_readings(expected: float, min_fraction: float) -> int:
    """Smallest reading count that keeps a window usable."""
    return math.ceil(min_fraction * expected)


def aggregate_windows(readings: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Per fixed time window: analog means, std of key signals, digital active fractions."""
    analog, digital = analog_columns(readings), digital_columns(readings)
    window_start = readings[TIMESTAMP_COLUMN].dt.floor(f"{window_seconds}s")
    grouped = readings.groupby(window_start, sort=True)
    means = grouped[analog].mean().add_suffix(MEAN_SUFFIX)
    stds = grouped[list(STD_SOURCE_COLUMNS)].std(ddof=0).add_suffix(STD_SUFFIX)
    active = grouped[digital].mean().add_suffix(ACTIVE_FRACTION_SUFFIX)
    table = pd.concat([means, stds, active], axis=1).astype(np.float32)
    table.insert(0, "reading_count", grouped.size().astype(np.int16))
    table.index.name = "window_start"
    return table.reset_index()


def flag_frozen_windows(
    window_starts: pd.Series, frozen_runs: pd.DataFrame, window_seconds: int
) -> np.ndarray:
    """True for windows that overlap any frozen-value run."""
    window_ends = window_starts + pd.Timedelta(seconds=window_seconds)
    flagged = np.zeros(len(window_starts), dtype=bool)
    for run in frozen_runs.itertuples():
        flagged |= ((window_starts <= run.end) & (window_ends > run.start)).to_numpy()
    return flagged


def model_feature_columns(table: pd.DataFrame) -> list[str]:
    """Analog means, analog stds and digital fractions, in table order."""
    suffixes = (MEAN_SUFFIX, STD_SUFFIX, ACTIVE_FRACTION_SUFFIX)
    return [c for c in table.columns if c.endswith(suffixes)]


def analog_feature_columns(table: pd.DataFrame) -> list[str]:
    """Analog means and stds (the features that are discretised into quantile bins)."""
    return [c for c in table.columns if c.endswith((MEAN_SUFFIX, STD_SUFFIX))]


def digital_feature_columns(table: pd.DataFrame) -> list[str]:
    """Digital active-fraction columns."""
    return [c for c in table.columns if c.endswith(ACTIVE_FRACTION_SUFFIX)]


def fit_standardiser(fit_features: pd.DataFrame) -> FeatureStandardiser:
    """Mean and std from fit windows only; zero-spread features get scale 1."""
    std = fit_features.std(ddof=0)
    constant = std < CONSTANT_STD_TOLERANCE
    scale = std.where(~constant, 1.0)
    return FeatureStandardiser(
        columns=list(fit_features.columns),
        mean=fit_features.mean().tolist(),
        scale=scale.tolist(),
        constant_columns=list(fit_features.columns[constant.to_numpy()]),
    )


def standardise(features: pd.DataFrame, standardiser: FeatureStandardiser) -> pd.DataFrame:
    """Apply a fitted standardiser (float32)."""
    mean = pd.Series(standardiser.mean, index=standardiser.columns)
    scale = pd.Series(standardiser.scale, index=standardiser.columns)
    return ((features[standardiser.columns] - mean) / scale).astype(np.float32)


def fit_bin_edges(
    fit_features: pd.DataFrame, columns: list[str], n_bins: int
) -> dict[str, list[float]]:
    """Quantile cut points from fit windows; duplicate cut points are merged."""
    levels = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return {
        column: sorted(set(np.quantile(fit_features[column].to_numpy(np.float64), levels).tolist()))
        for column in columns
    }


def discretise(features: pd.DataFrame, bin_edges: dict[str, list[float]]) -> pd.DataFrame:
    """Integer bin index per feature, using edges fitted on fit windows."""
    return pd.DataFrame(
        {
            column: np.digitize(features[column].to_numpy(np.float64), edges).astype(np.int8)
            for column, edges in bin_edges.items()
        },
        index=features.index,
    )


def binarise_digital(features: pd.DataFrame, fraction_columns: list[str]) -> pd.DataFrame:
    """1 if the digital signal is active for at least half of the window, else 0."""
    binary = (features[fraction_columns] >= BINARY_THRESHOLD).astype(np.uint8)
    return binary.rename(
        columns=lambda c: c.removesuffix(ACTIVE_FRACTION_SUFFIX) + ACTIVE_BINARY_SUFFIX
    )


def one_hot_visible(
    states: pd.DataFrame, bin_edges: dict[str, list[float]], digital_binary: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """RBM visible matrix: one-hot analog bins followed by binary digital signals."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for column, edges in bin_edges.items():
        n_states = len(edges) + 1
        blocks.append(np.eye(n_states, dtype=np.uint8)[states[column].to_numpy()])
        names += [f"{column}__state{i}" for i in range(n_states)]
    blocks.append(digital_binary.to_numpy(np.uint8))
    names += list(digital_binary.columns)
    return np.hstack(blocks), names


def assign_motor_state(motor_current_mean: pd.Series, cut_points: list[float]) -> np.ndarray:
    """Operating-state index of each window's mean motor current."""
    return np.digitize(motor_current_mean.to_numpy(np.float64), cut_points).astype(np.int8)
