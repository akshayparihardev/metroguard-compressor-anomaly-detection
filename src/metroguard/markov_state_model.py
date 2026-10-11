"""Markov chain over motor-current operating states: transition surprisal per window."""

import numpy as np
import pandas as pd

SURPRISAL = "max_transition_surprisal"


def assign_reading_states(motor_current: pd.Series, cut_points: list[float]) -> np.ndarray:
    """Operating-state index of every raw reading, using cut points fitted on fit data."""
    return np.digitize(motor_current.to_numpy(np.float64), cut_points).astype(np.int8)


def valid_transitions(timestamps: pd.Series, max_gap_seconds: float) -> np.ndarray:
    """True at reading t when the step from t-1 is no longer than the gap threshold."""
    deltas = timestamps.diff().dt.total_seconds().to_numpy()
    return np.nan_to_num(deltas, nan=np.inf) <= max_gap_seconds


def count_transitions(states: np.ndarray, in_fit: np.ndarray, valid: np.ndarray, n_states: int):
    """State-to-state counts over valid steps where both readings lie in fit windows."""
    counted = valid.copy()
    counted[0] = False
    counted[1:] &= in_fit[1:] & in_fit[:-1]
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (states[:-1][counted[1:]], states[1:][counted[1:]]), 1)
    return counts


def fit_transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalised transition probabilities with add-one (Laplace) smoothing."""
    smoothed = counts + 1.0
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def transition_surprisal(states: np.ndarray, matrix: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """-log P(s_t | s_{t-1}) per reading; NaN where the step is not a valid transition."""
    surprisal = np.full(len(states), np.nan)
    steps = np.flatnonzero(valid[1:]) + 1
    surprisal[steps] = -np.log(matrix[states[steps - 1], states[steps]])
    return surprisal


def window_scores(window_starts: pd.Series, surprisal: np.ndarray) -> pd.Series:
    """Window score = max surprisal of the transitions that end in the window (NaN if none)."""
    return pd.Series(surprisal).groupby(window_starts.to_numpy()).max()
