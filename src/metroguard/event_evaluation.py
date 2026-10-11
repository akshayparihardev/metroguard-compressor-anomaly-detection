"""Event-level evaluation of alert events against the four reported failure windows."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metroguard.alert_events import build_alert_events
from metroguard.alert_thresholds import alert_mask
from metroguard.chronological_split import TEST
from metroguard.failure_windows import REPORTED_FAILURE_WINDOWS, FailureWindow

PRIMARY_SCORE_NAMES = (
    "max_abs_z",
    "free_energy",
    "reconstruction_error",
    "max_transition_surprisal",
)


@dataclass(frozen=True)
class ScoredModel:
    """One model's score per window, ready for thresholding."""

    model: str
    score: str
    window_starts: pd.Series
    split: pd.Series
    usable: np.ndarray
    scores: np.ndarray

    @property
    def is_primary(self) -> bool:
        """True for the score the spec treats as the model's main anomaly score."""
        return self.score in PRIMARY_SCORE_NAMES


def scored_models(scores_dir: Path, rbm_k_values: list[int]) -> list[ScoredModel]:
    """Load every saved score table into a uniform list of (model, score) series."""
    layout = {
        "zscore_baseline_scores.parquet": [
            ("zscore_baseline", "max_abs_z", "max_abs_z"),
            ("zscore_baseline", "mean_abs_z", "mean_abs_z"),
        ],
        "rbm_scores.parquet": [
            (f"rbm_k{k}", "free_energy", f"free_energy_k{k}") for k in rbm_k_values
        ],
        "vae_scores.parquet": [
            ("vae", "reconstruction_error", "reconstruction_error"),
            ("vae", "negative_elbo", "negative_elbo"),
        ],
        "markov_scores.parquet": [
            ("markov_state_model", "max_transition_surprisal", "max_transition_surprisal")
        ],
    }
    models = []
    for file_name, columns in layout.items():
        path = scores_dir / file_name
        if not path.is_file():
            continue
        table = pd.read_parquet(path)
        for model, score, column in columns:
            models.append(
                ScoredModel(
                    model, score, table["window_start"], table["split"],
                    table["usable"].to_numpy(), table[column].to_numpy(np.float64),
                )
            )  # fmt: skip
    return models


def test_alert_events(
    scored: ScoredModel,
    threshold: float,
    window_seconds: int,
    merge_gap_minutes: float,
    min_event_windows: int,
) -> pd.DataFrame:
    """Alert events on usable test windows only."""
    in_test = (scored.split == TEST).to_numpy() & scored.usable
    return build_alert_events(
        scored.window_starts[in_test], scored.scores[in_test], threshold, window_seconds,
        merge_gap_minutes, min_event_windows,
    )  # fmt: skip


def reported_window_range(window: FailureWindow, lead_tolerance_hours: float):
    """Detection range [start - lead_tolerance, end] of a reported window."""
    return window.start - pd.Timedelta(hours=lead_tolerance_hours), window.end


def events_overlapping_range(
    events: pd.DataFrame, range_start: pd.Timestamp, range_end: pd.Timestamp
) -> pd.DataFrame:
    """Alert events that share at least one instant with the closed range."""
    return events[(events["start"] <= range_end) & (events["end"] >= range_start)]


def evaluate_events(
    events: pd.DataFrame,
    lead_tolerance_hours: float,
    windows: tuple[FailureWindow, ...] = REPORTED_FAILURE_WINDOWS,
) -> dict:
    """Detected events, detection delay per event and unconfirmed alert events."""
    result: dict = {}
    covered = np.zeros(len(events), dtype=bool)
    detected_ids = []
    for window in windows:
        range_start, range_end = reported_window_range(window, lead_tolerance_hours)
        overlapping = events_overlapping_range(events, range_start, range_end)
        covered[overlapping.index.to_numpy(dtype=int)] = True
        delay = np.nan
        if len(overlapping):
            detected_ids.append(window.event_id)
            first_alert = max(overlapping["start"].min(), range_start)
            delay = (first_alert - window.start) / pd.Timedelta(hours=1)
        result[f"delay_hours_event{window.event_id}"] = delay
    result["events_detected"] = len(detected_ids)
    result["detected_event_ids"] = ";".join(str(i) for i in detected_ids) or "none"
    result["n_alert_events"] = len(events)
    result["unconfirmed_alert_events"] = int((~covered).sum())
    return result


def evaluate_scored_model(
    scored: ScoredModel,
    thresholds: dict[str, float],
    window_seconds: int,
    merge_gap_minutes: float,
    min_event_windows: int,
    lead_tolerance_hours: float,
) -> list[dict]:
    """One metrics row per threshold percentile, computed on test windows only."""
    in_test = (scored.split == TEST).to_numpy()
    usable_test = in_test & scored.usable
    test_starts = scored.window_starts[in_test]
    test_days = (test_starts.max() + pd.Timedelta(seconds=window_seconds) - test_starts.min()) / (
        pd.Timedelta(days=1)
    )
    rows = []
    for percentile, threshold in thresholds.items():
        events = test_alert_events(scored, threshold, window_seconds, merge_gap_minutes,
                                   min_event_windows)  # fmt: skip
        n_alert_windows = int(alert_mask(np.nan_to_num(scored.scores[usable_test], nan=-np.inf),
                                         threshold).sum())  # fmt: skip
        row = {
            "model": scored.model,
            "score": scored.score,
            "is_primary_score": scored.is_primary,
            "percentile": float(percentile),
            "threshold": threshold,
            "n_usable_test_windows": int(usable_test.sum()),
            "n_alert_windows": n_alert_windows,
            "alert_window_fraction": n_alert_windows / int(usable_test.sum()),
        }
        row.update(evaluate_events(events, lead_tolerance_hours))
        row["test_days"] = test_days
        row["unconfirmed_alerts_per_day"] = row["unconfirmed_alert_events"] / test_days
        rows.append(row)
    return rows


def best_primary_model(rows: list[dict], primary_percentile: float) -> dict:
    """Primary-score row with most detected events, then fewest unconfirmed alerts per day."""
    candidates = [
        r for r in rows if r["is_primary_score"] and r["percentile"] == primary_percentile
    ]
    return min(candidates, key=lambda r: (-r["events_detected"], r["unconfirmed_alerts_per_day"]))


def events_to_explain(
    events: pd.DataFrame,
    max_events: int,
    windows: tuple[FailureWindow, ...] = REPORTED_FAILURE_WINDOWS,
) -> list[tuple[int, pd.Series]]:
    """For each reported window in order, its alert event with the highest peak score inside it."""
    chosen = []
    for window in windows:
        inside = events_overlapping_range(events, window.start, window.end)
        if len(inside) and len(chosen) < max_events:
            chosen.append((window.event_id, inside.loc[inside["peak_score"].idxmax()]))
    return chosen
