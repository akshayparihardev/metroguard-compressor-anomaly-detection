"""Turn window scores into alerts and merge neighbouring alerts into alert events."""

import numpy as np
import pandas as pd

from metroguard.alerting.alert_thresholds import alert_mask

EVENT_COLUMNS = ("start", "end", "n_alert_windows", "peak_score")


def build_alert_events(
    window_starts: pd.Series,
    scores: np.ndarray,
    threshold: float,
    window_seconds: int,
    merge_gap_minutes: float,
    min_event_windows: int,
) -> pd.DataFrame:
    """Merge alert windows separated by at most merge_gap; drop events with too few windows.

    An event runs from the start of its first alert window to the end of its last one.
    Windows with a NaN score never raise an alert.
    """
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(window_starts.to_numpy())
    starts = window_starts.to_numpy()[order]
    ordered_scores = scores[order]
    alerts = np.flatnonzero(alert_mask(np.nan_to_num(ordered_scores, nan=-np.inf), threshold))
    if alerts.size == 0:
        return pd.DataFrame(columns=list(EVENT_COLUMNS))
    alert_starts = pd.Series(starts[alerts])
    window = pd.Timedelta(seconds=window_seconds)
    gap = alert_starts.shift(-1) - (alert_starts + window)
    new_event = (gap.shift(1) > pd.Timedelta(minutes=merge_gap_minutes)).fillna(False)
    event_id = new_event.cumsum()
    grouped = pd.DataFrame(
        {"start": alert_starts, "score": ordered_scores[alerts], "event": event_id}
    ).groupby("event")
    events = pd.DataFrame(
        {
            "start": grouped["start"].min(),
            "end": grouped["start"].max() + window,
            "n_alert_windows": grouped.size(),
            "peak_score": grouped["score"].max(),
        }
    )
    return events[events["n_alert_windows"] >= min_event_windows].reset_index(drop=True)
