"""The four company-reported air-leak intervals and inclusive overlap checks."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FailureWindow:
    """One reported failure interval; both start and end are inclusive."""

    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp
    reported_issue: str
    maintenance_note: str


REPORTED_FAILURE_WINDOWS: tuple[FailureWindow, ...] = (
    FailureWindow(
        1, pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-18 23:59:00"), "Air leak", "-"
    ),
    FailureWindow(
        2,
        pd.Timestamp("2020-05-29 23:30:00"),
        pd.Timestamp("2020-05-30 06:00:00"),
        "Air leak",
        "Maintenance on 30Apr at 12:00",
    ),
    FailureWindow(
        3,
        pd.Timestamp("2020-06-05 10:00:00"),
        pd.Timestamp("2020-06-07 14:30:00"),
        "Air leak",
        "Maintenance on 8Jun at 16:00",
    ),
    FailureWindow(
        4,
        pd.Timestamp("2020-07-15 14:30:00"),
        pd.Timestamp("2020-07-15 19:00:00"),
        "Air leak",
        "Maintenance on 16Jul at 00:00",
    ),
)


def in_window_mask(timestamps: pd.Series, window: FailureWindow) -> np.ndarray:
    """Boolean mask of timestamps inside [start, end], both ends inclusive."""
    return ((timestamps >= window.start) & (timestamps <= window.end)).to_numpy()


def in_any_window_mask(
    timestamps: pd.Series, windows: tuple[FailureWindow, ...] = REPORTED_FAILURE_WINDOWS
) -> np.ndarray:
    """Boolean mask of timestamps inside at least one reported window."""
    mask = np.zeros(len(timestamps), dtype=bool)
    for window in windows:
        mask |= in_window_mask(timestamps, window)
    return mask


def intervals_overlap(
    start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp
) -> bool:
    """True when two closed intervals share at least one instant."""
    return start_a <= end_b and start_b <= end_a


def overlapping_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    windows: tuple[FailureWindow, ...] = REPORTED_FAILURE_WINDOWS,
) -> list[FailureWindow]:
    """Reported windows that overlap the closed interval [start, end]."""
    return [w for w in windows if intervals_overlap(start, end, w.start, w.end)]
