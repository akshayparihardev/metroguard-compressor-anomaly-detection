import numpy as np
import pandas as pd

from metroguard.alert_events import build_alert_events

MINUTE = 60


def minutes(*offsets: int) -> pd.Series:
    base = pd.Timestamp("2020-03-01 00:00:00")
    return pd.Series([base + pd.Timedelta(minutes=m) for m in offsets])


def test_alerts_within_merge_gap_form_one_event():
    starts = minutes(0, 1, 2, 3, 4, 5, 6, 7)
    scores = np.array([0, 5, 0, 0, 5, 0, 0, 0], dtype=float)
    events = build_alert_events(starts, scores, 1.0, MINUTE, 10, 1)
    assert len(events) == 1
    assert events.loc[0, "n_alert_windows"] == 2
    assert events.loc[0, "start"] == starts[1]
    assert events.loc[0, "end"] == starts[4] + pd.Timedelta(seconds=MINUTE)


def test_alerts_further_apart_than_merge_gap_are_separate_events():
    starts = minutes(0, 5, 30, 31)
    scores = np.array([9.0, 0.0, 9.0, 9.0])
    events = build_alert_events(starts, scores, 1.0, MINUTE, 10, 1)
    assert events["n_alert_windows"].tolist() == [1, 2]
    assert events["peak_score"].tolist() == [9.0, 9.0]


def test_gap_equal_to_merge_gap_merges():
    starts = minutes(0, 11)
    events = build_alert_events(starts, np.array([2.0, 2.0]), 1.0, MINUTE, 10, 1)
    assert len(events) == 1


def test_short_events_are_dropped_by_min_event_windows():
    starts = minutes(0, 1, 60)
    events = build_alert_events(starts, np.array([2.0, 2.0, 2.0]), 1.0, MINUTE, 10, 2)
    assert events["n_alert_windows"].tolist() == [2]


def test_score_equal_to_threshold_and_nan_do_not_alert():
    starts = minutes(0, 1, 2)
    events = build_alert_events(starts, np.array([1.0, np.nan, 0.5]), 1.0, MINUTE, 10, 1)
    assert events.empty


def test_unsorted_input_is_handled_chronologically():
    starts = minutes(5, 0, 1)
    events = build_alert_events(starts, np.array([0.0, 3.0, 4.0]), 1.0, MINUTE, 10, 1)
    assert events["n_alert_windows"].tolist() == [2]
    assert events.loc[0, "peak_score"] == 4.0
