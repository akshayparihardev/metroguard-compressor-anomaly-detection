import pandas as pd

from metroguard.failure_windows import (
    REPORTED_FAILURE_WINDOWS,
    in_any_window_mask,
    in_window_mask,
    intervals_overlap,
    overlapping_windows,
)

EVENT_1 = REPORTED_FAILURE_WINDOWS[0]


def test_four_windows_in_chronological_order():
    starts = [w.start for w in REPORTED_FAILURE_WINDOWS]
    assert [w.event_id for w in REPORTED_FAILURE_WINDOWS] == [1, 2, 3, 4]
    assert starts == sorted(starts)
    assert all(w.start < w.end for w in REPORTED_FAILURE_WINDOWS)


def test_window_boundaries_are_inclusive():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-04-17 23:59:50",
                "2020-04-18 00:00:00",
                "2020-04-18 12:00:00",
                "2020-04-18 23:59:00",
                "2020-04-18 23:59:10",
            ]
        )
    )
    assert in_window_mask(timestamps, EVENT_1).tolist() == [False, True, True, True, False]


def test_any_window_mask_combines_all_windows():
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2020-03-01 00:00:00",
                "2020-05-30 06:00:00",
                "2020-06-06 00:00:00",
                "2020-07-15 19:00:01",
            ]
        )
    )
    assert in_any_window_mask(timestamps).tolist() == [False, True, True, False]


def test_intervals_touching_at_one_instant_overlap():
    t = pd.Timestamp
    assert intervals_overlap(t("2020-01-01"), t("2020-01-02"), t("2020-01-02"), t("2020-01-03"))
    assert not intervals_overlap(
        t("2020-01-01"), t("2020-01-02"), t("2020-01-02 00:00:01"), t("2020-01-03")
    )


def test_overlapping_windows_finds_events_inside_range():
    found = overlapping_windows(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-06-05 10:00:00"))
    assert [w.event_id for w in found] == [2, 3]
    assert (
        overlapping_windows(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-29 23:59:59")) == []
    )
