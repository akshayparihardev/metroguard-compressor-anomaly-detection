import numpy as np
import pandas as pd
import pytest

from metroguard.data.chronological_split import (
    CALIB,
    FIT_TRAIN,
    FIT_VALIDATION,
    TEST,
    assign_split,
    training_period_end,
)

TRAIN_END = pd.Timestamp("2020-03-01")


def make_windows(n_train: int, n_test: int) -> pd.Series:
    train = pd.date_range("2020-02-01", periods=n_train, freq="min")
    test = pd.date_range("2020-03-01", periods=n_test, freq="min")
    return pd.Series(train.append(test))


def test_training_period_ends_at_start_of_next_month():
    end = training_period_end(pd.Timestamp("2020-02-01 00:00:00"), "first_calendar_month")
    assert end == TRAIN_END
    assert training_period_end(pd.Timestamp("2020-02-17 08:00:00"), "first_calendar_month") == end


def test_unknown_training_period_is_rejected():
    with pytest.raises(ValueError):
        training_period_end(pd.Timestamp("2020-02-01"), "first_30_days")


def test_split_is_chronological_with_requested_fractions():
    starts = make_windows(1000, 200)
    split = assign_split(starts, np.ones(len(starts), bool), TRAIN_END, 0.2, 0.1)
    counts = split.value_counts()
    assert counts[CALIB] == 200
    assert counts[FIT_VALIDATION] == 80
    assert counts[FIT_TRAIN] == 720
    assert counts[TEST] == 200
    codes = split.cat.codes.to_numpy()
    assert (np.diff(codes) >= 0).all(), "labels must never go back in time"
    assert (split[starts >= TRAIN_END] == TEST).all()
    assert (split[starts < TRAIN_END] != TEST).all()


def test_unusable_windows_do_not_count_towards_fractions():
    starts = make_windows(1000, 10)
    usable = np.ones(len(starts), bool)
    usable[:500] = False
    split = assign_split(starts, usable, TRAIN_END, 0.2, 0.1)
    usable_train = (starts < TRAIN_END).to_numpy() & usable
    assert (split[usable_train] == CALIB).sum() == 100
    assert (split[starts < TRAIN_END] == FIT_TRAIN).any()


def test_too_few_usable_windows_raises():
    starts = make_windows(5, 5)
    with pytest.raises(ValueError):
        assign_split(starts, np.ones(len(starts), bool), TRAIN_END, 0.2, 0.1)
