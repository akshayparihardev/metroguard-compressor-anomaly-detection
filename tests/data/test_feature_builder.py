import numpy as np
import pandas as pd

from metroguard.data.feature_builder import (
    aggregate_windows,
    binarise_digital,
    discretise,
    fit_bin_edges,
    fit_standardiser,
    flag_frozen_windows,
    minimum_readings,
    one_hot_visible,
    standardise,
)


def make_readings() -> pd.DataFrame:
    timestamps = pd.date_range("2020-02-01 00:00:00", periods=12, freq="10s")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "TP2": np.arange(12, dtype=np.float32),
            "TP3": np.full(12, 9.0, dtype=np.float32),
            "Motor_current": np.array([0.0] * 6 + [4.0] * 6, dtype=np.float32),
            "COMP": [True] * 9 + [False] * 3,
        }
    )


def test_aggregate_windows_means_stds_and_fractions():
    table = aggregate_windows(make_readings(), 60)
    assert table["reading_count"].tolist() == [6, 6]
    assert table["TP2_mean"].tolist() == [2.5, 8.5]
    assert table["Motor_current_mean"].tolist() == [0.0, 4.0]
    assert table["TP3_std"].tolist() == [0.0, 0.0]
    assert table["COMP_active_fraction"].tolist() == [1.0, 0.5]
    assert table["TP2_mean"].dtype == np.float32


def test_minimum_readings_rounds_up():
    assert minimum_readings(6.0, 0.5) == 3
    assert minimum_readings(5.0, 0.5) == 3


def test_standardiser_gives_constant_features_scale_one():
    fit = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
    standardiser = fit_standardiser(fit)
    assert standardiser.constant_columns == ["b"]
    assert standardiser.scale[1] == 1.0
    scaled = standardise(pd.DataFrame({"a": [2.0], "b": [7.0]}), standardiser)
    assert scaled["a"].iloc[0] == 0.0
    assert scaled["b"].iloc[0] == 2.0


def test_bin_edges_come_from_fit_data_only_and_merge_duplicates():
    fit = pd.DataFrame({"a": np.arange(9.0), "b": np.zeros(9)})
    edges = fit_bin_edges(fit, ["a", "b"], 3)
    assert len(edges["a"]) == 2
    assert len(edges["b"]) == 1
    later = pd.DataFrame({"a": [-100.0, 4.0, 1000.0], "b": [0.0, 0.0, 1.0]})
    states = discretise(later, edges)
    assert states["a"].tolist() == [0, 1, 2]
    assert fit_bin_edges(fit, ["a", "b"], 3) == edges


def test_binarise_digital_uses_half_window_threshold():
    fractions = pd.DataFrame({"COMP_active_fraction": [0.0, 0.49, 0.5, 1.0]})
    binary = binarise_digital(fractions, ["COMP_active_fraction"])
    assert binary["COMP_active"].tolist() == [0, 0, 1, 1]


def test_one_hot_visible_has_one_active_unit_per_analog_feature():
    edges = {"a": [1.0, 2.0], "b": [0.0]}
    states = pd.DataFrame({"a": np.array([0, 2], np.int8), "b": np.array([1, 0], np.int8)})
    digital = pd.DataFrame({"COMP_active": np.array([1, 0], np.uint8)})
    visible, names = one_hot_visible(states, edges, digital)
    assert visible.shape == (2, 3 + 2 + 1)
    assert names[-1] == "COMP_active"
    assert (visible[:, :5].sum(axis=1) == 2).all()


def test_frozen_windows_are_flagged_when_they_overlap_a_run():
    starts = pd.Series(pd.date_range("2020-02-01 00:00:00", periods=5, freq="min"))
    runs = pd.DataFrame(
        {
            "start": [pd.Timestamp("2020-02-01 00:01:30")],
            "end": [pd.Timestamp("2020-02-01 00:02:10")],
        }
    )
    assert flag_frozen_windows(starts, runs, 60).tolist() == [False, True, True, False, False]
