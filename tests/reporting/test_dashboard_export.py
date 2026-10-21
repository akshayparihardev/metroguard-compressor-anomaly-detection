import numpy as np
import pandas as pd
import pytest

from metroguard.alerting.event_evaluation import ScoredModel
from metroguard.reporting.dashboard_export import (
    build_latent,
    build_metrics,
    cell_value,
    compact_json,
    epoch_minutes,
    first_match,
    hourly_index,
    hourly_maxima,
    network_layout,
    read_markdown_sections,
    table_records,
)

MARKDOWN = """# Title

## 1. First section

Some paragraph.

- bullet one
- bullet two

| name | value |
|:--|--:|
| a | 1,250 |
| b | -inf |
| c | text |

## 2. Second section

| x |
|---|
| 0.5 |
"""


def make_scored(scores, usable, start="2020-03-01 00:00:00") -> ScoredModel:
    starts = pd.Series(pd.date_range(start, periods=len(scores), freq="min"))
    split = pd.Series(["test"] * len(scores))
    return ScoredModel(
        "vae", "reconstruction_error", starts, split, np.array(usable), np.array(scores)
    )


def test_epoch_minutes_matches_unix_time():
    assert epoch_minutes(pd.Timestamp("1970-01-01 00:01:00")) == 1
    assert epoch_minutes(pd.Timestamp("2020-02-01 00:00:00")) == 26_341_920


def test_markdown_sections_split_tables_bullets_and_paragraphs():
    sections = read_markdown_sections(MARKDOWN)
    first = sections["1. First section"]
    assert first["bullets"] == ["bullet one", "bullet two"]
    assert first["paragraphs"] == ["Some paragraph."]
    records = table_records(first["tables"][0])
    assert records == [
        {"name": "a", "value": 1250},
        {"name": "b", "value": None},
        {"name": "c", "value": "text"},
    ]
    assert table_records(sections["2. Second section"]["tables"][0]) == [{"x": 0.5}]


def test_cell_value_keeps_text_and_turns_infinity_into_none():
    assert cell_value("`COMP`") == "COMP"
    assert cell_value("inf") is None
    assert cell_value("12") == 12
    assert cell_value("1.5") == 1.5


def test_first_match_fails_loudly_when_report_format_changes():
    assert first_match(r"Shape: (\d+)", "Shape: 12 rows").group(1) == "12"
    with pytest.raises(ValueError):
        first_match(r"Shape: (\d+)", "different text")


def test_hourly_maxima_ignore_unusable_windows_and_mark_gaps():
    scores = [1.0, 5.0] + [2.0] * 58 + [9.0] * 60 + [3.0] * 60
    usable = [True, False] + [True] * 58 + [False] * 60 + [True] * 60
    scored = make_scored(scores, usable)
    values = hourly_maxima(scored, hourly_index(scored.window_starts))
    assert values == [2.0, None, 3.0]


def test_compact_json_rejects_nan():
    assert compact_json({"a": [1, None]}) == '{"a":[1,null]}'
    with pytest.raises(ValueError):
        compact_json({"a": float("nan")})


def test_detection_range_share_uses_test_days():
    metrics = pd.DataFrame(
        {
            "model": ["zscore_baseline"],
            "score": ["max_abs_z"],
            "test_days": [100.0],
            "n_usable_test_windows": [10],
        }
    )
    payload = build_metrics(metrics, {}, 24)
    assert payload["detection_range_days"] == pytest.approx(7.645138888, rel=1e-6)
    assert payload["detection_range_share"] == pytest.approx(0.0764513888, rel=1e-6)
    assert payload["rows"][0]["model_label"] == "Z-score baseline"


def test_network_layout_puts_parents_left_of_children_and_is_deterministic():
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "c"), ("b", "c"), ("c", "d"), ("a", "e")]
    layout = network_layout(nodes, edges)
    assert all(layout[parent][0] < layout[child][0] for parent, child in edges)
    assert all(0 <= v <= 1 for xy in layout.values() for v in xy)
    assert layout == network_layout(nodes, edges)


def test_latent_view_fits_axes_on_fit_windows_and_keeps_every_event_window():
    rng = np.random.default_rng(0)
    starts = pd.date_range("2020-04-17 00:00", periods=4000, freq="min")
    split = np.where(np.arange(4000) < 2000, "fit_train", "test")
    table = pd.DataFrame({"window_start": starts, "split": split, "usable": True})
    latent = rng.normal(size=(4000, 6)) * np.array([5, 3, 1, 0.5, 0.2, 0.1])
    view = build_latent(latent, table, np.zeros(4000, dtype=bool), seed=42, max_points=500)
    points = view["points"]
    in_event_1 = int(((starts >= "2020-04-18 00:00") & (starts <= "2020-04-18 23:59")).sum())
    assert points["event"].count(1) == in_event_1
    assert len(points["x"]) == max(500, in_event_1)
    assert set(points["split"]) == {0, 2}
    assert (
        view["explained_variance"][0]
        > view["explained_variance"][1]
        > view["explained_variance"][2]
    )
    assert view == build_latent(latent, table, np.zeros(4000, dtype=bool), seed=42, max_points=500)
