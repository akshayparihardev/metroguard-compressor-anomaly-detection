"""Step 1 audit measurements: schema, cadence, gaps, digital signals, events, split, motor."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metroguard.config_loader import AuditConfig
from metroguard.failure_windows import (
    REPORTED_FAILURE_WINDOWS,
    FailureWindow,
    in_any_window_mask,
    in_window_mask,
    overlapping_windows,
)
from metroguard.raw_data_loader import ColumnRoles

MOTOR_CURRENT_HYPOTHESIS_AMPS = (0.0, 4.0, 7.0, 9.0)
MOTOR_CURRENT_BIN_AMPS = 0.1
MOTOR_CURRENT_SMOOTHING_BINS = 5
MOTOR_CURRENT_MIN_PEAK_SHARE = 0.005
MOTOR_CURRENT_MAX_VALLEY_RATIO = 0.5
MOTOR_CURRENT_HYPOTHESIS_TOLERANCE_AMPS = 0.75


@dataclass(frozen=True)
class SplitCandidate:
    """One candidate training period and its overlap with reported windows."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    train_rows: int
    overlapping_event_ids: list[int]
    test_rows: int
    test_rows_in_windows: int


def schema_summary(raw: pd.DataFrame, roles: ColumnRoles, raw_csv: Path) -> dict:
    """File path/size, shape, dtypes, missing values and duplicate counts on the raw frame."""
    without_index = raw.drop(columns=roles.index_columns)
    return {
        "path": raw_csv,
        "size_mb": raw_csv.stat().st_size / 1e6,
        "n_rows": len(raw),
        "n_columns": raw.shape[1],
        "dtypes": raw.dtypes.astype(str).to_dict(),
        "missing_per_column": raw.isna().sum().to_dict(),
        "duplicate_rows_all_columns": int(raw.duplicated().sum()),
        "duplicate_rows_excluding_index": int(without_index.duplicated().sum()),
        "duplicate_timestamps": int(raw[roles.timestamp].duplicated().sum()),
        "head": raw.head(5),
        "tail": raw.tail(5),
    }


def index_column_summary(raw: pd.DataFrame, roles: ColumnRoles) -> dict[str, dict]:
    """Describe each unnamed index column (range, step, monotonicity)."""
    summary = {}
    for column in roles.index_columns:
        steps = raw[column].diff().dropna()
        summary[column] = {
            "min": raw[column].min(),
            "max": raw[column].max(),
            "most_common_step": steps.mode().iloc[0] if len(steps) else None,
            "strictly_increasing": bool((steps > 0).all()),
        }
    return summary


def timestamp_order_summary(parsed_in_file_order: pd.Series, parse_failures: int) -> dict:
    """Timestamp range and whether the file was already in time order."""
    steps = parsed_in_file_order.diff().dropna()
    return {
        "parse_failures": parse_failures,
        "min": parsed_in_file_order.min(),
        "max": parsed_in_file_order.max(),
        "span": parsed_in_file_order.max() - parsed_in_file_order.min(),
        "non_monotonic_steps_in_file_order": int((steps < pd.Timedelta(0)).sum()),
        "zero_steps_in_file_order": int((steps == pd.Timedelta(0)).sum()),
    }


def timestamp_deltas_seconds(sorted_timestamps: pd.Series) -> pd.Series:
    """Consecutive differences of the sorted timestamps, in seconds."""
    return sorted_timestamps.diff().dropna().dt.total_seconds()


def cadence_summary(deltas_seconds: pd.Series, audit: AuditConfig) -> dict:
    """Cadence statistics and gap classification of consecutive timestamp differences."""
    positive = deltas_seconds[deltas_seconds > 0]
    mode_seconds = float(positive.mode().iloc[0])
    gap_threshold = audit.gap_multiplier * mode_seconds
    long_gap_threshold = audit.long_gap_hours * 3600
    counts = positive.value_counts()
    common = pd.DataFrame(
        {
            "delta_seconds": counts.index,
            "count": counts.to_numpy(),
            "percent": 100 * counts.to_numpy() / len(deltas_seconds),
        }
    ).head(10)
    return {
        "n_deltas": len(deltas_seconds),
        "min_positive": float(positive.min()),
        "median_positive": float(positive.median()),
        "mean_positive": float(positive.mean()),
        "mode_positive": mode_seconds,
        "zero": int((deltas_seconds == 0).sum()),
        "negative": int((deltas_seconds < 0).sum()),
        "at_mode": int((positive == mode_seconds).sum()),
        "jitter": int(((positive != mode_seconds) & (positive <= gap_threshold)).sum()),
        "gap_threshold_seconds": gap_threshold,
        "gaps": int((positive > gap_threshold).sum()),
        "long_gap_threshold_seconds": long_gap_threshold,
        "long_gaps": int((positive >= long_gap_threshold).sum()),
        "seconds_lost_in_gaps": float(positive[positive > gap_threshold].sum()),
        "common_deltas": common,
    }


def longest_gaps(sorted_timestamps: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """The longest intervals without readings, with their start and end times."""
    deltas = sorted_timestamps.diff()
    order = deltas.nlargest(top_n).index
    return pd.DataFrame(
        {
            "gap_start": sorted_timestamps.shift(1).loc[order].to_numpy(),
            "gap_end": sorted_timestamps.loc[order].to_numpy(),
            "duration_hours": (deltas.loc[order].dt.total_seconds() / 3600).round(2).to_numpy(),
        }
    )


def frozen_value_runs(
    readings: pd.DataFrame, analog_columns: list[str], min_minutes: float
) -> pd.DataFrame:
    """Stretches where every analog sensor repeats exactly the same value for a long time."""
    unchanged = readings[analog_columns].diff().abs().sum(axis=1).eq(0).to_numpy()
    run_id = np.cumsum(~unchanged)
    frame = pd.DataFrame(
        {"run": run_id, "unchanged": unchanged, "timestamp": readings["timestamp"]}
    )
    runs = frame[frame["unchanged"]].groupby("run")["timestamp"].agg(["min", "max", "size"])
    runs["start"] = frame.groupby("run")["timestamp"].first().loc[runs.index]
    runs["duration_hours"] = (runs["max"] - runs["start"]).dt.total_seconds() / 3600
    long_runs = runs[runs["duration_hours"] >= min_minutes / 60]
    return pd.DataFrame(
        {
            "start": long_runs["start"].to_numpy(),
            "end": long_runs["max"].to_numpy(),
            "readings": (long_runs["size"] + 1).to_numpy(),
            "duration_hours": long_runs["duration_hours"].round(2).to_numpy(),
        }
    )


def daily_row_counts(sorted_timestamps: pd.Series) -> pd.Series:
    """Number of readings per calendar day, including days with zero readings."""
    counts = sorted_timestamps.dt.floor("D").value_counts().sort_index()
    full_range = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    return counts.reindex(full_range, fill_value=0)


def digital_signal_summary(
    raw: pd.DataFrame, roles: ColumnRoles, audit: AuditConfig
) -> pd.DataFrame:
    """Value set, binary check, active fraction and near-constant flag per digital column."""
    rows = []
    for column in roles.digital:
        value_counts = raw[column].value_counts(dropna=False).sort_index()
        unique = value_counts.index.tolist()
        binary = bool(np.isin([v for v in unique if pd.notna(v)], [0, 1]).all())
        active = float((raw[column] == 1).mean())
        rows.append(
            {
                "column": column,
                "value_counts": {repr(k): int(v) for k, v in value_counts.items()},
                "strictly_binary": binary,
                "active_fraction": active,
                "near_constant": active < audit.near_constant_fraction
                or active > 1 - audit.near_constant_fraction,
            }
        )
    return pd.DataFrame(rows)


def digital_activity_by_month(readings: pd.DataFrame, roles: ColumnRoles) -> pd.DataFrame:
    """Fraction of readings with value 1 per digital column and calendar month."""
    month = readings["timestamp"].dt.to_period("M")
    active = readings[roles.digital].astype(float).eq(1.0)
    return active.groupby(month).mean()


def event_window_coverage(
    sorted_timestamps: pd.Series, cadence_seconds: float, gap_threshold_seconds: float
) -> pd.DataFrame:
    """Readings per reported window, expected count, coverage and largest internal gap."""
    rows = []
    for window in REPORTED_FAILURE_WINDOWS:
        inside = sorted_timestamps[in_window_mask(sorted_timestamps, window)]
        duration = (window.end - window.start).total_seconds()
        expected = int(duration // cadence_seconds) + 1
        edges = pd.concat(
            [pd.Series([window.start]), inside, pd.Series([window.end])], ignore_index=True
        )
        largest_gap = edges.diff().dropna().dt.total_seconds().max()
        rows.append(
            {
                "event": window.event_id,
                "start": window.start,
                "end": window.end,
                "duration_hours": round(duration / 3600, 2),
                "readings": len(inside),
                "expected_at_cadence": expected,
                "coverage_percent": round(100 * len(inside) / expected, 1),
                "largest_gap_seconds": largest_gap,
                "fully_covered": bool(len(inside) > 0 and largest_gap <= gap_threshold_seconds),
            }
        )
    return pd.DataFrame(rows)


def readings_after_window_end_same_minute(
    sorted_timestamps: pd.Series, window: FailureWindow
) -> int:
    """Readings after a minute-precision end time but before the next minute (left outside)."""
    next_minute = window.end + pd.Timedelta(minutes=1)
    return int(((sorted_timestamps > window.end) & (sorted_timestamps < next_minute)).sum())


def split_candidates(sorted_timestamps: pd.Series) -> list[SplitCandidate]:
    """First calendar month and first 30 days as candidate training periods."""
    first = sorted_timestamps.iloc[0]
    month_start = first.normalize().replace(day=1)
    month_end = month_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
    thirty_days_end = first + pd.Timedelta(days=30) - pd.Timedelta(seconds=1)
    in_windows = in_any_window_mask(sorted_timestamps)
    candidates = []
    for name, start, end in (
        ("first_calendar_month", first, month_end),
        ("first_30_days", first, thirty_days_end),
    ):
        in_train = (sorted_timestamps >= start) & (sorted_timestamps <= end)
        in_test = (sorted_timestamps > end).to_numpy()
        candidates.append(
            SplitCandidate(
                name=name,
                start=start,
                end=end,
                train_rows=int(in_train.sum()),
                overlapping_event_ids=[w.event_id for w in overlapping_windows(start, end)],
                test_rows=int(in_test.sum()),
                test_rows_in_windows=int((in_test & in_windows).sum()),
            )
        )
    return candidates


def motor_current_summary(motor_current: pd.Series) -> dict:
    """Descriptive statistics, quantiles, unique and most common rounded values."""
    quantile_levels = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    rounded = motor_current.round(1).value_counts().head(10)
    return {
        "describe": motor_current.describe(),
        "unique_values": int(motor_current.nunique()),
        "quantiles": motor_current.quantile(quantile_levels),
        "common_rounded_0_1": rounded,
    }


def merge_shallow_peaks(peaks: list[int], smoothed: np.ndarray) -> list[int]:
    """Merge neighbouring peaks whose separating valley is not clearly lower than both."""
    merged: list[int] = []
    for peak in peaks:
        if merged:
            previous = merged[-1]
            valley = smoothed[previous : peak + 1].min()
            if valley > MOTOR_CURRENT_MAX_VALLEY_RATIO * min(smoothed[previous], smoothed[peak]):
                merged[-1] = previous if smoothed[previous] >= smoothed[peak] else peak
                continue
        merged.append(peak)
    return merged


def motor_current_candidate_states(motor_current: pd.Series) -> pd.DataFrame:
    """Peaks of a smoothed 0.1 A histogram and valley cut points between them."""
    values = motor_current.dropna().to_numpy()
    edges = np.arange(
        np.floor(values.min()),
        np.ceil(values.max()) + 2 * MOTOR_CURRENT_BIN_AMPS,
        MOTOR_CURRENT_BIN_AMPS,
    )
    counts, edges = np.histogram(values, bins=edges)
    kernel = np.ones(MOTOR_CURRENT_SMOOTHING_BINS) / MOTOR_CURRENT_SMOOTHING_BINS
    smoothed = np.convolve(counts, kernel, mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    min_height = MOTOR_CURRENT_MIN_PEAK_SHARE * len(values)
    padded = np.concatenate([[-np.inf], smoothed, [-np.inf]])
    peaks = [
        i
        for i in range(len(smoothed))
        if padded[i + 1] >= padded[i]
        and padded[i + 1] > padded[i + 2]
        and smoothed[i] >= min_height
    ]
    peaks = merge_shallow_peaks(peaks, smoothed)
    half = MOTOR_CURRENT_SMOOTHING_BINS // 2
    peaks = [
        max(0, p - half) + int(np.argmax(counts[max(0, p - half) : p + half + 1])) for p in peaks
    ]
    cut_points = [
        centres[left + int(np.argmin(smoothed[left : right + 1]))]
        for left, right in zip(peaks[:-1], peaks[1:], strict=True)
    ]
    bounds = [-np.inf, *cut_points, np.inf]
    rows = []
    for state, peak in enumerate(peaks):
        low, high = bounds[state], bounds[state + 1]
        share = float(((values > low) & (values <= high)).mean())
        rows.append(
            {
                "state": state,
                "peak_amps": round(float(centres[peak]), 2),
                "lower_cut_amps": low,
                "upper_cut_amps": high,
                "share_of_readings": share,
            }
        )
    return pd.DataFrame(rows)


def motor_current_hypothesis_shares(motor_current: pd.Series) -> pd.DataFrame:
    """Share of readings within a tolerance of each hypothesised 0/4/7/9 A level."""
    values = motor_current.dropna()
    return pd.DataFrame(
        {
            "level_amps": MOTOR_CURRENT_HYPOTHESIS_AMPS,
            "share_within_tolerance": [
                float(((values - level).abs() <= MOTOR_CURRENT_HYPOTHESIS_TOLERANCE_AMPS).mean())
                for level in MOTOR_CURRENT_HYPOTHESIS_AMPS
            ],
        }
    )


ANALYST_NOTES_HEADING = "## 11. Analyst notes (written by hand, preserved on re-run)"
EMPTY_ANALYST_NOTES = f"{ANALYST_NOTES_HEADING}\n\n_No notes yet._\n"


def markdown_table(frame: pd.DataFrame, float_format: str = "{:.4g}") -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""

    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return float_format.format(value)
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def preserved_analyst_notes(report_path: Path) -> str:
    """Return the hand-written analyst notes section of an existing report, if any."""
    if not report_path.exists():
        return EMPTY_ANALYST_NOTES
    text = report_path.read_text(encoding="utf-8")
    position = text.find(ANALYST_NOTES_HEADING)
    return text[position:] if position >= 0 else EMPTY_ANALYST_NOTES


def render_data_audit_report(results: dict, report_path: Path) -> str:
    """Build the markdown audit report from measured results; keeps existing analyst notes."""
    s, roles = results["schema"], results["roles"]
    ts, cad = results["timestamps"], results["cadence"]
    fig = {
        name: path.relative_to(report_path.parent).as_posix()
        for name, path in results["figures"].items()
    }
    in_order = ts["non_monotonic_steps_in_file_order"] == 0
    gap_classes = pd.DataFrame(
        {
            "class": ["normal (= cadence)", "jitter", "gap", "of which long gap"],
            "count": [cad["at_mode"], cad["jitter"], cad["gaps"], cad["long_gaps"]],
            "percent of deltas": [
                100 * cad[k] / cad["n_deltas"] for k in ("at_mode", "jitter", "gaps", "long_gaps")
            ],
        }
    )
    dtype_table = pd.DataFrame(
        {
            "column": list(s["dtypes"]),
            "dtype": list(s["dtypes"].values()),
            "missing": [s["missing_per_column"][c] for c in s["dtypes"]],
        }
    )
    split_table = pd.DataFrame(
        [
            {
                "candidate": c.name,
                "start": c.start,
                "end": c.end,
                "train rows": c.train_rows,
                "overlapping events": c.overlapping_event_ids or "none",
                "test rows": c.test_rows,
                "test rows in reported windows": c.test_rows_in_windows,
            }
            for c in results["split"]
        ]
    )
    digital = results["digital"].assign(
        active_fraction=results["digital"]["active_fraction"].map("{:.4f}".format)
    )
    by_month = results["digital_by_month"].reset_index().rename(columns={"timestamp": "month"})
    quantiles = results["motor"]["quantiles"].rename("Motor_current (A)")
    common_rounded = results["motor"]["common_rounded_0_1"].rename("readings")
    near_constant = results["near_constant_fraction"]

    lines = [
        "# MetroGuard: Step 1 Data Audit Report",
        "",
        "Generated by `pipeline/step01_audit_raw_data.py`. Every number below was computed from "
        "the raw CSV in that run. **No model was trained and no detection-performance claim is "
        "made in this audit.**",
        "",
        "## 1. Source, DOI, licence",
        "",
        "- Dataset: MetroPT-3 (Air Compressor), UCI Machine Learning Repository, "
        "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset",
        "- DOI: https://doi.org/10.24432/C5VW3R",
        "- Licence: CC BY 4.0. Attribution: MetroPT-3 Dataset, UCI Machine Learning Repository, "
        "https://doi.org/10.24432/C5VW3R.",
        "",
        "## 2. File and schema",
        "",
        f"- File: `{s['path'].name}` ({s['size_mb']:.1f} MB)",
        f"- Shape: {s['n_rows']:,} rows x {s['n_columns']} columns",
        f"- Timestamp column: `{roles.timestamp}`",
        "- Unnamed index column(s): "
        f"{', '.join(f'`{c}`' for c in roles.index_columns) or 'none'} "
        "(excluded from sensor analysis and from the parquet cache)",
    ]
    for column, info in results["index_columns"].items():
        lines.append(
            f"  - `{column}`: {info['min']:,} to {info['max']:,}, most common step "
            f"{info['most_common_step']}, strictly increasing: {info['strictly_increasing']}"
        )
    lines += [
        f"- Analog sensors ({len(roles.analog)}): {', '.join(roles.analog)}",
        f"- Digital sensors ({len(roles.digital)} of 8 expected; matched by header name, "
        f"verified by values): {', '.join(roles.digital)}",
        f"- Expected digital signals not found: {roles.expected_digital_missing or 'none'}",
        "- Columns outside the expected digital list whose values are strictly 0/1: "
        f"{roles.unexpected_binary or 'none'}",
        "",
        "Column dtypes as read and missing values:",
        "",
        markdown_table(dtype_table),
        "",
        "First five rows:",
        "",
        markdown_table(s["head"], "{:.6g}"),
        "",
        "Last five rows:",
        "",
        markdown_table(s["tail"], "{:.6g}"),
        "",
        "Numeric summary of sensor columns:",
        "",
        markdown_table(results["sensor_describe"].reset_index(names="statistic")),
        "",
        "## 3. Timestamps and cadence",
        "",
        f"- Parse failures with the explicit format `%Y-%m-%d %H:%M:%S`: {ts['parse_failures']}",
        f"- Range: {ts['min']} to {ts['max']} (span {ts['span']})",
        "- Timezone: not stated by the source. Assumption: timestamps and failure reports use the "
        "same local time.",
        "- UCI metadata gives conflicting sampling rates (1 Hz in one field, 0.1 Hz in another). "
        f"**The file itself shows a dominant interval of {cad['mode_positive']:g} s "
        f"({1 / cad['mode_positive']:g} Hz).**",
        f"- Positive deltas: min {cad['min_positive']:g} s, median {cad['median_positive']:g} s, "
        f"mean {cad['mean_positive']:.3f} s, mode {cad['mode_positive']:g} s",
        f"- Zero deltas (duplicate timestamps after sorting): {cad['zero']:,}; negative deltas "
        f"after sorting: {cad['negative']}",
        "",
        "Most common positive deltas:",
        "",
        markdown_table(cad["common_deltas"]),
        "",
        f"![timestamp deltas]({fig['timestamp_delta_distribution']})",
        "",
        "## 4. Missing values, duplicates, order, gaps",
        "",
        f"- Missing values: {sum(s['missing_per_column'].values()):,} in total "
        "(per column in section 2)",
        f"- Duplicate rows (all columns incl. index): {s['duplicate_rows_all_columns']:,}; "
        f"excluding the index column: {s['duplicate_rows_excluding_index']:,}",
        f"- Duplicate timestamps: {s['duplicate_timestamps']:,}",
        f"- File order: {ts['non_monotonic_steps_in_file_order']:,} backward and "
        f"{ts['zero_steps_in_file_order']:,} zero timestamp steps before sorting "
        f"({'already in time order' if in_order else 'not in time order'})",
        "",
        f"**Gap definition.** With the observed cadence c = {cad['mode_positive']:g} s: a delta "
        f"equal to c is *normal*; a delta in (c, {cad['gap_threshold_seconds']:g} s] is *jitter* "
        f"(up to {results['gap_multiplier']:g}x cadence, not an error); a delta above "
        f"{cad['gap_threshold_seconds']:g} s is a *gap*; a delta of at least "
        f"{cad['long_gap_threshold_seconds'] / 3600:g} h is a *long gap*.",
        "",
        markdown_table(gap_classes, "{:.3f}"),
        "",
        f"Time without readings inside gaps: {cad['seconds_lost_in_gaps'] / 86400:.2f} days.",
        "",
        "Ten longest gaps:",
        "",
        markdown_table(results["longest_gaps"]),
        "",
        f"![daily row counts]({fig['daily_row_counts']})",
        "",
        f"**Frozen-value runs.** Stretches of at least {results['frozen_run_min_minutes']:g} min "
        "in which all analog sensors repeat exactly the same value from reading to reading. "
        "Readings exist, but the values do not change, which suggests repeated or stuck logging "
        "rather than real behaviour. They are reported, not removed.",
        "",
        markdown_table(results["frozen_runs"]),
        "",
        f"Total readings in frozen runs: {int(results['frozen_runs']['readings'].sum()):,} "
        f"({100 * results['frozen_runs']['readings'].sum() / s['n_rows']:.2f}% of rows).",
        "",
        "## 5. Digital signals",
        "",
        markdown_table(digital),
        "",
        f"Near-constant flag: active fraction below {near_constant:.1%} or above "
        f"{1 - near_constant:.1%}. No column was recoded or dropped.",
        "",
        "Active fraction per month:",
        "",
        markdown_table(by_month, "{:.3f}"),
        "",
        f"![digital signal activity]({fig['digital_signal_activity']})",
        "",
        "## 6. Reported failure windows and coverage",
        "",
        "Windows are treated as reported intervals, not dense labels. Boundaries are inclusive "
        "at both ends: [start, end].",
        "",
        markdown_table(results["events"]),
        "",
        f"- `expected_at_cadence` assumes one reading every {cad['mode_positive']:g} s. "
        "`fully_covered` means no gap above the gap threshold inside the window, counting from "
        "the window start to the first reading and from the last reading to the window end.",
        "- Event 1 ends at 23:59:00 (minute precision). Under the inclusive rule, "
        f"{results['event1_after_end']} readings between 23:59:00 and 00:00:00 fall outside the "
        "window. They are reported here; the window is not extended.",
        "- Missing coverage inside a window is a property of the data file, not a model result; "
        "no model exists yet.",
        "",
        f"![analog signals overview]({fig['analog_signals_overview']})",
        "",
    ]
    for window in REPORTED_FAILURE_WINDOWS:
        key = f"event{window.event_id}_zoom"
        lines += [f"![event {window.event_id} zoom]({fig[key]})", ""]
    lines += [
        "UCI metadata ambiguities (reported, not resolved):",
        "",
        '- UCI numbers two events as "#1"; they are renumbered 1-4 here for convenience only.',
        '- The maintenance note for event 2 reads "Maintenance on 30Apr at 12:00", which '
        "precedes the event (29-30 May). It may be a typo for 30 May; it is not corrected.",
        "- The timezone of timestamps and reports is not stated (see section 3).",
        "",
        "## 7. Candidate chronological split",
        "",
        markdown_table(split_table),
        "",
        "No train/test files were written and no scaler was fitted in step 1.",
        "",
        "## 8. Motor current",
        "",
        f"- Unique values: {results['motor']['unique_values']:,}",
        "",
        markdown_table(quantiles.rename_axis("quantile").reset_index()),
        "",
        "Most common values rounded to 0.1 A:",
        "",
        markdown_table(common_rounded.rename_axis("Motor_current (A)").reset_index()),
        "",
        "Candidate states from a transparent rule: peaks of a 0.1 A histogram smoothed over "
        f"{MOTOR_CURRENT_SMOOTHING_BINS} bins, keeping peaks whose smoothed bin holds at least "
        f"{MOTOR_CURRENT_MIN_PEAK_SHARE:.1%} of all readings; neighbouring peaks are merged "
        f"unless the valley between them is below {MOTOR_CURRENT_MAX_VALLEY_RATIO:.0%} of the "
        "lower peak; `peak_amps` is the most populated raw bin near each peak; cut points are "
        "the minima between peaks. Provisional only, not a final discretisation.",
        "",
        markdown_table(results["motor_states"], "{:.3g}"),
        "",
        f"Share of readings within ±{MOTOR_CURRENT_HYPOTHESIS_TOLERANCE_AMPS} A of the "
        "hypothesised 0/4/7/9 A levels:",
        "",
        markdown_table(results["motor_hypothesis"], "{:.4f}"),
        "",
        f"![motor current histogram]({fig['motor_current_histogram']})",
        "",
        f"![motor current time series]({fig['motor_current_timeseries']})",
        "",
        "## 9. Caveats",
        "",
        "- Failure windows are company-reported intervals; rows inside them are not uniformly "
        "faulty and rows outside them are not guaranteed normal.",
        "- Sensor units are not stated in the CSV; plots label raw values.",
        "- Visual observations, unanswered questions and step 2 recommendations are in section "
        "11 and in `reports/decision_log.md`.",
        "",
        "## 10. Statement",
        "",
        "No model was trained, no threshold was chosen and no detection-performance or causal "
        "claim is made in this audit.",
        "",
    ]
    return "\n".join(lines) + "\n" + preserved_analyst_notes(report_path)
