"""All figure functions, sharing one consistent style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from metroguard.failure_windows import REPORTED_FAILURE_WINDOWS, FailureWindow  # noqa: E402

FIGURE_DPI = 120
WINDOW_SHADE_COLOUR = "tab:red"
WINDOW_SHADE_ALPHA = 0.2

plt.rcParams.update({"figure.dpi": FIGURE_DPI, "axes.grid": True, "grid.alpha": 0.3})


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as PNG, close it and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def shade_failure_windows(ax: plt.Axes, windows: tuple[FailureWindow, ...]) -> None:
    """Shade reported failure windows on a time axis, labelled once in the legend."""
    for i, window in enumerate(windows):
        ax.axvspan(
            window.start,
            window.end,
            color=WINDOW_SHADE_COLOUR,
            alpha=WINDOW_SHADE_ALPHA,
            label="reported failure window" if i == 0 else None,
        )


def break_lines_at_gaps(view: pd.DataFrame, gap_seconds: float) -> pd.DataFrame:
    """Insert an empty row after each gap so line plots do not bridge missing data."""
    gap_after = view["timestamp"].diff().shift(-1).dt.total_seconds() > gap_seconds
    breaks = view.loc[gap_after, ["timestamp"]].assign(
        timestamp=lambda f: f["timestamp"] + pd.Timedelta(seconds=1)
    )
    return pd.concat([view, breaks]).sort_values("timestamp", kind="stable")


def plot_timestamp_delta_distribution(deltas_seconds: pd.Series, path: Path) -> Path:
    """Bar chart of positive timestamp differences, log count axis."""
    positive = deltas_seconds[deltas_seconds > 0]
    counts = positive.value_counts().sort_index()
    top = counts.nlargest(15).sort_index()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(top))
    ax.bar(positions, top.to_numpy(), color="tab:blue")
    ax.set_xticks(positions, [f"{v:g}" for v in top.index])
    ax.set_yscale("log")
    ax.set_xlabel("time difference between consecutive readings (s)")
    ax.set_ylabel("count (log scale)")
    ax.set_title(
        f"Timestamp deltas: 15 most common positive values "
        f"(of {len(counts)} distinct; non-positive deltas reported separately)"
    )
    ax.tick_params(axis="x", rotation=45)
    return save_figure(fig, path)


def plot_daily_row_counts(daily_counts: pd.Series, expected_per_day: float, path: Path) -> Path:
    """Readings per day with the full-cadence expectation and failure windows."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(
        daily_counts.index,
        daily_counts.to_numpy(),
        width=1.0,
        color="tab:blue",
        label="readings per day",
    )
    ax.axhline(
        expected_per_day,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"expected at observed cadence ({expected_per_day:,.0f})",
    )
    shade_failure_windows(ax, REPORTED_FAILURE_WINDOWS)
    ax.set_xlabel("date")
    ax.set_ylabel("readings")
    ax.set_title("Data coverage: readings per calendar day")
    ax.legend(loc="lower left")
    return save_figure(fig, path)


def plot_digital_signal_activity(activity_by_month: pd.DataFrame, path: Path) -> Path:
    """Heatmap of the active fraction of each digital signal per month."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(activity_by_month.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(activity_by_month.index)), [str(m) for m in activity_by_month.index])
    ax.set_yticks(range(len(activity_by_month.columns)), activity_by_month.columns)
    for (row, col), value in np.ndenumerate(activity_by_month.T.to_numpy()):
        ax.text(
            col,
            row,
            f"{value:.2f}",
            ha="center",
            va="center",
            color="white" if value < 0.6 else "black",
            fontsize=8,
        )
    ax.grid(False)
    ax.set_xlabel("month")
    ax.set_title("Digital signals: fraction of readings with value 1, per month")
    fig.colorbar(image, ax=ax, label="active fraction")
    return save_figure(fig, path)


def plot_analog_signals_overview(
    display_frame: pd.DataFrame, columns: list[str], resample_minutes: int, path: Path
) -> Path:
    """One panel per analog signal over the full period, failure windows shaded."""
    fig, axes = plt.subplots(len(columns), 1, figsize=(13, 1.9 * len(columns)), sharex=True)
    for ax, column in zip(axes, columns, strict=True):
        ax.plot(display_frame.index, display_frame[column], linewidth=0.5, color="tab:blue")
        shade_failure_windows(ax, REPORTED_FAILURE_WINDOWS)
        ax.set_ylabel(column, fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Analog signals, full period (display only: {resample_minutes}-min means; "
        "units not stated in source metadata)"
    )
    axes[-1].set_xlabel("time")
    return save_figure(fig, path)


def plot_event_window_zoom(
    readings: pd.DataFrame,
    columns: list[str],
    window: FailureWindow,
    context: pd.Timedelta,
    gap_seconds: float,
    path: Path,
) -> Path:
    """Raw analog readings around one reported window, window shaded, lines broken at gaps."""
    start, end = window.start - context, window.end + context
    view = readings[(readings["timestamp"] >= start) & (readings["timestamp"] <= end)]
    view = break_lines_at_gaps(view, gap_seconds)
    fig, axes = plt.subplots(len(columns), 1, figsize=(13, 1.9 * len(columns)), sharex=True)
    for ax, column in zip(axes, columns, strict=True):
        ax.plot(view["timestamp"], view[column], linewidth=0.5, color="tab:blue")
        shade_failure_windows(ax, (window,))
        ax.set_ylabel(column, fontsize=8)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Event {window.event_id}: raw readings, reported window "
        f"{window.start} to {window.end} shaded, ±{context} context"
    )
    axes[-1].set_xlabel("time")
    axes[-1].set_xlim(start, end)
    return save_figure(fig, path)


def plot_motor_current_histogram(
    motor_current: pd.Series, cut_points: list[float], path: Path
) -> Path:
    """Histogram of Motor_current (log count axis) with candidate state cut points."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(
        motor_current.dropna(),
        bins=np.arange(motor_current.min(), motor_current.max() + 0.1, 0.1),
        color="tab:blue",
    )
    for i, cut in enumerate(cut_points):
        ax.axvline(
            cut,
            color="black",
            linestyle="--",
            linewidth=1,
            label="candidate state cut point" if i == 0 else None,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Motor_current (A, 0.1 A bins)")
    ax.set_ylabel("readings (log scale)")
    ax.set_title("Motor_current distribution, all readings")
    if cut_points:
        ax.legend()
    return save_figure(fig, path)


def plot_motor_current_timeseries(
    readings: pd.DataFrame, start: pd.Timestamp, days: int, gap_seconds: float, path: Path
) -> Path:
    """Raw Motor_current over a short sample period, lines broken at gaps."""
    end = start + pd.Timedelta(days=days)
    view = readings[(readings["timestamp"] >= start) & (readings["timestamp"] < end)]
    view = break_lines_at_gaps(view[["timestamp", "Motor_current"]], gap_seconds)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(view["timestamp"], view["Motor_current"], linewidth=0.4, color="tab:blue")
    ax.set_xlabel("time")
    ax.set_ylabel("Motor_current (A)")
    ax.set_title(f"Motor_current, raw readings, sample {start:%Y-%m-%d} + {days} days")
    return save_figure(fig, path)
