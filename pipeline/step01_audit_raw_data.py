"""Step 1: audit the raw MetroPT-3 CSV, cache it as parquet and write the audit report."""

import argparse
import sys

import pandas as pd

from metroguard import plotting, raw_data_audit
from metroguard.config_loader import load_config
from metroguard.failure_windows import REPORTED_FAILURE_WINDOWS
from metroguard.raw_data_loader import (
    RawDataMissingError,
    build_typed_readings,
    cache_path,
    identify_column_roles,
    parse_timestamps,
    read_raw_csv,
    require_raw_csv,
    save_readings_cache,
)
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step01_audit_raw_data"
OVERVIEW_COLUMNS = [
    "Motor_current",
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
]
MOTOR_CURRENT_SAMPLE_DAYS = 2


def main() -> None:
    """Run the audit end to end, or skip it when outputs are up to date."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths = config.paths
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    try:
        require_raw_csv(paths.raw_csv)
    except RawDataMissingError as error:
        log.error("%s", error)
        sys.exit(1)

    figures_dir = paths.figures_dir / "data_audit"
    report_path = paths.reports_dir / "data_audit_report.md"
    fingerprint = config_fingerprint(
        {"paths": config.sections["paths"], "audit": config.sections["audit"]}
    )
    outputs = [report_path, cache_path(paths.processed_dir)]
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, [paths.raw_csv], outputs, fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        log.info("Reading %s", paths.raw_csv)
        raw = read_raw_csv(paths.raw_csv)
        roles = identify_column_roles(raw)
        parsed_in_file_order, parse_failures = parse_timestamps(raw[roles.timestamp])
        readings = build_typed_readings(raw, roles)
        cache = save_readings_cache(readings, paths.processed_dir)
        log.info("Cached %d typed readings to %s", len(readings), cache)

        ts = readings["timestamp"]
        deltas = raw_data_audit.timestamp_deltas_seconds(ts)
        cadence = raw_data_audit.cadence_summary(deltas, config.audit)
        cadence_seconds = cadence["mode_positive"]
        daily = raw_data_audit.daily_row_counts(ts)
        digital_by_month = raw_data_audit.digital_activity_by_month(readings, roles)
        motor_states = raw_data_audit.motor_current_candidate_states(readings["Motor_current"])
        overview_columns = [c for c in OVERVIEW_COLUMNS if c in readings.columns]
        missing_overview = sorted(set(OVERVIEW_COLUMNS) - set(overview_columns))
        if missing_overview:
            log.warning("Overview columns not in the file (not substituted): %s", missing_overview)
        display_frame = (
            readings.set_index("timestamp")[overview_columns]
            .resample(f"{config.audit.display_resample_minutes}min")
            .mean()
        )
        context = pd.Timedelta(hours=config.audit.event_context_hours)

        figures = {
            "timestamp_delta_distribution": plotting.plot_timestamp_delta_distribution(
                deltas, figures_dir / "timestamp_delta_distribution.png"
            ),
            "daily_row_counts": plotting.plot_daily_row_counts(
                daily, 86400 / cadence_seconds, figures_dir / "daily_row_counts.png"
            ),
            "digital_signal_activity": plotting.plot_digital_signal_activity(
                digital_by_month, figures_dir / "digital_signal_activity.png"
            ),
            "analog_signals_overview": plotting.plot_analog_signals_overview(
                display_frame,
                overview_columns,
                config.audit.display_resample_minutes,
                figures_dir / "analog_signals_overview.png",
            ),
            "motor_current_histogram": plotting.plot_motor_current_histogram(
                readings["Motor_current"],
                motor_states["upper_cut_amps"].iloc[:-1].tolist(),
                figures_dir / "motor_current_histogram.png",
            ),
            "motor_current_timeseries": plotting.plot_motor_current_timeseries(
                readings,
                ts.iloc[0].normalize(),
                MOTOR_CURRENT_SAMPLE_DAYS,
                cadence["gap_threshold_seconds"],
                figures_dir / "motor_current_timeseries.png",
            ),
        }
        for window in REPORTED_FAILURE_WINDOWS:
            figures[f"event{window.event_id}_zoom"] = plotting.plot_event_window_zoom(
                readings,
                overview_columns,
                window,
                context,
                cadence["gap_threshold_seconds"],
                figures_dir / f"event{window.event_id}_zoom_analog_signals.png",
            )

        results = {
            "schema": raw_data_audit.schema_summary(raw, roles, paths.raw_csv),
            "roles": roles,
            "index_columns": raw_data_audit.index_column_summary(raw, roles),
            "sensor_describe": raw[roles.analog + roles.digital].describe(),
            "timestamps": raw_data_audit.timestamp_order_summary(
                parsed_in_file_order, parse_failures
            ),
            "cadence": cadence,
            "gap_multiplier": config.audit.gap_multiplier,
            "longest_gaps": raw_data_audit.longest_gaps(ts),
            "frozen_runs": raw_data_audit.frozen_value_runs(
                readings, roles.analog, config.audit.frozen_run_min_minutes
            ),
            "frozen_run_min_minutes": config.audit.frozen_run_min_minutes,
            "digital": raw_data_audit.digital_signal_summary(raw, roles, config.audit),
            "digital_by_month": digital_by_month,
            "near_constant_fraction": config.audit.near_constant_fraction,
            "events": raw_data_audit.event_window_coverage(
                ts, cadence_seconds, cadence["gap_threshold_seconds"]
            ),
            "event1_after_end": raw_data_audit.readings_after_window_end_same_minute(
                ts, REPORTED_FAILURE_WINDOWS[0]
            ),
            "split": raw_data_audit.split_candidates(ts),
            "motor": raw_data_audit.motor_current_summary(readings["Motor_current"]),
            "motor_states": motor_states,
            "motor_hypothesis": raw_data_audit.motor_current_hypothesis_shares(
                readings["Motor_current"]
            ),
            "figures": figures,
        }
        report_path.write_text(
            raw_data_audit.render_data_audit_report(results, report_path), encoding="utf-8"
        )
        log.info("Wrote %s, %d figures in %s and %s", report_path, len(figures), figures_dir, cache)
        log.info(
            "Rows %d | cadence %.0f s | gaps %d (long %d) | duplicate timestamps %d | "
            "digital columns %d | motor-current candidate states %d",
            len(raw),
            cadence_seconds,
            cadence["gaps"],
            cadence["long_gaps"],
            results["schema"]["duplicate_timestamps"],
            len(roles.digital),
            len(motor_states),
        )


if __name__ == "__main__":
    main()
