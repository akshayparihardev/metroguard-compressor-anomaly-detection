"""Step 2: aggregate readings into windows, split chronologically and build model inputs."""

import argparse
import json
from dataclasses import asdict

import numpy as np

from metroguard import chronological_split as split_lib
from metroguard import feature_builder as fb
from metroguard import raw_data_audit
from metroguard.config_loader import load_config
from metroguard.raw_data_loader import cache_path, load_readings_cache
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step02_build_features"
WINDOW_FEATURES_FILE = "window_features.parquet"
STANDARDISED_FEATURES_FILE = "window_standardised_features.parquet"
DISCRETE_STATES_FILE = "window_discrete_states.parquet"
RBM_VISIBLE_FILE = "window_rbm_visible.npz"
STANDARDISER_FILE = "feature_standardiser.json"
BIN_EDGES_FILE = "discretisation_bin_edges.json"
MOTOR_CUT_POINTS_FILE = "motor_current_state_cut_points.json"


def main() -> None:
    """Build windowed features and fit every transformation on the fit windows only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    processed = paths.processed_dir
    outputs = [
        processed / WINDOW_FEATURES_FILE,
        processed / STANDARDISED_FEATURES_FILE,
        processed / DISCRETE_STATES_FILE,
        processed / RBM_VISIBLE_FILE,
        paths.models_dir / STANDARDISER_FILE,
        paths.models_dir / BIN_EDGES_FILE,
        paths.models_dir / MOTOR_CUT_POINTS_FILE,
    ]
    fingerprint = config_fingerprint(
        {k: sections[k] for k in ("seed", "audit", "split", "windows", "discretise")}
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, [cache_path(processed)], outputs, fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        readings = load_readings_cache(processed)
        window_seconds = sections["windows"]["size_seconds"]
        table = fb.aggregate_windows(readings, window_seconds)
        expected = fb.expected_readings_per_window(readings[fb.TIMESTAMP_COLUMN], window_seconds)
        min_readings = fb.minimum_readings(expected, sections["windows"]["min_readings_fraction"])

        frozen_runs = raw_data_audit.frozen_value_runs(
            readings, fb.analog_columns(readings), config.audit.frozen_run_min_minutes
        )
        table["in_frozen_run"] = fb.flag_frozen_windows(
            table["window_start"], frozen_runs, window_seconds
        )
        table["too_few_readings"] = (table["reading_count"] < min_readings).to_numpy()
        table["usable"] = ~(table["in_frozen_run"] | table["too_few_readings"]).to_numpy()

        train_end = split_lib.training_period_end(
            readings[fb.TIMESTAMP_COLUMN].iloc[0], sections["split"]["train_period"]
        )
        table["split"] = split_lib.assign_split(
            table["window_start"],
            table["usable"].to_numpy(),
            train_end,
            sections["split"]["calib_fraction"],
            sections["split"]["fit_validation_fraction"],
        )
        fit_rows = (table["usable"] & table["split"].isin(split_lib.FIT_SPLITS)).to_numpy()
        calib_start = split_lib.split_start_time(table["window_start"], table["split"], "calib")

        features = fb.model_feature_columns(table)
        analog_features = fb.analog_feature_columns(table)
        standardiser = fb.fit_standardiser(table.loc[fit_rows, features])
        bin_edges = fb.fit_bin_edges(
            table.loc[fit_rows], analog_features, sections["discretise"]["n_bins"]
        )

        fit_motor_current = readings.loc[
            readings[fb.TIMESTAMP_COLUMN] < calib_start, "Motor_current"
        ]
        motor_states = raw_data_audit.motor_current_candidate_states(fit_motor_current)
        motor_cut_points = motor_states["upper_cut_amps"].iloc[:-1].tolist()
        table["motor_current_state"] = fb.assign_motor_state(
            table["Motor_current_mean"], motor_cut_points
        )

        standardised = fb.standardise(table, standardiser)
        states = fb.discretise(table, bin_edges)
        digital_binary = fb.binarise_digital(table, fb.digital_feature_columns(table))
        visible, visible_names = fb.one_hot_visible(states, bin_edges, digital_binary)

        processed.mkdir(parents=True, exist_ok=True)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        table.to_parquet(processed / WINDOW_FEATURES_FILE, index=False)
        standardised.insert(0, "window_start", table["window_start"])
        standardised.to_parquet(processed / STANDARDISED_FEATURES_FILE, index=False)
        discrete = states.join(digital_binary)
        discrete.insert(0, "window_start", table["window_start"])
        discrete.to_parquet(processed / DISCRETE_STATES_FILE, index=False)
        np.savez_compressed(
            processed / RBM_VISIBLE_FILE,
            window_start=table["window_start"].to_numpy(),
            visible=visible,
            column_names=np.array(visible_names),
        )
        (paths.models_dir / STANDARDISER_FILE).write_text(
            json.dumps(asdict(standardiser), indent=2), encoding="utf-8"
        )
        (paths.models_dir / BIN_EDGES_FILE).write_text(
            json.dumps(bin_edges, indent=2), encoding="utf-8"
        )
        (paths.models_dir / MOTOR_CUT_POINTS_FILE).write_text(
            json.dumps(
                {"cut_points_amps": motor_cut_points, "states": motor_states.to_dict("records")},
                indent=2,
            ),
            encoding="utf-8",
        )

        log.info(
            "Window size %d s, expected %.1f readings, minimum %d",
            window_seconds,
            expected,
            min_readings,
        )
        log.info("Training period ends (exclusive) %s; calib starts %s", train_end, calib_start)
        summary = table.groupby("split", observed=True).agg(
            windows=("usable", "size"),
            usable=("usable", "sum"),
            too_few_readings=("too_few_readings", "sum"),
            in_frozen_run=("in_frozen_run", "sum"),
            first=("window_start", "min"),
            last=("window_start", "max"),
        )
        log.info("Windows per split:\n%s", summary.to_string())
        log.info("Constant features in fit (scale set to 1): %s", standardiser.constant_columns)
        log.info("Bins per feature: %s", {c: len(e) + 1 for c, e in bin_edges.items()})
        log.info("RBM visible units: %d (%s)", visible.shape[1], visible.dtype)
        log.info(
            "Motor-current state cut points from fit only: %s (%d states)",
            motor_cut_points,
            len(motor_states),
        )
        log.info("Wrote %d files to %s and %s", len(outputs), processed, paths.models_dir)


if __name__ == "__main__":
    main()
