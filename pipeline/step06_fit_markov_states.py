"""Step 6: fit a Markov chain on motor-current states (fit only) and score each window."""

import argparse
import json

import numpy as np
import pandas as pd

from metroguard import markov_state_model as msm
from metroguard.alert_thresholds import fit_thresholds, freeze_thresholds
from metroguard.chronological_split import FIT_SPLITS, TEST
from metroguard.config_loader import load_config
from metroguard.feature_builder import TIMESTAMP_COLUMN
from metroguard.raw_data_loader import cache_path, load_readings_cache
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step06_fit_markov_states"
MODEL_NAME = "markov_state_model"
WINDOW_FEATURES_FILE = "window_features.parquet"
CUT_POINTS_FILE = "motor_current_state_cut_points.json"
MATRIX_FILE = "markov_transition_matrix.json"
SCORES_FILE = "markov_scores.parquet"


def main() -> None:
    """Estimate transitions on fit readings, freeze calib thresholds, then score test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    features_path = paths.processed_dir / WINDOW_FEATURES_FILE
    cut_points_path = paths.models_dir / CUT_POINTS_FILE
    matrix_path = paths.models_dir / MATRIX_FILE
    scores_path = paths.scores_dir / SCORES_FILE
    inputs = [cache_path(paths.processed_dir), features_path, cut_points_path]
    fingerprint = config_fingerprint(
        {k: sections[k] for k in ("seed", "audit", "split", "windows", "thresholds")}
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, inputs, [matrix_path, scores_path], fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        readings = load_readings_cache(paths.processed_dir)[[TIMESTAMP_COLUMN, "Motor_current"]]
        table = pd.read_parquet(features_path, columns=["window_start", "split", "usable"])
        cut_points = json.loads(cut_points_path.read_text())["cut_points_amps"]
        n_states = len(cut_points) + 1
        if n_states < 2:
            raise RuntimeError("Audit found no distinct motor-current states; skip this step")

        window_seconds = sections["windows"]["size_seconds"]
        reading_window = readings[TIMESTAMP_COLUMN].dt.floor(f"{window_seconds}s")
        window_row = pd.Index(table["window_start"]).get_indexer(reading_window)
        in_fit_window = (table["split"].isin(FIT_SPLITS) & table["usable"]).to_numpy()[window_row]
        cadence = readings[TIMESTAMP_COLUMN].diff().dt.total_seconds().median()
        max_gap = config.audit.gap_multiplier * cadence

        states = msm.assign_reading_states(readings["Motor_current"], cut_points)
        valid = msm.valid_transitions(readings[TIMESTAMP_COLUMN], max_gap)
        counts = msm.count_transitions(states, in_fit_window, valid, n_states)
        matrix = msm.fit_transition_matrix(counts)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        matrix_path.write_text(
            json.dumps({"cut_points_amps": cut_points, "counts": counts.tolist(),
                        "transition_matrix": matrix.tolist()}, indent=2)
        )  # fmt: skip
        log.info("Transition counts on fit readings:\n%s", counts)
        log.info("Transition matrix (add-one smoothing):\n%s", np.round(matrix, 5))

        surprisal = msm.transition_surprisal(states, matrix, valid)
        scores = msm.window_scores(reading_window, surprisal).reindex(table["window_start"])
        scores = scores.to_numpy(dtype=np.float32)
        is_test = (table["split"] == TEST).to_numpy()
        is_calib = (table["split"] == "calib").to_numpy() & table["usable"].to_numpy()
        calib_scores = scores[is_calib]
        calib_scores = calib_scores[~np.isnan(calib_scores)]
        thresholds = fit_thresholds(calib_scores, sections["thresholds"]["percentiles"])
        frozen_at = freeze_thresholds(
            paths.tables_dir / "thresholds.json", MODEL_NAME, msm.SURPRISAL, thresholds,
            len(calib_scores),
        )  # fmt: skip
        log.info("Frozen %s thresholds %s at %s", MODEL_NAME, thresholds, frozen_at)

        output = table.assign(**{msm.SURPRISAL: scores})
        output.loc[is_test, msm.SURPRISAL] = scores[is_test]
        paths.scores_dir.mkdir(parents=True, exist_ok=True)
        output.to_parquet(scores_path, index=False)
        log.info("Windows without a valid transition (NaN score): %d of %d",
                 int(np.isnan(scores).sum()), len(scores))  # fmt: skip
        log.info("Wrote %s and %s", matrix_path, scores_path)


if __name__ == "__main__":
    main()
