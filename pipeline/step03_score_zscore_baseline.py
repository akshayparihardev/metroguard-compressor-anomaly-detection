"""Step 3: z-score baseline scores; thresholds are frozen from calib before test is scored."""

import argparse
import json

import pandas as pd

from metroguard import zscore_baseline as zb
from metroguard.alert_thresholds import fit_thresholds, freeze_thresholds
from metroguard.chronological_split import CALIB, FIT_SPLITS, TEST
from metroguard.config_loader import load_config
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step03_score_zscore_baseline"
MODEL_NAME = "zscore_baseline"
WINDOW_FEATURES_FILE = "window_features.parquet"
SCORES_FILE = "zscore_baseline_scores.parquet"
REFERENCE_FILE = "zscore_reference.json"


def main() -> None:
    """Fit the z-score reference on fit, freeze calib thresholds, then score test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    features_path = paths.processed_dir / WINDOW_FEATURES_FILE
    scores_path = paths.scores_dir / SCORES_FILE
    reference_path = paths.models_dir / REFERENCE_FILE
    thresholds_path = paths.tables_dir / "thresholds.json"
    fingerprint = config_fingerprint({k: sections[k] for k in ("seed", "split", "thresholds")})
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, [features_path], [scores_path, reference_path], fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        table = pd.read_parquet(features_path)
        usable = table["usable"].to_numpy()
        is_fit = (table["split"].isin(FIT_SPLITS)).to_numpy() & usable
        is_test = (table["split"] == TEST).to_numpy()
        mean_columns = zb.analog_mean_columns(table)

        reference = zb.fit_zscore_reference(table.loc[is_fit, mean_columns])
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(json.dumps(reference.to_dict(orient="index"), indent=2))

        scores = pd.DataFrame(index=table.index, columns=list(zb.SCORE_NAMES), dtype="float32")
        before_test = ~is_test
        scores.loc[before_test] = zb.zscore_scores(table.loc[before_test, mean_columns], reference)

        is_calib = (table["split"] == CALIB).to_numpy() & usable
        for score_name in zb.SCORE_NAMES:
            thresholds = fit_thresholds(
                scores.loc[is_calib, score_name].to_numpy(), sections["thresholds"]["percentiles"]
            )
            frozen_at = freeze_thresholds(
                thresholds_path, MODEL_NAME, score_name, thresholds, int(is_calib.sum())
            )
            log.info(
                "Frozen %s %s thresholds %s at %s", MODEL_NAME, score_name, thresholds, frozen_at
            )

        scores.loc[is_test] = zb.zscore_scores(table.loc[is_test, mean_columns], reference)
        output = scores.join(table[["window_start", "split", "usable"]])
        output = output[["window_start", "split", "usable", *zb.SCORE_NAMES]]
        paths.scores_dir.mkdir(parents=True, exist_ok=True)
        output.to_parquet(scores_path, index=False)
        log.info("Scored %d windows (%d usable); wrote %s", len(output), usable.sum(), scores_path)


if __name__ == "__main__":
    main()
