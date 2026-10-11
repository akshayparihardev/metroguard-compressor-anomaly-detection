"""Step 4: train RBMs with CD-k (k = 1, 5, 10) on the GPU and score windows by free energy."""

import argparse

import numpy as np
import pandas as pd
import torch

from metroguard import artifact_names as artifacts
from metroguard.alerting.alert_thresholds import fit_thresholds, freeze_thresholds
from metroguard.config_loader import load_config
from metroguard.data.chronological_split import CALIB, FIT_TRAIN, FIT_VALIDATION, TEST
from metroguard.models.restricted_boltzmann_machine import (
    RestrictedBoltzmannMachine,
    free_energy_scores,
    select_device,
    train_rbm,
)
from metroguard.reporting import plotting
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step04_train_rbm"
CURVES_FIGURE = "rbm_training_curves_cd_k.png"
SCORE_NAME = "free_energy"


def main() -> None:
    """Train one RBM per k, freeze calib thresholds, then score the test windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    rbm_config = sections["rbm"]
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    visible_path = paths.processed_dir / artifacts.RBM_VISIBLE_FILE
    features_path = paths.processed_dir / artifacts.WINDOW_FEATURES_FILE
    scores_path = paths.scores_dir / artifacts.RBM_SCORES_FILE
    history_path = paths.tables_dir / artifacts.RBM_HISTORY_FILE
    figure_path = paths.figures_dir / "rbm" / CURVES_FIGURE
    weight_paths = [
        paths.models_dir / artifacts.rbm_weights_file(k) for k in rbm_config["k_values"]
    ]
    outputs = [scores_path, history_path, figure_path, *weight_paths]
    fingerprint = config_fingerprint(
        {k: sections[k] for k in ("seed", "split", "discretise", "windows", "rbm", "thresholds")}
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, [visible_path, features_path], outputs, fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        device = select_device()
        log.info(
            "Device: %s%s",
            device,
            f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "",
        )
        table = pd.read_parquet(features_path, columns=["window_start", "split", "usable"])
        with np.load(visible_path) as archive:
            visible = torch.from_numpy(archive["visible"]).to(device=device, dtype=torch.float32)
            same_windows = np.array_equal(archive["window_start"], table["window_start"].to_numpy())
        if not same_windows:
            raise ValueError("RBM visible matrix and window table are not row-aligned")

        usable = table["usable"].to_numpy()
        masks = {
            name: torch.from_numpy((table["split"] == name).to_numpy() & usable).to(device)
            for name in (FIT_TRAIN, FIT_VALIDATION, CALIB)
        }
        is_test = torch.from_numpy((table["split"] == TEST).to_numpy()).to(device)
        fit_data, validation_data = visible[masks[FIT_TRAIN]], visible[masks[FIT_VALIDATION]]
        log.info(
            "Visible units %d; fit_train %d, fit_validation %d, calib %d windows",
            visible.shape[1],
            len(fit_data),
            len(validation_data),
            int(masks[CALIB].sum()),
        )

        histories: dict[int, pd.DataFrame] = {}
        scores = pd.DataFrame(index=table.index)
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        for k, weight_path in zip(rbm_config["k_values"], weight_paths, strict=True):
            generator = torch.Generator(device=device).manual_seed(config.seed)
            rbm = RestrictedBoltzmannMachine(
                visible.shape[1], rbm_config["n_hidden"], device, generator
            )
            history = train_rbm(
                rbm,
                fit_data,
                validation_data,
                k,
                rbm_config["epochs"],
                rbm_config["batch_size"],
                rbm_config["learning_rate"],
                rbm_config["momentum"],
                rbm_config["weight_decay"],
            )
            histories[k] = history
            torch.save(rbm.state_dict(), weight_path)
            last = history.iloc[-1]
            log.info(
                "k=%d epoch %d: fit recon %.5f, F %.3f | validation recon %.5f, F %.3f",
                k,
                int(last["epoch"]),
                last["fit_reconstruction_error"],
                last["fit_mean_free_energy"],
                last["validation_reconstruction_error"],
                last["validation_mean_free_energy"],
            )

            k_scores = torch.full((len(table),), float("nan"), device=device)
            k_scores[~is_test] = free_energy_scores(rbm, visible[~is_test])
            calib_scores = k_scores[masks[CALIB]].cpu().numpy()
            thresholds = fit_thresholds(calib_scores, sections["thresholds"]["percentiles"])
            frozen_at = freeze_thresholds(
                paths.tables_dir / artifacts.THRESHOLDS_FILE,
                f"rbm_k{k}",
                SCORE_NAME,
                thresholds,
                len(calib_scores),
            )
            log.info("Frozen rbm_k%d thresholds %s at %s", k, thresholds, frozen_at)
            k_scores[is_test] = free_energy_scores(rbm, visible[is_test])
            scores[f"{SCORE_NAME}_k{k}"] = k_scores.cpu().numpy()

        output = pd.concat([table, scores], axis=1)
        paths.scores_dir.mkdir(parents=True, exist_ok=True)
        output.to_parquet(scores_path, index=False)
        paths.tables_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(histories, names=["k"]).reset_index(level=0).to_csv(history_path, index=False)
        plotting.plot_rbm_training_curves(histories, figure_path)
        log.info("Wrote %s, %s and %s", scores_path, history_path, figure_path)


if __name__ == "__main__":
    main()
