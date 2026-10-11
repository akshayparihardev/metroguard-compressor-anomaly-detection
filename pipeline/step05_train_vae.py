"""Step 5: train a VAE on the GPU; score windows; freeze calib thresholds before test scoring."""

import argparse

import pandas as pd
import torch

from metroguard import artifact_names as artifacts
from metroguard.alerting.alert_thresholds import fit_thresholds, freeze_thresholds
from metroguard.config_loader import load_config
from metroguard.data.chronological_split import CALIB, FIT_TRAIN, FIT_VALIDATION, TEST
from metroguard.models import variational_autoencoder as vae_lib
from metroguard.models.restricted_boltzmann_machine import select_device
from metroguard.reporting import plotting
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step05_train_vae"
MODEL_NAME = "vae"


def main() -> None:
    """Train the VAE with early stopping on fit_validation, then score all windows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    vae_config = sections["vae"]
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    inputs = [
        paths.processed_dir / artifacts.STANDARDISED_FEATURES_FILE,
        paths.processed_dir / artifacts.WINDOW_FEATURES_FILE,
    ]
    weights_path = paths.models_dir / artifacts.VAE_WEIGHTS_FILE
    scores_path = paths.scores_dir / artifacts.VAE_SCORES_FILE
    history_path = paths.tables_dir / artifacts.VAE_HISTORY_FILE
    figure_path = paths.figures_dir / "vae" / "vae_training_curves.png"
    fingerprint = config_fingerprint(
        {k: sections[k] for k in ("seed", "split", "discretise", "windows", "vae", "thresholds")}
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME,
        paths.logs_dir,
        inputs,
        [weights_path, scores_path, history_path, figure_path],
        fingerprint,
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        device = select_device()
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        log.info("Device: %s (%s)", device, device_name)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        table = pd.read_parquet(inputs[1], columns=["window_start", "split", "usable"])
        standardised = pd.read_parquet(inputs[0])
        if not standardised["window_start"].equals(table["window_start"]):
            raise ValueError("Standardised features and window table are not row-aligned")
        features = standardised.drop(columns="window_start").to_numpy(dtype="float32")
        data = torch.from_numpy(features).to(device)

        usable = table["usable"].to_numpy()
        masks = {
            name: torch.from_numpy((table["split"] == name).to_numpy() & usable).to(device)
            for name in (FIT_TRAIN, FIT_VALIDATION, CALIB)
        }
        is_test = torch.from_numpy((table["split"] == TEST).to_numpy()).to(device)
        log.info(
            "Features %d; fit_train %d, fit_validation %d, calib %d windows",
            data.shape[1],
            int(masks[FIT_TRAIN].sum()),
            int(masks[FIT_VALIDATION].sum()),
            int(masks[CALIB].sum()),
        )

        model = vae_lib.VariationalAutoencoder(
            data.shape[1], vae_config["hidden_sizes"], vae_config["latent_dim"]
        ).to(device)
        history = vae_lib.train_vae(
            model,
            data[masks[FIT_TRAIN]],
            data[masks[FIT_VALIDATION]],
            vae_config["epochs"],
            vae_config["batch_size"],
            vae_config["learning_rate"],
            vae_config["beta"],
            vae_config["early_stopping_patience"],
        )
        best = history.loc[history["validation_loss"].idxmin()]
        log.info(
            "Stopped after %d of %d epochs; best epoch %d (validation loss %.4f)",
            len(history),
            vae_config["epochs"],
            int(best["epoch"]),
            best["validation_loss"],
        )
        paths.models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weights_path)

        all_scores = torch.full((len(table), 2), float("nan"), device=device)
        all_scores[~is_test] = vae_lib.anomaly_scores(model, data[~is_test])
        for column, score_name in enumerate(vae_lib.SCORE_NAMES):
            calib_scores = all_scores[masks[CALIB], column].cpu().numpy()
            thresholds = fit_thresholds(calib_scores, sections["thresholds"]["percentiles"])
            frozen_at = freeze_thresholds(
                paths.tables_dir / artifacts.THRESHOLDS_FILE,
                MODEL_NAME,
                score_name,
                thresholds,
                len(calib_scores),
            )
            log.info(
                "Frozen %s %s thresholds %s at %s", MODEL_NAME, score_name, thresholds, frozen_at
            )
        all_scores[is_test] = vae_lib.anomaly_scores(model, data[is_test])

        output = pd.concat(
            [table, pd.DataFrame(all_scores.cpu().numpy(), columns=list(vae_lib.SCORE_NAMES))],
            axis=1,
        )
        paths.scores_dir.mkdir(parents=True, exist_ok=True)
        output.to_parquet(scores_path, index=False)
        paths.tables_dir.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)
        plotting.plot_vae_training_curves(history, figure_path)
        log.info("Wrote %s, %s and %s", scores_path, history_path, figure_path)


if __name__ == "__main__":
    main()
