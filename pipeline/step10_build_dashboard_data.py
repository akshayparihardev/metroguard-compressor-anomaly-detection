"""Step 10: export saved pipeline results to dashboard/data for the read-only static site."""

import argparse
import json

import pandas as pd
import torch

from metroguard import artifact_names as artifacts
from metroguard.alerting.event_evaluation import scored_models
from metroguard.config_loader import PROJECT_ROOT, load_config
from metroguard.data.failure_windows import REPORTED_FAILURE_WINDOWS
from metroguard.models import variational_autoencoder as vae_lib
from metroguard.reporting import dashboard_export as export
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step10_build_dashboard_data"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"


def main() -> None:
    """Write one JSON file per dashboard view from files the pipeline already produced."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if up to date")
    args = parser.parse_args()

    config = load_config()
    paths, sections = config.paths, config.sections
    log = configure_step_logging(STEP_NAME, paths.logs_dir)
    tables = paths.tables_dir
    inputs = [
        tables / artifacts.METRICS_CSV_FILE,
        tables / artifacts.THRESHOLDS_FILE,
        tables / artifacts.RBM_HISTORY_FILE,
        tables / artifacts.VAE_HISTORY_FILE,
        tables / artifacts.NETWORK_EDGES_FILE,
        tables / artifacts.NETWORK_DEMONSTRATIONS_FILE,
        tables / artifacts.NETWORK_ALERT_CONTEXT_FILE,
        paths.reports_dir / "data_audit_report.md",
        *[paths.scores_dir / f for f in artifacts.SCORE_FILES],
        paths.processed_dir / artifacts.STANDARDISED_FEATURES_FILE,
        paths.models_dir / artifacts.VAE_WEIGHTS_FILE,
    ]
    event_files = [f"event_{w.event_id}.json" for w in REPORTED_FAILURE_WINDOWS]
    outputs = [
        DASHBOARD_DATA_DIR / name
        for name in (
            "timeline.json",
            "metrics.json",
            "training.json",
            "audit.json",
            "network.json",
            "context.json",
            "latent.json",
            *event_files,
        )
    ]
    fingerprint = config_fingerprint(
        {
            k: sections[k]
            for k in ("seed", "windows", "alerts", "evaluation", "thresholds", "rbm", "vae")
        }
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, inputs, outputs, fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        thresholds = json.loads((tables / artifacts.THRESHOLDS_FILE).read_text(encoding="utf-8"))
        models = scored_models(paths.scores_dir, sections["rbm"]["k_values"])
        lead_hours = sections["evaluation"]["lead_tolerance_hours"]
        primary = float(sections["thresholds"]["primary_percentile"])
        metrics = pd.read_csv(tables / artifacts.METRICS_CSV_FILE)
        edges = pd.read_csv(tables / artifacts.NETWORK_EDGES_FILE)
        metrics_payload = export.build_metrics(metrics, thresholds, lead_hours)

        files = {
            "timeline.json": export.build_timeline(models, thresholds, primary),
            "metrics.json": metrics_payload,
            "training.json": export.build_training(
                pd.read_csv(tables / artifacts.RBM_HISTORY_FILE),
                pd.read_csv(tables / artifacts.VAE_HISTORY_FILE),
            ),
            "audit.json": export.build_audit(
                (paths.reports_dir / "data_audit_report.md").read_text(encoding="utf-8")
            ),
            "network.json": export.build_network(
                edges,
                sections["bayesian_network"]["nodes"],
                (tables / artifacts.NETWORK_DEMONSTRATIONS_FILE).read_text(encoding="utf-8"),
                (tables / artifacts.NETWORK_ALERT_CONTEXT_FILE).read_text(encoding="utf-8"),
            ),
            "context.json": {
                "lead_tolerance_hours": lead_hours,
                "merge_gap_minutes": sections["alerts"]["merge_gap_minutes"],
                "window_seconds": sections["windows"]["size_seconds"],
                "primary_percentile": primary,
                "seed": config.seed,
                "windows": export.failure_windows_payload(),
                "runtimes_seconds": export.read_runtimes(paths.logs_dir),
                "model_labels": export.MODEL_LABELS,
                "score_labels": export.SCORE_LABELS,
            },
        }
        files["latent.json"] = latent_view(paths, sections, thresholds, primary, config.seed)
        for window in REPORTED_FAILURE_WINDOWS:
            files[f"event_{window.event_id}.json"] = export.build_event_series(
                models, thresholds, window.event_id, lead_hours, primary
            )
        DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (DASHBOARD_DATA_DIR / name).write_text(export.compact_json(payload), encoding="utf-8")
        sizes = {name: (DASHBOARD_DATA_DIR / name).stat().st_size for name in files}
        log.info(
            "Wrote %d files to %s (%.0f KB total)",
            len(files),
            DASHBOARD_DATA_DIR,
            sum(sizes.values()) / 1024,
        )


def latent_view(paths, sections: dict, thresholds: dict, primary: float, seed: int) -> dict:
    """Latent means from the saved VAE weights (no retraining), projected to 3-D."""
    standardised = pd.read_parquet(paths.processed_dir / artifacts.STANDARDISED_FEATURES_FILE)
    scores = pd.read_parquet(paths.scores_dir / artifacts.VAE_SCORES_FILE)
    if not standardised["window_start"].equals(scores["window_start"]):
        raise ValueError("Standardised features and VAE scores are not row-aligned")
    features = torch.from_numpy(standardised.drop(columns="window_start").to_numpy(dtype="float32"))
    vae_config = sections["vae"]
    model = vae_lib.VariationalAutoencoder(
        features.shape[1], vae_config["hidden_sizes"], vae_config["latent_dim"]
    )
    weights = paths.models_dir / artifacts.VAE_WEIGHTS_FILE
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    latent = vae_lib.latent_means(model, features).numpy()
    line = thresholds["vae"]["reconstruction_error"]["thresholds"][f"{primary:g}"]
    alert = (scores["reconstruction_error"] > line).to_numpy()
    return export.build_latent(latent, scores[["window_start", "split", "usable"]], alert, seed)


if __name__ == "__main__":
    main()
