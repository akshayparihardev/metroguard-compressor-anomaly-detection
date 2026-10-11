"""Step 10: export saved pipeline results to dashboard/data for the read-only static site."""

import argparse
import json

import pandas as pd

from metroguard import dashboard_export as export
from metroguard.config_loader import PROJECT_ROOT, load_config
from metroguard.event_evaluation import scored_models
from metroguard.failure_windows import REPORTED_FAILURE_WINDOWS
from metroguard.step_run_tracking import (
    config_fingerprint,
    configure_step_logging,
    is_step_up_to_date,
    track_step_run,
)

STEP_NAME = "step10_build_dashboard_data"
DASHBOARD_DATA_DIR = PROJECT_ROOT / "dashboard" / "data"
SCORE_FILES = (
    "zscore_baseline_scores.parquet",
    "rbm_scores.parquet",
    "vae_scores.parquet",
    "markov_scores.parquet",
)


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
        tables / "metrics.csv",
        tables / "thresholds.json",
        tables / "rbm_training_history.csv",
        tables / "vae_training_history.csv",
        tables / "bayesian_network_edges.csv",
        tables / "bayesian_network_demonstrations.md",
        tables / "bayesian_network_alert_context.md",
        paths.reports_dir / "data_audit_report.md",
        *[paths.scores_dir / f for f in SCORE_FILES],
    ]
    event_files = [f"event_{w.event_id}.json" for w in REPORTED_FAILURE_WINDOWS]
    outputs = [
        DASHBOARD_DATA_DIR / name
        for name in (
            "timeline.json", "metrics.json", "training.json", "audit.json", "network.json",
            "context.json", *event_files,
        )
    ]  # fmt: skip
    fingerprint = config_fingerprint(
        {k: sections[k] for k in ("seed", "windows", "alerts", "evaluation", "thresholds", "rbm")}
    )
    if not args.force and is_step_up_to_date(
        STEP_NAME, paths.logs_dir, inputs, outputs, fingerprint
    ):
        log.info("%s is up to date; use --force to recompute", STEP_NAME)
        return

    with track_step_run(STEP_NAME, paths.logs_dir, fingerprint):
        thresholds = json.loads((tables / "thresholds.json").read_text(encoding="utf-8"))
        models = scored_models(paths.scores_dir, sections["rbm"]["k_values"])
        lead_hours = sections["evaluation"]["lead_tolerance_hours"]
        primary = float(sections["thresholds"]["primary_percentile"])
        metrics = pd.read_csv(tables / "metrics.csv")
        edges = pd.read_csv(tables / "bayesian_network_edges.csv")
        metrics_payload = export.build_metrics(metrics, thresholds, lead_hours)

        files = {
            "timeline.json": export.build_timeline(models, thresholds, primary),
            "metrics.json": metrics_payload,
            "training.json": export.build_training(
                pd.read_csv(tables / "rbm_training_history.csv"),
                pd.read_csv(tables / "vae_training_history.csv"),
            ),
            "audit.json": export.build_audit(
                (paths.reports_dir / "data_audit_report.md").read_text(encoding="utf-8")
            ),
            "network.json": export.build_network(
                edges,
                sections["bayesian_network"]["nodes"],
                (tables / "bayesian_network_demonstrations.md").read_text(encoding="utf-8"),
                (tables / "bayesian_network_alert_context.md").read_text(encoding="utf-8"),
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
        for window in REPORTED_FAILURE_WINDOWS:
            files[f"event_{window.event_id}.json"] = export.build_event_series(
                models, thresholds, window.event_id, lead_hours, primary
            )
        DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (DASHBOARD_DATA_DIR / name).write_text(export.compact_json(payload), encoding="utf-8")
        sizes = {name: (DASHBOARD_DATA_DIR / name).stat().st_size for name in files}
        log.info("Wrote %d files to %s (%.0f KB total)", len(files), DASHBOARD_DATA_DIR,
                 sum(sizes.values()) / 1024)  # fmt: skip


if __name__ == "__main__":
    main()
