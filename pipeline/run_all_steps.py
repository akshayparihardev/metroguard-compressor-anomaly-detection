"""Run steps 2 to 8 in order; each step skips itself when its outputs are up to date."""

import argparse
import subprocess
import sys
from pathlib import Path

from metroguard.config_loader import load_config
from metroguard.step_run_tracking import config_fingerprint, configure_step_logging, track_step_run

STEP_NAME = "run_all_steps"
PIPELINE_DIR = Path(__file__).resolve().parent
STEP_SCRIPTS = (
    "step02_build_features.py",
    "step03_score_zscore_baseline.py",
    "step04_train_rbm.py",
    "step05_train_vae.py",
    "step06_fit_markov_states.py",
    "step07_build_bayesian_network.py",
    "step08_evaluate_alerts.py",
)


def main() -> None:
    """Run every step after the audit with the current Python interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute every step")
    args = parser.parse_args()

    config = load_config()
    log = configure_step_logging(STEP_NAME, config.paths.logs_dir)
    with track_step_run(STEP_NAME, config.paths.logs_dir, config_fingerprint(config.sections)):
        for script in STEP_SCRIPTS:
            log.info("Running %s", script)
            command = [sys.executable, str(PIPELINE_DIR / script)]
            if args.force:
                command.append("--force")
            if subprocess.run(command, check=False).returncode != 0:
                log.error("%s failed; stopping", script)
                sys.exit(1)


if __name__ == "__main__":
    main()
