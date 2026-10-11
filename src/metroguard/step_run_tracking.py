"""Skip-if-up-to-date checks and per-step runtime logging."""

import hashlib
import json
import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def config_fingerprint(config_values: Any) -> str:
    """Return a stable hash of the config values a step depends on."""
    encoded = json.dumps(config_values, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_path(logs_dir: Path, step_name: str) -> Path:
    """Path of the JSON file that records a step's last successful run."""
    return logs_dir / f"{step_name}_manifest.json"


def is_step_up_to_date(
    step_name: str,
    logs_dir: Path,
    inputs: Iterable[Path],
    outputs: Iterable[Path],
    fingerprint: str,
) -> bool:
    """True when all outputs exist, are newer than all inputs and the config is unchanged."""
    outputs = list(outputs)
    manifest = manifest_path(logs_dir, step_name)
    if not manifest.exists() or not all(path.exists() for path in outputs):
        return False
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    if recorded.get("config_fingerprint") != fingerprint:
        return False
    newest_input = max((path.stat().st_mtime for path in inputs), default=0.0)
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return oldest_output >= newest_input


def configure_step_logging(step_name: str, logs_dir: Path) -> logging.Logger:
    """Log to the console and to outputs/logs/<step_name>.log."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / f"{step_name}.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger(step_name)


@contextmanager
def track_step_run(step_name: str, logs_dir: Path, fingerprint: str) -> Iterator[None]:
    """Time a step and, on success, write its manifest with runtime and config fingerprint."""
    started = time.perf_counter()
    yield
    runtime_seconds = round(time.perf_counter() - started, 2)
    manifest = {
        "step": step_name,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "config_fingerprint": fingerprint,
    }
    manifest_path(logs_dir, step_name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.getLogger(step_name).info("%s finished in %.1f s", step_name, runtime_seconds)
