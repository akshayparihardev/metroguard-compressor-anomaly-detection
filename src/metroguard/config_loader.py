"""Read config.yaml into a typed, read-only configuration object."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class PathsConfig:
    """Absolute project paths, resolved relative to the project root."""

    raw_csv: Path
    processed_dir: Path
    models_dir: Path
    scores_dir: Path
    logs_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class AuditConfig:
    """Settings used only by the step 1 data audit."""

    gap_multiplier: float
    long_gap_hours: float
    near_constant_fraction: float
    event_context_hours: float
    display_resample_minutes: int
    frozen_run_min_minutes: float


@dataclass(frozen=True)
class ProjectConfig:
    """All configuration values; `sections` keeps the raw YAML for change detection."""

    seed: int
    paths: PathsConfig
    audit: AuditConfig
    sections: dict[str, Any]


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load config.yaml and resolve every path against the project root."""
    with config_path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    root = config_path.resolve().parent
    paths = PathsConfig(**{name: root / value for name, value in raw["paths"].items()})
    return ProjectConfig(
        seed=int(raw["seed"]),
        paths=paths,
        audit=AuditConfig(**raw["audit"]),
        sections=raw,
    )
