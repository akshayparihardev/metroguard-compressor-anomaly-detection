"""Read the raw MetroPT-3 CSV once, parse timestamps and cache a typed parquet copy."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
CACHE_FILE_NAME = "metropt3_sensor_readings.parquet"

# Expected digital signals from the UCI description, keyed by a normalised name.
# "DV_eletric" (as spelled in the file) and "DV_electric" both map to the same signal.
EXPECTED_DIGITAL_SIGNALS: dict[str, str] = {
    "comp": "COMP",
    "dv_eletric": "DV_eletric",
    "dv_electric": "DV_eletric",
    "towers": "Towers",
    "mpg": "MPG",
    "lps": "LPS",
    "pressure_switch": "Pressure_switch",
    "oil_level": "Oil_level",
    "caudal_impulses": "Caudal_impulses",
}


class RawDataMissingError(FileNotFoundError):
    """Raised when the official raw CSV is not where config.yaml says it is."""


@dataclass(frozen=True)
class ColumnRoles:
    """Which actual CSV headers play which role."""

    timestamp: str
    index_columns: list[str]
    digital: list[str]
    analog: list[str]
    expected_digital_missing: list[str]
    unexpected_binary: list[str]


def require_raw_csv(raw_csv: Path) -> None:
    """Stop with a clear message if the raw CSV is missing; never substitute data."""
    if not raw_csv.is_file():
        raise RawDataMissingError(
            f"Raw data not found at {raw_csv}. Download MetroPT3(AirCompressor).csv from "
            "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset and place it there. "
            "No synthetic or substitute data will be used."
        )


def read_raw_csv(raw_csv: Path) -> pd.DataFrame:
    """Read the CSV exactly as stored, keeping the timestamp column as text."""
    require_raw_csv(raw_csv)
    header = pd.read_csv(raw_csv, nrows=0).columns
    timestamp_column = find_timestamp_column(list(header))
    return pd.read_csv(raw_csv, dtype={timestamp_column: str})


def find_timestamp_column(columns: list[str]) -> str:
    """Return the single header whose name is 'timestamp' (case-insensitive)."""
    matches = [c for c in columns if c.strip().lower() == "timestamp"]
    if len(matches) != 1:
        raise ValueError(f"Expected one timestamp column, found {matches} in {columns}")
    return matches[0]


def parse_timestamps(text: pd.Series) -> tuple[pd.Series, int]:
    """Parse timestamps with the explicit format; return parsed values and failure count."""
    parsed = pd.to_datetime(text, format=TIMESTAMP_FORMAT, errors="coerce")
    return parsed, int(parsed.isna().sum())


def is_strictly_binary(values: pd.Series) -> bool:
    """True when every non-missing value is exactly 0 or 1."""
    unique = pd.unique(values.dropna())
    return len(unique) > 0 and bool(np.isin(unique, [0, 1]).all())


def identify_column_roles(raw: pd.DataFrame) -> ColumnRoles:
    """Map actual headers to timestamp, index, digital and analog roles, verified by values."""
    columns = list(raw.columns)
    timestamp = find_timestamp_column(columns)
    index_columns = [c for c in columns if c.startswith("Unnamed:") or c.strip() == ""]
    sensor_columns = [c for c in columns if c != timestamp and c not in index_columns]
    digital = [c for c in sensor_columns if c.strip().lower() in EXPECTED_DIGITAL_SIGNALS]
    found_canonical = {EXPECTED_DIGITAL_SIGNALS[c.strip().lower()] for c in digital}
    expected_missing = sorted(set(EXPECTED_DIGITAL_SIGNALS.values()) - found_canonical)
    analog = [c for c in sensor_columns if c not in digital]
    unexpected_binary = [c for c in analog if is_strictly_binary(raw[c])]
    return ColumnRoles(
        timestamp, index_columns, digital, analog, expected_missing, unexpected_binary
    )


def build_typed_readings(raw: pd.DataFrame, roles: ColumnRoles) -> pd.DataFrame:
    """Working copy sorted by time: float32 analog, bool digital (only if strictly 0/1)."""
    parsed, _ = parse_timestamps(raw[roles.timestamp])
    typed = pd.DataFrame({"timestamp": parsed})
    for column in roles.analog:
        typed[column] = raw[column].astype(np.float32)
    for column in roles.digital:
        typed[column] = raw[column].astype(bool) if is_strictly_binary(raw[column]) else raw[column]
    return typed.sort_values("timestamp", kind="stable").reset_index(drop=True)


def cache_path(processed_dir: Path) -> Path:
    """Location of the typed parquet cache."""
    return processed_dir / CACHE_FILE_NAME


def save_readings_cache(readings: pd.DataFrame, processed_dir: Path) -> Path:
    """Write the typed readings to parquet and return the path."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(processed_dir)
    readings.to_parquet(path, index=False)
    return path


def load_readings_cache(processed_dir: Path) -> pd.DataFrame:
    """Read the typed readings cache written by step 1 (never re-reads the raw CSV)."""
    path = cache_path(processed_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run pipeline/step01_audit_raw_data.py first.")
    return pd.read_parquet(path)
