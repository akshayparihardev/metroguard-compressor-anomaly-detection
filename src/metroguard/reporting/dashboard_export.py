"""Step 10: reshape saved pipeline results into the small JSON files the dashboard reads."""

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from metroguard.alerting.event_evaluation import ScoredModel
from metroguard.data.chronological_split import CALIB, TEST
from metroguard.data.failure_windows import REPORTED_FAILURE_WINDOWS

LOG_SCORES = ("max_abs_z", "mean_abs_z", "reconstruction_error", "negative_elbo")
MODEL_LABELS = {
    "zscore_baseline": "Z-score baseline",
    "rbm_k1": "RBM, k = 1",
    "rbm_k5": "RBM, k = 5",
    "rbm_k10": "RBM, k = 10",
    "vae": "VAE",
    "markov_state_model": "Markov chain",
}
SCORE_LABELS = {
    "max_abs_z": "max |z|",
    "mean_abs_z": "mean |z|",
    "free_energy": "free energy",
    "reconstruction_error": "reconstruction error",
    "negative_elbo": "negative ELBO",
    "max_transition_surprisal": "transition surprisal",
}
LATENT_POINTS = 12_000
LATENT_SPLIT_CODES = {"fit_train": 0, "fit_validation": 0, CALIB: 1, TEST: 2}
EVENT_PAD_BEFORE_HOURS = 3
EVENT_PAD_AFTER_HOURS = 12
VALUE_DECIMALS = 4
LAYOUT_SWEEPS = 4


def compact_json(payload: object) -> str:
    """Serialise without whitespace; NaN must already have been turned into None."""
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def epoch_minutes(timestamp: pd.Timestamp) -> int:
    """Minutes since 1970-01-01 for a naive timestamp (the dashboard treats times as local)."""
    return int(timestamp.value // 60_000_000_000)


def read_markdown_sections(text: str) -> dict[str, dict]:
    """Split a generated markdown file into {heading: {tables, bullets, paragraphs}}."""
    sections: dict[str, dict] = {}
    current = sections.setdefault("", {"tables": [], "bullets": [], "paragraphs": []})
    block: list[str] = []

    def flush_table() -> None:
        if len(block) >= 2:
            header = _split_row(block[0])
            rows = [_split_row(line) for line in block[2:]]
            current["tables"].append({"columns": header, "rows": rows})
        block.clear()

    for line in text.splitlines():
        if line.startswith("|"):
            block.append(line)
            continue
        flush_table()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            current = sections.setdefault(heading, {"tables": [], "bullets": [], "paragraphs": []})
        elif line.lstrip().startswith("- "):
            current["bullets"].append(line.lstrip()[2:].strip())
        elif line.strip() and not line.startswith("!["):
            current["paragraphs"].append(line.strip())
    flush_table()
    return sections


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def cell_value(cell: str) -> object:
    """Turn a markdown cell into a number when it is one, otherwise keep the text."""
    plain = cell.replace(",", "").replace("`", "")
    try:
        number = float(plain)
    except ValueError:
        return cell.replace("`", "")
    if not np.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def table_records(table: dict) -> list[dict]:
    """A parsed markdown table as a list of {column: value} records."""
    return [
        {column: cell_value(cell) for column, cell in zip(table["columns"], row, strict=False)}
        for row in table["rows"]
    ]


def find_section(sections: dict[str, dict], prefix: str) -> dict:
    """The section whose heading starts with the prefix; fails loudly if the format drifted."""
    matches = [value for heading, value in sections.items() if heading.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one markdown section starting with {prefix!r}")
    return matches[0]


def first_match(pattern: str, text: str) -> re.Match:
    """Regex match that fails loudly so a report-format change cannot silently blank a number."""
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Pattern not found in report: {pattern}")
    return match


def hourly_maxima(scored: ScoredModel, index: pd.DatetimeIndex) -> list[float | None]:
    """Hourly maximum of the usable-window scores on a shared hourly index (None = no data)."""
    series = pd.Series(
        np.where(scored.usable, scored.scores, np.nan),
        index=pd.DatetimeIndex(scored.window_starts),
    )
    hourly = series.resample("1h").max().reindex(index)
    return [None if np.isnan(v) else round(float(v), VALUE_DECIMALS) for v in hourly]


def hourly_index(window_starts: pd.Series) -> pd.DatetimeIndex:
    """Hourly index covering every window."""
    return pd.date_range(window_starts.min().floor("h"), window_starts.max().floor("h"), freq="1h")


def usable_per_hour(scored: ScoredModel, index: pd.DatetimeIndex) -> list[int]:
    """Number of usable 60 s windows in each hour (0 means a gap or frozen data)."""
    counts = (
        pd.Series(scored.usable.astype(int), index=pd.DatetimeIndex(scored.window_starts))
        .resample("1h")
        .sum()
        .reindex(index, fill_value=0)
    )
    return [int(v) for v in counts]


def build_timeline(models: list[ScoredModel], thresholds: dict, primary: float) -> dict:
    """Hourly maxima per primary model, thresholds, coverage and the split boundaries."""
    primary_models = [m for m in models if m.is_primary]
    reference = primary_models[0]
    index = hourly_index(reference.window_starts)
    split_starts = {
        name: epoch_minutes(reference.window_starts[(reference.split == name).to_numpy()].min())
        for name in (CALIB, TEST)
    }
    return {
        "start_minute": epoch_minutes(index[0]),
        "step_minutes": 60,
        "coverage": usable_per_hour(reference, index),
        "calib_start_minute": split_starts[CALIB],
        "test_start_minute": split_starts[TEST],
        "windows": failure_windows_payload(),
        "models": [
            {
                "id": m.model,
                "label": MODEL_LABELS[m.model],
                "score": m.score,
                "score_label": SCORE_LABELS[m.score],
                "scale": "log" if m.score in LOG_SCORES else "linear",
                "threshold_p99": thresholds[m.model][m.score]["thresholds"][f"{primary:g}"],
                "threshold_p99_5": thresholds[m.model][m.score]["thresholds"]["99.5"],
                "values": hourly_maxima(m, index),
            }
            for m in primary_models
        ],
    }


def failure_windows_payload() -> list[dict]:
    """The four reported intervals, in epoch minutes."""
    return [
        {
            "event": w.event_id,
            "start_minute": epoch_minutes(w.start),
            "end_minute": epoch_minutes(w.end),
            "issue": w.reported_issue,
            "maintenance_note": w.maintenance_note,
        }
        for w in REPORTED_FAILURE_WINDOWS
    ]


def build_event_series(
    models: list[ScoredModel], thresholds: dict, event_id: int, lead_hours: float, primary: float
) -> dict:
    """Window-level scores of every primary model around one reported window."""
    window = next(w for w in REPORTED_FAILURE_WINDOWS if w.event_id == event_id)
    start = window.start - pd.Timedelta(hours=lead_hours + EVENT_PAD_BEFORE_HOURS)
    end = window.end + pd.Timedelta(hours=EVENT_PAD_AFTER_HOURS)
    series = []
    for m in (m for m in models if m.is_primary):
        times = pd.Series(m.window_starts)
        keep = ((times >= start) & (times <= end)).to_numpy() & m.usable & np.isfinite(m.scores)
        series.append(
            {
                "id": m.model,
                "minutes": [epoch_minutes(t) for t in times[keep]],
                "values": [round(float(v), VALUE_DECIMALS) for v in m.scores[keep]],
                "threshold_p99": thresholds[m.model][m.score]["thresholds"][f"{primary:g}"],
                "threshold_p99_5": thresholds[m.model][m.score]["thresholds"]["99.5"],
            }
        )
    return {
        "event": event_id,
        "start_minute": epoch_minutes(window.start),
        "end_minute": epoch_minutes(window.end),
        "lead_start_minute": epoch_minutes(window.start - pd.Timedelta(hours=lead_hours)),
        "series": series,
    }


def build_metrics(metrics: pd.DataFrame, thresholds: dict, lead_hours: float) -> dict:
    """Every metrics row, labels, frozen thresholds and the detection-range share."""
    test_days = float(metrics["test_days"].iloc[0])
    range_days = sum(
        (w.end - (w.start - pd.Timedelta(hours=lead_hours))) / pd.Timedelta(days=1)
        for w in REPORTED_FAILURE_WINDOWS
    )
    rows = json.loads(metrics.to_json(orient="records"))
    for row in rows:
        row["model_label"] = MODEL_LABELS[row["model"]]
        row["score_label"] = SCORE_LABELS[row["score"]]
    frozen = [
        {
            "model": MODEL_LABELS[model],
            "score": SCORE_LABELS[score],
            "frozen_at": entry["frozen_at"],
            "n_calib_windows": entry["n_calib_windows"],
        }
        for model, scores in thresholds.items()
        for score, entry in scores.items()
    ]
    return {
        "rows": rows,
        "test_days": test_days,
        "n_usable_test_windows": int(metrics["n_usable_test_windows"].iloc[0]),
        "detection_range_days": range_days,
        "detection_range_share": range_days / test_days,
        "frozen_thresholds": frozen,
    }


def build_training(rbm_history: pd.DataFrame, vae_history: pd.DataFrame) -> dict:
    """RBM and VAE training curves as column-wise arrays."""
    rbm = {
        int(k): {c: [round(float(v), 6) for v in g[c]] for c in g.columns if c != "k"}
        for k, g in rbm_history.groupby("k")
    }
    vae = {c: [round(float(v), 6) for v in vae_history[c]] for c in vae_history.columns}
    best = int(vae_history.loc[vae_history["validation_loss"].idxmin(), "epoch"])
    return {"rbm": rbm, "vae": vae, "vae_best_epoch": best}


def build_audit(report_text: str) -> dict:
    """Key tables and facts parsed from the step 1 audit report (fails if its format drifts)."""
    sections = read_markdown_sections(report_text)
    schema = find_section(sections, "2. File and schema")
    cadence = find_section(sections, "3. Timestamps and cadence")
    gaps = find_section(sections, "4. Missing values")
    digital = find_section(sections, "5. Digital signals")
    events = find_section(sections, "6. Reported failure windows")
    split = find_section(sections, "7. Candidate chronological split")
    motor = find_section(sections, "8. Motor current")
    schema_text = " ".join(schema["bullets"])
    cadence_text = " ".join(cadence["bullets"])
    gap_text = " ".join(gaps["paragraphs"] + gaps["bullets"])
    shape = first_match(r"Shape: ([\d,]+) rows x (\d+) columns", schema_text)
    span = first_match(r"Range: (\S+ \S+) to (\S+ \S+) \(span ([^)]+)\)", cadence_text)
    mode = first_match(r"dominant interval of (\d+) s", cadence_text)
    lost = first_match(r"Time without readings inside gaps: ([\d.]+) days", gap_text)
    analog_bullet = next(b for b in schema["bullets"] if b.startswith("Analog sensors"))
    digital_bullet = next(b for b in schema["bullets"] if b.startswith("Digital sensors"))
    digital_bullet = digital_bullet.replace("verified by values): ", "verified by values):")
    frozen_total = first_match(
        r"Total readings in frozen runs: ([\d,]+) \(([\d.]+)% of rows\)", gap_text
    )
    return {
        "rows": int(shape.group(1).replace(",", "")),
        "columns": int(shape.group(2)),
        "first_timestamp": span.group(1),
        "last_timestamp": span.group(2),
        "span": span.group(3),
        "cadence_seconds": int(mode.group(1)),
        "days_without_readings": float(lost.group(1)),
        "frozen_readings": int(frozen_total.group(1).replace(",", "")),
        "frozen_percent": float(frozen_total.group(2)),
        "analog": [c.strip() for c in analog_bullet.split(":", 1)[1].split(",")],
        "digital": [c.strip() for c in digital_bullet.split(":", 1)[1].split(",")],
        "common_deltas": table_records(cadence["tables"][0]),
        "gap_classes": table_records(gaps["tables"][0]),
        "longest_gaps": table_records(gaps["tables"][1]),
        "frozen_runs": table_records(gaps["tables"][2]),
        "digital_signals": table_records(digital["tables"][0]),
        "digital_by_month": table_records(digital["tables"][1]),
        "event_coverage": table_records(events["tables"][0]),
        "split_candidates": table_records(split["tables"][0]),
        "motor_states": table_records(motor["tables"][2]),
    }


def network_layout(nodes: list[str], edges: list[tuple[str, str]]) -> dict[str, list[float]]:
    """Deterministic layered positions in [0, 1]; every parent sits left of its children."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    layers = [sorted(generation) for generation in nx.topological_generations(graph)]
    column = {node: i for i, layer in enumerate(layers) for node in layer}
    for _ in range(LAYOUT_SWEEPS):
        for i in range(1, len(layers)):
            order = {n: j for j, n in enumerate(layers[i - 1])}
            layers[i].sort(key=lambda n: _barycentre(graph.predecessors(n), order))
        for i in range(len(layers) - 2, -1, -1):
            order = {n: j for j, n in enumerate(layers[i + 1])}
            layers[i].sort(key=lambda n: _barycentre(graph.successors(n), order))
    last = max(len(layers) - 1, 1)
    return {
        node: [round(column[node] / last, 4), round((j + 0.5) / len(layer), 4)]
        for layer in layers
        for j, node in enumerate(layer)
    }


def _barycentre(neighbours, order: dict[str, int]) -> float:
    """Mean position of the neighbours that sit in the adjacent layer (0 if there are none)."""
    positions = [order[n] for n in neighbours if n in order]
    return sum(positions) / len(positions) if positions else 0.0


def build_network(edges: pd.DataFrame, nodes: list[str], demonstrations: str, context: str) -> dict:
    """Edges, layout and the parsed demonstration and alert-context tables."""
    edge_list = [(r.parent, r.child) for r in edges.itertuples()]
    demo = read_markdown_sections(demonstrations)
    demo_head = first_match(
        r"Demonstration edge: `(\S+) -> (\S+)` .*?([\d.]+) nats", demonstrations
    )
    causal = find_section(demo, "Causal-direction query")
    evidential = find_section(demo, "Evidential query")
    local = find_section(demo, "Local independence")
    patterns = find_section(demo, "d-separation in chain")
    events = [
        {
            "heading": heading,
            "unusual_rule": [
                p for p in section_paragraphs(context) if p.startswith("Per variable")
            ],
            "table": table_records(section["tables"][0]),
        }
        for heading, section in read_markdown_sections(context).items()
        if heading.startswith("Reported window")
    ]
    model_used = first_match(r"Model used: `(\S+)` \((\S+),", context)
    return {
        "nodes": [
            {"id": n, "x": xy[0], "y": xy[1]} for n, xy in network_layout(nodes, edge_list).items()
        ],
        "edges": [{"parent": a, "child": b} for a, b in edge_list],
        "demonstration": {
            "parent": demo_head.group(1),
            "child": demo_head.group(2),
            "mutual_information": float(demo_head.group(3)),
            "causal_heading": next(h for h in demo if h.startswith("Causal-direction query")),
            "causal_table": {
                "columns": causal["tables"][0]["columns"],
                "rows": causal["tables"][0]["rows"],
            },
            "evidential_heading": next(h for h in demo if h.startswith("Evidential query")),
            "evidential_table": {
                "columns": evidential["tables"][0]["columns"],
                "rows": evidential["tables"][0]["rows"],
            },
            "local_independence": local["bullets"],
            "d_separation": patterns["bullets"],
        },
        "alert_context": {
            "model": model_used.group(1),
            "score": model_used.group(2),
            "events": events,
        },
    }


def build_latent(
    latent: np.ndarray,
    table: pd.DataFrame,
    alert: np.ndarray,
    seed: int,
    max_points: int = LATENT_POINTS,
) -> dict:
    """3-D PCA view of the VAE latent means, in the same shape as the browser engine's latent.json.

    The axes are fitted on the fit windows only. Every usable window inside a reported window
    is kept; the rest is a seeded random sample.
    """
    usable = table["usable"].to_numpy()
    split = table["split"].to_numpy()
    fit = usable & np.isin(split, ["fit_train", "fit_validation"])
    mean = latent[fit].mean(axis=0)
    covariance = np.cov(latent[fit] - mean, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0, None), vectors[:, order]
    axes = vectors[:, :3]
    starts = pd.to_datetime(table["window_start"])
    event = np.zeros(len(table), dtype=int)
    for window in REPORTED_FAILURE_WINDOWS:
        event[((starts >= window.start) & (starts <= window.end)).to_numpy()] = window.event_id
    candidates = np.flatnonzero(usable)
    keep = candidates[event[candidates] > 0]
    rest = candidates[event[candidates] == 0]
    sampled = min(len(rest), max(0, max_points - len(keep)))
    rest = np.random.default_rng(seed).permutation(rest)[:sampled]
    rows = np.sort(np.concatenate([keep, rest]))
    projected = (latent[rows] - mean) @ axes
    return {
        "explained_variance": [round(float(v), 4) for v in values[:3] / max(values.sum(), 1e-12)],
        "points": {
            "x": [round(float(v), 3) for v in projected[:, 0]],
            "y": [round(float(v), 3) for v in projected[:, 1]],
            "z": [round(float(v), 3) for v in projected[:, 2]],
            "split": [LATENT_SPLIT_CODES[s] for s in split[rows]],
            "alert": [int(a) for a in alert[rows]],
            "event": [int(e) for e in event[rows]],
            "minute": [epoch_minutes(t) for t in starts.iloc[rows]],
        },
    }


def section_paragraphs(text: str) -> list[str]:
    """Paragraph lines of a markdown file that sit outside tables and headings."""
    return [line.strip() for line in text.splitlines() if line.strip() and line[0] not in "#|-"]


def read_runtimes(logs_dir: Path) -> dict[str, float]:
    """Wall-clock seconds per step from the step manifests written by the pipeline."""
    runtimes = {}
    for path in sorted(logs_dir.glob("*_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["step"] != "step10_build_dashboard_data":
            runtimes[manifest["step"]] = manifest["runtime_seconds"]
    return runtimes
