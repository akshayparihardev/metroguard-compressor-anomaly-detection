// Shapes the browser pipeline's results into the same JSON documents that pipeline/step10 writes for the
// MetroGuard data, so every dashboard page can show an uploaded dataset without special cases.

import { createRng } from "./numeric.js";
import { formatNaive, formatSpan, localNow } from "./timestamps.js";

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;
const EVENT_PAD_BEFORE_HOURS = 3;
const EVENT_PAD_AFTER_HOURS = 12;
const LATENT_POINTS = 12_000;
const LOG_SCORES = new Set(["max_abs_z", "mean_abs_z", "reconstruction_error", "negative_elbo"]);
export const SCORE_LABELS = { max_abs_z: "max |z|", mean_abs_z: "mean |z|", free_energy: "free energy", reconstruction_error: "reconstruction error", negative_elbo: "negative ELBO", max_transition_surprisal: "transition surprisal" };

export function modelLabel(model) {
  if (model.startsWith("rbm_k")) return `RBM, k = ${model.slice(5)}`;
  return { zscore_baseline: "Z-score baseline", vae: "VAE", markov_state_model: "Markov chain" }[model] ?? model;
}

const r4 = (v) => (Number.isFinite(v) ? Number(v.toFixed(4)) : null);
const minute = (ms) => Math.floor(ms / 60000);

function lowerBound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < target) lo = mid + 1; else hi = mid; }
  return lo;
}

/** Eigen-decomposition of a small symmetric matrix (Jacobi rotations). */
function symmetricEigen(matrix) {
  const n = matrix.length;
  const a = matrix.map((row) => row.slice());
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1);
      const s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
    }
  }
  return Array.from({ length: n }, (_, i) => ({ value: a[i][i], vector: v.map((row) => row[i]) })).sort((x, y) => y.value - x.value);
}

/** 3-D PCA projection of the VAE latent means (fitted on fit windows) for a sample of windows. */
export function latentProjection({ latent, dims, fitRows, candidates, inWindow, alert, split, starts, seed }) {
  const mean = new Array(dims).fill(0);
  fitRows.forEach((r) => { for (let d = 0; d < dims; d++) mean[d] += latent[r * dims + d] / fitRows.length; });
  const cov = Array.from({ length: dims }, () => new Array(dims).fill(0));
  fitRows.forEach((r) => { for (let i = 0; i < dims; i++) for (let j = 0; j < dims; j++) cov[i][j] += ((latent[r * dims + i] - mean[i]) * (latent[r * dims + j] - mean[j])) / fitRows.length; });
  const eigen = symmetricEigen(cov);
  const total = eigen.reduce((s, e) => s + Math.max(e.value, 0), 0) || 1;
  const axes = eigen.slice(0, 3);
  const keep = candidates.filter((r) => inWindow[r]);
  const rest = candidates.filter((r) => !inWindow[r]);
  const rng = createRng(seed);
  const order = rng.permutation(rest.length);
  const sampled = Math.min(rest.length, Math.max(0, LATENT_POINTS - keep.length));
  for (let i = 0; i < sampled; i++) keep.push(rest[order[i]]);
  keep.sort((a, b) => a - b);
  const points = { x: [], y: [], z: [], split: [], alert: [], event: [], minute: [] };
  const splitCode = { fit_train: 0, fit_validation: 0, calib: 1, test: 2 };
  for (const r of keep) {
    const c = axes.map((axis) => axis.vector.reduce((s, w, d) => s + w * (latent[r * dims + d] - mean[d]), 0));
    points.x.push(Number(c[0].toFixed(3))); points.y.push(Number(c[1].toFixed(3))); points.z.push(Number((c[2] ?? 0).toFixed(3)));
    points.split.push(splitCode[split[r]]); points.alert.push(alert[r] ? 1 : 0); points.event.push(inWindow[r]); points.minute.push(minute(starts[r]));
  }
  return { explained_variance: axes.map((a) => Number((Math.max(a.value, 0) / total).toFixed(4))), points };
}

export function buildBundle(ctx) {
  const { meta, options: o, columns, cadence, frozenRuns, prepared, windows, scored, metricRows, frozenThresholds, rbmHistory, vaeResult, markov, network, runtimes, notes, settings } = ctx;
  const { agg, n, usable, split } = prepared;
  const starts = agg.windowStart;
  const primary = scored.filter((s) => s.primary);
  const [p1, p2] = o.thresholds.percentiles.map(String);
  const firstOf = (name) => { for (let i = 0; i < n; i++) if (split[i] === name) return starts[i]; return NaN; };

  // timeline -------------------------------------------------------------------------------------------------
  const stepMinutes = Math.max(60, Math.ceil(settings.windowSeconds / 60));
  const stepMs = stepMinutes * 60000;
  const first = Math.floor(starts[0] / stepMs) * stepMs;
  const buckets = Math.floor((starts[n - 1] - first) / stepMs) + 1;
  const coverage = new Array(buckets).fill(0);
  for (let i = 0; i < n; i++) if (usable[i]) coverage[Math.floor((starts[i] - first) / stepMs)]++;
  const windowPayload = windows.map((w) => ({ event: w.event, start_minute: minute(w.start), end_minute: minute(w.end), issue: w.label || "Reported interval", maintenance_note: "-" }));
  const timeline = {
    start_minute: minute(first), step_minutes: stepMinutes, coverage,
    calib_start_minute: minute(firstOf("calib")), test_start_minute: minute(firstOf("test")), windows: windowPayload,
    models: primary.map((s) => {
      const values = new Array(buckets).fill(null);
      for (let i = 0; i < n; i++) {
        if (!usable[i] || !Number.isFinite(s.values[i])) continue;
        const b = Math.floor((starts[i] - first) / stepMs);
        if (values[b] === null || s.values[i] > values[b]) values[b] = s.values[i];
      }
      return { id: s.model, label: modelLabel(s.model), score: s.score, score_label: SCORE_LABELS[s.score], scale: LOG_SCORES.has(s.score) ? "log" : "linear", threshold_p99: s.thresholds[p1], threshold_p99_5: s.thresholds[p2], values: values.map(r4) };
    }),
  };

  // per-event window-level series ----------------------------------------------------------------------------
  const events = Object.fromEntries(windows.map((w) => {
    const from = w.start - (settings.leadHours + EVENT_PAD_BEFORE_HOURS) * HOUR_MS;
    const to = w.end + EVENT_PAD_AFTER_HOURS * HOUR_MS;
    const series = primary.map((s) => {
      const minutes = [], values = [];
      for (let i = lowerBound(starts, from); i < n && starts[i] <= to; i++) if (usable[i] && Number.isFinite(s.values[i])) { minutes.push(minute(starts[i])); values.push(r4(s.values[i])); }
      return { id: s.model, minutes, values, threshold_p99: s.thresholds[p1], threshold_p99_5: s.thresholds[p2] };
    });
    return [`event_${w.event}`, { event: w.event, start_minute: minute(w.start), end_minute: minute(w.end), lead_start_minute: minute(w.start - settings.leadHours * HOUR_MS), series }];
  }));

  // metrics ----------------------------------------------------------------------------------------------------
  const testDays = metricRows[0].test_days;
  const rangeDays = windows.reduce((s, w) => s + (w.end - (w.start - settings.leadHours * HOUR_MS)) / DAY_MS, 0);
  const metrics = {
    rows: metricRows.map((r) => ({ ...r, model_label: modelLabel(r.model), score_label: SCORE_LABELS[r.score] })),
    test_days: testDays,
    n_usable_test_windows: metricRows[0].n_usable_test_windows,
    detection_range_days: rangeDays,
    detection_range_share: rangeDays / testDays,
    frozen_thresholds: frozenThresholds.map((f) => ({ model: modelLabel(f.model), score: SCORE_LABELS[f.score], frozen_at: f.frozen_at, n_calib_windows: f.n_calib_windows })),
  };

  // training curves ------------------------------------------------------------------------------------------------
  const vh = vaeResult.history;
  const bestEpoch = vh.epoch[vh.validation_loss.indexOf(Math.min(...vh.validation_loss))];
  const training = { rbm: rbmHistory, vae: vh, vae_best_epoch: bestEpoch };

  // audit ----------------------------------------------------------------------------------------------------------
  const time = columns.time;
  const frozenReadings = frozenRuns.reduce((s, r) => s + r.readings, 0);
  const monthKey = (ms) => new Date(ms).toISOString().slice(0, 7);
  const months = [...new Set(Array.from(time, monthKey))];
  const digitalNames = Object.keys(columns.digital);
  const activeFraction = (values, from = 0, to = values.length) => { let on = 0, k = 0; for (let i = from; i < to; i++) if (Number.isFinite(values[i])) { k++; if (values[i] >= 0.5) on++; } return k ? on / k : NaN; };
  const audit = {
    rows: meta.rowsRead, columns: meta.columnCount, columns_used: 1 + Object.keys(columns.analog).length + digitalNames.length,
    first_timestamp: formatNaive(time[0]), last_timestamp: formatNaive(time[time.length - 1]), span: formatSpan(time[time.length - 1] - time[0]),
    cadence_seconds: cadence.modeSeconds, days_without_readings: cadence.daysWithoutReadings,
    frozen_readings: frozenReadings, frozen_percent: Number(((100 * frozenReadings) / time.length).toFixed(2)),
    analog: Object.keys(columns.analog), digital: digitalNames,
    common_deltas: cadence.commonDeltas, gap_classes: cadence.gapClasses, longest_gaps: cadence.longestGaps,
    frozen_runs: frozenRuns.map((r) => ({ start: formatNaive(r.start), end: formatNaive(r.end), readings: r.readings, duration_hours: r.duration_hours })),
    digital_signals: digitalNames.map((name) => {
      const values = columns.digital[name];
      const f = activeFraction(values);
      const binary = values.every((v) => !Number.isFinite(v) || v === 0 || v === 1);
      return { column: name, strictly_binary: binary ? "True" : "False", active_fraction: Number(f.toFixed(4)), near_constant: f < o.audit.nearConstantFraction || f > 1 - o.audit.nearConstantFraction ? "True" : "False" };
    }),
    digital_by_month: months.map((m) => {
      const from = lowerBound(time, Date.parse(`${m}-01T00:00:00Z`));
      const [y, mo] = m.split("-").map(Number);
      const to = lowerBound(time, Date.UTC(y, mo, 1));
      return Object.fromEntries([["month", m], ...digitalNames.map((d) => [d, Number(activeFraction(columns.digital[d], from, to).toFixed(3))])]);
    }),
    event_coverage: windows.map((w) => {
      const from = lowerBound(time, w.start);
      const to = lowerBound(time, w.end + 1);
      const inside = to - from;
      const expected = Math.floor((w.end - w.start) / 1000 / cadence.modeSeconds) + 1;
      const edges = [w.start, ...Array.from(time.subarray(from, to)), w.end];
      let largest = 0;
      for (let i = 1; i < edges.length; i++) largest = Math.max(largest, (edges[i] - edges[i - 1]) / 1000);
      return { event: w.event, start: formatNaive(w.start), end: formatNaive(w.end), duration_hours: Number(((w.end - w.start) / HOUR_MS).toFixed(2)), readings: inside, expected_at_cadence: expected, coverage_percent: Number(((100 * inside) / expected).toFixed(1)), largest_gap_seconds: largest, fully_covered: inside > 0 && largest <= cadence.gapThresholdSeconds ? "True" : "False" };
    }),
    state_column: markov?.stateColumn ?? null,
    motor_states: (markov?.fullRecordStates ?? []).map((s) => ({ state: s.state, peak_amps: s.peak, lower_cut_amps: s.lower, upper_cut_amps: s.upper, share_of_readings: Number(s.share.toFixed(3)) })),
  };

  // network ---------------------------------------------------------------------------------------------------------
  const net = network;
  const networkDoc = {
    nodes: net.layout,
    edges: net.edges.map(([parent, child]) => ({ parent, child })),
    demonstration: net.strongest && net.causal ? {
      parent: net.strongest.parent, child: net.strongest.child, mutual_information: Number(net.strongest.mi.toFixed(3)),
      causal_heading: `Causal-direction query: P(${net.strongest.child} | ${net.strongest.parent} = state)`, causal_table: net.causal,
      evidential_heading: `Evidential query: P(${net.strongest.parent} | ${net.strongest.child} = state)`, evidential_table: net.evidential,
      local_independence: net.localIndependence, d_separation: net.patterns,
    } : null,
    alert_context: {
      model: net.contextModel.model, score: net.contextModel.score,
      events: net.contextEvents.map((e) => ({ heading: `${e.label}: alert event ${formatNaive(e.event.start)} to ${formatNaive(e.event.end)} (${e.rows} usable windows)`, table: e.table })),
    },
  };

  // latent space ------------------------------------------------------------------------------------------------------
  const vaeScored = scored.find((s) => s.model === "vae" && s.score === "reconstruction_error");
  const inWindow = new Int32Array(n);
  windows.forEach((w) => { for (let i = lowerBound(starts, w.start); i < n && starts[i] <= w.end; i++) inWindow[i] = w.event; });
  const latent = latentProjection({
    latent: vaeResult.latent, dims: o.vae.latentDim, fitRows: prepared.fitRows,
    candidates: Array.from({ length: n }, (_, i) => i).filter((i) => usable[i]),
    inWindow, alert: Uint8Array.from(vaeScored.values, (v) => (v > vaeScored.thresholds[p1] ? 1 : 0)), split, starts, seed: o.seed,
  });

  const context = {
    lead_tolerance_hours: settings.leadHours, merge_gap_minutes: settings.mergeGapMinutes, window_seconds: settings.windowSeconds,
    primary_percentile: o.thresholds.primary, seed: o.seed, windows: windowPayload, runtimes_seconds: runtimes,
    model_labels: Object.fromEntries(scored.map((s) => [s.model, modelLabel(s.model)])), score_labels: SCORE_LABELS,
    source: {
      kind: "upload", file_name: meta.fileName, file_kind: meta.fileKind, analysed_at: localNow().slice(0, 16).replace("T", " "),
      timestamp_note: meta.timestampNote, train_period: o.split.trainPeriod === "first_calendar_month" ? "First calendar month" : `First ${Math.round(o.split.trainShare * 100)}% of the record`,
      calib_start: formatNaive(firstOf("calib")), test_start: formatNaive(firstOf("test")), state_column: markov?.stateColumn ?? null, notes,
    },
  };

  return { timeline, metrics, training, audit, network: networkDoc, context, latent, ...events };
}
