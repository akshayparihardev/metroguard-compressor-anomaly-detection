// Steps 1 and 2 of the pipeline in the browser: sort, audit (cadence, gaps, frozen stretches), 60 s windows,
// the chronological split, and every transformation fitted on the fit windows only. Mirrors
// src/metroguard/data/{raw_data_audit,feature_builder,chronological_split}.py.

import { digitize, meanStd, median, quantileSorted, sortedFinite } from "./numeric.js";
import { formatNaive } from "./timestamps.js";

export const FIT_TRAIN = "fit_train";
export const FIT_VALIDATION = "fit_validation";
export const CALIB = "calib";
export const TEST = "test";
const HOUR_MS = 3_600_000;
const CONSTANT_STD = 1e-8;

/** Python's round(): half to even. */
export function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Sort every column by time (stable) and report whether the file was already in order. */
export function sortByTime(columns) {
  const { time } = columns;
  let backwards = 0;
  for (let i = 1; i < time.length; i++) if (time[i] < time[i - 1]) backwards++;
  if (!backwards) return { columns, backwardSteps: 0 };
  const order = Array.from({ length: time.length }, (_, i) => i).sort((a, b) => time[a] - time[b] || a - b);
  const reorder = (arr) => { const out = new arr.constructor(arr.length); order.forEach((src, dst) => { out[dst] = arr[src]; }); return out; };
  const map = (obj) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, reorder(v)]));
  return { columns: { ...columns, time: reorder(time), analog: map(columns.analog), digital: map(columns.digital) }, backwardSteps: backwards };
}

/** Cadence statistics and gap classes of consecutive timestamp differences (seconds). */
export function auditCadence(time, gapMultiplier, longGapHours) {
  const n = time.length - 1;
  const deltas = new Float64Array(Math.max(n, 0));
  for (let i = 0; i < n; i++) deltas[i] = (time[i + 1] - time[i]) / 1000;
  const counts = new Map();
  let zero = 0;
  for (const d of deltas) { if (d > 0) counts.set(d, (counts.get(d) ?? 0) + 1); else zero++; }
  let mode = NaN;
  let best = -1;
  for (const [d, c] of [...counts.entries()].sort((a, b) => a[0] - b[0])) if (c > best) { best = c; mode = d; }
  const gapThreshold = gapMultiplier * mode;
  const longThreshold = longGapHours * 3600;
  let atMode = 0, jitter = 0, gaps = 0, longGaps = 0, lost = 0;
  for (const d of deltas) {
    if (d <= 0) continue;
    if (d === mode) atMode++;
    else if (d <= gapThreshold) jitter++;
    if (d > gapThreshold) { gaps++; lost += d; }
    if (d >= longThreshold) longGaps++;
  }
  const positive = deltas.filter((d) => d > 0);
  const longest = Array.from(deltas.keys()).sort((a, b) => deltas[b] - deltas[a]).slice(0, 10).map((i) => ({
    gap_start: formatNaive(time[i]), gap_end: formatNaive(time[i + 1]), duration_hours: Number((deltas[i] / 3600).toFixed(2)),
  }));
  const common = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
    .map(([d, c]) => ({ delta_seconds: d, count: c, percent: (100 * c) / n }));
  const pct = (c) => Number(((100 * c) / n).toFixed(3));
  return {
    nDeltas: n,
    modeSeconds: mode,
    medianSeconds: median(positive),
    medianAllSeconds: median(deltas),
    gapThresholdSeconds: gapThreshold,
    zero,
    daysWithoutReadings: Number((lost / 86400).toFixed(2)),
    gapClasses: [
      { class: "normal (= cadence)", count: atMode, "percent of deltas": pct(atMode) },
      { class: "jitter", count: jitter, "percent of deltas": pct(jitter) },
      { class: "gap", count: gaps, "percent of deltas": pct(gaps) },
      { class: "of which long gap", count: longGaps, "percent of deltas": pct(longGaps) },
    ],
    longestGaps: longest,
    commonDeltas: common,
  };
}

/** Stretches where every analog sensor repeats exactly the same value for at least `minMinutes`. */
export function frozenValueRuns(time, analogColumns, minMinutes) {
  const series = Object.values(analogColumns);
  const runs = [];
  let runStart = 0;
  let unchangedCount = 0;
  let lastUnchanged = -1;
  const close = () => {
    if (unchangedCount > 0) {
      const hours = (time[lastUnchanged] - time[runStart]) / HOUR_MS;
      if (hours >= minMinutes / 60) runs.push({ start: time[runStart], end: time[lastUnchanged], readings: unchangedCount + 1, duration_hours: Number(hours.toFixed(2)) });
    }
  };
  for (let i = 0; i < time.length; i++) {
    let unchanged = true;
    if (i > 0) for (const s of series) { const d = Math.abs(s[i] - s[i - 1]); if (Number.isFinite(d) && d !== 0) { unchanged = false; break; } }
    if (unchanged) { unchangedCount++; lastUnchanged = i; }
    else { close(); runStart = i; unchangedCount = 0; lastUnchanged = -1; }
  }
  close();
  return runs;
}

/**
 * Leave out sensors that cannot help: blank in most of the training period, or never changing in it. Judged on the
 * training-period readings only (first calendar month, or the first share of readings), so no test data is used.
 */
export function screenSensors(columns, splitOptions, minFilledShare = 0.5) {
  const { time, analog } = columns;
  let trainRows = time.length;
  if (splitOptions.trainPeriod === "first_calendar_month") {
    const first = new Date(time[0]);
    const end = Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 1);
    trainRows = time.findIndex((t) => t >= end);
    if (trainRows < 0) trainRows = time.length;
  } else {
    trainRows = Math.max(1, Math.round(splitOptions.trainShare * time.length));
  }
  const kept = {};
  const dropped = [];
  for (const [name, values] of Object.entries(analog)) {
    let filled = 0;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < trainRows; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      filled++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (filled < minFilledShare * trainRows) dropped.push({ name, reason: `is blank in ${Math.round(100 * (1 - filled / trainRows))}% of the training-period rows` });
    else if (max === min) dropped.push({ name, reason: "never changes in the training period" });
    else kept[name] = values;
  }
  return { columns: { ...columns, analog: kept }, dropped };
}

/** Aggregate readings into fixed windows: analog means, std of selected signals, digital active fractions. */
export function aggregateWindows(columns, windowSeconds, stdColumns) {
  const { time, analog, digital } = columns;
  const windowMs = windowSeconds * 1000;
  const analogNames = Object.keys(analog);
  const digitalNames = Object.keys(digital);
  const starts = [];
  const counts = [];
  const sums = analogNames.map(() => []);
  const squares = analogNames.map(() => []);
  const valid = analogNames.map(() => []);
  const dSums = digitalNames.map(() => []);
  const dValid = digitalNames.map(() => []);
  const readingWindow = new Int32Array(time.length);
  let current = NaN;
  let w = -1;
  for (let i = 0; i < time.length; i++) {
    const start = Math.floor(time[i] / windowMs) * windowMs;
    if (start !== current) {
      current = start; w++;
      starts.push(start); counts.push(0);
      sums.forEach((a) => a.push(0)); squares.forEach((a) => a.push(0)); valid.forEach((a) => a.push(0));
      dSums.forEach((a) => a.push(0)); dValid.forEach((a) => a.push(0));
    }
    counts[w]++;
    readingWindow[i] = w;
    analogNames.forEach((name, c) => { const v = analog[name][i]; if (Number.isFinite(v)) { sums[c][w] += v; squares[c][w] += v * v; valid[c][w]++; } });
    digitalNames.forEach((name, c) => { const v = digital[name][i]; if (Number.isFinite(v)) { dSums[c][w] += v >= 0.5 ? 1 : 0; dValid[c][w]++; } });
  }
  const features = {};
  analogNames.forEach((name, c) => { features[`${name}_mean`] = Float64Array.from(sums[c], (s, i) => (valid[c][i] ? s / valid[c][i] : NaN)); });
  analogNames.forEach((name, c) => {
    if (!stdColumns.includes(name)) return;
    features[`${name}_std`] = Float64Array.from(sums[c], (s, i) => {
      const k = valid[c][i];
      if (!k) return NaN;
      const m = s / k;
      return Math.sqrt(Math.max(0, squares[c][i] / k - m * m));
    });
  });
  digitalNames.forEach((name, c) => { features[`${name}_active_fraction`] = Float64Array.from(dSums[c], (s, i) => (dValid[c][i] ? s / dValid[c][i] : NaN)); });
  return { windowStart: Float64Array.from(starts), readingCount: Int32Array.from(counts), readingWindow, features, analogNames, digitalNames };
}

/** True for windows that overlap any frozen-value run. */
export function flagFrozenWindows(windowStart, runs, windowSeconds) {
  const flagged = new Uint8Array(windowStart.length);
  const windowMs = windowSeconds * 1000;
  for (const run of runs) for (let i = 0; i < windowStart.length; i++) if (windowStart[i] <= run.end && windowStart[i] + windowMs > run.start) flagged[i] = 1;
  return flagged;
}

/** Exclusive end of the training period. */
export function trainingPeriodEnd(windowStart, usable, options) {
  if (options.trainPeriod === "first_calendar_month") {
    const first = new Date(windowStart[0]);
    return Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 1);
  }
  const usableStarts = windowStart.filter((_, i) => usable[i]);
  const index = Math.min(usableStarts.length - 1, Math.max(1, Math.round(options.trainShare * usableStarts.length)));
  return usableStarts[index];
}

/** Label every window by time; usable training windows decide the fraction boundaries (mirrors assign_split). */
export function assignSplit(windowStart, usable, trainEnd, calibFraction, validationFraction) {
  const train = [];
  for (let i = 0; i < windowStart.length; i++) if (windowStart[i] < trainEnd && usable[i]) train.push(windowStart[i]);
  const nCalib = pyRound(calibFraction * train.length);
  const nFit = train.length - nCalib;
  const nValidation = pyRound(validationFraction * nFit);
  if (nFit - nValidation < 1 || nCalib < 1 || nValidation < 1) {
    throw new Error(`Only ${train.length} usable windows fall in the training period, which is too few to learn from. Choose “First share” with a larger share, or use a file that covers more time.`);
  }
  const calibStart = train[nFit];
  const validationStart = train[nFit - nValidation];
  const split = Array.from(windowStart, (t) => (t >= trainEnd ? TEST : t >= calibStart ? CALIB : t >= validationStart ? FIT_VALIDATION : FIT_TRAIN));
  return { split, calibStart, validationStart };
}

/** Mean and scale per feature from fit rows; zero-spread features get scale 1. */
export function fitStandardiser(features, names, fitRows) {
  return names.map((name) => {
    const values = fitRows.map((i) => features[name][i]);
    const { mean, std } = meanStd(values);
    return { name, mean, scale: std < CONSTANT_STD ? 1 : std, constant: std < CONSTANT_STD };
  });
}

export function standardise(features, standardiser, n) {
  const out = new Float32Array(n * standardiser.length);
  standardiser.forEach((s, j) => { const col = features[s.name]; for (let i = 0; i < n; i++) out[i * standardiser.length + j] = (col[i] - s.mean) / s.scale; });
  return out;
}

/** Quantile cut points from fit rows (numpy linear method); duplicates merged. */
export function fitBinEdges(features, names, fitRows, nBins) {
  const levels = Array.from({ length: nBins - 1 }, (_, i) => (i + 1) / nBins);
  return Object.fromEntries(names.map((name) => {
    const sorted = sortedFinite(fitRows.map((i) => features[name][i]));
    return [name, [...new Set(levels.map((q) => quantileSorted(sorted, q)))].sort((a, b) => a - b)];
  }));
}

export function discretise(features, binEdges, n) {
  return Object.fromEntries(Object.entries(binEdges).map(([name, edges]) => [name, Int8Array.from({ length: n }, (_, i) => digitize(features[name][i], edges))]));
}

export function binariseDigital(features, digitalNames, n) {
  return Object.fromEntries(digitalNames.map((name) => [`${name}_active`, Uint8Array.from({ length: n }, (_, i) => (features[`${name}_active_fraction`][i] >= 0.5 ? 1 : 0))]));
}

/** RBM visible matrix: one-hot analog bins followed by binary digital signals (row-major Float32). */
export function oneHotVisible(states, binEdges, digitalBinary, n) {
  const blocks = Object.entries(binEdges).map(([name, edges]) => ({ name, states: states[name], width: edges.length + 1 }));
  const digitalNames = Object.keys(digitalBinary);
  const width = blocks.reduce((s, b) => s + b.width, 0) + digitalNames.length;
  const visible = new Float32Array(n * width);
  for (let i = 0; i < n; i++) {
    let offset = 0;
    for (const b of blocks) { visible[i * width + offset + b.states[i]] = 1; offset += b.width; }
    digitalNames.forEach((name, j) => { visible[i * width + offset + j] = digitalBinary[name][i]; });
  }
  return { visible, width };
}
