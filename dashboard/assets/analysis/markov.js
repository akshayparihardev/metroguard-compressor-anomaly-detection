// Operating states from a histogram-peak rule and a Markov chain over them
// (mirrors raw_data_audit.motor_current_candidate_states and models/markov_state_model.py).

import { digitize, niceStep } from "./numeric.js";

const SMOOTHING_BINS = 5;
const MIN_PEAK_SHARE = 0.005;
const MAX_VALLEY_RATIO = 0.5;

function mergeShallowPeaks(peaks, smoothed) {
  const merged = [];
  for (const peak of peaks) {
    if (merged.length) {
      const previous = merged[merged.length - 1];
      let valley = Infinity;
      for (let i = previous; i <= peak; i++) valley = Math.min(valley, smoothed[i]);
      if (valley > MAX_VALLEY_RATIO * Math.min(smoothed[previous], smoothed[peak])) {
        merged[merged.length - 1] = smoothed[previous] >= smoothed[peak] ? previous : peak;
        continue;
      }
    }
    merged.push(peak);
  }
  return merged;
}

/**
 * Candidate operating states of one signal: peaks of a smoothed histogram, merged unless separated by a clear
 * valley, with cut points at the minima between peaks. `binWidth` defaults to a "nice" 1/100 of the range.
 */
export function candidateStates(values, binWidth) {
  const finite = values.filter(Number.isFinite);
  if (finite.length < 10) return { cutPoints: [], states: [] };
  let lo = Infinity, hi = -Infinity;
  for (const v of finite) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const width = binWidth ?? niceStep(Math.max(hi - lo, 1e-9), 100);
  const start = Math.floor(lo / width) * width;
  const stop = Math.ceil(hi / width) * width + 2 * width;
  const nEdges = Math.max(2, Math.ceil((stop - start) / width - 1e-9));
  const edges = Array.from({ length: nEdges }, (_, i) => start + i * width);
  const nBins = edges.length - 1;
  const counts = new Float64Array(nBins);
  for (const v of finite) {
    let idx = Math.floor((v - start) / width);
    if (idx >= nBins) idx = nBins - 1;
    if (idx > 0 && v < edges[idx]) idx--;
    else if (idx < nBins - 1 && v >= edges[idx + 1]) idx++;
    counts[Math.max(0, idx)]++;
  }
  const half = SMOOTHING_BINS >> 1;
  const smoothed = Float64Array.from(counts, (_, i) => {
    let s = 0;
    for (let k = i - half; k <= i + half; k++) if (k >= 0 && k < nBins) s += counts[k];
    return s / SMOOTHING_BINS;
  });
  const centres = Array.from({ length: nBins }, (_, i) => (edges[i] + edges[i + 1]) / 2);
  const minHeight = MIN_PEAK_SHARE * finite.length;
  const padded = [-Infinity, ...smoothed, -Infinity];
  let peaks = [];
  for (let i = 0; i < nBins; i++) if (padded[i + 1] >= padded[i] && padded[i + 1] > padded[i + 2] && smoothed[i] >= minHeight) peaks.push(i);
  peaks = mergeShallowPeaks(peaks, smoothed).map((p) => {
    const from = Math.max(0, p - half);
    let best = from;
    for (let k = from; k <= Math.min(nBins - 1, p + half); k++) if (counts[k] > counts[best]) best = k;
    return best;
  });
  const cutPoints = [];
  for (let n = 0; n + 1 < peaks.length; n++) {
    let best = peaks[n];
    for (let k = peaks[n]; k <= peaks[n + 1]; k++) if (smoothed[k] < smoothed[best]) best = k;
    cutPoints.push(centres[best]);
  }
  const bounds = [-Infinity, ...cutPoints, Infinity];
  const states = peaks.map((p, s) => {
    let inside = 0;
    for (const v of finite) if (v > bounds[s] && v <= bounds[s + 1]) inside++;
    return { state: s, peak: Number(centres[p].toFixed(2)), lower: Number.isFinite(bounds[s]) ? bounds[s] : null, upper: Number.isFinite(bounds[s + 1]) ? bounds[s + 1] : null, share: inside / finite.length };
  });
  return { cutPoints, states };
}

/**
 * Markov chain over per-reading states: transition matrix from fit readings (add-one smoothing), score per
 * transition = -log P(s_t | s_{t-1}), window score = max over transitions that end in the window.
 */
export function markovWindowScores({ time, values, cutPoints, readingWindow, inFitWindow, maxGapSeconds, nWindows }) {
  const nStates = cutPoints.length + 1;
  const states = Int8Array.from(values, (v) => digitize(v, cutPoints));
  const counts = Array.from({ length: nStates }, () => new Float64Array(nStates));
  const valid = (i) => i > 0 && (time[i] - time[i - 1]) / 1000 <= maxGapSeconds;
  for (let i = 1; i < states.length; i++) if (valid(i) && inFitWindow[i] && inFitWindow[i - 1]) counts[states[i - 1]][states[i]]++;
  const matrix = counts.map((row) => { const total = row.reduce((s, c) => s + c + 1, 0); return Array.from(row, (c) => (c + 1) / total); });
  const scores = new Float64Array(nWindows).fill(NaN);
  for (let i = 1; i < states.length; i++) {
    if (!valid(i)) continue;
    const s = -Math.log(matrix[states[i - 1]][states[i]]);
    const w = readingWindow[i];
    if (!(scores[w] >= s)) scores[w] = s;
  }
  return { scores, matrix, counts: counts.map((r) => Array.from(r)) };
}
