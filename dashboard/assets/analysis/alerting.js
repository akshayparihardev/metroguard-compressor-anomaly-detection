// Thresholds from calib, alert events and event-level evaluation (mirrors src/metroguard/alerting/).

import { percentile } from "./numeric.js";

const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

/** Threshold = percentile of the score on calib windows (never on test). */
export function fitThresholds(calibScores, percentiles) {
  const finite = calibScores.filter(Number.isFinite);
  if (!finite.length) throw new Error("No finite calibration scores to fit a threshold on.");
  return Object.fromEntries(percentiles.map((p) => [String(p), percentile(finite, p)]));
}

/** Merge alert windows separated by at most mergeGap; drop events with fewer than minWindows windows. */
export function buildAlertEvents(starts, scores, threshold, windowSeconds, mergeGapMinutes, minWindows) {
  const windowMs = windowSeconds * 1000;
  const events = [];
  let current = null;
  for (let i = 0; i < starts.length; i++) {
    if (!(scores[i] > threshold)) continue;
    if (current && starts[i] - current.end <= mergeGapMinutes * 60000) {
      current.end = starts[i] + windowMs; current.n += 1; current.peak = Math.max(current.peak, scores[i]);
    } else {
      if (current) events.push(current);
      current = { start: starts[i], end: starts[i] + windowMs, n: 1, peak: scores[i] };
    }
  }
  if (current) events.push(current);
  return events.filter((e) => e.n >= minWindows);
}

/** Detected windows, delay per window and unconfirmed alert events. */
export function evaluateEvents(events, windows, leadHours) {
  const covered = new Uint8Array(events.length);
  const result = {};
  const detected = [];
  for (const w of windows) {
    const rangeStart = w.start - leadHours * HOUR_MS;
    let first = Infinity;
    events.forEach((e, i) => { if (e.start <= w.end && e.end >= rangeStart) { covered[i] = 1; first = Math.min(first, e.start); } });
    if (Number.isFinite(first)) detected.push(w.event);
    result[`delay_hours_event${w.event}`] = Number.isFinite(first) ? (Math.max(first, rangeStart) - w.start) / HOUR_MS : null;
  }
  result.events_detected = detected.length;
  result.detected_event_ids = detected.length ? detected.join(";") : "none";
  result.n_alert_events = events.length;
  result.unconfirmed_alert_events = covered.reduce((s, c) => s + (c ? 0 : 1), 0);
  return result;
}

/** One metrics row per percentile on usable test windows (same fields as metrics.csv). */
export function evaluateModel({ model, score, isPrimary, starts, scores, split, usable, thresholds, windows, settings }) {
  const testIdx = [];
  for (let i = 0; i < starts.length; i++) if (split[i] === "test") testIdx.push(i);
  const usableTest = testIdx.filter((i) => usable[i]);
  const testDays = (starts[testIdx[testIdx.length - 1]] + settings.windowSeconds * 1000 - starts[testIdx[0]]) / DAY_MS;
  const ts = usableTest.map((i) => starts[i]);
  const ss = usableTest.map((i) => scores[i]);
  return Object.entries(thresholds).map(([percentileKey, threshold]) => {
    const events = buildAlertEvents(ts, ss, threshold, settings.windowSeconds, settings.mergeGapMinutes, settings.minEventWindows);
    const nAlert = ss.filter((v) => v > threshold).length;
    const row = {
      model, score, is_primary_score: isPrimary, percentile: Number(percentileKey), threshold,
      n_usable_test_windows: usableTest.length, n_alert_windows: nAlert, alert_window_fraction: nAlert / usableTest.length,
      ...evaluateEvents(events, windows, settings.leadHours),
      test_days: testDays,
    };
    row.unconfirmed_alerts_per_day = row.unconfirmed_alert_events / testDays;
    return { row, events };
  });
}
