// The whole MetroGuard pipeline in the browser, in the same order and with the same rules as the Python
// pipeline: nothing is fitted on test data, and each model's thresholds are fitted on calib and frozen before
// that model scores a test window.

import { buildBundle } from "./bundle.js";
import { learnStructure, fitNetwork, conditionalTable, strongestEdge, localIndependenceCases, structurePatterns, blanketSurprisal, layeredLayout } from "./bayesian-network.js";
import { fitThresholds, evaluateModel } from "./alerting.js";
import { candidateStates, markovWindowScores } from "./markov.js";
import { createRng, meanStd, percentile } from "./numeric.js";
import * as prep from "./preparation.js";
import { createRbm, trainRbm, freeEnergy } from "./rbm.js";
import { localNow } from "./timestamps.js";
import { createVae, trainVae, vaeScores } from "./vae.js";

export const DEFAULTS = Object.freeze({
  seed: 42,
  audit: { gapMultiplier: 2, longGapHours: 1, nearConstantFraction: 0.005, frozenRunMinMinutes: 60 },
  split: { trainPeriod: "first_calendar_month", trainShare: 0.2, calibFraction: 0.2, validationFraction: 0.1 },
  windows: { sizeSeconds: "auto", minReadingsFraction: 0.5 },
  discretise: { nBins: 3 },
  rbm: { nHidden: 32, kValues: [1, 5, 10], epochs: 30, batchSize: 256, learningRate: 0.01, momentum: 0.5, weightDecay: 0.0001 },
  vae: { latentDim: 6, hiddenSizes: [64, 32], epochs: 50, batchSize: 256, learningRate: 0.001, beta: 1.0, patience: 5 },
  thresholds: { percentiles: [99, 99.5], primary: 99 },
  alerts: { mergeGapMinutes: 10, minEventWindows: 1 },
  evaluation: { leadHours: 24 },
  network: { maxIndegree: 3, ess: 10, contextEvents: 2, contextPercentile: 95, maxNodes: 12, maxAnalogNodes: 8 },
});

const METROPT_STD_COLUMNS = ["Motor_current", "TP2", "TP3"];
export const WINDOW_CHOICES = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400];
const UNUSUAL_SHARE_FACTOR = 2;

/** Window length: 60 s for readings every 10 s or faster, otherwise about six readings per window. */
export function chooseWindowSeconds(cadenceSeconds) {
  if (!(cadenceSeconds > 10)) return 60;
  return WINDOW_CHOICES.find((w) => w >= 6 * cadenceSeconds) ?? 86400;
}

const isMetroPtSchema = (analogNames) => METROPT_STD_COLUMNS.every((c) => analogNames.includes(c));
const indices = (n, keep) => { const out = []; for (let i = 0; i < n; i++) if (keep(i)) out.push(i); return out; };

function gather(matrix, width, rows) {
  const out = new Float32Array(rows.length * width);
  rows.forEach((r, i) => out.set(matrix.subarray(r * width, (r + 1) * width), i * width));
  return out;
}

/**
 * Run the full analysis on typed columns. `report(stage, fraction, message)` receives progress.
 * Returns the bundle of dashboard JSON documents.
 */
export async function runPipeline(rawColumns, meta, options = DEFAULTS, report = () => {}) {
  const o = options;
  const runtimes = {};
  const timed = async (key, fn) => { const t0 = performance.now(); const r = await fn(); runtimes[key] = Number(((performance.now() - t0) / 1000).toFixed(2)); return r; };
  const yieldToLoop = () => new Promise((resolve) => setTimeout(resolve, 0));
  const notes = [...(meta.notes ?? [])];

  // ---- step 1: audit -------------------------------------------------------------------------------------
  report("audit", 0.02, "Sorting by time and auditing the record");
  const { columns, cadence, frozenRuns, backwardSteps } = await timed("step01_audit_raw_data", () => {
    const sorted = prep.sortByTime(rawColumns);
    if (sorted.columns.time.length < 100) throw new Error("Fewer than 100 rows with a valid timestamp. Check the timestamp column.");
    return {
      columns: sorted.columns,
      backwardSteps: sorted.backwardSteps,
      cadence: prep.auditCadence(sorted.columns.time, o.audit.gapMultiplier, o.audit.longGapHours),
      frozenRuns: Object.keys(sorted.columns.analog).length ? prep.frozenValueRuns(sorted.columns.time, sorted.columns.analog, o.audit.frozenRunMinMinutes) : [],
    };
  });
  if (backwardSteps) notes.push(`The file was not in time order (${backwardSteps} backward steps); rows were sorted by time.`);
  const analogNames = Object.keys(columns.analog);
  const digitalNames = Object.keys(columns.digital);
  if (!analogNames.length) throw new Error("Choose at least one sensor (numeric) column to analyse.");

  // ---- step 2: windows, split, fit-only transforms -------------------------------------------------------
  report("features", 0.08, "Building windows and the chronological split");
  await yieldToLoop();
  const prepared = await timed("step02_build_features", () => {
    const windowSeconds = o.windows.sizeSeconds === "auto" ? chooseWindowSeconds(cadence.medianSeconds) : o.windows.sizeSeconds;
    const stdColumns = isMetroPtSchema(analogNames) ? METROPT_STD_COLUMNS : analogNames;
    const agg = prep.aggregateWindows(columns, windowSeconds, stdColumns);
    const n = agg.windowStart.length;
    const featureNames = Object.keys(agg.features);
    const frozen = prep.flagFrozenWindows(agg.windowStart, frozenRuns, windowSeconds);
    const minReadings = Math.ceil(o.windows.minReadingsFraction * (windowSeconds / cadence.medianSeconds));
    const usable = Uint8Array.from({ length: n }, (_, i) => (!frozen[i] && agg.readingCount[i] >= minReadings && featureNames.every((f) => Number.isFinite(agg.features[f][i])) ? 1 : 0));
    const trainEnd = prep.trainingPeriodEnd(agg.windowStart, usable, o.split);
    const { split, calibStart } = prep.assignSplit(agg.windowStart, usable, trainEnd, o.split.calibFraction, o.split.validationFraction);
    const fitRows = indices(n, (i) => usable[i] && (split[i] === prep.FIT_TRAIN || split[i] === prep.FIT_VALIDATION));
    const standardiser = prep.fitStandardiser(agg.features, featureNames, fitRows);
    const standardised = prep.standardise(agg.features, standardiser, n);
    const analogFeatures = featureNames.filter((f) => f.endsWith("_mean") || f.endsWith("_std"));
    const binEdges = prep.fitBinEdges(agg.features, analogFeatures, fitRows, o.discretise.nBins);
    const states = prep.discretise(agg.features, binEdges, n);
    const digitalBinary = prep.binariseDigital(agg.features, digitalNames, n);
    const { visible, width } = prep.oneHotVisible(states, binEdges, digitalBinary, n);
    return { windowSeconds, agg, n, featureNames, frozen, minReadings, usable, trainEnd, split, calibStart, fitRows, standardiser, standardised, binEdges, states, digitalBinary, visible, visibleWidth: width };
  });
  const { n, usable, split, agg, windowSeconds } = prepared;
  const rowsOf = (name) => indices(n, (i) => usable[i] && split[i] === name);
  const fitTrain = rowsOf(prep.FIT_TRAIN);
  const fitValidation = rowsOf(prep.FIT_VALIDATION);
  const calibRows = rowsOf(prep.CALIB);
  const isTest = Uint8Array.from(split, (s) => (s === prep.TEST ? 1 : 0));
  if (!isTest.some((v) => v)) throw new Error("No windows fall after the training period, so there is nothing to test. Choose a shorter training period.");
  const percentiles = o.thresholds.percentiles;
  const frozenThresholds = [];
  const scored = [];
  const freeze = (model, score, values) => {
    const thresholds = fitThresholds(calibRows.map((i) => values[i]), percentiles);
    frozenThresholds.push({ model, score, frozen_at: localNow(), n_calib_windows: calibRows.filter((i) => Number.isFinite(values[i])).length, thresholds });
    return thresholds;
  };

  // ---- step 3: z-score baseline --------------------------------------------------------------------------
  report("zscore", 0.12, "Scoring the z-score baseline");
  await timed("step03_score_zscore_baseline", () => {
    const means = analogNames.map((a) => `${a}_mean`);
    const ref = means.map((f) => { const { mean, std } = meanStd(prepared.fitRows.map((i) => agg.features[f][i])); return { f, mean, std: std < 1e-8 ? 1 : std }; });
    const scoreRow = (i) => { let max = 0, sum = 0; for (const r of ref) { const z = Math.abs((agg.features[r.f][i] - r.mean) / r.std); if (z > max) max = z; sum += z; } return [max, sum / ref.length]; };
    const maxZ = new Float64Array(n).fill(NaN);
    const meanZ = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) if (!isTest[i]) [maxZ[i], meanZ[i]] = scoreRow(i);
    const tMax = freeze("zscore_baseline", "max_abs_z", maxZ);
    const tMean = freeze("zscore_baseline", "mean_abs_z", meanZ);
    for (let i = 0; i < n; i++) if (isTest[i]) [maxZ[i], meanZ[i]] = scoreRow(i);
    scored.push({ model: "zscore_baseline", score: "max_abs_z", primary: true, values: maxZ, thresholds: tMax });
    scored.push({ model: "zscore_baseline", score: "mean_abs_z", primary: false, values: meanZ, thresholds: tMean });
  });

  // ---- step 4: RBM with CD-k -------------------------------------------------------------------------------
  const vw = prepared.visibleWidth;
  const rbmHistory = {};
  const fitTrainVisible = gather(prepared.visible, vw, fitTrain);
  const validationVisible = gather(prepared.visible, vw, fitValidation);
  await timed("step04_train_rbm", async () => {
    for (const [index, k] of o.rbm.kValues.entries()) {
      const base = 0.15 + (0.35 * index) / o.rbm.kValues.length;
      report("rbm", base, `Training the RBM with CD-${k}`);
      await yieldToLoop();
      const rng = createRng(o.seed);
      const rbm = createRbm(vw, o.rbm.nHidden, rng);
      rbmHistory[k] = trainRbm(rbm, fitTrainVisible, fitTrain.length, validationVisible, fitValidation.length, { k, ...o.rbm }, rng,
        (f) => report("rbm", base + (0.35 / o.rbm.kValues.length) * f, `Training the RBM with CD-${k}`));
      const nonTest = indices(n, (i) => !isTest[i]);
      const testRows = indices(n, (i) => isTest[i]);
      const energy = new Float64Array(n).fill(NaN);
      freeEnergy(rbm, gather(prepared.visible, vw, nonTest), nonTest.length).forEach((v, j) => { energy[nonTest[j]] = v; });
      const thresholds = freeze(`rbm_k${k}`, "free_energy", energy);
      freeEnergy(rbm, gather(prepared.visible, vw, testRows), testRows.length).forEach((v, j) => { energy[testRows[j]] = v; });
      scored.push({ model: `rbm_k${k}`, score: "free_energy", primary: true, values: energy, thresholds });
    }
  });

  // ---- step 5: VAE -----------------------------------------------------------------------------------------
  report("vae", 0.52, "Training the VAE");
  await yieldToLoop();
  const fw = prepared.featureNames.length;
  const vaeResult = await timed("step05_train_vae", () => {
    const rng = createRng(o.seed);
    const vae = createVae(fw, o.vae.hiddenSizes, o.vae.latentDim, rng);
    const history = trainVae(vae, gather(prepared.standardised, fw, fitTrain), fitTrain.length, gather(prepared.standardised, fw, fitValidation), fitValidation.length, o.vae, rng,
      (f) => report("vae", 0.52 + 0.2 * f, "Training the VAE"));
    const nonTest = indices(n, (i) => !isTest[i]);
    const testRows = indices(n, (i) => isTest[i]);
    const reconstruction = new Float64Array(n).fill(NaN);
    const negativeElbo = new Float64Array(n).fill(NaN);
    const latent = new Float32Array(n * o.vae.latentDim);
    const fill = (rows) => {
      const s = vaeScores(vae, gather(prepared.standardised, fw, rows), rows.length);
      rows.forEach((r, j) => { reconstruction[r] = s.reconstruction[j]; negativeElbo[r] = s.negativeElbo[j]; latent.set(s.latent.subarray(j * o.vae.latentDim, (j + 1) * o.vae.latentDim), r * o.vae.latentDim); });
    };
    fill(nonTest);
    const tRec = freeze("vae", "reconstruction_error", reconstruction);
    const tElbo = freeze("vae", "negative_elbo", negativeElbo);
    fill(testRows);
    scored.push({ model: "vae", score: "reconstruction_error", primary: true, values: reconstruction, thresholds: tRec });
    scored.push({ model: "vae", score: "negative_elbo", primary: false, values: negativeElbo, thresholds: tElbo });
    return { history, latent };
  });

  // ---- step 6: Markov chain over operating states ---------------------------------------------------------
  report("markov", 0.74, "Fitting the Markov chain");
  await yieldToLoop();
  const markov = await timed("step06_fit_markov_states", () => {
    const fitReadings = indices(columns.time.length, (i) => columns.time[i] < prepared.calibStart);
    const binWidthFor = (name) => (name === "Motor_current" ? 0.1 : undefined);
    const statesFor = (name) => candidateStates(fitReadings.map((i) => columns.analog[name][i]), binWidthFor(name));
    let stateColumn = meta.stateColumn && meta.stateColumn !== "auto" ? meta.stateColumn : null;
    if (!stateColumn) {
      // MetroPT's motor current is the documented state signal; otherwise use the sensor with the most levels.
      const counts = analogNames.map((name) => ({ name, states: statesFor(name).cutPoints.length + 1 }));
      const motor = counts.find((c) => c.name === "Motor_current" && c.states > 1);
      const most = counts.reduce((best, c) => (c.states > best.states ? c : best), { name: null, states: 1 });
      stateColumn = motor?.name ?? most.name;
    }
    if (!stateColumn) { notes.push("No sensor showed two or more distinct operating states in the training period, so the Markov chain was skipped."); return null; }
    const { cutPoints } = statesFor(stateColumn);
    if (!cutPoints.length) { notes.push(`${stateColumn} has a single operating state in the training period, so the Markov chain was skipped.`); return null; }
    const inFitWindow = Uint8Array.from(agg.readingWindow, (w) => (usable[w] && (split[w] === prep.FIT_TRAIN || split[w] === prep.FIT_VALIDATION) ? 1 : 0));
    const { scores, matrix, counts } = markovWindowScores({ time: columns.time, values: columns.analog[stateColumn], cutPoints, readingWindow: agg.readingWindow, inFitWindow, maxGapSeconds: o.audit.gapMultiplier * cadence.medianAllSeconds, nWindows: n });
    const thresholds = freeze("markov_state_model", "max_transition_surprisal", scores);
    scored.push({ model: "markov_state_model", score: "max_transition_surprisal", primary: true, values: scores, thresholds });
    return { stateColumn, cutPoints, matrix, counts, fullRecordStates: candidateStates(Array.from(columns.analog[stateColumn]), binWidthFor(stateColumn)).states };
  });

  // ---- step 8 (evaluation) runs before the network so the case study can pick a model -------------------
  report("evaluate", 0.8, "Evaluating alerts at event level");
  await yieldToLoop();
  const settings = { windowSeconds, mergeGapMinutes: o.alerts.mergeGapMinutes, minEventWindows: o.alerts.minEventWindows, leadHours: o.evaluation.leadHours };
  const windows = (meta.failureWindows ?? []).map((w, i) => ({ event: i + 1, start: w.start, end: w.end, label: w.label }));
  const evaluation = await timed("step08_evaluate_alerts", () => scored.map((s) => ({
    s,
    results: evaluateModel({ model: s.model, score: s.score, isPrimary: s.primary, starts: agg.windowStart, scores: s.values, split, usable, thresholds: s.thresholds, windows, settings }),
  })));
  const metricRows = evaluation.flatMap((e) => e.results.map((r) => r.row));

  // ---- step 7: Bayesian network -----------------------------------------------------------------------------
  report("network", 0.86, "Learning the Bayesian network");
  await yieldToLoop();
  const network = await timed("step07_build_bayesian_network", () => {
    const fitRows = prepared.fitRows;
    const fitFraction = (name) => fitRows.reduce((s, i) => s + prepared.digitalBinary[`${name}_active`][i], 0) / fitRows.length;
    const analogNodes = analogNames.slice(0, o.network.maxAnalogNodes).map((a) => `${a}_mean`);
    const digitalNodes = digitalNames.filter((d) => { const f = fitFraction(d); return f >= o.audit.nearConstantFraction && f <= 1 - o.audit.nearConstantFraction; }).map((d) => `${d}_active`);
    const nodes = [...analogNodes, ...digitalNodes].slice(0, o.network.maxNodes);
    const columnOf = (node) => prepared.states[node] ?? prepared.digitalBinary[node];
    const cards = Object.fromEntries(nodes.map((nd) => [nd, prepared.binEdges[nd] ? prepared.binEdges[nd].length + 1 : 2]));
    const fitData = Object.fromEntries(nodes.map((nd) => [nd, Int8Array.from(fitRows, (i) => columnOf(nd)[i])]));
    const structure = learnStructure(fitData, nodes, cards, o.network.maxIndegree, o.seed);
    const edges = structure.edges;
    const model = fitNetwork(edges, fitData, nodes, cards, o.network.ess);
    const strongest = edges.length ? strongestEdge(model, fitData) : null;
    const assignment = (i) => Object.fromEntries(nodes.map((nd) => [nd, columnOf(nd)[i]]));

    const primaryRows = metricRows.filter((r) => r.is_primary_score && r.percentile === o.thresholds.primary);
    // Without reported windows, explain the most selective model that raised any alert at all.
    const alerting = windows.length ? primaryRows : primaryRows.filter((r) => r.n_alert_events > 0);
    const best = [...(alerting.length ? alerting : primaryRows)].sort((a, b) => b.events_detected - a.events_detected || a.unconfirmed_alerts_per_day - b.unconfirmed_alerts_per_day)[0];
    const bestEvaluation = evaluation.find((e) => e.s.model === best.model && e.s.score === best.score);
    const events = bestEvaluation.results.find((r) => r.row.percentile === o.thresholds.primary).events;
    let explained;
    if (windows.length) {
      explained = windows.map((w) => ({ w, inside: events.filter((e) => e.start <= w.end && e.end >= w.start) }))
        .filter((x) => x.inside.length).slice(0, o.network.contextEvents)
        .map(({ w, inside }) => ({ label: `Reported window ${w.event}`, event: inside.reduce((m, e) => (e.peak > m.peak ? e : m)) }));
    } else {
      explained = [...events].sort((a, b) => b.peak - a.peak).slice(0, o.network.contextEvents).map((event, i) => ({ label: `Alert event ${i + 1}`, event }));
    }
    const calibAssignments = calibRows.map(assignment);
    const calibSurprisal = Object.fromEntries(nodes.map((nd) => [nd, blanketSurprisal(model, calibAssignments, nd)]));
    const chance = UNUSUAL_SHARE_FACTOR * (1 - o.network.contextPercentile / 100);
    const contextEvents = explained.map(({ label, event }) => {
      const rows = indices(n, (i) => usable[i] && agg.windowStart[i] >= event.start && agg.windowStart[i] < event.end);
      const eventAssignments = rows.map(assignment);
      const table = nodes.map((nd) => {
        const observed = blanketSurprisal(model, eventAssignments, nd);
        const high = percentile(calibSurprisal[nd], o.network.contextPercentile);
        const share = observed.filter((v) => v > high).length / Math.max(observed.length, 1);
        const mean = (arr) => arr.reduce((s, v) => s + v, 0) / Math.max(arr.length, 1);
        return { variable: nd, event_mean: Number(mean(observed).toFixed(4)), calib_mean: Number(mean(calibSurprisal[nd]).toFixed(4)), [`calib_p${o.network.contextPercentile}`]: Number(high.toFixed(4)), [`share_above_calib_p${o.network.contextPercentile}`]: Number(share.toFixed(4)), unusual: share > chance ? "yes" : "no" };
      }).sort((a, b) => (b.event_mean - b.calib_mean) - (a.event_mean - a.calib_mean));
      return { label, event, rows: rows.length, table };
    });
    return {
      nodes, edges, model, strongest, layout: layeredLayout(nodes, edges),
      causal: strongest ? conditionalTable(model, strongest.child, strongest.parent) : null,
      evidential: strongest ? conditionalTable(model, strongest.parent, strongest.child) : null,
      localIndependence: localIndependenceCases(model),
      patterns: structurePatterns(model),
      contextModel: { model: best.model, score: best.score },
      bic: { best: structure.score, restarts: structure.scores },
      contextEvents,
    };
  });

  // ---- step 10: dashboard documents --------------------------------------------------------------------------
  report("export", 0.97, "Preparing the dashboard");
  await yieldToLoop();
  runtimes.run_all_steps = Number(Object.entries(runtimes).filter(([k]) => k !== "step01_audit_raw_data").reduce((s, [, v]) => s + v, 0).toFixed(2));
  const bundle = buildBundle({ meta, options: o, columns, cadence, frozenRuns, prepared, windows, scored, metricRows, frozenThresholds, rbmHistory, vaeResult, markov, network, runtimes, notes, settings });
  report("done", 1, "Done");
  return bundle;
}
