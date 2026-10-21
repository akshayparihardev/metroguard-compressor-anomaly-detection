// Unit tests for the in-browser analysis engine.
// Run from the repository root: node --test --disable-warning=MODULE_TYPELESS_PACKAGE_JSON "dashboard/tests/*.test.mjs"
import assert from "node:assert/strict";
import { test } from "node:test";

import { buildAlertEvents, evaluateEvents, fitThresholds } from "../assets/analysis/alerting.js";
import { fitNetwork, layeredLayout, learnStructure, structurePatterns, blanketSurprisal } from "../assets/analysis/bayesian-network.js";
import { candidateStates, markovWindowScores } from "../assets/analysis/markov.js";
import { createRng, digitize, percentile, quantileSorted } from "../assets/analysis/numeric.js";
import { DEFAULTS, runPipeline } from "../assets/analysis/pipeline.js";
import { aggregateWindows, assignSplit, frozenValueRuns, pyRound, screenSensors } from "../assets/analysis/preparation.js";
import { createRbm, freeEnergy, trainRbm } from "../assets/analysis/rbm.js";
import { createCsvParser, detectColumns } from "../assets/analysis/table-reader.js";
import { detectTimestampFormat, parserForKind } from "../assets/analysis/timestamps.js";
import { createVae, trainVae, vaeScores } from "../assets/analysis/vae.js";

const parseAll = (text, delimiter = ",") => { const rows = []; const p = createCsvParser(delimiter, (r) => rows.push(r)); p.push(text); p.end(); return rows; };

test("CSV parser handles quotes, escaped quotes, CRLF and chunk boundaries", () => {
  assert.deepEqual(parseAll('a,b\r\n"x, y","he said ""hi"""\r\n1,2\n'), [["a", "b"], ["x, y", 'he said "hi"'], ["1", "2"]]);
  const rows = [];
  const p = createCsvParser(";", (r) => rows.push(r));
  p.push('t;"v'); p.push('al";3\n4;5'); p.end();
  assert.deepEqual(rows, [["t", "val", "3"], ["4", "5"]]);
});

test("timestamp formats are detected and parsed as naive wall-clock time", () => {
  assert.equal(detectTimestampFormat(["2020-02-01 00:00:10", "2020-02-01 00:00:20"]).kind, "iso");
  assert.equal(parserForKind("iso")("2020-02-01 00:00:10"), Date.UTC(2020, 1, 1, 0, 0, 10));
  assert.equal(parserForKind("iso")("2020-02-01T05:30:00+05:30"), Date.UTC(2020, 1, 1, 0, 0, 0));
  assert.equal(detectTimestampFormat(["13/02/2020 10:00", "14/02/2020 10:00"]).kind, "dmy");
  assert.equal(detectTimestampFormat(["02/13/2020 10:00", "02/14/2020 10:00"]).kind, "mdy");
  assert.match(detectTimestampFormat(["01/02/2020 10:00"]).note, /assumed/);
  assert.equal(detectTimestampFormat(["1580515200", "1580515260"]).kind, "epoch-s");
  assert.equal(detectTimestampFormat(["12.5", "13.1"]), null);
});

test("column roles are detected from values, and unnamed index columns are ignored", () => {
  const header = ["", "timestamp", "pressure", "valve"];
  const rows = [["0", "2020-02-01 00:00:00", "9.3", "1"], ["10", "2020-02-01 00:00:10", "9.1", "0"]];
  assert.deepEqual(detectColumns(header, rows).map((c) => c.role), ["ignore", "timestamp", "analog", "digital"]);
});

test("numeric helpers match numpy conventions", () => {
  assert.equal(quantileSorted(Float64Array.from([1, 2, 3, 4]), 1 / 3), 2);
  assert.equal(percentile([1, 2, 3, 4, 5], 50), 3);
  assert.equal(digitize(5, [1, 5, 9]), 2);
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
});

test("chronological split follows the Python rules", () => {
  const starts = Float64Array.from({ length: 1200 }, (_, i) => Date.UTC(2020, 1, 1) + i * 60000);
  const trainEnd = starts[1000];
  const { split } = assignSplit(starts, new Uint8Array(1200).fill(1), trainEnd, 0.2, 0.1);
  const count = (name) => split.filter((s) => s === name).length;
  assert.deepEqual([count("fit_train"), count("fit_validation"), count("calib"), count("test")], [720, 80, 200, 200]);
  const order = ["fit_train", "fit_validation", "calib", "test"];
  for (let i = 1; i < split.length; i++) assert.ok(order.indexOf(split[i]) >= order.indexOf(split[i - 1]));
});

test("frozen runs and window aggregation", () => {
  const time = Float64Array.from({ length: 30 }, (_, i) => i * 10_000);
  const a = Float32Array.from({ length: 30 }, (_, i) => (i >= 5 && i < 25 ? 7 : i));
  assert.equal(frozenValueRuns(time, { a }, 2).length, 1);
  const agg = aggregateWindows({ time, analog: { a }, digital: { d: Float32Array.from({ length: 30 }, (_, i) => i % 2) } }, 60, ["a"]);
  assert.equal(agg.windowStart.length, 5);
  assert.equal(agg.readingCount[0], 6);
  assert.equal(agg.features.d_active_fraction[0], 0.5);
  assert.equal(agg.readingWindow[29], 4);
});

test("sensors blank or constant in the training period are left out, judged on training rows only", () => {
  const time = Float64Array.from({ length: 60 * 24 * 40 }, (_, i) => Date.UTC(2021, 0, 1) + i * 60000);
  const n = time.length;
  const janEnd = time.findIndex((t) => t >= Date.UTC(2021, 1, 1));
  const varying = Float32Array.from({ length: n }, (_, i) => Math.sin(i / 50));
  const blank = Float32Array.from({ length: n }, (_, i) => (i % 4 === 0 ? 1 + (i % 7) : NaN));
  const flatThenMoving = Float32Array.from({ length: n }, (_, i) => (i < janEnd ? 3 : i % 5));
  const { columns, dropped } = screenSensors({ time, analog: { varying, blank, flatThenMoving }, digital: {} }, { trainPeriod: "first_calendar_month", trainShare: 0.2 });
  assert.deepEqual(Object.keys(columns.analog), ["varying"]);
  assert.deepEqual(dropped.map((d) => d.name), ["blank", "flatThenMoving"]);
  assert.match(dropped[1].reason, /never changes/);
});

test("RBM free energy equals brute-force -log sum_h exp(-E(v,h))", () => {
  const rng = createRng(1);
  const rbm = createRbm(4, 3, rng);
  rbm.W.forEach((_, i) => { rbm.W[i] = rng.normal(); });
  rbm.a.forEach((_, i) => { rbm.a[i] = rng.normal(); });
  rbm.b.forEach((_, i) => { rbm.b[i] = rng.normal(); });
  const visible = new Float32Array(16 * 4);
  for (let s = 0; s < 16; s++) for (let i = 0; i < 4; i++) visible[s * 4 + i] = (s >> i) & 1;
  const fe = freeEnergy(rbm, visible, 16);
  for (let s = 0; s < 16; s++) {
    let total = 0;
    for (let h = 0; h < 8; h++) {
      let negE = 0;
      for (let i = 0; i < 4; i++) negE += rbm.a[i] * visible[s * 4 + i];
      for (let j = 0; j < 3; j++) { const hj = (h >> j) & 1; negE += rbm.b[j] * hj; for (let i = 0; i < 4; i++) negE += visible[s * 4 + i] * rbm.W[i * 3 + j] * hj; }
      total += Math.exp(negE);
    }
    assert.ok(Math.abs(fe[s] + Math.log(total)) < 1e-9);
  }
});

test("CD-k training lowers the free energy of the data", () => {
  const rng = createRng(1);
  const rbm = createRbm(4, 3, rng);
  const data = new Float32Array(128 * 4);
  for (let s = 0; s < 128; s++) data.set(s < 64 ? [1, 1, 0, 0] : [0, 0, 1, 1], s * 4);
  const before = freeEnergy(rbm, data, 128).reduce((x, y) => x + y) / 128;
  const history = trainRbm(rbm, data, 128, data, 128, { k: 1, epochs: 30, batchSize: 32, learningRate: 0.1, momentum: 0.5, weightDecay: 0 }, rng);
  assert.ok(freeEnergy(rbm, data, 128).reduce((x, y) => x + y) / 128 < before);
  assert.ok(history.fit_reconstruction_error.at(-1) < history.fit_reconstruction_error[0]);
});

test("VAE training reduces validation loss and scores are deterministic", () => {
  const rng = createRng(3);
  const f = 4;
  const data = Float32Array.from({ length: 400 * f }, (_, i) => Math.sin(i * 0.37) + (i % f));
  const vae = createVae(f, [8, 6], 2, rng);
  const history = trainVae(vae, data, 400, data, 400, { epochs: 30, batchSize: 32, learningRate: 0.01, beta: 1, patience: 30 }, rng);
  assert.ok(Math.min(...history.validation_loss) < history.validation_loss[0]);
  const a = vaeScores(vae, data, 400);
  const b = vaeScores(vae, data, 400);
  assert.deepEqual(Array.from(a.reconstruction), Array.from(b.reconstruction));
  assert.equal(a.latent.length, 400 * 2);
});

test("candidate states find separated operating levels and the Markov chain scores rare transitions", () => {
  const rng = createRng(5);
  const values = Array.from({ length: 6000 }, (_, i) => [0, 4, 7][Math.floor(i / 200) % 3] + rng.normal() * 0.1);
  const { cutPoints } = candidateStates(values);
  assert.equal(cutPoints.length, 2);
  const time = Float64Array.from({ length: 6000 }, (_, i) => i * 10_000);
  const readingWindow = Int32Array.from({ length: 6000 }, (_, i) => Math.floor(i / 6));
  const { scores } = markovWindowScores({ time, values: Float32Array.from(values), cutPoints, readingWindow, inFitWindow: new Uint8Array(6000).fill(1), maxGapSeconds: 20, nWindows: 1000 });
  assert.ok(Number.isNaN(scores[0]) === false);
  assert.ok(Math.max(...scores.filter(Number.isFinite)) > Math.min(...scores.filter(Number.isFinite)));
});

test("structure search recovers a chain, d-separation behaves, blanket probabilities are proper", () => {
  const rng = createRng(7);
  const n = 3000;
  const A = Int8Array.from({ length: n }, () => (rng.uniform() < 0.5 ? 0 : 1));
  const B = Int8Array.from(A, (a) => (rng.uniform() < 0.9 ? a : 1 - a));
  const C = Int8Array.from(B, (b) => (rng.uniform() < 0.9 ? b : 1 - b));
  const data = { A, B, C };
  const cards = { A: 2, B: 2, C: 2 };
  const { edges } = learnStructure(data, ["A", "B", "C"], cards, 3);
  const skeleton = new Set(edges.map(([x, y]) => [x, y].sort().join("-")));
  assert.deepEqual([...skeleton].sort(), ["A-B", "B-C"]);
  const model = fitNetwork(edges, data, ["A", "B", "C"], cards, 10);
  const pattern = structurePatterns(model).find((p) => p.includes(" B "));
  if (pattern?.startsWith("chain") || pattern?.startsWith("fork")) assert.match(pattern, /without evidence = True; connected given B = False/);
  const p = blanketSurprisal(model, [{ A: 0, B: 0, C: 0 }, { A: 0, B: 1, C: 0 }], "B").map((s) => Math.exp(-s));
  assert.ok(Math.abs(p[0] + p[1] - 1) < 1e-9);
  assert.equal(layeredLayout(["A", "B", "C"], edges).length, 3);
});

test("alert events merge within the gap and early alerts inside the lead tolerance count as detections", () => {
  const m = (k) => Date.UTC(2020, 2, 1) + k * 60000;
  const starts = [m(0), m(1), m(2), m(3), m(4), m(30)];
  const events = buildAlertEvents(starts, [0, 5, 0, 0, 5, 9], 1, 60, 10, 1);
  assert.equal(events.length, 2);
  assert.equal(events[0].n, 2);
  assert.deepEqual(buildAlertEvents([m(0), m(11)], [2, 2], 1, 60, 10, 1).length, 1);
  assert.equal(buildAlertEvents([m(0), m(1)], [1, NaN], 1, 60, 10, 1).length, 0);
  const result = evaluateEvents(events, [{ event: 1, start: m(60), end: m(90) }], 24);
  assert.equal(result.events_detected, 1);
  assert.ok(Math.abs(result.delay_hours_event1 - -59 / 60) < 1e-12);
  assert.equal(result.unconfirmed_alert_events, 0);
  assert.deepEqual(Object.keys(fitThresholds([1, 2, 3, 4], [99, 99.5])), ["99", "99.5"]);
});

test("end to end on a small fixture: valid bundle, thresholds frozen, no NaN in the JSON", async () => {
  const rng = createRng(11);
  const n = 3 * 30 * 24 * 60;
  const time = Float64Array.from({ length: n }, (_, i) => Date.UTC(2021, 0, 1) + i * 60000);
  const state = Float32Array.from({ length: n }, (_, i) => (Math.floor(i / 90) % 3 === 0 ? 0.1 : Math.floor(i / 90) % 3 === 1 ? 4 : 6) + rng.normal() * 0.05);
  const pressure = Float32Array.from(state, (s, i) => 9 - 0.1 * s + rng.normal() * 0.05 + (i > n - 3000 && i < n - 2500 ? 3 : 0));
  const valve = Float32Array.from(state, (s) => (s > 1 ? 1 : 0));
  const failure = { start: time[n - 2900], end: time[n - 2600] };
  const options = { ...DEFAULTS, windows: { sizeSeconds: 300, minReadingsFraction: 0.5 }, rbm: { ...DEFAULTS.rbm, kValues: [1], epochs: 3 }, vae: { ...DEFAULTS.vae, epochs: 3 } };
  const bundle = await runPipeline({ time, analog: { pressure, state }, digital: { valve }, rowsRead: n, badTimestamps: 0 }, { fileName: "fixture.csv", fileKind: "csv", rowsRead: n, columnCount: 4, failureWindows: [failure], stateColumn: "auto" }, options);
  for (const key of ["timeline", "metrics", "training", "audit", "network", "context", "latent", "event_1"]) assert.ok(bundle[key], key);
  assert.doesNotThrow(() => JSON.stringify(bundle, (k, v) => { if (typeof v === "number" && !Number.isFinite(v)) throw new Error(`non-finite ${k}`); return v; }));
  assert.equal(bundle.context.source.kind, "upload");
  assert.ok(bundle.metrics.frozen_thresholds.length >= 4);
  assert.ok(bundle.metrics.rows.some((r) => r.model === "markov_state_model"));
});
