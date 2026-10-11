import { el, eventIds, eventPills, fmt, footer, initShell, isUpload, loadJSON, plainTable, showError } from "./app.js";
import { lineMap } from "./linemap.js";

const PIPELINE = [
  ["step01_audit_raw_data", "Audit the raw file"],
  ["step02_build_features", (context) => `Build ${fmt.windowLength(context.window_seconds)} windows`],
  ["step03_score_zscore_baseline", "Z-score baseline"],
  ["step04_train_rbm", "RBM, Gibbs and CD-k"],
  ["step05_train_vae", "VAE"],
  ["step06_fit_markov_states", "Markov chain"],
  ["step07_build_bayesian_network", "Bayesian network"],
  ["step08_evaluate_alerts", "Evaluate alerts"],
];

const range = (values) => [Math.min(...values), Math.max(...values)];
const flagged = (r) => fmt.pct(r.alert_window_fraction, r.alert_window_fraction < 0.01 ? 2 : 1);

function metroGuardStory(context, metrics, byModel, primary, ids) {
  const rbm = primary.filter((r) => r.model.startsWith("rbm_k"));
  const [rbmLow, rbmHigh] = range(rbm.map((r) => r.alert_window_fraction));
  const allEvents = primary.filter((r) => r.events_detected === ids.length).map((r) => r.model_label);
  const markov = byModel.markov_state_model;
  return [
    el("p", { class: "lede" },
      `The ${context.primary_percentile}th-percentile thresholds were fixed on the training month, so each model was set to flag about 1% of windows. On the ${fmt.num(metrics.test_days, 0)} calendar days of test data the Z-score baseline flags ${fmt.pct(byModel.zscore_baseline.alert_window_fraction)} of windows, the RBMs ${fmt.pct(rbmLow)} to ${fmt.pct(rbmHigh)}, the VAE ${fmt.pct(byModel.vae.alert_window_fraction)}, and the Markov chain ${fmt.pct(markov.alert_window_fraction, 2)}.`),
    el("p", {},
      `${allEvents.length === 5 ? "Five of six models" : `${allEvents.length} models`} detect all four reported windows, but the detection ranges together cover only ${fmt.pct(metrics.detection_range_share)} of the test period. A model that flags a tenth or more of all windows will overlap them by chance. The Markov chain is the only one selective enough for its detections to carry weight, and it misses event ${ids.find((e) => !markov.detected_event_ids.split(";").includes(String(e)))}.`),
  ];
}

function uploadStory(context, metrics, primary, ids) {
  const sorted = [...primary].sort((a, b) => a.alert_window_fraction - b.alert_window_fraction);
  const rates = sorted.map((r) => `${r.model_label}: ${flagged(r)}`).join("; ");
  const expected = 100 - context.primary_percentile;
  const story = [
    el("p", { class: "lede" },
      `The ${context.primary_percentile}th-percentile thresholds were fixed on the held-out end of the training period, so on data like it each model would flag about ${expected}% of windows. On the ${fmt.num(metrics.test_days, 1)} days of test data the share flagged is: ${rates}.`),
    el("p", {}, "A share far above that level means the test period differs from the training period for that model. That can be a fault, a change in how the machine is used, or a training period that did not cover all normal behaviour; the scores alone cannot say which."),
  ];
  if (ids.length) {
    story.push(el("p", {}, `The detection ranges of the ${ids.length} reported window${ids.length === 1 ? "" : "s"} (each start minus ${context.lead_tolerance_hours} h, to its end) cover ${fmt.pct(metrics.detection_range_share)} of the test period. A model that flags a large share of all windows overlaps them by chance, so read detections together with the share flagged.`));
  } else {
    story.push(el("p", {}, "No reported windows were given, so detection cannot be measured. The tables show how often each model alerts."));
  }
  return story;
}

try {
  const [context, metrics, timeline, audit] = await Promise.all([
    loadJSON("context"), loadJSON("metrics"), loadJSON("timeline"), loadJSON("audit"),
  ]);
  const upload = isUpload(context);
  const ids = eventIds(context);
  const days = audit.span.split(" ")[0];
  const { wrap } = initShell(
    "index.html",
    upload ? "Your analysis" : "MetroGuard",
    upload
      ? `What the models learned from ${days} days of ${context.source.file_name}, how often they flagged windows in the test period, and how far that evidence goes.`
      : `Anomaly screening for a metro train's air compressor. This site shows what four models learned from ${days} days of sensor data, how often they flagged windows, and how far that evidence goes.`,
  );

  const primary = metrics.rows.filter((r) => r.is_primary_score && r.percentile === context.primary_percentile);
  const byModel = Object.fromEntries(primary.map((r) => [r.model, r]));
  const end = timeline.start_minute + timeline.coverage.length * timeline.step_minutes;

  wrap.append(
    el("section", { class: "reveal", "aria-labelledby": "map-h" },
      el("h2", { id: "map-h", class: "muted small", style: "font-size:.82rem;font-weight:600;margin-top:2.2rem;letter-spacing:0" }, `The record, ${fmt.date(timeline.start_minute)} to ${fmt.date(end)}`),
      el("div", { class: "linemap", id: "linemap" }),
      el("div", { class: "legend" },
        el("span", {}, el("i", { class: "sw-line" }), timeline.step_minutes === 60 ? "Hours with scored data" : "Time with scored data"),
        el("span", {}, el("i", { class: "sw-gap" }), "Gap in the record"),
        ids.length ? el("span", {}, el("i", { class: "sw-station" }), "Reported failure window (open it)") : null,
        el("span", {}, el("i", { class: "sw-rib" }), "Share of each day the baseline is above its threshold"),
      ),
    ),
  );
  lineMap(document.getElementById("linemap"), timeline, { trainingLabel: upload ? "Training period" : "Training month" });

  const columns = [
    { label: "Model", key: "model_label", render: (r) => el("span", {}, el("b", {}, r.model_label), el("span", { class: "muted small" }, ` ${r.score_label}`)) },
    { label: "Windows flagged", key: "f", align: "right", render: (r) => el("span", { class: "cell-bar" }, el("span", { class: "num" }, flagged(r)), el("span", { class: "meter" }, el("i", { style: `width:${Math.min(100, (r.alert_window_fraction / 0.25) * 100)}%;--i:${primary.indexOf(r)}` }))) },
    ids.length ? { label: "Reported windows detected", key: "d", render: (r) => el("span", {}, eventPills(r.detected_event_ids, ids.length), el("span", { class: "muted small" }, `  ${r.events_detected} of ${ids.length}`)) } : null,
    { label: "Unconfirmed alerts per day", key: "u", align: "right", render: (r) => fmt.num(r.unconfirmed_alerts_per_day, 1) },
  ].filter(Boolean);

  wrap.append(
    el("h2", {}, "Read the detection numbers next to the alert rates"),
    ...(upload ? uploadStory(context, metrics, primary, ids) : metroGuardStory(context, metrics, byModel, primary, ids)),
    el("div", { class: "section-head" }, el("h3", {}, "All models at the primary threshold"), el("a", { href: "metrics.html" }, "Full metrics and the 99.5th percentile row")),
    plainTable({
      caption: `${context.primary_percentile}th-percentile threshold, usable test windows`,
      note: `${fmt.int(metrics.n_usable_test_windows)} windows over ${fmt.num(metrics.test_days, upload ? 1 : 0)} calendar days`,
      columns,
      rows: primary,
    }),
    el("div", { class: "note" },
      el("p", {}, el("strong", {}, "Unconfirmed alert"), upload
        ? " means an alert outside every reported window plus its lead tolerance. It is not known to be a false alarm: the reported windows are only the problems someone wrote down."
        : " means an alert outside every reported window plus its lead tolerance. It is not known to be a false alarm: the reported windows are four company-listed intervals, not a complete record of the compressor's condition."),
    ),

    el("h2", {}, "How the result was produced"),
    el("p", {}, upload
      ? `Eight steps, run in order in this browser. The models learn from the training period only (${context.source.train_period.toLowerCase()}, ending ${context.source.calib_start.slice(0, 10)} plus a held-out part); thresholds are fixed on that held-out part before any later data is scored.`
      : "Eight steps, run in order and each logged. The models learn from the first month only; thresholds are fixed on a held-out part of that month before any later data is scored."),
    el("ol", { class: "steps" },
      PIPELINE.map(([id, label]) => el("li", {}, el("b", {}, typeof label === "function" ? label(context) : label), el("span", {}, fmt.seconds(context.runtimes_seconds[id])))),
    ),
    el("p", { class: "muted small" }, upload
      ? `Steps 2 to 8 together took ${fmt.seconds(context.runtimes_seconds.run_all_steps)} in this browser.`
      : `Steps 2 to 8 together took ${fmt.seconds(context.runtimes_seconds.run_all_steps)}; the RBM and the VAE were trained on a laptop GPU.`),

    el("h2", {}, "Where to look next"),
    el("ul", { class: "list" },
      el("li", {}, el("a", { href: "scores.html" }, "Scores over time"), ids.length ? ": drag to zoom, or open any reported window." : ": drag to zoom into any stretch of the record."),
      el("li", {}, el("a", { href: "network.html" }, "Bayesian network"), ": which sensor states look unusual together, and why the arrows are not causes."),
      el("li", {}, el("a", { href: "data.html" }, "Data and training"), ": the gaps and frozen stretches in the raw file, and the training curves."),
      el("li", {}, el("a", { href: "claims.html" }, "Claims and limits"), ": the short list of things this evidence does and does not support."),
      upload ? null : el("li", {}, el("a", { href: "analyse.html" }, "Analyse your data"), ": run the same pipeline on your own time-series file, in your browser."),
    ),
    footer(context),
  );
} catch (error) {
  initShell("index.html", "MetroGuard");
  showError(error);
}
