import { el, eventPills, fmt, footer, initShell, loadJSON, plainTable, segmented, showError, sortableTable } from "./app.js";

const PERCENTILES = [99, 99.5];
const SETS = [["primary", "Primary scores"], ["secondary", "Secondary scores"], ["all", "All scores"]];
const MAX_FLAGGED_SCALE = 0.25;

try {
  const [context, metrics] = await Promise.all([loadJSON("context"), loadJSON("metrics")]);
  const { wrap } = initShell(
    "metrics.html",
    "Metrics",
    `Event-level results on ${fmt.int(metrics.n_usable_test_windows)} usable test windows (${fmt.num(metrics.test_days, 0)} calendar days). Thresholds were fitted on the training month and frozen before each model scored test windows.`,
  );

  let percentile = context.primary_percentile;
  let scoreSet = "primary";
  const controls = el(
    "div",
    { class: "controls" },
    segmented({ label: "Threshold percentile", options: PERCENTILES.map((p) => [p, `${p}th`]), value: percentile, onChange: (v) => { percentile = v; paint(); } }),
    segmented({ label: "Scores", options: SETS, value: scoreSet, onChange: (v) => { scoreSet = v; paint(); } }),
  );
  const tableHost = el("div");
  const delayHost = el("div");
  const rows = metrics.rows;

  function selected() {
    return rows.filter(
      (r) => r.percentile === percentile && (scoreSet === "all" || (scoreSet === "primary") === r.is_primary_score),
    );
  }

  function paint() {
    const data = selected();
    tableHost.replaceChildren(
      sortableTable({
        caption: `Results at the ${percentile}th-percentile threshold`,
        note: "Click a column heading to sort. Threshold values are in each model's own score units.",
        columns: [
          { key: "model_label", label: "Model", render: (r) => el("span", {}, el("b", {}, r.model_label), el("span", { class: "muted small" }, ` ${r.score_label}`)) },
          { key: "threshold", label: "Threshold", align: "right", render: (r) => fmt.num(r.threshold, Math.abs(r.threshold) < 10 ? 3 : 1) },
          { key: "alert_window_fraction", label: "Windows flagged", align: "right", render: (r) => el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.alert_window_fraction, r.alert_window_fraction < 0.01 ? 2 : 1)), el("span", { class: "meter" }, el("i", { style: `width:${Math.min(100, (r.alert_window_fraction / MAX_FLAGGED_SCALE) * 100)}%;--i:${data.indexOf(r)}` }))) },
          { key: "events_detected", label: "Reported windows detected", render: (r) => el("span", {}, eventPills(r.detected_event_ids), el("span", { class: "muted small" }, `  ${r.events_detected} of 4`)) },
          { key: "n_alert_events", label: "Alert events", align: "right", render: (r) => fmt.int(r.n_alert_events) },
          { key: "unconfirmed_alerts_per_day", label: "Unconfirmed alerts per day", align: "right", render: (r) => fmt.num(r.unconfirmed_alerts_per_day, 1) },
        ],
        rows: data,
      }),
    );
    delayHost.replaceChildren(
      plainTable({
        caption: `Detection delay at the ${percentile}th percentile`,
        note: "First alert time in [start − 24 h, end] minus the reported start. Negative means before the reported start.",
        columns: [
          { key: "model_label", label: "Model", render: (r) => el("b", {}, r.model_label) },
          ...[1, 2, 3, 4].map((e) => ({ key: `e${e}`, label: `Event ${e}`, align: "right", render: (r) => fmt.hours(r[`delay_hours_event${e}`]) })),
        ],
        rows: rows.filter((r) => r.percentile === percentile && r.is_primary_score),
      }),
    );
  }
  paint();

  const primary = (p) => rows.filter((r) => r.is_primary_score && r.percentile === p);
  const sensitivity = primary(99).map((a) => ({ a, b: primary(99.5).find((r) => r.model === a.model) }));
  const rbm = primary(context.primary_percentile).filter((r) => r.model.startsWith("rbm_k"));
  const maxPerDay = Math.max(...rbm.map((r) => r.unconfirmed_alerts_per_day));

  wrap.append(
    controls,
    tableHost,
    el("div", { class: "note accent" },
      el("p", {}, el("strong", {}, "How to read detection. "), `The four detection ranges (start minus ${context.lead_tolerance_hours} h to end) cover ${fmt.num(metrics.detection_range_days, 1)} of the ${fmt.num(metrics.test_days, 0)} test days, ${fmt.pct(metrics.detection_range_share)} of the period. A model that flags a large share of all windows overlaps them by chance, so read "reported windows detected" together with "windows flagged" and "unconfirmed alerts per day". Only the Markov chain flags few enough windows for its detections to be selective.`),
    ),
    el("h2", {}, "Detection delay"),
    delayHost,
    el("p", { class: "muted small" }, `A delay close to −${context.lead_tolerance_hours} h means an alert event was already running when the lead range began; the delay is clipped to the start of the range. It does not show that the failure was anticipated. This project makes no prediction claim.`),

    el("h2", {}, "Does the threshold choice matter?"),
    plainTable({
      caption: "99th against 99.5th percentile, primary scores",
      note: "Moving to the 99.5th percentile lowers the flagged share for some models and barely moves others.",
      columns: [
        { label: "Model", render: (r) => el("b", {}, r.a.model_label) },
        { label: "Flagged at 99th", align: "right", render: (r) => fmt.pct(r.a.alert_window_fraction, 2) },
        { label: "Flagged at 99.5th", align: "right", render: (r) => fmt.pct(r.b.alert_window_fraction, 2) },
        { label: "Unconfirmed per day, 99th", align: "right", render: (r) => fmt.num(r.a.unconfirmed_alerts_per_day, 1) },
        { label: "Unconfirmed per day, 99.5th", align: "right", render: (r) => fmt.num(r.b.unconfirmed_alerts_per_day, 1) },
        { label: "Detected", render: (r) => `${r.a.events_detected} / ${r.b.events_detected} of 4` },
      ],
      rows: sensitivity,
    }),

    el("h2", {}, "RBM: does the number of Gibbs steps matter?"),
    el("p", {}, `The RBM was trained with contrastive divergence using k = 1, 5 and 10 Gibbs steps, with the same seed, epochs and hidden units. At the ${context.primary_percentile}th percentile the three behave almost identically, and the models were still improving when training ended, so this says little about CD-k in general.`),
    el("div", { class: "table-scroll" },
      el("table", {},
        el("caption", {}, "RBM variants", el("span", {}, "Bars are scaled to the largest value in each column.")),
        el("thead", {}, el("tr", {}, ["Variant", "Windows flagged", "Unconfirmed alerts per day"].map((h) => el("th", { scope: "col" }, h)))),
        el("tbody", {}, rbm.map((r) => el("tr", {},
          el("td", {}, el("b", {}, r.model_label)),
          el("td", {}, el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.alert_window_fraction)), el("span", { class: "meter brand" }, el("i", { style: `width:${(r.alert_window_fraction / Math.max(...rbm.map((x) => x.alert_window_fraction))) * 100}%` })))),
          el("td", {}, el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.num(r.unconfirmed_alerts_per_day, 1)), el("span", { class: "meter brand" }, el("i", { style: `width:${(r.unconfirmed_alerts_per_day / maxPerDay) * 100}%` })))),
        ))),
      ),
    ),

    el("h2", {}, "Evidence that thresholds were frozen first"),
    el("p", {}, "Each model's thresholds were fitted on the held-out end of the training month and written to disk before that model scored any test window. These are the recorded times."),
    plainTable({
      caption: "Frozen thresholds",
      columns: [
        { label: "Model", render: (r) => el("b", {}, r.model) },
        { label: "Score", key: "score" },
        { label: "Frozen at", render: (r) => r.frozen_at.replace("T", " ") },
        { label: "Calibration windows", align: "right", render: (r) => fmt.int(r.n_calib_windows) },
      ],
      rows: metrics.frozen_thresholds,
    }),
    footer(context),
  );
} catch (error) {
  initShell("metrics.html", "Metrics");
  showError(error);
}
