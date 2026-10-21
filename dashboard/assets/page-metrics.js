import { compact, el, eventIds, eventPills, fmt, footer, initShell, isUpload, loadJSON, plainTable, segmented, showError, sortableTable } from "./app.js";

const SETS = [["primary", "Main score"], ["secondary", "Second score"], ["all", "Both"]];
const MAX_FLAGGED_SCALE = 0.25;

try {
  const [context, metrics] = await Promise.all([loadJSON("context"), loadJSON("metrics")]);
  const upload = isUpload(context);
  const ids = eventIds(context);
  const trainingPeriod = upload ? "training period" : "training month";
  const { wrap } = initShell(
    "metrics.html",
    "Results table",
    `Every number behind the overview. Each row is one detector on the ${fmt.num(metrics.test_days, upload ? 1 : 0)} test days (${fmt.int(metrics.n_usable_test_windows)} one-minute windows). The alert lines were fixed on the ${trainingPeriod} before the test was scored.`,
  );

  const rows = metrics.rows;
  const percentiles = [...new Set(rows.map((r) => r.percentile))].sort((a, b) => a - b);
  let percentile = context.primary_percentile;
  let scoreSet = "primary";
  const controls = el(
    "div",
    { class: "controls" },
    segmented({ label: "Alert line", options: percentiles.map((p, i) => [p, i === 0 ? "Normal" : "Stricter"]), value: percentile, onChange: (v) => { percentile = v; paint(); } }),
    segmented({ label: "Score", options: SETS, value: scoreSet, onChange: (v) => { scoreSet = v; paint(); } }),
  );
  const tableHost = el("div");
  const delayHost = el("div");

  function selected() {
    return rows.filter(
      (r) => r.percentile === percentile && (scoreSet === "all" || (scoreSet === "primary") === r.is_primary_score),
    );
  }

  function paint() {
    const data = selected();
    tableHost.replaceChildren(
      sortableTable({
        caption: `Results with the ${percentile === percentiles[0] ? "normal" : "stricter"} alert line`,
        note: "Click a column heading to sort.",
        columns: [
          { key: "model_label", label: "Model", render: (r) => el("span", {}, el("b", {}, r.model_label), el("span", { class: "muted small" }, ` ${r.score_label}`)) },
          { key: "threshold", label: "Alert line (score)", align: "right", render: (r) => fmt.num(r.threshold, Math.abs(r.threshold) < 10 ? 3 : 1) },
          { key: "alert_window_fraction", label: "Time with an alert", align: "right", render: (r) => el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.alert_window_fraction, r.alert_window_fraction < 0.01 ? 2 : 1)), el("span", { class: "meter" }, el("i", { style: `width:${Math.min(100, (r.alert_window_fraction / MAX_FLAGGED_SCALE) * 100)}%;--i:${data.indexOf(r)}` }))) },
          ids.length ? { key: "events_detected", label: upload ? "Problems caught" : "Leaks caught", render: (r) => el("span", {}, eventPills(r.detected_event_ids, ids.length), el("span", { class: "muted small" }, `  ${r.events_detected} of ${ids.length}`)) } : null,
          { key: "n_alert_events", label: "Alert episodes", align: "right", render: (r) => fmt.int(r.n_alert_events) },
          { key: "unconfirmed_alerts_per_day", label: "Unconfirmed alerts per day", align: "right", render: (r) => fmt.num(r.unconfirmed_alerts_per_day, 1) },
        ].filter(Boolean),
        rows: data,
      }),
    );
    if (!ids.length) return;
    delayHost.replaceChildren(
      plainTable({
        caption: `Detection delay at the ${percentile}th percentile`,
        note: `First alert time in [start − ${context.lead_tolerance_hours} h, end] minus the reported start. Negative means before the reported start.`,
        columns: [
          { key: "model_label", label: "Model", render: (r) => el("b", {}, r.model_label) },
          ...ids.map((e) => ({ key: `e${e}`, label: `Event ${e}`, align: "right", render: (r) => fmt.hours(r[`delay_hours_event${e}`]) })),
        ],
        rows: rows.filter((r) => r.percentile === percentile && r.is_primary_score),
      }),
    );
  }
  paint();

  const primary = (p) => rows.filter((r) => r.is_primary_score && r.percentile === p);
  const [low, high] = [percentiles[0], percentiles.at(-1)];
  const sensitivity = primary(low).map((a) => ({ a, b: primary(high).find((r) => r.model === a.model) }));
  const rbm = primary(context.primary_percentile).filter((r) => r.model.startsWith("rbm_k"));
  const kList = rbm.map((r) => r.model.slice(5));
  const maxFlagged = Math.max(...rbm.map((x) => x.alert_window_fraction));
  const maxPerDay = Math.max(...rbm.map((r) => r.unconfirmed_alerts_per_day));
  const markov = primary(context.primary_percentile).find((r) => r.model === "markov_state_model");

  const detectionNote = ids.length
    ? el("div", { class: "note accent" },
        el("p", {}, el("strong", {}, "How to read detection. "),
          `The ${upload ? ids.length : "four"} detection range${ids.length === 1 ? "" : "s"} (start minus ${context.lead_tolerance_hours} h to end) cover ${fmt.num(metrics.detection_range_days, 1)} of the ${fmt.num(metrics.test_days, upload ? 1 : 0)} test days, ${fmt.pct(metrics.detection_range_share)} of the period. A model that flags a large share of all windows overlaps them by chance, so read "reported windows detected" together with "windows flagged" and "unconfirmed alerts per day".`,
          upload ? "" : " Only the Markov chain flags few enough windows for its detections to be selective."))
    : el("div", { class: "note" }, el("p", {}, "No reported windows were given for this file, so detection and delay cannot be measured. Every alert counts as unconfirmed."));

  wrap.append(...compact(
    controls,
    el("div", { class: "how-to" },
      el("b", {}, "What the columns mean"),
      el("ul", { class: "glossary" },
        el("li", {}, el("b", {}, "Alert line"), ": the score above which a minute counts as an alert, in the detector's own units."),
        el("li", {}, el("b", {}, "Time with an alert"), ": share of test minutes above the line. On normal data this would be about 1% (normal line) or 0.5% (stricter)."),
        ids.length ? el("li", {}, el("b", {}, upload ? "Problems caught" : "Leaks caught"), `: reported windows with an alert during them or up to ${context.lead_tolerance_hours} h before.`) : null,
        el("li", {}, el("b", {}, "Alert episodes"), `: alerts less than ${context.merge_gap_minutes} minutes apart joined into one.`),
        el("li", {}, el("b", {}, "Unconfirmed alerts per day"), ": episodes outside every reported window. Nobody checked the machine then, so they are not known to be false alarms."),
        el("li", {}, el("b", {}, "Main / second score"), ": some detectors give two scores (for example, the VAE's rebuild error and its ELBO); the main one is used everywhere else."),
      )),
    tableHost,
    detectionNote,
    ids.length ? el("h2", {}, "How early was the first alert?") : null,
    ids.length ? delayHost : null,
    ids.length ? el("p", { class: "muted small" }, `A delay close to −${context.lead_tolerance_hours} h means an alert event was already running when the lead range began; the delay is clipped to the start of the range. It does not show that the failure was anticipated. This project makes no prediction claim.`) : null,

    el("h2", {}, "Normal versus stricter alert line"),
    plainTable({
      caption: `${low}th against ${high}th percentile, primary scores`,
      note: `Moving to the ${high}th percentile lowers the flagged share for some models and barely moves others.`,
      columns: [
        { label: "Model", render: (r) => el("b", {}, r.a.model_label) },
        { label: `Flagged at ${low}th`, align: "right", render: (r) => fmt.pct(r.a.alert_window_fraction, 2) },
        { label: `Flagged at ${high}th`, align: "right", render: (r) => fmt.pct(r.b.alert_window_fraction, 2) },
        { label: `Unconfirmed per day, ${low}th`, align: "right", render: (r) => fmt.num(r.a.unconfirmed_alerts_per_day, 1) },
        { label: `Unconfirmed per day, ${high}th`, align: "right", render: (r) => fmt.num(r.b.unconfirmed_alerts_per_day, 1) },
        ids.length ? { label: "Detected", render: (r) => `${r.a.events_detected} / ${r.b.events_detected} of ${ids.length}` } : null,
      ].filter(Boolean),
      rows: sensitivity,
    }),

    el("h2", {}, "RBM: does the number of Gibbs steps matter?"),
    el("p", {}, upload
      ? `The RBM was trained with contrastive divergence using k = ${kList.join(", ")} Gibbs step${kList.length === 1 ? "" : "s"}, with the same seed, epochs and hidden units. Compare the rows below; with the fixed number of epochs this says little about CD-k in general.`
      : `The RBM was trained with contrastive divergence using k = 1, 5 and 10 Gibbs steps, with the same seed, epochs and hidden units. At the ${context.primary_percentile}th percentile the three behave almost identically, and the models were still improving when training ended, so this says little about CD-k in general.`),
    el("div", { class: "table-scroll" },
      el("table", {},
        el("caption", {}, "RBM variants", el("span", {}, "Bars are scaled to the largest value in each column.")),
        el("thead", {}, el("tr", {}, ["Variant", "Windows flagged", "Unconfirmed alerts per day"].map((h) => el("th", { scope: "col" }, h)))),
        el("tbody", {}, rbm.map((r) => el("tr", {},
          el("td", {}, el("b", {}, r.model_label)),
          el("td", {}, el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.alert_window_fraction)), el("span", { class: "meter brand" }, el("i", { style: `width:${maxFlagged ? (r.alert_window_fraction / maxFlagged) * 100 : 0}%` })))),
          el("td", {}, el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.num(r.unconfirmed_alerts_per_day, 1)), el("span", { class: "meter brand" }, el("i", { style: `width:${maxPerDay ? (r.unconfirmed_alerts_per_day / maxPerDay) * 100 : 0}%` })))),
        ))),
      ),
    ),
    upload && !markov ? el("p", { class: "muted small" }, "The Markov chain was skipped for this file: no sensor showed two or more distinct operating states in the training period.") : null,

    el("h2", {}, "Proof that the alert lines were locked first"),
    el("p", {}, `Each model's thresholds were fitted on the held-out end of the ${trainingPeriod} and ${upload ? "recorded" : "written to disk"} before that model scored any test window. These are the recorded times.`),
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
  ));
} catch (error) {
  initShell("metrics.html", "Results table");
  showError(error);
}
