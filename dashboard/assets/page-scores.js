import { createStack } from "./charts.js";
import { el, fmt, footer, initShell, loadJSON, plainTable, showError } from "./app.js";

const DEFAULT_MODELS = ["zscore_baseline", "rbm_k5", "vae", "markov_state_model"];
const PERCENTILES = [["99", "99th"], ["99.5", "99.5th"]];
const EVENT_PAD_BEFORE_MINUTES = 180;
const EVENT_PAD_AFTER_MINUTES = 720;

try {
  const [context, timeline, metrics, audit] = await Promise.all([
    loadJSON("context"), loadJSON("timeline"), loadJSON("metrics"), loadJSON("audit"),
  ]);
  const { wrap } = initShell(
    "scores.html",
    "Scores over time",
    "Each panel is one model's anomaly score. Orange marks scores above the threshold fixed on the training month, and red bands are the reported failure windows.",
  );

  const params = new URLSearchParams(location.search);
  const startEvent = Number(params.get("event"));
  const state = {
    models: new Set(DEFAULT_MODELS),
    percentile: "99",
    range: startEvent >= 1 && startEvent <= 4 ? startEvent : "full",
    zoom: null,
  };
  const eventCache = new Map();
  const loadEvent = (n) => {
    if (!eventCache.has(n)) eventCache.set(n, loadJSON(`event_${n}`));
    return eventCache.get(n);
  };

  const controls = el("div", { class: "controls" });
  const chartHost = el("div");
  const facts = el("div");
  const hint = el("p", { class: "muted small" });
  const full = timeline.models;
  const stack = createStack(chartHost, {
    onZoom: (a, b) => {
      state.zoom = a == null ? null : [a, b];
      draw();
    },
  });

  const rangeOptions = [["full", "Full record"], ...[1, 2, 3, 4].map((n) => [n, `Event ${n}`])];

  function paintControls() {
    const range = el("div", {}, el("span", { class: "control-label" }, "Range"),
      el("div", { class: "seg", role: "group", "aria-label": "Range" },
        rangeOptions.map(([value, label]) => el("button", { type: "button", "aria-pressed": String(state.range === value), onclick: () => { state.range = value; state.zoom = null; paintControls(); draw(); } }, label))));
    const pct = el("div", {}, el("span", { class: "control-label" }, "Threshold"),
      el("div", { class: "seg", role: "group", "aria-label": "Threshold percentile" },
        PERCENTILES.map(([value, label]) => el("button", { type: "button", "aria-pressed": String(state.percentile === value), onclick: () => { state.percentile = value; paintControls(); draw(); } }, label))));
    const chips = el("div", { class: "chips", role: "group", "aria-label": "Models shown" },
      full.map((m) => el("button", { class: "chip", type: "button", "aria-pressed": String(state.models.has(m.id)), onclick: () => { state.models.has(m.id) ? state.models.size > 1 && state.models.delete(m.id) : state.models.add(m.id); paintControls(); draw(); } }, m.label)));
    controls.replaceChildren(range, pct, chips);
  }

  const thresholdOf = (series, model) => (state.percentile === "99" ? series.threshold_p99 : series.threshold_p99_5) ?? model.threshold_p99;

  async function draw() {
    const shown = full.filter((m) => state.models.has(m.id));
    const base = {
      windows: timeline.windows,
      thresholdLabel: `p${state.percentile}`,
    };
    let spec;
    if (state.range === "full") {
      const total = timeline.coverage.length;
      const xs = Array.from({ length: total }, (_, i) => timeline.start_minute + i * timeline.step_minutes);
      spec = {
        ...base,
        mode: "hourly",
        xMin: state.zoom?.[0] ?? xs[0],
        xMax: state.zoom?.[1] ?? xs[total - 1],
        markers: [{ minute: timeline.test_start_minute, label: "Test period begins", colour: "ink", dash: [5, 4] }],
        panels: shown.map((m, i) => ({
          label: m.label, sub: `${m.score_label}, hourly maximum`, scale: m.scale, showMarkerLabels: i === 0,
          threshold: state.percentile === "99" ? m.threshold_p99 : m.threshold_p99_5,
          series: { xs, ys: m.values },
        })),
      };
      hint.textContent = "Hourly maximum of the window scores (display only; metrics use every window). Drag across a chart to zoom, double-click to reset.";
    } else {
      const event = await loadEvent(state.range);
      spec = {
        ...base,
        mode: "window",
        xMin: state.zoom?.[0] ?? event.lead_start_minute - EVENT_PAD_BEFORE_MINUTES,
        xMax: state.zoom?.[1] ?? event.end_minute + EVENT_PAD_AFTER_MINUTES,
        markers: [{ minute: event.lead_start_minute, label: `${context.lead_tolerance_hours} h lead range begins`, colour: "ink", dash: [3, 4] }],
        panels: shown.map((m, i) => {
          const series = event.series.find((s) => s.id === m.id);
          return {
            label: m.label, sub: `${m.score_label}, every ${context.window_seconds} s window`, scale: m.scale, showMarkerLabels: i === 0,
            threshold: thresholdOf(series, m), series: { xs: series.minutes, ys: series.values },
          };
        }),
      };
      hint.textContent = "Every 60 s window around the reported window. Orange dots are alert windows. Blank stretches are gaps or frozen data that were not scored.";
    }
    stack.render(spec);
    paintFacts();
  }

  function paintFacts() {
    if (state.range === "full") {
      facts.replaceChildren(
        el("div", { class: "note" },
          el("p", {}, "The training month is on the left of the dashed line. Scores jump after it for most models because the later months differ from the training month, not because every later window is abnormal. Open an event to see its scores window by window.")),
      );
      return;
    }
    const w = timeline.windows.find((x) => x.event === state.range);
    const cover = audit.event_coverage.find((c) => c.event === state.range);
    const rows = metrics.rows.filter((r) => r.is_primary_score && r.percentile === Number(state.percentile));
    facts.replaceChildren(
      el("h2", {}, `Event ${w.event}: ${fmt.dateTime(w.start_minute)} to ${fmt.dateTime(w.end_minute)}`),
      el("p", {}, `Reported issue: ${w.issue.toLowerCase()}. ${w.maintenance_note !== "-" ? `Maintenance note as published: “${w.maintenance_note}”.` : "No maintenance note was published for this event."} The file holds ${fmt.int(cover.readings)} readings inside the window (${cover.coverage_percent}% of one every ${audit.cadence_seconds} s); the longest silent stretch inside it is ${fmt.seconds(cover.largest_gap_seconds)}.`),
      plainTable({
        caption: `Detection at the ${state.percentile}th percentile`,
        note: "Delay is clipped to the start of the lead range (see Metrics).",
        columns: [
          { label: "Model", render: (r) => el("b", {}, r.model_label) },
          { label: "Detected", render: (r) => (r.detected_event_ids.split(";").includes(String(w.event)) ? el("span", { class: "tag ok" }, "yes") : el("span", { class: "tag warn" }, "no")) },
          { label: "Delay", align: "right", render: (r) => fmt.hours(r[`delay_hours_event${w.event}`]) },
          { label: "Windows flagged overall", align: "right", render: (r) => fmt.pct(r.alert_window_fraction, r.alert_window_fraction < 0.01 ? 2 : 1) },
        ],
        rows,
      }),
    );
  }

  paintControls();
  wrap.append(
    controls,
    el("div", { class: "legend" },
      el("span", {}, el("i", { class: "sw-line" }), "Score below threshold"),
      el("span", {}, el("i", { class: "sw-alert" }), "Score above threshold"),
      el("span", {}, el("i", { class: "sw-dash" }), "Frozen threshold"),
      el("span", {}, el("i", { class: "sw-fail" }), "Reported failure window"),
    ),
    chartHost,
    hint,
    facts,
    footer(context),
  );
  await draw();
} catch (error) {
  initShell("scores.html", "Scores over time");
  showError(error);
}
