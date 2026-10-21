import { createStack } from "./charts.js";
import { compact, el, eventIds, fmt, footer, initShell, isUpload, loadJSON, plainTable, segmented, showError } from "./app.js";

// One detector at a time, in plain words: the score over time, the alert line, and the reported windows.

const DETECTORS = [
  ["zscore_baseline", "Simple statistics"],
  ["rbm_k5", "RBM"],
  ["vae", "VAE"],
  ["markov_state_model", "Markov chain"],
];
const EVENT_PAD_BEFORE_MINUTES = 180;
const EVENT_PAD_AFTER_MINUTES = 720;

const plainName = (id, fallback) => DETECTORS.find(([d]) => d === id)?.[1] ?? fallback;

function delayText(hours) {
  if (hours == null) return "no alert";
  const h = Math.abs(hours);
  const amount = h < 1 ? `${Math.round(h * 60)} min` : `${h.toFixed(1)} h`;
  return hours <= 0 ? `${amount} before it started` : `${amount} after it started`;
}

try {
  const [context, timeline, metrics, audit] = await Promise.all([
    loadJSON("context"), loadJSON("timeline"), loadJSON("metrics"), loadJSON("audit"),
  ]);
  const upload = isUpload(context);
  const ids = eventIds(context);
  const leak = upload ? "Problem" : "Leak";
  const { wrap } = initShell(
    "scores.html",
    "Timeline",
    "Pick a detector. The line is its score for each moment: the higher, the more unusual. When the line goes above the dashed alert line, that moment is an alert.",
  );

  const available = timeline.models.map((m) => m.id);
  const detectors = DETECTORS.filter(([id]) => available.includes(id));
  const choices = detectors.length ? detectors : timeline.models.map((m) => [m.id, m.label]);
  const params = new URLSearchParams(location.search);
  const startEvent = Number(params.get("event"));
  const state = {
    model: choices[0][0],
    percentile: "99",
    range: ids.includes(startEvent) ? startEvent : "full",
    zoom: null,
  };
  const eventCache = new Map();
  const loadEvent = (n) => {
    if (!eventCache.has(n)) eventCache.set(n, loadJSON(`event_${n}`));
    return eventCache.get(n);
  };

  const chartHost = el("div");
  const facts = el("div");
  const hint = el("p", { class: "muted small" });
  const stack = createStack(chartHost, {
    onZoom: (a, b) => {
      state.zoom = a == null ? null : [a, b];
      draw(false);
    },
  });

  const controls = el("div", { class: "controls" }, ...compact(
    segmented({ label: "Detector", options: choices, value: state.model, onChange: (v) => { state.model = v; draw(true); } }),
    ids.length ? segmented({ label: "Show", options: [["full", "Whole record"], ...ids.map((n) => [n, `${leak} ${n}`])], value: state.range, onChange: (v) => { state.range = v; state.zoom = null; draw(true); } }) : null,
    segmented({ label: "Alert line", options: [["99", "Normal"], ["99.5", "Stricter"]], value: state.percentile, onChange: (v) => { state.percentile = v; draw(false); } }),
  ));

  const howTo = el("div", { class: "how-to" },
    el("b", {}, "How to read the chart"),
    el("ul", {},
      el("li", {}, el("i", { class: "sw-line" }), "Teal line: the detector's score. Up = more unusual."),
      el("li", {}, el("i", { class: "sw-dash" }), "Dashed line: the alert line, fixed before the test months."),
      el("li", {}, el("i", { class: "sw-alert" }), "Orange: the score is above the line, so it is an alert."),
      ids.length ? el("li", {}, el("i", { class: "sw-fail" }), `Red band: a reported ${leak.toLowerCase()}.`) : null,
    ),
  );

  async function draw(animate = false) {
    const model = timeline.models.find((m) => m.id === state.model);
    const name = plainName(model.id, model.label);
    const threshold = (series) => (state.percentile === "99" ? series.threshold_p99 : series.threshold_p99_5) ?? model.threshold_p99;
    const base = { windows: timeline.windows, windowLabel: leak, thresholdLabel: "alert line" };
    let spec;
    if (state.range === "full") {
      const total = timeline.coverage.length;
      const xs = Array.from({ length: total }, (_, i) => timeline.start_minute + i * timeline.step_minutes);
      const per = timeline.step_minutes === 60 ? "hour" : fmt.windowLength(timeline.step_minutes * 60);
      spec = {
        ...base,
        mode: "hourly",
        unit: timeline.step_minutes === 60 ? "hours" : "periods",
        xMin: state.zoom?.[0] ?? xs[0],
        xMax: state.zoom?.[1] ?? xs[total - 1],
        markers: [{ minute: timeline.test_start_minute, label: `Test starts (${upload ? "learning period" : "February"} is on the left)`, colour: "ink", dash: [5, 4] }],
        panels: [{ label: name, sub: `highest score in each ${per}`, scale: model.scale, showMarkerLabels: true, threshold: threshold(model), series: { xs, ys: model.values } }],
      };
      hint.textContent = "Tip: drag across the chart to zoom in, double-click to zoom out. Hover to see the exact value.";
    } else {
      const event = await loadEvent(state.range);
      const series = event.series.find((s) => s.id === model.id);
      spec = {
        ...base,
        mode: "window",
        unit: "windows",
        xMin: state.zoom?.[0] ?? event.lead_start_minute - EVENT_PAD_BEFORE_MINUTES,
        xMax: state.zoom?.[1] ?? event.end_minute + EVENT_PAD_AFTER_MINUTES,
        markers: [{ minute: event.lead_start_minute, label: `${context.lead_tolerance_hours} h before the ${leak.toLowerCase()}: alerts from here count`, colour: "ink", dash: [3, 4] }],
        panels: [{ label: name, sub: `score of every ${fmt.windowLength(context.window_seconds)} window`, scale: model.scale, showMarkerLabels: true, threshold: threshold(series), series: { xs: series.minutes, ys: series.values } }],
      };
      hint.textContent = "Gaps in the line are times with no usable data. Drag to zoom, double-click to zoom out.";
    }
    stack.render(spec, { animate });
    paintFacts();
  }

  function paintFacts() {
    const rows = metrics.rows.filter((r) => r.is_primary_score && r.percentile === Number(state.percentile));
    if (state.range === "full") {
      const row = rows.find((r) => r.model === state.model);
      facts.replaceChildren(
        el("div", { class: "note" },
          el("p", {}, el("strong", {}, "What this shows. "), row ? `In the test months, ${plainName(row.model, row.model_label)} was above the alert line ${fmt.pct(row.alert_window_fraction, row.alert_window_fraction < 0.01 ? 2 : 1)} of the time, although the line was set so that normal data crosses it only ${state.percentile === "99" ? "1%" : "0.5%"} of the time.` : "",
            upload
              ? " A score that stays high after the learning period means the later data differs from it, which is not the same as every later moment being abnormal."
              : " Most detectors stay high after February because the later months simply look different, not because every later minute is broken."),
          ids.length ? el("p", {}, `Pick a ${leak.toLowerCase()} above to zoom into it.`) : null,
        ),
      );
      return;
    }
    const w = timeline.windows.find((x) => x.event === state.range);
    const cover = audit.event_coverage.find((c) => c.event === state.range);
    facts.replaceChildren(
      el("h2", {}, `${leak} ${w.event}: ${fmt.dateTime(w.start_minute)} to ${fmt.dateTime(w.end_minute)}`),
      el("p", {}, `${upload ? "Reported" : "Reported issue"}: ${w.issue.toLowerCase()}. ${w.maintenance_note !== "-" ? `Maintenance note: “${w.maintenance_note}”. ` : ""}${cover.coverage_percent >= 99 ? "Data was recorded through the whole window." : `Data covers ${cover.coverage_percent}% of this window.`}`),
      plainTable({
        caption: `Did each detector catch ${leak.toLowerCase()} ${w.event}?`,
        note: `Caught = an alert during the ${leak.toLowerCase()} or up to ${context.lead_tolerance_hours} h before it. The first alert time is counted from the reported start.`,
        columns: [
          { label: "Detector", render: (r) => el("b", {}, plainName(r.model, r.model_label)) },
          { label: "Caught?", render: (r) => (r.detected_event_ids.split(";").includes(String(w.event)) ? el("span", { class: "tag ok" }, "yes") : el("span", { class: "tag warn" }, "no")) },
          { label: "First alert", render: (r) => delayText(r[`delay_hours_event${w.event}`]) },
          { label: "Alerts overall (test months)", align: "right", render: (r) => fmt.pct(r.alert_window_fraction, r.alert_window_fraction < 0.01 ? 2 : 1) },
        ],
        rows,
      }),
      el("p", { class: "muted small" }, `An alert ${context.lead_tolerance_hours} h early is not a prediction: a detector that alerts often will usually have one running already.`),
    );
  }

  wrap.append(controls, howTo, chartHost, hint, facts, footer(context));
  await draw(true);
} catch (error) {
  initShell("scores.html", "Timeline");
  showError(error);
}
