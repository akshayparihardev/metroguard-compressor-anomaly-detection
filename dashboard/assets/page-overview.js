import { el, eventIds, fmt, footer, initShell, isUpload, loadJSON, showError } from "./app.js";
import { lineMap } from "./linemap.js";

// Plain-language names for the detectors, grouped by family (the three RBM variants share one row).
const FAMILIES = [
  { id: "zscore", match: (m) => m === "zscore_baseline", name: "Simple statistics", how: "How far is each sensor from its usual average?" },
  { id: "rbm", match: (m) => m.startsWith("rbm_k"), name: "RBM (neural network)", how: "Is this combination of sensor states one it rarely saw?" },
  { id: "vae", match: (m) => m === "vae", name: "VAE (neural network)", how: "Can it rebuild these readings from what it learned?" },
  { id: "markov", match: (m) => m === "markov_state_model", name: "Markov chain", how: "Did the motor switch between its states in an unusual way?" },
];

const pct = (f) => fmt.pct(f, f < 0.01 ? 2 : 1);

function familyRows(primary) {
  return FAMILIES.map((family) => {
    const rows = primary.filter((r) => family.match(r.model));
    if (!rows.length) return null;
    const fractions = rows.map((r) => r.alert_window_fraction);
    const lo = Math.min(...fractions);
    const hi = Math.max(...fractions);
    const detected = Math.min(...rows.map((r) => r.events_detected));
    return { ...family, lo, hi, detected, label: lo === hi ? pct(lo) : `${pct(lo).replace("%", "")}–${pct(hi)}` };
  }).filter(Boolean);
}

function problemCards(upload, audit, context, days) {
  const sensors = audit.analog.length + audit.digital.length;
  const n = context.windows.length;
  if (upload) {
    return [
      ["📈", `${fmt.int(audit.rows)} readings`, `${sensors} signals from ${context.source.file_name}, over ${days} days.`],
      ["🏷️", n ? `${n} reported problem${n === 1 ? "" : "s"}` : "No reported problems", n ? "Only time windows were given, not a label for every reading." : "There is nothing to check the alerts against, so only alert rates are shown."],
      ["🔍", "Too much to read by hand", "The question: which moments look unusual compared with normal operation?"],
    ];
  }
  return [
    ["🚇", "A metro train's air compressor", "It supplies air for the brakes and doors. If it leaks, the train is taken out of service."],
    ["📈", `${fmt.int(audit.rows)} sensor readings`, `${sensors} sensors, one reading every ${audit.cadence_seconds} seconds, for ${days} days.`],
    ["🏷️", `Only ${n} known failures`, `${n} air leaks were reported, as time windows. Nobody labelled the other 1.5 million readings.`],
  ];
}

try {
  const [context, metrics, timeline, audit] = await Promise.all([
    loadJSON("context"), loadJSON("metrics"), loadJSON("timeline"), loadJSON("audit"),
  ]);
  const upload = isUpload(context);
  const ids = eventIds(context);
  const days = audit.span.split(" ")[0];
  const trainingName = upload ? "the training period" : "the first month";

  const { wrap } = initShell(
    "index.html",
    upload ? "Your analysis" : "MetroGuard",
    upload
      ? `Which moments in ${context.source.file_name} look unusual, found by learning what normal looks like first.`
      : "Can sensor data tell us when a train's air compressor starts to behave unusually, before anyone has labelled what a failure looks like?",
  );

  const primary = metrics.rows.filter((r) => r.is_primary_score && r.percentile === context.primary_percentile);
  const families = familyRows(primary);
  const zscore = families.find((f) => f.id === "zscore");
  const neural = families.filter((f) => f.id === "rbm" || f.id === "vae");
  const selective = [...families].filter((f) => f.detected > 0).sort((a, b) => a.hi - b.hi)[0];
  const scale = Math.max(...families.map((f) => f.hi), 0.01);

  const section = (n, title, ...body) => el("section", { class: "story reveal" }, el("div", { class: "story-head" }, el("span", { class: "story-n" }, String(n)), el("h2", {}, title)), ...body);

  const findings = [];
  if (ids.length && selective) {
    findings.push(`The ${selective.name} was the most precise: it caught ${selective.detected} of ${ids.length} ${upload ? "reported problems" : "leaks"} while alerting only ${selective.label} of the time.`);
    findings.push(`Catching every ${upload ? "problem" : "leak"} is easier than it sounds: the reported windows, plus ${context.lead_tolerance_hours} hours before each, cover ${fmt.pct(metrics.detection_range_share)} of the test period, so a detector that alerts often will land in them by chance.`);
  }
  if (zscore && neural.length) {
    const neuralLo = Math.min(...neural.map((f) => f.lo));
    const beaten = neural.every((f) => f.detected <= zscore.detected) && neuralLo >= zscore.lo;
    findings.push(beaten
      ? `The neural networks did not beat simple statistics: they raised alerts more often (${neural.map((f) => f.label).join(" and ")} against ${zscore.label}) and caught no more ${upload ? "problems" : "leaks"}.`
      : `Compare the neural networks (${neural.map((f) => f.label).join(" and ")} of the time) with simple statistics (${zscore.label}) before trusting either.`);
  }
  findings.push(upload
    ? `Every detector was set to alert on about ${100 - context.primary_percentile}% of normal data. Rates far above that mean the later data differs from ${trainingName}.`
    : `Every detector was set to alert on about ${100 - context.primary_percentile}% of normal data, yet most alerted far more often after February. The later months simply look different from the first one, so "unusual" does not always mean "broken".`);

  wrap.append(
    section(1, upload ? "The question" : "The problem",
      el("div", { class: "cards-3" }, problemCards(upload, audit, context, days).map(([icon, title, text]) =>
        el("div", { class: "card" }, el("span", { class: "card-icon", "aria-hidden": "true" }, icon), el("b", {}, title), el("p", {}, text)))),
      el("p", { class: "lede" }, upload
        ? "Instead of guessing rules, the models learn what normal looks like and point at moments that do not fit."
        : "With so few known failures you cannot train an ordinary classifier. So MetroGuard does the opposite: it learns what normal looks like and points at the moments that do not fit."),
    ),

    section(2, "How we solved it",
      el("ol", { class: "plan" },
        el("li", {}, el("b", {}, `Learn normal from ${trainingName}`), el("span", {}, `${upload ? "The models study" : "February holds no reported failure, so the models study"} only this period, summarised into ${fmt.windowLength(context.window_seconds)} windows.`)),
        el("li", {}, el("b", {}, "Score every later window"), el("span", {}, "Four detectors each ask a different question about how unusual a window is (below).")),
        el("li", {}, el("b", {}, "Draw an alert line on normal data"), el("span", {}, "The line is set on held-out normal data and locked before any later data is scored, so the test is fair.")),
        el("li", {}, el("b", {}, ids.length ? `Check alerts against the ${ids.length} reported ${upload ? "problems" : "leaks"}` : "Count the alerts"), el("span", {}, ids.length ? `An alert counts if it falls inside a reported window or up to ${context.lead_tolerance_hours} hours before it.` : "Without reported problems, only how often each detector alerts can be measured.")),
      ),
    ),

    section(3, "What we found",
      el("div", { class: "verdict" },
        el("div", { class: "verdict-head" }, el("span", {}, "Detector"), el("span", {}, "Share of time with an alert"), ids.length ? el("span", {}, upload ? "Problems caught" : "Leaks caught") : null),
        families.map((f, i) => el("div", { class: `verdict-row${f === selective && ids.length ? " best" : ""}` },
          el("div", {}, el("b", {}, f.name), el("span", { class: "muted small" }, f.how)),
          el("div", { class: "verdict-bar" }, el("span", { class: "meter" }, el("i", { style: `width:${Math.max(1.5, (f.hi / scale) * 100)}%;--i:${i}` })), el("b", { class: "num" }, f.label)),
          ids.length ? el("div", { class: "verdict-caught num" }, el("b", {}, `${f.detected} of ${ids.length}`)) : null,
        )),
      ),
      el("ul", { class: "findings" }, findings.map((t) => el("li", {}, t))),
      el("h3", {}, `The whole record at a glance, ${fmt.date(timeline.start_minute)} to ${fmt.date(timeline.start_minute + timeline.coverage.length * timeline.step_minutes)}`),
      el("div", { class: "linemap", id: "linemap" }),
      el("div", { class: "legend" },
        el("span", {}, el("i", { class: "sw-line" }), "Data recorded"),
        el("span", {}, el("i", { class: "sw-gap" }), "No data"),
        ids.length ? el("span", {}, el("i", { class: "sw-station" }), upload ? "Reported problem (click to open)" : "Reported leak (click to open)") : null,
        el("span", {}, el("i", { class: "sw-rib" }), "Darker = more alerts that day (simple statistics)"),
      ),
    ),

    section(4, "Want the details?",
      el("div", { class: "more" },
        [
          ["scores.html", "Timeline", "Zoom into the alert scores around each reported window."],
          ["metrics.html", "Results table", "Every number behind the summary above."],
          ["network.html", "Sensor links", "Which sensors move together, from the Bayesian network."],
          ["data.html", "The data", "Gaps, frozen sensors and how the models trained."],
          ["claims.html", "Honest limits", "What this can and cannot prove."],
          upload ? null : ["analyse.html", "Try your own data", "Run the same method on your file, in your browser."],
        ].filter(Boolean).map(([href, title, text]) => el("a", { class: "more-card", href }, el("b", {}, title), el("span", {}, text))),
      ),
    ),
    footer(context),
  );
  lineMap(document.getElementById("linemap"), timeline, { trainingLabel: upload ? "Training period" : "Training month" });
} catch (error) {
  initShell("index.html", "MetroGuard");
  showError(error);
}
