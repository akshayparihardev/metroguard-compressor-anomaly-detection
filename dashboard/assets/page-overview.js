import { compact, el, eventIds, fmt, footer, initShell, isUpload, loadJSON, showError } from "./app.js";
import { lineMap } from "./linemap.js";

// The overview tells the whole method as one journey, from the raw file to the result, in plain words.
// Every number comes from dashboard/data; the small pictures are explanations, not data.

const FAMILIES = [
  { id: "zscore", match: (m) => m === "zscore_baseline", name: "Simple statistics", concept: "Z-score (statistics)",
    idea: "For each sensor, how many standard deviations is this minute from its normal average? The biggest distance is the score." },
  { id: "rbm", match: (m) => m.startsWith("rbm_k"), name: "RBM", concept: "Restricted Boltzmann Machine: Gibbs sampling, contrastive divergence, free energy",
    idea: "A neural network that learns which combinations of sensor states are common. Its 'free energy' is low for familiar minutes and high for strange ones." },
  { id: "vae", match: (m) => m === "vae", name: "VAE", concept: "Variational autoencoder",
    idea: "A neural network that squeezes a minute into 6 numbers and rebuilds it. It rebuilds normal minutes well; a large rebuild error means unusual." },
  { id: "markov", match: (m) => m === "markov_state_model", name: "Markov chain", concept: "Markov chain (state transitions)",
    idea: "The motor is off, on at low load or on at high load. It learns how often each switch happens; a rare switch gives a high 'surprise' score." },
];
const SENSOR_MEANINGS = {
  TP2: "air pressure inside the compressor", TP3: "air pressure at the output panel", H1: "pressure drop at the filter valve",
  DV_pressure: "pressure drop at the air dryer", Reservoirs: "pressure in the air tanks", Oil_temperature: "oil temperature",
  Motor_current: "electric current of the motor", COMP: "air intake valve", DV_eletric: "outlet valve", Towers: "which dryer tower is working",
  MPG: "start signal when pressure falls low", LPS: "low-pressure alarm", Pressure_switch: "pressure switch", Oil_level: "oil level alarm",
  Caudal_impulses: "air-flow meter pulses",
};

const windowWords = (s) => (s < 3600 ? `${s / 60}-minute` : s < 86400 ? `${s / 3600}-hour` : `${s / 86400}-day`);
const pct = (f) => fmt.pct(f, f < 0.01 ? 2 : 1);

function familyRows(primary) {
  return FAMILIES.map((family) => {
    const rows = primary.filter((r) => family.match(r.model));
    if (!rows.length) return null;
    const fractions = rows.map((r) => r.alert_window_fraction);
    const lo = Math.min(...fractions);
    const hi = Math.max(...fractions);
    return { ...family, lo, hi, detected: Math.min(...rows.map((r) => r.events_detected)), label: lo === hi ? pct(lo) : `${pct(lo).replace("%", "")}–${pct(hi)}` };
  }).filter(Boolean);
}

/** Usable windows before calib, in calib and in test, counted from the timeline coverage. */
function splitCounts(timeline) {
  const index = (minute) => Math.floor((minute - timeline.start_minute) / timeline.step_minutes);
  const sum = (a, b) => timeline.coverage.slice(a, b).reduce((s, v) => s + v, 0);
  const calib = index(timeline.calib_start_minute);
  const test = index(timeline.test_start_minute);
  return { learn: sum(0, calib), line: sum(calib, test), test: sum(test), total: sum(0) };
}

// ---- small explanatory pictures (inline SVG, theme colours) ------------------------------------------------
function windowPicture(seconds, cadence) {
  const dots = Math.max(2, Math.min(8, Math.round(seconds / cadence)));
  const gap = 34;
  const box = 40 + dots * gap + 30;
  const w = box + 160;
  return el("figure", { class: "pic", html: `<svg viewBox="0 0 ${w} 96" role="img" aria-label="${dots} readings become one window">
    <defs><marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--ink-2)"/></marker></defs>
    <rect x="10" y="20" width="${dots * gap}" height="52" rx="12" fill="none" stroke="var(--line)" stroke-dasharray="4 4"/>
    ${Array.from({ length: dots }, (_, i) => `<circle cx="${27 + i * gap}" cy="46" r="8" fill="var(--brand)" opacity="${0.55 + (i % 3) * 0.15}"/>`).join("")}
    <text x="10" y="92" class="pic-t">${dots} readings, one every ${cadence} s</text>
    <path d="M${box - 24} 46 h16" stroke="var(--ink-2)" stroke-width="2" marker-end="url(#arr)"/>
    <rect x="${box}" y="18" width="150" height="56" rx="12" fill="var(--brand-wash)" stroke="var(--brand)"/>
    <text x="${box + 75}" y="42" text-anchor="middle" class="pic-b">1 window</text>
    <text x="${box + 75}" y="60" text-anchor="middle" class="pic-t">averages + spread</text>
  </svg>` });
}

function splitPicture(counts, upload) {
  const parts = [
    ["learn", counts.learn, upload ? "Learn normal" : "Learn normal (Feb)"],
    ["line", counts.line, "Set the alert line"],
    ["test", counts.test, upload ? "Test (everything later)" : "Test (Mar to Aug)"],
  ];
  const raw = parts.map(([, n]) => Math.max(0.26, n / counts.total));
  const scale = 100 / raw.reduce((a, b) => a + b, 0);
  return el("div", { class: "split-bar", role: "img", "aria-label": parts.map(([, n, label]) => `${label}: ${fmt.int(n)} windows`).join("; ") },
    parts.map(([key, n, label], i) => el("div", { class: `split-part ${key}`, style: `flex-basis:${raw[i] * scale}%` },
      el("b", {}, label), el("span", { class: "num" }, `${fmt.int(n)} windows`))));
}

function thresholdPicture() {
  const ys = [62, 58, 66, 55, 60, 64, 57, 61, 30, 22, 59, 63, 56, 60, 65, 58, 62, 57];
  const path = ys.map((y, i) => `${i ? "L" : "M"}${8 + i * 16} ${y}`).join(" ");
  return el("figure", { class: "pic", html: `<svg viewBox="0 0 300 100" role="img" aria-label="A score line crosses the alert line twice, which raises an alert">
    <line x1="0" x2="300" y1="40" y2="40" stroke="var(--ink-2)" stroke-dasharray="5 4"/>
    <text x="298" y="34" text-anchor="end" class="pic-t">alert line</text>
    <path d="${path}" fill="none" stroke="var(--brand)" stroke-width="2.4" stroke-linejoin="round"/>
    <circle cx="136" cy="30" r="5" fill="var(--alert)"/><circle cx="152" cy="22" r="5" fill="var(--alert)"/>
    <text x="144" y="12" text-anchor="middle" class="pic-b" style="fill:var(--alert)">alert</text>
    <text x="8" y="94" class="pic-t">score of each window over time (higher = more unusual)</text>
  </svg>` });
}

function step(n, id, title, { what, why, concept, result, extra }) {
  return el("section", { class: "journey-step reveal", id },
    el("div", { class: "journey-marker", "aria-hidden": "true" }, el("span", {}, String(n))),
    el("div", { class: "journey-body" },
      el("h2", {}, title),
      el("p", { class: "journey-what" }, what),
      extra ?? null,
      el("dl", { class: "journey-facts" },
        result ? el("div", { class: "result" }, el("dt", {}, "Result"), el("dd", {}, result)) : null,
        why ? el("div", {}, el("dt", {}, "Why"), el("dd", {}, why)) : null,
        concept ? el("div", {}, el("dt", {}, "Concept"), el("dd", {}, concept)) : null,
      ),
    ),
  );
}

try {
  const [context, metrics, timeline, audit, network] = await Promise.all([
    loadJSON("context"), loadJSON("metrics"), loadJSON("timeline"), loadJSON("audit"), loadJSON("network"),
  ]);
  const upload = isUpload(context);
  const ids = eventIds(context);
  const days = audit.span.split(" ")[0];
  const counts = splitCounts(timeline);
  const sensors = [...audit.analog, ...audit.digital];
  const primary = metrics.rows.filter((r) => r.is_primary_score && r.percentile === context.primary_percentile);
  const families = familyRows(primary);
  const zscore = families.find((f) => f.id === "zscore");
  const neural = families.filter((f) => f.id === "rbm" || f.id === "vae");
  const selective = [...families].filter((f) => f.detected > 0).sort((a, b) => a.hi - b.hi)[0];
  const scale = Math.max(...families.map((f) => f.hi), 0.01);
  const leaks = upload ? "reported problems" : "leaks";
  const leak = upload ? "reported problem" : "leak";
  const windowName = windowWords(context.window_seconds);
  const unusual = [...new Set((network.alert_context?.events ?? []).flatMap((e) => e.table.filter((r) => r.unusual === "yes").map((r) => r.variable.replace(/_(mean|active)$/, ""))))];

  const { wrap } = initShell(
    "index.html",
    upload ? "Your analysis" : "MetroGuard",
    upload
      ? `How ${context.source.file_name} was analysed, step by step, and what came out.`
      : "We taught computers what a healthy metro-train air compressor looks like, so they can point at the moments when it starts behaving differently. Here is the whole journey, step by step.",
  );

  const stepsList = [
    ["data", "Get the data"], ["check", "Check it"], ["windows", "Make windows"], ["split", "Split by time"],
    ["score", "Score windows"], ["line", "Alert line"], ["compare", `Compare with ${leaks}`], ["explain", "Explain alerts"], ["result", "Result"],
  ];

  wrap.append(...compact(
    el("nav", { class: "flow reveal", "aria-label": "The steps on this page" },
      stepsList.map(([id, label], i) => el("a", { href: `#${id}` }, el("span", { class: "flow-n" }, i === stepsList.length - 1 ? "✓" : String(i + 1)), label))),

    upload ? null : el("div", { class: "problem reveal" },
      el("p", {}, el("b", {}, "The problem. "), `A metro train's air compressor supplies air for its brakes and doors. When it leaks, the train has to be taken out of service. Sensors record the compressor every ${audit.cadence_seconds} seconds, but in ${days} days only ${ids.length} air leaks were ever reported, so there is almost nothing labelled to learn "failure" from.`),
      el("p", {}, el("b", {}, "The idea. "), "Instead of learning what failure looks like, learn what normal looks like, and flag the minutes that do not fit."),
    ),

    step(1, "data", "Get the data", {
      what: upload
        ? `${context.source.file_name}: ${fmt.int(audit.rows)} rows over ${days} days, with ${sensors.length} signals.`
        : `MetroPT-3, a public dataset from the UCI Machine Learning Repository: a real compressor on a Porto metro train, ${fmt.int(audit.rows)} readings from 1 Feb to 1 Sep 2020. Each reading is one row with ${sensors.length} sensor values.`,
      extra: el("details", { class: "sensors" }, el("summary", {}, `What the ${sensors.length} sensors measure`),
        el("ul", {}, sensors.map((s) => el("li", {}, el("code", {}, s), ` ${(!upload && SENSOR_MEANINGS[s]) || (audit.digital.includes(s) ? "on/off signal" : "measurement")}`)))),
      why: `${audit.analog.length} of them are measurements (pressure, temperature, current); ${audit.digital.length} are on/off switches.`,
    }),

    step(2, "check", "Check the data", {
      what: "Before any model, we looked for problems in the file itself: missing time and sensors that got stuck.",
      result: `${fmt.num(audit.days_without_readings, 1)} days have no readings at all. In ${audit.frozen_runs.length} stretches every sensor repeated exactly the same value for hours (${fmt.int(audit.frozen_readings)} readings, ${fmt.num(audit.frozen_percent, 1)}%). That looks like broken logging, so those stretches are left out.`,
      concept: "Data audit",
    }),

    step(3, "windows", `Make ${windowName} windows`, {
      what: `One reading every ${audit.cadence_seconds} seconds is noisy. So we group the readings of each ${windowName} stretch into one row, called a window: the average of each measurement, how much it moved, and the share of time each switch was on.`,
      extra: windowPicture(context.window_seconds, audit.cadence_seconds),
      result: `${fmt.int(audit.rows)} readings become ${fmt.int(counts.total)} usable windows. From here on, everything works on windows.`,
      concept: "Feature engineering (aggregation)",
    }),

    step(4, "split", "Split by time, like an exam", {
      what: upload
        ? "The models study the first part only. Its last part is kept aside to set the alert line. Everything after it is the test, which the models never see while learning."
        : "The models study February only (no leak was reported then). The last part of February is kept aside to set the alert line. March to August is the test, which the models never see while learning.",
      extra: splitPicture(counts, upload),
      why: "If a model had seen the test data, a good result would mean nothing. Splitting by time, never shuffling, keeps the test honest.",
      concept: "Chronological train / calibration / test split",
    }),

    step(5, "score", "Give every window an 'unusual' score", {
      what: "Four different detectors look at each window and answer one question: how unusual is it compared with normal? Each judges in its own way:",
      extra: el("div", { class: "detectors" }, families.map((f) => el("div", { class: "detector" }, el("b", {}, f.name), el("p", {}, f.idea), el("span", { class: "tag" }, f.concept)))),
      result: "Every window gets four scores. Higher always means more unusual.",
    }),

    step(6, "line", "Draw the alert line", {
      what: `On the windows kept aside, we find the score that only ${100 - context.primary_percentile}% of normal windows go above, and draw the alert line there. A window above the line is an alert. Alerts less than ${context.merge_gap_minutes} minutes apart are joined into one alert episode.`,
      extra: thresholdPicture(),
      why: "The line is locked and saved before the test months are scored, so nobody can move it after seeing the answers.",
      concept: `${context.primary_percentile}th percentile threshold (calibration)`,
    }),

    step(7, "compare", ids.length ? `Compare with the ${ids.length} ${leaks}` : "Count the alerts", {
      what: ids.length
        ? `A ${leak} counts as caught if an alert episode happens during it or up to ${context.lead_tolerance_hours} hours before it. Alerts anywhere else are called "unconfirmed": nobody checked the machine then, so they are not known to be wrong.`
        : "No reported problems were given, so we can only count how often each detector raised an alert.",
      concept: "Event-level evaluation",
    }),

    step(8, "explain", "Explain the alerts", {
      what: "A fifth model does not raise alerts. It learns which sensor states usually go together. For an alert, it shows which sensors were in an unlikely state given their neighbours.",
      result: unusual.length ? `For the ${upload ? "alerts it examined" : "alerts during leaks 1 and 2"}, the unusual signals were ${unusual.join(" and ")}.` : null,
      why: "It shows which sensors move together, not what caused what.",
      concept: "Bayesian network (structure learning, Markov blanket)",
    }),

    el("section", { class: "journey-step reveal final", id: "result" },
      el("div", { class: "journey-marker", "aria-hidden": "true" }, el("span", {}, "✓")),
      el("div", { class: "journey-body" },
        el("h2", {}, "Result"),
        el("p", { class: "journey-what" }, `Each detector was set to alert on about ${100 - context.primary_percentile}% of normal minutes. Here is what it actually did on the ${fmt.num(metrics.test_days, 0)} test days:`),
        el("div", { class: "verdict" },
          el("div", { class: "verdict-head" }, el("span", {}, "Detector"), el("span", {}, "Share of test time with an alert"), ids.length ? el("span", {}, upload ? "Problems caught" : "Leaks caught") : null),
          families.map((f, i) => el("div", { class: `verdict-row${f === selective && ids.length ? " best" : ""}` },
            el("div", {}, el("b", {}, f.name)),
            el("div", { class: "verdict-bar" }, el("span", { class: "meter" }, el("i", { style: `width:${Math.max(1.5, (f.hi / scale) * 100)}%;--i:${i}` })), el("b", { class: "num" }, f.label)),
            ids.length ? el("div", { class: "verdict-caught num" }, el("b", {}, `${f.detected} of ${ids.length}`)) : null,
          )),
        ),
        el("ul", { class: "findings" }, [
          ids.length && selective ? `The ${selective.name} was the most precise: it caught ${selective.detected} of ${ids.length} ${leaks} while alerting only ${selective.label} of the time.` : null,
          zscore && neural.length && neural.every((f) => f.detected <= zscore.detected && f.lo >= zscore.lo)
            ? `The neural networks (RBM and VAE) did not beat simple statistics: they alerted about twice as often and caught no more ${leaks}.` : null,
          ids.length ? `Catching every ${leak} is easier than it sounds: the ${leaks} plus ${context.lead_tolerance_hours} hours before each cover ${fmt.pct(metrics.detection_range_share)} of the test time, so a detector that alerts often will hit them by luck.` : null,
          upload
            ? `Alert rates far above ${100 - context.primary_percentile}% mean the later data looks different from the learning period, which is not the same as being broken.`
            : `Most detectors alerted far more than 1% of the time because March to August simply looks different from February. "Unusual" does not always mean "broken".`,
        ].filter(Boolean).map((t) => el("li", {}, t))),
        el("h3", {}, "The whole record on one line"),
        el("p", { class: "muted small" }, `Left to right is time. The thick line shows when data exists${ids.length ? `; red circles are the ${leaks} (click one to see its scores)` : ""}; the orange strip gets darker on days with more alerts from simple statistics.`),
        el("div", { class: "linemap", id: "linemap" }),
      ),
    ),

    el("section", { class: "reveal" },
      el("h2", {}, "Look closer"),
      el("div", { class: "more" },
        [
          ["scores.html", "Timeline", "One detector's score and alert line over time; zoom into each leak."],
          ["metrics.html", "Results table", "The full numbers, including a stricter alert line."],
          ["network.html", "Sensor links", "The Bayesian network: which sensors move together."],
          ["data.html", "The data", "Gaps, stuck sensors, and how the models learned."],
          ["claims.html", "Honest limits", "What this can and cannot prove."],
          upload ? null : ["analyse.html", "Try your own data", "Run the same steps on your own file, in your browser."],
        ].filter(Boolean).map(([href, title, text]) => el("a", { class: "more-card", href }, el("b", {}, title), el("span", {}, text))),
      ),
    ),
    footer(context),
  ));
  lineMap(document.getElementById("linemap"), timeline, { trainingLabel: upload ? "Learning period" : "Learning month" });
} catch (error) {
  initShell("index.html", "MetroGuard");
  showError(error);
}
