// The signature component: the whole record drawn as a metro line.
// Solid track = hours with scored data, dashed = gaps, red stations = reported failure windows,
// bar = end of the training month, ribbon = share of each day the baseline spends above its threshold.

import { fmt, svg } from "./app.js";

const MIN_GAP_MINUTES = 360;
const LINE_Y = 66;
const MAX_DATED_STATIONS = 6;
const MONTH_MINUTES = (year, month) => Date.UTC(year, month, 1) / 60000;
let hasDrawn = false;
let lastWidth = 0;

/** Buckets with scored data; gaps shorter than six hours are drawn as track. */
function presentBuckets(coverage, stepMinutes) {
  const minGap = Math.max(1, Math.round(MIN_GAP_MINUTES / stepMinutes));
  const present = coverage.map((c) => c > 0);
  let i = 0;
  while (i < present.length) {
    if (present[i]) { i++; continue; }
    let j = i;
    while (j < present.length && !present[j]) j++;
    if (j - i < minGap && i > 0 && j < present.length) for (let k = i; k < j; k++) present[k] = true;
    i = j;
  }
  return present;
}

function runs(flags) {
  const out = [];
  let start = null;
  flags.forEach((f, i) => {
    if (f && start == null) start = i;
    if (!f && start != null) { out.push([start, i]); start = null; }
  });
  if (start != null) out.push([start, flags.length]);
  return out;
}

export function lineMap(container, timeline, { ribbonModel = "zscore_baseline", trainingLabel = "Training month" } = {}) {
  const step = timeline.step_minutes;
  const draw = () => {
    const width = Math.max(container.clientWidth, 320);
    lastWidth = width;
    const left = 18;
    const right = 18;
    const plotW = width - left - right;
    const n = timeline.coverage.length;
    const xMin = timeline.start_minute;
    const xMax = xMin + n * step;
    const x = (minute) => left + ((minute - xMin) / (xMax - xMin)) * plotW;
    const height = 154;
    const narrow = width < 560;
    const days = Math.round((xMax - xMin) / 1440);
    const nWindows = timeline.windows.length;
    const root = svg("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `Timeline of the ${days}-day record with data gaps, ${nWindows ? `${nWindows} reported failure window${nWindows === 1 ? "" : "s"}` : "no reported failure windows"} and the end of the training period` });

    const present = presentBuckets(timeline.coverage, step);
    const gaps = runs(present.map((p) => !p));
    for (const [a, b] of gaps) {
      if (a === 0 || b === n) continue;
      root.append(svg("path", { class: "gap", d: `M${x(xMin + a * step)} ${LINE_Y} H${x(xMin + b * step)}` }));
    }
    const track = svg("path", {
      class: `track${hasDrawn ? "" : " draw"}`,
      d: runs(present).map(([a, b]) => `M${x(xMin + a * step)} ${LINE_Y} H${x(xMin + b * step)}`).join(" "),
    });
    root.append(track);

    const model = timeline.models.find((m) => m.id === ribbonModel);
    const perDay = new Map();
    model.values.forEach((v, i) => {
      if (v == null) return;
      const day = Math.floor((xMin + i * step) / 1440);
      const entry = perDay.get(day) ?? { hours: 0, over: 0 };
      entry.hours += 1;
      if (v > model.threshold_p99) entry.over += 1;
      perDay.set(day, entry);
    });
    const ribbonY = LINE_Y + 62;
    for (const [day, { hours, over }] of perDay) {
      const share = over / hours;
      root.append(
        svg("rect", {
          x: x(day * 1440), y: ribbonY, width: Math.max(1, (plotW / (xMax - xMin)) * 1440 - 0.6), height: 9, rx: 1.5,
          fill: "var(--alert)", opacity: (0.08 + 0.92 * share).toFixed(2),
        }, svg("title", {}, `${fmt.date(day * 1440)}: ${fmt.pct(share, 0)} of scored ${step === 60 ? "hours" : "periods"} above threshold`)),
      );
    }

    const startDate = new Date(xMin * 60000);
    let year = startDate.getUTCFullYear();
    let month = startDate.getUTCMonth();
    for (;;) {
      const minute = MONTH_MINUTES(year, month);
      if (minute > xMax) break;
      if (minute >= xMin) {
        root.append(svg("line", { class: "tick", x1: x(minute), x2: x(minute), y1: ribbonY + 12, y2: ribbonY + 20 }));
        root.append(svg("text", { class: "t-muted", x: x(minute) + 3, y: ribbonY + 32 }, fmt.monthName(minute)));
      }
      month += 1;
      if (month > 11) { month = 0; year += 1; }
    }

    const tx = x(timeline.test_start_minute);
    root.append(
      svg("g", { class: "terminus" },
        svg("line", { x1: tx, x2: tx, y1: LINE_Y - 22, y2: LINE_Y + 22 }),
        svg("text", { class: "t-strong", x: tx - 6, y: LINE_Y - 30, "text-anchor": "end" }, narrow ? "Training" : trainingLabel),
        svg("text", { class: "t-muted", x: tx + 6, y: LINE_Y - 30 }, narrow ? "Test" : "Test period"),
      ),
    );

    const dated = nWindows <= MAX_DATED_STATIONS;
    for (const w of timeline.windows) {
      const cx = x((w.start_minute + w.end_minute) / 2);
      const above = w.event % 2 === 1;
      const labelY = above ? LINE_Y - 30 : LINE_Y + 36;
      const anchor = cx > width - 70 ? "end" : cx < 70 ? "start" : "middle";
      const dates = fmt.shortDate(w.start_minute) === fmt.shortDate(w.end_minute) ? fmt.shortDate(w.start_minute) : `${fmt.shortDate(w.start_minute)}–${fmt.shortDate(w.end_minute)}`;
      const link = svg("a", { href: `scores.html?event=${w.event}`, "aria-label": `Event ${w.event}, ${dates}: open the scores around this window` },
        svg("g", { class: "station", tabindex: "-1" },
          svg("circle", { cx, cy: LINE_Y, r: 9 }),
          svg("text", { class: "t-strong", x: cx, y: labelY, "text-anchor": anchor }, dated ? `Event ${w.event}` : String(w.event)),
          dated ? svg("text", { class: "t-muted", x: cx, y: labelY + 14, "text-anchor": anchor }, dates) : null,
        ),
      );
      root.append(link);
    }
    container.replaceChildren(root);
    if (!hasDrawn) {
      const length = track.getTotalLength();
      track.style.setProperty("--len", String(Math.ceil(length)));
      hasDrawn = true;
    }
  };
  draw();
  new ResizeObserver(() => {
    if (Math.max(container.clientWidth, 320) !== lastWidth) draw();
  }).observe(container);
}
