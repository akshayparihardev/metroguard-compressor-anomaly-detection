// Canvas charts: a stack of time-series panels sharing one x axis, and a plain multi-line chart.
// Alerts are marked by shape (diamonds) and by a text summary, never by colour alone.

import { cssVar, el, fmt } from "./app.js";

const DPR = () => Math.min(window.devicePixelRatio || 1, 2);
const WINDOW_GAP_MINUTES = 5;
const HOURLY_TOLERANCE_MINUTES = 75;
const MONTH_STARTS_THRESHOLD_DAYS = 100;
const REVEAL_MS = 650;
const DIAMOND = 3.6;
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const easeOutExpo = (t) => (t >= 1 ? 1 : 1 - 2 ** (-10 * t));

function niceStep(range, target) {
  const raw = range / target;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / magnitude;
  return (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * magnitude;
}

function linearTicks(lo, hi, target = 4) {
  const step = niceStep(hi - lo, target);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(Number(v.toPrecision(10)));
  return ticks;
}

function logTicks(lo, hi) {
  const ticks = [];
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** p;
      if (v >= lo && v <= hi) ticks.push(v);
    }
  }
  return ticks.length > 6 ? ticks.filter((v) => Math.log10(v) % 1 === 0) : ticks;
}

function formatValue(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString("en-GB", { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(2);
}

function timeTicks(xMin, xMax) {
  const spanDays = (xMax - xMin) / 1440;
  if (spanDays > MONTH_STARTS_THRESHOLD_DAYS) {
    const ticks = [];
    const d = new Date(xMin * 60000);
    let year = d.getUTCFullYear();
    let month = d.getUTCMonth() + 1;
    for (;;) {
      const minute = Date.UTC(year, month, 1) / 60000;
      if (minute > xMax) break;
      ticks.push({ minute, label: fmt.monthName(minute) });
      month += 1;
      if (month > 11) { month = 0; year += 1; }
    }
    return ticks;
  }
  const steps = [180, 360, 720, 1440, 2880, 10080];
  const step = steps.find((s) => (xMax - xMin) / s <= 9) ?? steps[steps.length - 1];
  const ticks = [];
  for (let m = Math.ceil(xMin / step) * step; m <= xMax; m += step) {
    ticks.push({ minute: m, label: step >= 1440 ? fmt.shortDate(m) : fmt.dateTime(m).replace(/^\d+ \w+ /, `${fmt.shortDate(m)} `) });
  }
  return ticks;
}

function lowerBound(xs, target) {
  let lo = 0;
  let hi = xs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function nearestIndex(xs, minute) {
  const i = lowerBound(xs, minute);
  if (i === 0) return 0;
  if (i === xs.length) return xs.length - 1;
  return minute - xs[i - 1] <= xs[i] - minute ? i - 1 : i;
}

function palette() {
  return {
    line: cssVar("--brand"),
    alert: cssVar("--alert"),
    fail: cssVar("--fail"),
    failWash: cssVar("--fail-wash"),
    ink: cssVar("--ink"),
    ink2: cssVar("--ink-2"),
    muted: cssVar("--muted"),
    grid: cssVar("--line"),
    surface: cssVar("--surface"),
    font: cssVar("--font"),
    mono: cssVar("--mono"),
  };
}

function setupCanvas(canvas, width, height) {
  const ratio = DPR();
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function makeYScale(panel, visibleValues, top, bottom) {
  const finite = visibleValues.filter((v) => Number.isFinite(v));
  const all = finite.concat([panel.threshold].filter((v) => Number.isFinite(v)));
  if (!all.length) return null;
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (panel.scale === "log") {
    lo = Math.max(lo, 1e-6) / 1.25;
    hi *= 1.25;
    const a = Math.log10(lo);
    const b = Math.log10(hi);
    return { y: (v) => bottom - ((Math.log10(Math.max(v, lo)) - a) / (b - a)) * (bottom - top), ticks: logTicks(lo, hi) };
  }
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  return { y: (v) => bottom - ((v - lo) / (hi - lo)) * (bottom - top), ticks: linearTicks(lo, hi) };
}

function visibleRange(xs, spec) {
  return [Math.max(0, lowerBound(xs, spec.xMin) - 1), Math.min(xs.length, lowerBound(xs, spec.xMax) + 1)];
}

function summarise(panel, spec) {
  const { xs, ys } = panel.series;
  const [first, last] = visibleRange(xs, spec);
  let scored = 0;
  let above = 0;
  for (let i = first; i < last; i++) {
    if (xs[i] < spec.xMin || xs[i] > spec.xMax || ys[i] == null) continue;
    scored += 1;
    if (ys[i] > panel.threshold) above += 1;
  }
  return scored
    ? `${fmt.int(above)} of ${fmt.int(scored)} scored ${spec.unit} above the alert line (${fmt.pct(above / scored, 0)})`
    : "no scored data in this range";
}

function drawPanel(ctx, width, height, panel, spec, isLast, reveal) {
  const c = palette();
  const left = 58;
  const right = 14;
  const top = 30;
  const bottom = height - (isLast ? 24 : 8);
  const plotW = width - left - right;
  const x = (minute) => left + ((minute - spec.xMin) / (spec.xMax - spec.xMin)) * plotW;
  const { xs, ys } = panel.series;
  const [first, last] = visibleRange(xs, spec);
  ctx.clearRect(0, 0, width, height);

  const scale = makeYScale(panel, ys.slice(first, last), top, bottom);
  const geometry = { left, right, plotW, x, y: scale ? scale.y : null, fromX: (px) => spec.xMin + ((px - left) / plotW) * (spec.xMax - spec.xMin) };
  if (!scale) {
    ctx.font = `12px ${c.font}`;
    ctx.fillStyle = c.muted;
    ctx.fillText("No scored windows in this range", left + 8, (top + bottom) / 2);
    return geometry;
  }

  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.font = `10.5px ${c.mono}`;
  for (const t of scale.ticks) {
    const py = Math.round(scale.y(t)) + 0.5;
    if (py < top - 2 || py > bottom + 2) continue;
    ctx.strokeStyle = c.grid;
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(width - right, py);
    ctx.stroke();
    ctx.fillStyle = c.muted;
    ctx.fillText(formatValue(t), left - 8, py);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of timeTicks(spec.xMin, spec.xMax)) {
    const px = Math.round(x(t.minute)) + 0.5;
    ctx.strokeStyle = c.grid;
    ctx.beginPath();
    ctx.moveTo(px, top);
    ctx.lineTo(px, bottom);
    ctx.stroke();
    if (isLast) {
      ctx.fillStyle = c.muted;
      ctx.font = `10.5px ${c.font}`;
      ctx.fillText(t.label, px, bottom + 6);
    }
  }

  ctx.font = `10.5px ${c.font}`;
  for (const w of spec.windows) {
    if (w.end_minute < spec.xMin || w.start_minute > spec.xMax) continue;
    const a = Math.max(left, x(w.start_minute));
    const b = Math.min(width - right, Math.max(x(w.end_minute), a + 2));
    ctx.fillStyle = c.failWash;
    ctx.fillRect(a, top, b - a, bottom - top);
    ctx.strokeStyle = c.fail;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(a + 0.5, top);
    ctx.lineTo(a + 0.5, bottom);
    ctx.stroke();
    ctx.globalAlpha = 1;
    if (panel.showMarkerLabels && width >= 560 && b - a >= 22) {
      ctx.fillStyle = c.fail;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${spec.windowLabel} ${w.event}`, a + 4, bottom - 4);
    }
  }
  for (const m of spec.markers ?? []) {
    if (m.minute < spec.xMin || m.minute > spec.xMax) continue;
    const px = Math.round(x(m.minute)) + 0.5;
    ctx.strokeStyle = c.ink2;
    ctx.setLineDash(m.dash ?? [2, 4]);
    ctx.beginPath();
    ctx.moveTo(px, top);
    ctx.lineTo(px, bottom);
    ctx.stroke();
    ctx.setLineDash([]);
    if (m.label && panel.showMarkerLabels && width >= 560) {
      ctx.fillStyle = c.ink2;
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(m.label, px + 5, top + 2);
    }
  }

  const revealX = left + plotW * reveal;
  const thresholdY = Number.isFinite(panel.threshold) ? scale.y(panel.threshold) : null;
  const path = () => {
    ctx.beginPath();
    let penDown = false;
    let previous = null;
    for (let i = first; i < last; i++) {
      const v = ys[i];
      if (v == null || !Number.isFinite(v)) { penDown = false; previous = null; continue; }
      const px = x(xs[i]);
      const py = scale.y(v);
      const broken = previous != null && spec.mode === "window" && xs[i] - previous > WINDOW_GAP_MINUTES;
      if (!penDown || broken) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
      penDown = true;
      previous = xs[i];
    }
  };
  ctx.lineJoin = "round";
  ctx.lineWidth = spec.mode === "window" ? 1.15 : 1.25;
  const strokeClipped = (colour, y0, y1) => {
    ctx.save();
    ctx.beginPath();
    ctx.rect(left, y0, Math.max(0, revealX - left), Math.max(0, y1 - y0));
    ctx.clip();
    ctx.strokeStyle = colour;
    path();
    ctx.stroke();
    ctx.restore();
  };
  if (thresholdY == null) strokeClipped(c.line, top - 2, bottom + 2);
  else {
    strokeClipped(c.line, thresholdY, bottom + 2);
    strokeClipped(c.alert, top - 2, thresholdY);
  }
  if (spec.mode === "window" && thresholdY != null) {
    ctx.fillStyle = c.alert;
    for (let i = first; i < last; i++) {
      const px = x(xs[i]);
      if (ys[i] == null || ys[i] <= panel.threshold || px > revealX) continue;
      const py = scale.y(ys[i]);
      ctx.beginPath();
      ctx.moveTo(px, py - DIAMOND);
      ctx.lineTo(px + DIAMOND, py);
      ctx.lineTo(px, py + DIAMOND);
      ctx.lineTo(px - DIAMOND, py);
      ctx.closePath();
      ctx.fill();
    }
  }
  if (thresholdY != null) {
    const ty = Math.round(thresholdY) + 0.5;
    ctx.strokeStyle = c.ink2;
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(left, ty);
    ctx.lineTo(width - right, ty);
    ctx.stroke();
    ctx.setLineDash([]);
    const label = spec.thresholdLabel;
    ctx.font = `10.5px ${c.font}`;
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = c.surface;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(width - right - textW - 12, ty - 17, textW + 10, 15);
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.ink2;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, width - right - 7, ty - 4);
  }
  return geometry;
}

export function createStack(container, { onZoom }) {
  let spec = null;
  let built = [];
  let animation = 0;
  let cursorMinute = null;
  const crosshair = el("div", { class: "xhair" });
  const tip = el("div", { class: "tip", role: "status", "aria-live": "polite" });
  const selection = el("div", { class: "xhair", style: "background:var(--brand);opacity:.18;width:0" });
  const stack = el("div", { class: "chart-stack" }, crosshair, tip, selection);
  container.replaceChildren(stack);

  function build() {
    stack.querySelectorAll(".panel").forEach((n) => n.remove());
    built = [];
    const width = stack.clientWidth;
    spec.panels.forEach((panel, index) => {
      const isLast = index === spec.panels.length - 1;
      const height = spec.panels.length === 1 ? 300 : isLast ? 178 : 152;
      const summary = summarise(panel, spec);
      const canvas = el("canvas", { role: "img", tabindex: "0", "aria-label": `${panel.label}, ${panel.sub}. ${summary}. Use the left and right arrow keys to inspect values.` });
      const dot = el("div", { class: "dot" });
      const holder = el("div", { class: "panel" }, el("header", {}, el("b", {}, panel.label), el("span", {}, panel.sub), el("span", { class: "summary" }, summary)), canvas, dot);
      stack.append(holder);
      built.push({ panel, canvas, dot, ctx: setupCanvas(canvas, width, height), width, height, isLast, geometry: null });
      bind(canvas, index);
    });
  }

  function paint(reveal = 1) {
    built.forEach((b) => {
      b.geometry = drawPanel(b.ctx, b.width, b.height, b.panel, spec, b.isLast, reveal);
    });
  }

  function render(next, { animate = false } = {}) {
    spec = next;
    cancelAnimationFrame(animation);
    build();
    hide();
    if (!animate || reducedMotion()) { paint(1); return; }
    const started = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - started) / REVEAL_MS);
      paint(easeOutExpo(t));
      if (t < 1) animation = requestAnimationFrame(step);
    };
    animation = requestAnimationFrame(step);
  }

  function minuteAt(event, canvas, index) {
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    return { minute: built[index].geometry.fromX(px), px };
  }

  function bind(canvas, index) {
    let dragStart = null;
    canvas.addEventListener("pointermove", (event) => {
      const { minute, px } = minuteAt(event, canvas, index);
      const g = built[index].geometry;
      if (px < g.left || px > g.left + g.plotW) { hide(); return; }
      if (dragStart != null) {
        const stackRect = stack.getBoundingClientRect();
        const rect = canvas.getBoundingClientRect();
        const a = Math.min(dragStart, px);
        selection.style.display = "block";
        selection.style.left = `${rect.left - stackRect.left + a}px`;
        selection.style.width = `${Math.abs(px - dragStart)}px`;
        tip.style.display = "none";
        return;
      }
      showAt(minute, event, index);
    });
    canvas.addEventListener("pointerleave", () => { if (dragStart == null) hide(); });
    canvas.addEventListener("pointerdown", (event) => {
      if (event.pointerType === "touch") {
        const { minute, px } = minuteAt(event, canvas, index);
        const g = built[index].geometry;
        if (px >= g.left && px <= g.left + g.plotW) showAt(minute, event, index);
        return;
      }
      if (event.button !== 0) return;
      dragStart = minuteAt(event, canvas, index).px;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointerup", (event) => {
      if (dragStart == null) return;
      const { px } = minuteAt(event, canvas, index);
      const g = built[index].geometry;
      const a = Math.min(dragStart, px);
      const b = Math.max(dragStart, px);
      dragStart = null;
      selection.style.display = "none";
      if (b - a > 8 && onZoom) onZoom(Math.round(g.fromX(a)), Math.round(g.fromX(b)));
    });
    canvas.addEventListener("dblclick", () => onZoom && onZoom(null, null));
    canvas.addEventListener("keydown", (event) => {
      const unit = spec.mode === "window" ? 1 : 60;
      const factor = event.shiftKey ? 10 : 1;
      if (cursorMinute == null) cursorMinute = (spec.xMin + spec.xMax) / 2;
      if (event.key === "ArrowRight") cursorMinute += unit * factor;
      else if (event.key === "ArrowLeft") cursorMinute -= unit * factor;
      else if (event.key === "Home") cursorMinute = spec.xMin;
      else if (event.key === "End") cursorMinute = spec.xMax;
      else if (event.key === "Escape") { hide(); return; }
      else return;
      event.preventDefault();
      cursorMinute = Math.min(spec.xMax, Math.max(spec.xMin, cursorMinute));
      showAt(cursorMinute, null, index);
    });
    canvas.addEventListener("blur", hide);
  }

  function hide() {
    crosshair.style.display = "none";
    tip.style.display = "none";
    built.forEach((b) => { b.dot.style.display = "none"; });
  }

  function showAt(minute, event, anchorIndex) {
    const stackRect = stack.getBoundingClientRect();
    cursorMinute = minute;
    const tolerance = spec.mode === "window" ? WINDOW_GAP_MINUTES : HOURLY_TOLERANCE_MINUTES;
    const rows = built.map(({ panel, dot, geometry, canvas }) => {
      const { xs, ys } = panel.series;
      const i = xs.length ? nearestIndex(xs, minute) : -1;
      const hit = i >= 0 && Math.abs(xs[i] - minute) <= tolerance && ys[i] != null;
      if (hit && geometry.y) {
        dot.style.display = "block";
        dot.style.left = `${geometry.x(xs[i])}px`;
        dot.style.top = `${canvas.offsetTop + geometry.y(ys[i])}px`;
        dot.classList.toggle("over", ys[i] > panel.threshold);
      } else dot.style.display = "none";
      return { panel, value: hit ? ys[i] : null };
    });
    const anchor = built[anchorIndex];
    const canvasRect = anchor.canvas.getBoundingClientRect();
    const x = anchor.geometry.x(minute);
    crosshair.style.display = "block";
    crosshair.style.left = `${canvasRect.left - stackRect.left + x}px`;
    tip.replaceChildren(
      el("b", {}, fmt.dateTime(Math.round(minute))),
      ...rows.map(({ panel, value }) =>
        el("div", { class: "row" }, el("span", {}, panel.label), el("span", { class: value != null && value > panel.threshold ? "over" : "" }, value == null ? "no data" : `${formatValue(value)}${value > panel.threshold ? " · alert" : " · below line"}`)),
      ),
    );
    tip.style.display = "block";
    const pointerX = event ? event.clientX - stackRect.left : canvasRect.left - stackRect.left + x;
    const pointerY = event ? event.clientY - stackRect.top : canvasRect.top - stackRect.top + 40;
    const w = tip.offsetWidth;
    tip.style.left = `${pointerX + 16 + w > stackRect.width ? pointerX - w - 16 : pointerX + 16}px`;
    tip.style.top = `${Math.max(8, pointerY - 20)}px`;
  }

  new ResizeObserver(() => { if (spec) { build(); paint(1); } }).observe(container);
  window.addEventListener("themechange", () => { if (spec) paint(1); });
  return { render };
}

export function lineChart(container, { title, subtitle, series, logY = false, xLabel, yLabel, vline, height = 260 }) {
  const canvas = el("canvas", { role: "img", "aria-label": `${title}. ${subtitle ?? ""}` });
  const legend = el(
    "div",
    { class: "legend" },
    series.map((s) => el("span", {}, el("i", { class: "sw-line", style: `background:var(${s.colour});${s.dash ? "opacity:.55" : ""}` }), s.label)),
  );
  container.replaceChildren(
    el("div", { class: "chart-single" }, el("div", { class: "chart-title" }, title, subtitle ? el("span", {}, subtitle) : null), canvas, legend),
  );

  function draw() {
    const c = palette();
    const width = canvas.parentElement.clientWidth - 8;
    const ctx = setupCanvas(canvas, width, height);
    const left = 62;
    const right = 16;
    const top = 14;
    const bottom = height - 34;
    const plotW = width - left - right;
    const allX = series.flatMap((s) => s.xs);
    const allY = series.flatMap((s) => s.ys).filter((v) => Number.isFinite(v) && (!logY || v > 0));
    const xMin = Math.min(...allX);
    const xMax = Math.max(...allX);
    let lo = Math.min(...allY);
    let hi = Math.max(...allY);
    const a = Math.log10(lo);
    const b = Math.log10(hi);
    if (!logY) { const pad = (hi - lo || 1) * 0.06; lo -= pad; hi += pad; }
    const px = (v) => left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
    const py = (v) => (logY ? bottom - ((Math.log10(v) - a) / (b - a || 1)) * (bottom - top) : bottom - ((v - lo) / (hi - lo)) * (bottom - top));
    ctx.font = `10.5px ${c.mono}`;
    const ticks = logY ? logTicks(lo, hi) : linearTicks(lo, hi, 5);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of ticks) {
      const y = Math.round(py(t)) + 0.5;
      if (y < top - 2 || y > bottom + 2) continue;
      ctx.strokeStyle = c.grid;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
      ctx.fillStyle = c.muted;
      ctx.fillText(formatValue(t), left - 8, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const step = Math.max(1, niceStep(xMax - xMin, 6));
    for (let v = Math.ceil(xMin / step) * step; v <= xMax; v += step) {
      ctx.fillStyle = c.muted;
      ctx.fillText(String(Math.round(v)), px(v), bottom + 6);
    }
    ctx.font = `11px ${c.font}`;
    ctx.fillText(xLabel, left + plotW / 2, bottom + 20);
    ctx.save();
    ctx.translate(12, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    if (vline) {
      ctx.strokeStyle = c.ink2;
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(px(vline.x), top); ctx.lineTo(px(vline.x), bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = c.ink2;
      ctx.textAlign = "left";
      ctx.fillText(vline.label, px(vline.x) + 5, top + 2);
    }
    for (const s of series) {
      ctx.strokeStyle = cssVar(s.colour);
      ctx.lineWidth = 1.8;
      ctx.setLineDash(s.dash ?? []);
      ctx.beginPath();
      s.xs.forEach((xv, i) => (i ? ctx.lineTo(px(xv), py(s.ys[i])) : ctx.moveTo(px(xv), py(s.ys[i]))));
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  new ResizeObserver(draw).observe(container);
  window.addEventListener("themechange", draw);
  draw();
}
