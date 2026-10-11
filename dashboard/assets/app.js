// Shared shell, theme, data loading and formatting for every dashboard page.

import { loadAnalysis, setShowingAnalysis, showingAnalysis } from "./saved-analysis.js";

const NAV = [
  ["index.html", "Overview"],
  ["metrics.html", "Metrics"],
  ["scores.html", "Scores over time"],
  ["network.html", "Bayesian network"],
  ["data.html", "Data and training"],
  ["claims.html", "Claims and limits"],
  ["analyse.html", "Analyse your data"],
];

const cache = new Map();

// The visitor's own analysis, when they chose to view it; otherwise the pages read the MetroGuard results.
let analysis = null;
const analysisReady = showingAnalysis()
  ? loadAnalysis()
      .then((bundle) => {
        analysis = bundle ?? null;
        if (!analysis) setShowingAnalysis(false);
      })
      .catch(() => setShowingAnalysis(false))
  : Promise.resolve();

export function loadJSON(name) {
  if (!cache.has(name)) {
    cache.set(
      name,
      analysisReady.then(() => {
        if (analysis) {
          if (!(name in analysis)) throw new Error(`The saved analysis has no "${name}" section.`);
          return analysis[name];
        }
        return fetch(`data/${name}.json`).then((response) => {
          if (!response.ok) throw new Error(`Could not load data/${name}.json (${response.status})`);
          return response.json();
        });
      }),
    );
  }
  return cache.get(name);
}

/** True when the page is showing the visitor's own analysis rather than the MetroGuard results. */
export const isUpload = (context) => context.source?.kind === "upload";

/** Reported window numbers, 1..n. */
export const eventIds = (context) => context.windows.map((w) => w.event);

export function showMetroGuardResults() {
  setShowingAnalysis(false);
  location.reload();
}

export function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === false || value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Drops null and false so optional sections can be passed straight to append() or replaceChildren(). */
export const compact = (...nodes) => nodes.filter((node) => node != null && node !== false);

export function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const two = (n) => String(n).padStart(2, "0");
const utc = (minute) => new Date(minute * 60000);

export const fmt = {
  int: (n) => Math.round(n).toLocaleString("en-GB"),
  num: (n, digits = 2) =>
    n == null ? "n/a" : Number(n).toLocaleString("en-GB", { minimumFractionDigits: digits, maximumFractionDigits: digits }),
  pct: (fraction, digits = 1) => `${(fraction * 100).toFixed(digits)}%`,
  hours: (h) => (h == null ? "not detected" : `${h > 0 ? "+" : "−"}${Math.abs(h).toFixed(1)} h`),
  date: (minute) => {
    const d = utc(minute);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  },
  shortDate: (minute) => {
    const d = utc(minute);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
  },
  dateTime: (minute) => {
    const d = utc(minute);
    return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${two(d.getUTCHours())}:${two(d.getUTCMinutes())}`;
  },
  monthName: (minute) => MONTHS[utc(minute).getUTCMonth()],
  seconds: (s) => {
    if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
    const whole = Math.round(s);
    return `${Math.floor(whole / 60)} min ${whole % 60} s`;
  },
  windowLength: (s) => (s <= 120 ? `${s} s` : s < 3600 ? `${s / 60} min` : s < 86400 ? `${s / 3600} h` : `${s / 86400} day`),
};

const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function setTheme(theme, animate) {
  const apply = () => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "dark" ? "#000000" : "#f2f5f6");
    window.dispatchEvent(new CustomEvent("themechange"));
  };
  if (animate && document.startViewTransition && !reduceMotion()) document.startViewTransition(apply);
  else apply();
}

function storedTheme() {
  try {
    return localStorage.getItem("mg-theme");
  } catch {
    return null;
  }
}

// Light is the default for every visitor; dark is only used after the visitor picks it.
function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

const themeToggleLabel = (theme) => (theme === "dark" ? "Light theme" : "Dark theme");

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function initShell(page, title, description) {
  document.title = title === "MetroGuard" ? "MetroGuard" : `${title} · MetroGuard`;
  document.querySelector(".skeleton")?.remove();
  if (!document.documentElement.dataset.theme) setTheme(storedTheme() === "dark" ? "dark" : "light", false);

  const toggle = el(
    "button",
    {
      class: "btn",
      type: "button",
      "aria-pressed": String(currentTheme() === "dark"),
      onclick: () => {
        const next = currentTheme() === "dark" ? "light" : "dark";
        setTheme(next, true);
        toggle.setAttribute("aria-pressed", String(next === "dark"));
        toggle.textContent = themeToggleLabel(next);
        try {
          localStorage.setItem("mg-theme", next);
        } catch {
          /* storage unavailable: the choice just won't persist */
        }
      },
    },
    themeToggleLabel(currentTheme()),
  );

  const rail = el(
    "aside",
    { class: "rail" },
    el(
      "a",
      { class: "brand", href: "index.html", "aria-label": "MetroGuard overview" },
      el("span", { html: '<svg width="34" height="34" viewBox="0 0 34 34" aria-hidden="true"><rect width="34" height="34" rx="9" fill="var(--brand)"/><path d="M6 22 H28" stroke="#fff" stroke-width="3.2" stroke-linecap="round"/><circle cx="11" cy="22" r="3.6" fill="var(--brand)" stroke="#fff" stroke-width="2.4"/><circle cx="23" cy="22" r="3.6" fill="var(--brand)" stroke="#fff" stroke-width="2.4"/><path d="M8 13 L14 9.5 L20 13 L26 9" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity=".85"/></svg>' }),
      el("span", {}, el("b", {}, "MetroGuard"), el("span", { class: "brand-sub" }, "Compressor telemetry evidence")),
    ),
    el(
      "nav",
      { class: "nav", "aria-label": "Pages" },
      NAV.map(([href, label]) => el("a", { href, "aria-current": href === page ? "page" : false }, label)),
    ),
    el(
      "div",
      { class: "rail-foot" },
      toggle,
      analysis
        ? el("span", { class: "src" }, "Data: your file, analysed in this browser.")
        : el(
            "span",
            { class: "src" },
            "Data: ",
            el("a", { href: "https://doi.org/10.24432/C5VW3R", rel: "noopener" }, "MetroPT-3, UCI"),
            ", CC BY 4.0.",
          ),
    ),
  );

  const shell = document.getElementById("shell");
  shell.prepend(rail);
  const wrap = document.getElementById("wrap");
  wrap.prepend(
    el("header", { class: "reveal" }, el("h1", {}, title), description ? el("p", { class: "lede" }, description) : null),
  );
  if (analysis && page !== "analyse.html") wrap.prepend(sourceBar(analysis.context.source));
  return { wrap };
}

function sourceBar(source) {
  return el(
    "div",
    { class: "source-bar reveal", role: "status" },
    el("span", { class: "source-dot", "aria-hidden": "true" }),
    el("span", { class: "source-text" }, "Showing your analysis of ", el("b", {}, source.file_name), `, run in this browser on ${source.analysed_at}.`),
    el(
      "span",
      { class: "source-actions" },
      el("a", { class: "btn", href: "analyse.html" }, "Analyse another file"),
      el("button", { class: "btn", type: "button", onclick: showMetroGuardResults }, "Back to MetroGuard results"),
    ),
  );
}

export function footer(context) {
  const total = context.runtimes_seconds.run_all_steps + context.runtimes_seconds.step01_audit_raw_data;
  if (isUpload(context)) {
    return el(
      "footer",
      { class: "footer" },
      el("p", {}, `Every number on these pages was computed in this browser from ${context.source.file_name} on ${context.source.analysed_at}. The file was not sent anywhere, and the results are stored only in this browser.`),
      el("p", {}, `Seed ${context.seed}. Analysis time in this browser: ${fmt.seconds(total)}. Exploratory results from one record; no claim of failure prediction or causation.`),
    );
  }
  return el(
    "footer",
    { class: "footer" },
    el(
      "p",
      {},
      "Every number on this site is read from files the pipeline produced (",
      el("code", {}, "dashboard/data"),
      "). Nothing is estimated here, and no model runs in your browser.",
    ),
    el(
      "p",
      {},
      `Seed ${context.seed}. Full pipeline run time on the author's machine: ${fmt.seconds(total)}. Exploratory results from four reported events on one compressor; no claim of failure prediction or causation.`,
    ),
  );
}

export function showError(error) {
  document.getElementById("wrap").append(
    el(
      "div",
      { class: "note warn", role: "alert" },
      el("strong", {}, "Could not load the dashboard data. "),
      String(error.message || error),
      " If you opened this file directly from disk, serve the folder over HTTP instead (for example ",
      el("code", {}, "python -m http.server"),
      " inside ",
      el("code", {}, "dashboard/"),
      ").",
    ),
  );
}

export function sortableTable({ columns, rows, caption, note, initialSort }) {
  let sort = initialSort ?? null;
  const tbody = el("tbody");
  const headRow = el("tr");
  const table = el(
    "table",
    {},
    el("caption", {}, caption, note ? el("span", {}, note) : null),
    el("thead", {}, headRow),
    tbody,
  );

  function paint() {
    let data = rows.slice();
    if (sort) {
      const column = columns.find((c) => c.key === sort.key);
      const value = column.sortValue ?? ((row) => row[column.key]);
      data.sort((a, b) => {
        const x = value(a);
        const y = value(b);
        const bad = (v) => v == null || (typeof v === "number" && Number.isNaN(v));
        if (bad(x) && bad(y)) return 0;
        if (bad(x)) return 1;
        if (bad(y)) return -1;
        return (x < y ? -1 : x > y ? 1 : 0) * (sort.dir === "asc" ? 1 : -1);
      });
    }
    tbody.replaceChildren(
      ...data.map((row) =>
        el(
          "tr",
          {},
          columns.map((c) => el("td", { class: c.align === "right" ? "r" : "" }, c.render ? c.render(row) : row[c.key])),
        ),
      ),
    );
    headRow.replaceChildren(
      ...columns.map((c) => {
        const active = sort && sort.key === c.key;
        return el(
          "th",
          {
            scope: "col",
            class: c.align === "right" ? "r" : "",
            "aria-sort": active ? (sort.dir === "asc" ? "ascending" : "descending") : false,
          },
          c.sortable === false
            ? c.label
            : el(
                "button",
                {
                  type: "button",
                  onclick: () => {
                    sort = { key: c.key, dir: active && sort.dir === "desc" ? "asc" : "desc" };
                    paint();
                  },
                },
                c.label,
              ),
        );
      }),
    );
  }
  paint();
  return el("div", { class: "table-scroll" }, table);
}

export function plainTable({ columns, rows, caption, note, className }) {
  return el(
    "div",
    { class: "table-scroll" },
    el(
      "table",
      { class: className },
      caption ? el("caption", {}, caption, note ? el("span", {}, note) : null) : null,
      el("thead", {}, el("tr", {}, columns.map((c) => el("th", { scope: "col", class: c.align === "right" ? "r" : "" }, c.label)))),
      el(
        "tbody",
        {},
        rows.map((row) =>
          el("tr", {}, columns.map((c) => el("td", { class: c.align === "right" ? "r" : "" }, c.render ? c.render(row) : row[c.key]))),
        ),
      ),
    ),
  );
}

export function segmented({ label, options, value, onChange }) {
  const group = el("div", { class: "seg", role: "group", "aria-label": label });
  const thumb = el("span", { class: "thumb instant", "aria-hidden": "true" });
  const buttons = options.map(([optionValue, text]) =>
    el("button", { type: "button", "data-value": String(optionValue), onclick: () => select(optionValue, true) }, text),
  );
  group.append(thumb, ...buttons);
  let current = value;

  function place(instant) {
    const active = buttons.find((b) => b.getAttribute("aria-pressed") === "true");
    if (!active || !active.offsetWidth) return;
    thumb.classList.toggle("instant", instant || reduceMotion());
    thumb.style.width = `${active.offsetWidth}px`;
    thumb.style.transform = `translateX(${active.offsetLeft}px)`;
  }
  function select(next, notify) {
    if (next === current && notify) return;
    current = next;
    buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === next)));
    place(!notify);
    if (notify) onChange(next);
  }
  buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(options[i][0] === value)));
  new ResizeObserver(() => place(true)).observe(group);
  const wrapper = el("div", {}, el("span", { class: "control-label" }, label), group);
  wrapper.select = (next) => select(next, false);
  return wrapper;
}

export function eventPills(ids, nEvents) {
  const detected = new Set(String(ids).split(";").filter((v) => v !== "none").map(Number));
  return el(
    "span",
    { class: "pills", role: "img", "aria-label": detected.size ? `Events detected: ${[...detected].join(", ")}` : "No events detected" },
    Array.from({ length: nEvents }, (_, i) => el("span", { class: `pill${detected.has(i + 1) ? " on" : ""}`, style: `--i:${i}` }, String(i + 1))),
  );
}
