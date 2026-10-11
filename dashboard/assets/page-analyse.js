// "Analyse your data": choose a file, check how its columns are read, set the few choices that depend on the
// data, then run the whole pipeline in a worker and keep the result in this browser.

import { DEFAULTS, WINDOW_CHOICES } from "./analysis/pipeline.js";
import { detectTimestampFormat, parserForKind } from "./analysis/timestamps.js";
import { compact, el, fmt, initShell, segmented, showError } from "./app.js";
import { ACCEPTED, readUploadedFile, withSheet } from "./file-import.js";
import { deleteAnalysis, loadAnalysis, saveAnalysis, setShowingAnalysis, showingAnalysis } from "./saved-analysis.js";

const ROLES = [["timestamp", "Timestamp"], ["analog", "Sensor (numeric)"], ["digital", "On/off (0 or 1)"], ["ignore", "Ignore"]];
const FORMATS = [
  ["iso", "Year first, e.g. 2020-02-01 13:45:00"],
  ["dmy", "Day first, e.g. 01/02/2020 13:45"],
  ["mdy", "Month first, e.g. 02/01/2020 13:45"],
  ["epoch-s", "Unix time in seconds"],
  ["epoch-ms", "Unix time in milliseconds"],
  ["spreadsheet-date", "Spreadsheet date cells"],
];
const STAGES = [
  ["read", "Read the file"], ["audit", "Audit the record"], ["features", "Windows and split"], ["zscore", "Z-score baseline"],
  ["rbm", "RBM with CD-k"], ["vae", "VAE"], ["markov", "Markov chain"], ["evaluate", "Evaluate alerts"],
  ["network", "Bayesian network"], ["export", "Prepare the dashboard"],
];
const DELIMITER_NAMES = { ",": "commas", ";": "semicolons", "\t": "tabs", "|": "vertical bars" };
const K_CHOICES = [1, 5, 10];
const PREVIEW_ROWS = 8;
const SAMPLE_ROWS = 200;
const MIN_PARSED_SHARE = 0.9;

const UPLOAD_ICON = '<svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true"><rect x="1" y="1" width="38" height="38" rx="11" fill="var(--brand-wash)"/><path d="M20 26V13m0 0-5 5m5-5 5 5M13 28h14" stroke="var(--brand-ink)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const LOCK_ICON = '<svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true"><rect x="3.5" y="8" width="11" height="8" rx="2" stroke="currentColor" stroke-width="1.6"/><path d="M6 8V6a3 3 0 0 1 6 0v2" stroke="currentColor" stroke-width="1.6"/></svg>';

const formatBytes = (bytes) => (bytes < 1e6 ? `${Math.max(1, Math.round(bytes / 1e3))} kB` : `${(bytes / 1e6).toFixed(1)} MB`);
const cellText = (v) => {
  const text = v instanceof Date ? v.toLocaleString("en-GB") : String(v ?? "");
  return text.length > 28 ? `${text.slice(0, 27)}…` : text;
};
const naiveMs = (value) => (value ? Date.parse(`${value.length === 16 ? `${value}:00` : value}Z`) : NaN);
const stepHead = (n, title) => el("div", { class: "step-head" }, el("span", { class: "step-n", "aria-hidden": "true" }, String(n)), el("h2", {}, title));

function field(label, control, hint) {
  return el("label", { class: "field" }, el("span", { class: "field-label" }, label), control, hint ? el("span", { class: "field-hint" }, hint) : null);
}

function options(list, current) {
  return list.map(([value, label]) => el("option", { value, selected: String(current) === String(value) }, label));
}

try {
  const { wrap } = initShell(
    "analyse.html",
    "Analyse your data",
    "Run the same pipeline on your own time-series file: the z-score baseline, the RBM, the VAE, the Markov chain and the Bayesian network, with thresholds frozen before any test window is scored.",
  );

  const state = {
    file: null, table: null, roles: [], timestampKind: null, timestampNote: "", confirmed: true,
    trainPeriod: DEFAULTS.split.trainPeriod, trainShare: 20, windowSeconds: "auto", stateColumn: "auto",
    kValues: new Set(DEFAULTS.rbm.kValues), windows: [], worker: null, timer: null,
  };

  const savedHost = el("div");
  const status = el("p", { class: "drop-status", role: "status", "aria-live": "polite" });
  const input = el("input", { type: "file", id: "file-input", class: "visually-hidden", accept: ACCEPTED, onchange: () => input.files[0] && openFile(input.files[0]) });
  const drop = el("label", { class: "drop", for: "file-input" },
    el("span", { html: UPLOAD_ICON }),
    el("b", {}, "Drop a file here or choose one"),
    el("span", { class: "muted small" }, "CSV, TSV or text · Excel (.xlsx, .xls) · PDF with a text table"),
  );
  ["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (event) => { event.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) openFile(file); });

  const fileSection = el("section", { class: "step-card reveal" },
    stepHead(1, "Choose a file"),
    input, drop, status,
    el("div", { class: "privacy" }, el("span", { html: LOCK_ICON }),
      el("p", {}, "Your file is read by this page and never uploaded. The analysis runs in a background thread of this tab, and the results are saved only in this browser until you delete them.")),
    el("details", { class: "needs" },
      el("summary", {}, "What the file needs"),
      el("ul", { class: "list" },
        el("li", {}, "One row per reading and a timestamp column: ISO dates, day-first or month-first dates, Unix time, or spreadsheet date cells."),
        el("li", {}, "At least one numeric sensor column. Columns holding only 0 and 1 are read as on/off signals."),
        el("li", {}, "Enough history for a training period and a later test period. By default the models learn from the first calendar month and are tested on everything after it."),
        el("li", {}, "For PDFs, a table with selectable text (not a scan). Reading tables from PDF is approximate, so the preview must be checked."),
      )),
  );
  const mappingSection = el("section", { class: "step-card", hidden: true });
  const settingsSection = el("section", { class: "step-card", hidden: true });
  const runSection = el("section", { class: "step-card", hidden: true });
  wrap.append(savedHost, fileSection, mappingSection, settingsSection, runSection);
  paintSaved();

  async function paintSaved() {
    let saved = null;
    try {
      saved = await loadAnalysis();
    } catch {
      saved = null;
    }
    if (!saved) { savedHost.replaceChildren(); return; }
    const source = saved.context.source;
    const showing = showingAnalysis();
    savedHost.replaceChildren(
      el("div", { class: "saved reveal" },
        el("div", {}, el("b", {}, "Saved analysis: "), source.file_name, el("span", { class: "muted small" }, ` · run on ${source.analysed_at}${showing ? " · shown on the other pages" : ""}`)),
        el("div", { class: "saved-actions" },
          showing
            ? el("button", { class: "btn", type: "button", onclick: () => { setShowingAnalysis(false); paintSaved(); } }, "Show MetroGuard results instead")
            : el("a", { class: "btn primary", href: "index.html", onclick: () => setShowingAnalysis(true) }, "View its results"),
          el("button", { class: "btn", type: "button", onclick: async () => { await deleteAnalysis(); setShowingAnalysis(false); paintSaved(); } }, "Delete it"),
        ),
      ),
    );
  }

  async function openFile(file) {
    status.classList.remove("error");
    status.textContent = `Opening ${file.name}…`;
    try {
      const table = await readUploadedFile(file, (message) => { status.textContent = message; });
      state.file = file;
      adoptTable(table);
      status.textContent = `${file.name} · ${formatBytes(file.size)} · ${describe(table)}`;
    } catch (error) {
      status.textContent = String(error.message ?? error);
      status.classList.add("error");
    }
  }

  function describe(table) {
    if (table.kind === "csv") return `${table.header.length} columns separated by ${DELIMITER_NAMES[table.csv.delimiter]}`;
    if (table.kind === "excel") return `${fmt.int(table.rows.length)} rows on sheet “${table.sheet}”`;
    return `${fmt.int(table.rows.length)} table rows from ${table.pages} page${table.pages === 1 ? "" : "s"}`;
  }

  function adoptTable(table) {
    state.table = table;
    const roles = table.columns.map((c) => c.role);
    const first = roles.indexOf("timestamp");
    state.roles = roles.map((role, i) => (role === "timestamp" && i !== first ? "ignore" : role));
    setTimestampColumn(first);
    state.confirmed = table.kind !== "pdf";
    state.stateColumn = "auto";
    paintMapping();
    paintSettings();
    paintRun();
    mappingSection.hidden = settingsSection.hidden = runSection.hidden = false;
    mappingSection.classList.add("reveal");
  }

  const sampleValues = (index) => state.table.rows.slice(0, SAMPLE_ROWS).map((r) => r[index]).filter((v) => v !== "" && v != null);

  function setTimestampColumn(index) {
    if (index < 0) { state.timestampKind = null; state.timestampNote = ""; return; }
    const detected = state.table.columns[index].timestamp ?? detectTimestampFormat(sampleValues(index));
    state.timestampKind = detected?.kind ?? "iso";
    state.timestampNote = detected?.note ?? "Format not recognised, choose it here";
  }

  function parsedShare(index) {
    const values = sampleValues(index);
    const parse = parserForKind(state.timestampKind);
    return values.length ? values.filter((v) => Number.isFinite(parse(v))).length / values.length : 0;
  }

  function setRole(index, role) {
    const previous = state.roles.indexOf("timestamp");
    if (role === "timestamp" && previous >= 0 && previous !== index) {
      const column = state.table.columns[previous];
      state.roles[previous] = column.binary ? "digital" : column.numeric ? "analog" : "ignore";
    }
    state.roles[index] = role;
    setTimestampColumn(state.roles.indexOf("timestamp"));
    paintMapping();
    paintSettings();
    paintRun();
  }

  function paintMapping() {
    const { table } = state;
    const timestampIndex = state.roles.indexOf("timestamp");
    const head = table.header.map((name, i) => {
      const select = el("select", { "aria-label": `Role of column ${name}`, onchange: () => setRole(i, select.value) }, options(ROLES, state.roles[i]));
      return el("th", { scope: "col", class: `col-${state.roles[i]}` }, el("span", { class: "col-name", title: name }, name), select);
    });
    const body = table.rows.slice(0, PREVIEW_ROWS).map((row) =>
      el("tr", {}, table.header.map((_, i) => el("td", { class: `col-${state.roles[i]}` }, cellText(row[i])))));
    const sheetSelect = table.sheets?.length > 1
      ? field("Sheet", el("select", { onchange: (event) => { adoptTable(withSheet(table, event.target.value)); status.textContent = `${state.file.name} · ${describe(state.table)}`; } }, options(table.sheets.map((s) => [s, s]), table.sheet)))
      : null;
    const formats = FORMATS.filter(([kind]) => kind !== "spreadsheet-date" || table.kind === "excel");
    const formatSelect = el("select", { onchange: () => { state.timestampKind = formatSelect.value; state.timestampNote = "Chosen by you"; paintMapping(); paintRun(); } }, options(formats, state.timestampKind));
    const share = timestampIndex >= 0 ? parsedShare(timestampIndex) : 0;

    mappingSection.replaceChildren(...compact(
      stepHead(2, "Check how the columns are read"),
      el("p", { class: "muted" }, "Each column's role was guessed from its first rows. Change any guess that is wrong; ignored columns are not read."),
      sheetSelect ? el("div", { class: "field-row" }, sheetSelect) : null,
      el("div", { class: "table-scroll preview" }, el("table", {}, el("thead", {}, el("tr", {}, head)), el("tbody", {}, body))),
      el("p", { class: "muted small" }, `First ${Math.min(PREVIEW_ROWS, table.rows.length)} rows shown.`),
      timestampIndex >= 0
        ? el("div", { class: "field-row" },
            field("Timestamp format", formatSelect, `${state.timestampNote}. ${Math.round(share * 100)}% of the first ${fmt.int(sampleValues(timestampIndex).length)} timestamps can be read with it. Times are used as written, with no timezone conversion.`))
        : null,
      table.kind === "pdf"
        ? el("label", { class: "check" },
            el("input", { type: "checkbox", checked: state.confirmed, onchange: (event) => { state.confirmed = event.target.checked; paintRun(); } }),
            el("span", {}, `I compared this preview with the PDF and the columns line up. ${fmt.int(table.skippedLines)} lines that did not fit the table, such as titles, notes and page numbers, were left out.`))
        : null,
    ));
  }

  function paintSettings() {
    const analogNames = state.table.header.filter((_, i) => state.roles[i] === "analog");
    if (!analogNames.includes(state.stateColumn)) state.stateColumn = "auto";
    const shareInput = el("input", { type: "number", min: "5", max: "80", step: "5", value: String(state.trainShare), disabled: state.trainPeriod !== "share", oninput: () => { state.trainShare = Number(shareInput.value); paintRun(); } });
    const windowSelect = el("select", { onchange: () => { state.windowSeconds = windowSelect.value === "auto" ? "auto" : Number(windowSelect.value); } },
      options([["auto", "Auto (from the reading interval)"], ...WINDOW_CHOICES.map((s) => [s, fmt.windowLength(s)])], state.windowSeconds));
    const stateSelect = el("select", { onchange: () => { state.stateColumn = stateSelect.value; } },
      options([["auto", "Auto (most distinct levels)"], ...analogNames.map((n) => [n, n])], state.stateColumn));
    const kChips = K_CHOICES.map((k) => el("button", {
      class: "chip", type: "button", "aria-pressed": String(state.kValues.has(k)),
      onclick: (event) => {
        if (state.kValues.has(k)) state.kValues.delete(k); else state.kValues.add(k);
        event.currentTarget.setAttribute("aria-pressed", String(state.kValues.has(k)));
        paintRun();
      },
    }, `k = ${k}`));

    const windowsHost = el("div", { class: "windows-list" });
    const paintWindows = () => {
      windowsHost.replaceChildren(...state.windows.map((w, i) => el("div", { class: "window-row" },
        el("span", { class: "window-n", "aria-hidden": "true" }, String(i + 1)),
        field("Start", el("input", { type: "datetime-local", step: "1", value: w.start, oninput: (event) => { w.start = event.target.value; paintRun(); } })),
        field("End", el("input", { type: "datetime-local", step: "1", value: w.end, oninput: (event) => { w.end = event.target.value; paintRun(); } })),
        field("What was reported", el("input", { type: "text", value: w.label, maxlength: "60", placeholder: "e.g. Air leak", oninput: (event) => { w.label = event.target.value; } })),
        el("button", { class: "btn", type: "button", "aria-label": `Remove reported window ${i + 1}`, onclick: () => { state.windows.splice(i, 1); paintWindows(); paintRun(); } }, "Remove"),
      )));
    };
    paintWindows();

    settingsSection.replaceChildren(
      stepHead(3, "Choose the settings"),
      el("div", { class: "settings-grid" },
        el("div", { class: "field" },
          segmented({ label: "Training period", options: [["first_calendar_month", "First calendar month"], ["share", "First share"]], value: state.trainPeriod, onChange: (v) => { state.trainPeriod = v; shareInput.disabled = v !== "share"; paintRun(); } }),
          field("Share of the record used for training (%)", shareInput, "The models learn only from the training period. Its last 20% is held out to fix the thresholds, and everything after it is the test period.")),
        field("Window length", windowSelect, "Readings are summarised per window. Auto uses 60 s windows for readings every 10 s or faster, otherwise about six readings per window."),
        field("Operating-state column for the Markov chain", stateSelect, "Auto picks the sensor with the most distinct levels in the training period (Motor_current, when present and it has several)."),
        el("fieldset", { class: "field" },
          el("legend", { class: "field-label" }, "RBM variants (Gibbs steps k)"),
          el("div", { class: "chips" }, kChips),
          el("span", { class: "field-hint" }, "Each k trains one RBM with contrastive divergence. Fewer variants finish sooner.")),
      ),
      el("h3", {}, "Reported windows (optional)"),
      el("p", { class: "muted small" }, "Intervals in which a problem was reported, in the same clock time as your file. Alerts are evaluated against them. Without any, the dashboard still shows the scores and how often each model alerts, but it cannot measure detection."),
      windowsHost,
      el("button", { class: "btn", type: "button", onclick: () => { state.windows.push({ start: "", end: "", label: "" }); paintWindows(); paintRun(); } }, "Add a reported window"),
    );
  }

  function problems() {
    const list = [];
    const timestampIndex = state.roles.indexOf("timestamp");
    if (timestampIndex < 0) list.push("Mark one column as the timestamp.");
    else if (parsedShare(timestampIndex) < MIN_PARSED_SHARE) list.push("Most timestamps cannot be read with the chosen format. Pick another format or another column.");
    if (!state.roles.includes("analog")) list.push("Mark at least one column as a numeric sensor.");
    if (!state.confirmed) list.push("Confirm that the PDF preview matches the table.");
    if (state.trainPeriod === "share" && !(state.trainShare >= 5 && state.trainShare <= 80)) list.push("Set the training share between 5% and 80%.");
    if (!state.kValues.size) list.push("Choose at least one RBM variant.");
    state.windows.forEach((w, i) => {
      const start = naiveMs(w.start);
      const end = naiveMs(w.end);
      if (!Number.isFinite(start) || !Number.isFinite(end)) list.push(`Reported window ${i + 1} needs a start and an end.`);
      else if (end <= start) list.push(`Reported window ${i + 1} ends before it starts.`);
    });
    return list;
  }

  function paintRun(message) {
    const issues = problems();
    runSection.replaceChildren(...compact(
      stepHead(4, "Run the analysis"),
      message,
      issues.length ? el("ul", { class: "problems" }, issues.map((text) => el("li", {}, text))) : null,
      el("div", { class: "run-row" },
        el("button", { class: "btn primary big", type: "button", disabled: issues.length > 0, onclick: run }, "Run the analysis"),
        el("span", { class: "muted small" }, `${state.kValues.size} RBM variant${state.kValues.size === 1 ? "" : "s"}, VAE, Markov chain, Bayesian network. Large files take a few minutes; keep this tab open.`)),
    ));
  }

  function buildRequest() {
    const columnsWith = (role) => state.table.header.map((name, index) => ({ name, index })).filter(({ index }) => state.roles[index] === role);
    const timestampIndex = state.roles.indexOf("timestamp");
    const { table, file } = state;
    return {
      source: table.kind === "csv" ? { kind: "csv", file, ...table.csv } : { kind: "rows", rows: table.rows },
      mapping: { timestampIndex, timestampKind: state.timestampKind, analog: columnsWith("analog"), digital: columnsWith("digital") },
      meta: {
        fileName: file.name, fileKind: table.kind, columnCount: table.header.length, stateColumn: state.stateColumn, timestampNote: state.timestampNote,
        failureWindows: state.windows.map((w) => ({ start: naiveMs(w.start), end: naiveMs(w.end), label: w.label.trim() })),
      },
      options: {
        ...DEFAULTS,
        split: { ...DEFAULTS.split, trainPeriod: state.trainPeriod, trainShare: state.trainShare / 100 },
        windows: { ...DEFAULTS.windows, sizeSeconds: state.windowSeconds },
        rbm: { ...DEFAULTS.rbm, kValues: [...state.kValues].sort((a, b) => a - b) },
      },
    };
  }

  function setEditing(enabled) {
    for (const section of [fileSection, mappingSection, settingsSection]) section.inert = !enabled;
  }

  function stopWorker() {
    state.worker?.terminate();
    state.worker = null;
    clearInterval(state.timer);
    setEditing(true);
  }

  function run() {
    const started = performance.now();
    const bar = el("i");
    const percent = el("b", { class: "num" }, "0%");
    const message = el("span", {}, "Starting");
    const elapsed = el("span", { class: "muted small num" }, "0 s");
    const stageItems = STAGES.map(([, label]) => el("li", {}, label));
    const meter = el("div", { class: "progress-bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "0", "aria-label": "Analysis progress" }, bar);
    runSection.replaceChildren(
      stepHead(4, "Running the analysis"),
      el("div", { class: "progress" },
        el("div", { class: "progress-top" }, percent, message, elapsed),
        meter,
        el("ol", { class: "stage-list" }, stageItems),
        el("div", { class: "run-row" },
          el("button", { class: "btn", type: "button", onclick: () => { stopWorker(); paintRun(el("p", { class: "note" }, "The analysis was cancelled. Nothing was saved.")); } }, "Cancel"),
          el("span", { class: "muted small" }, "Everything runs in this tab. Closing or leaving it stops the analysis.")),
      ),
    );
    setEditing(false);
    state.timer = setInterval(() => { elapsed.textContent = fmt.seconds((performance.now() - started) / 1000); }, 1000);

    const worker = new Worker(new URL("./analysis/worker.js", import.meta.url), { type: "module" });
    state.worker = worker;
    worker.onmessage = async ({ data }) => {
      if (data.type === "progress") {
        const value = Math.round(data.fraction * 100);
        bar.style.transform = `scaleX(${data.fraction})`;
        meter.setAttribute("aria-valuenow", String(value));
        percent.textContent = `${value}%`;
        message.textContent = data.message;
        const current = STAGES.findIndex(([key]) => key === data.stage);
        stageItems.forEach((item, i) => { item.className = data.stage === "done" || i < current ? "done" : i === current ? "active" : ""; });
      } else if (data.type === "error") {
        stopWorker();
        paintRun(el("div", { class: "note warn", role: "alert" }, el("p", {}, el("strong", {}, "The analysis stopped. "), data.message)));
      } else if (data.type === "done") {
        const seconds = (performance.now() - started) / 1000;
        stopWorker();
        await finish(data.bundle, seconds);
      }
    };
    worker.onerror = (event) => {
      stopWorker();
      paintRun(el("div", { class: "note warn", role: "alert" }, el("p", {}, el("strong", {}, "The analysis stopped. "), event.message || "The background thread failed to start.")));
    };
    worker.postMessage(buildRequest());
  }

  async function finish(bundle, seconds) {
    let saved = true;
    try {
      await saveAnalysis(bundle);
      setShowingAnalysis(true);
    } catch {
      saved = false;
    }
    const { context, metrics, audit } = bundle;
    const primary = metrics.rows.filter((r) => r.is_primary_score && r.percentile === context.primary_percentile);
    const nWindows = context.windows.length;
    runSection.replaceChildren(...compact(
      stepHead(4, "Analysis complete"),
      el("p", {}, `Finished in ${fmt.seconds(seconds)}. ${fmt.int(audit.rows)} rows were read; the test period has ${fmt.int(metrics.n_usable_test_windows)} usable windows over ${fmt.num(metrics.test_days, 1)} days.`),
      el("div", { class: "facts" },
        el("div", { class: "fact" }, el("b", {}, String(primary.length)), el("span", {}, "models scored")),
        el("div", { class: "fact" }, el("b", {}, fmt.windowLength(context.window_seconds)), el("span", {}, "window length")),
        el("div", { class: "fact" }, el("b", {}, context.source.test_start.slice(0, 10)), el("span", {}, "test period starts")),
        el("div", { class: "fact" }, el("b", {}, String(nWindows)), el("span", {}, nWindows === 1 ? "reported window" : "reported windows")),
      ),
      context.source.notes.length ? el("div", { class: "note" }, context.source.notes.map((n) => el("p", {}, n))) : null,
      saved
        ? el("div", { class: "run-row" },
            el("a", { class: "btn primary big", href: "index.html" }, "Open the results"),
            el("a", { class: "btn", href: "scores.html" }, "Scores over time"),
            el("button", { class: "btn", type: "button", onclick: () => { paintRun(); paintSaved(); } }, "Change settings and run again"))
        : el("div", { class: "note warn", role: "alert" }, el("p", {}, "This browser would not store the results, so the other pages cannot show them. Allow this site to store data (or leave private browsing) and run the analysis again.")),
    ));
    paintSaved();
  }
} catch (error) {
  initShell("analyse.html", "Analyse your data");
  showError(error);
}
