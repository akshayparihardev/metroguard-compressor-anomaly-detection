// Timestamp detection and parsing. Times are kept as "naive" milliseconds: the wall-clock time written in the
// file is stored as if it were UTC, so the dashboard shows exactly what the file says (no timezone shifts).

const ISO = /^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$/;
const DMY = /^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?)?$/;

function naive(year, month, day, hour = 0, minute = 0, second = 0, fraction = "") {
  const ms = fraction ? Math.round(Number(`0.${fraction}`) * 1000) : 0;
  if (month < 1 || month > 12 || day < 1 || day > 31 || hour > 23 || minute > 59 || second > 60) return NaN;
  return Date.UTC(year, month - 1, day, hour, minute, second, ms);
}

function parseIso(text) {
  const m = ISO.exec(text);
  if (!m) return NaN;
  let value = naive(+m[1], +m[2], +m[3], +(m[4] ?? 0), +(m[5] ?? 0), +(m[6] ?? 0), m[7] ?? "");
  if (m[8] && m[8] !== "Z") {
    const sign = m[8][0] === "-" ? -1 : 1;
    const digits = m[8].slice(1).replace(":", "");
    value -= sign * (Number(digits.slice(0, 2)) * 60 + Number(digits.slice(2))) * 60000;
  }
  return value;
}

function parseDayFirst(text, dayFirst) {
  const m = DMY.exec(text);
  if (!m) return NaN;
  const [a, b] = [+m[1], +m[2]];
  return naive(+m[3], dayFirst ? b : a, dayFirst ? a : b, +(m[4] ?? 0), +(m[5] ?? 0), +(m[6] ?? 0), m[7] ?? "");
}

function epochKind(value) {
  if (value > 1e11 && value < 1e14) return "epoch-ms";
  if (value > 1e8 && value < 1e11) return "epoch-s";
  return null;
}

const spreadsheetDate = (d) =>
  d instanceof Date ? naive(d.getFullYear(), d.getMonth() + 1, d.getDate(), d.getHours(), d.getMinutes(), d.getSeconds()) : NaN;

/** The parser for a detected format kind (kinds are plain strings so they can be sent to a worker). */
export function parserForKind(kind) {
  switch (kind) {
    case "spreadsheet-date": return spreadsheetDate;
    case "iso": return (v) => parseIso(String(v).trim());
    case "epoch-ms": return (v) => (v === "" || v == null ? NaN : Number(v));
    case "epoch-s": return (v) => (v === "" || v == null ? NaN : Number(v) * 1000);
    case "dmy": return (v) => parseDayFirst(String(v).trim(), true);
    case "mdy": return (v) => parseDayFirst(String(v).trim(), false);
    default: throw new Error(`Unknown timestamp format: ${kind}`);
  }
}

/**
 * Pick one timestamp format for a column from sample values. Returns { kind, note } or null when the column
 * does not look like timestamps. Ambiguous day/month dates default to day-first and say so in the note.
 */
export function detectTimestampFormat(samples) {
  const values = samples.map((v) => (v instanceof Date ? v : String(v ?? "").trim())).filter((v) => v !== "");
  if (!values.length) return null;
  const ok = (parse) => values.filter((v) => Number.isFinite(parse(v))).length / values.length >= 0.95;
  if (values.every((v) => v instanceof Date)) return { kind: "spreadsheet-date", note: "Spreadsheet dates" };
  if (ok(parserForKind("iso"))) return { kind: "iso", note: "ISO dates (year first)" };
  const numeric = values.map(Number);
  if (numeric.every(Number.isFinite)) {
    const kinds = new Set(numeric.map(epochKind));
    if (kinds.size !== 1 || kinds.has(null)) return null;
    const kind = [...kinds][0];
    return { kind, note: kind === "epoch-ms" ? "Unix time in milliseconds" : "Unix time in seconds" };
  }
  if (ok(parserForKind("dmy")) || ok(parserForKind("mdy"))) {
    const parts = values.map((v) => DMY.exec(v)).filter(Boolean);
    const firstOver12 = parts.some((m) => +m[1] > 12);
    const secondOver12 = parts.some((m) => +m[2] > 12);
    const dayFirst = firstOver12 || !secondOver12;
    const note = firstOver12 || secondOver12 ? `${dayFirst ? "Day" : "Month"} first (from the values)` : "Day first (ambiguous in the file; assumed)";
    return { kind: dayFirst ? "dmy" : "mdy", note };
  }
  return null;
}

const two = (n) => String(n).padStart(2, "0");

/** "YYYY-MM-DD HH:MM:SS" for naive milliseconds. */
export function formatNaive(ms) {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}-${two(d.getUTCMonth() + 1)}-${two(d.getUTCDate())} ${two(d.getUTCHours())}:${two(d.getUTCMinutes())}:${two(d.getUTCSeconds())}`;
}

/** The visitor's current local time as "YYYY-MM-DDTHH:MM:SS". */
export function localNow() {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
}

/** Pandas-style timedelta text, e.g. "213 days 03:59:50". */
export function formatSpan(ms) {
  const total = Math.round(ms / 1000);
  const days = Math.floor(total / 86400);
  const rest = total % 86400;
  return `${days} days ${two(Math.floor(rest / 3600))}:${two(Math.floor((rest % 3600) / 60))}:${two(rest % 60)}`;
}
