// Reading tabular data: a quote-aware streaming CSV parser, delimiter sniffing, column-type detection, and
// conversion of the chosen columns into typed arrays. Works on a File/Blob in the browser and in Node.

import { detectTimestampFormat, parserForKind } from "./timestamps.js";

const DELIMITERS = [",", ";", "\t", "|"];
const SNIFF_BYTES = 1 << 20;
const PREVIEW_ROWS = 400;

/** Parse CSV text chunks into records; `onRecord(fields)` is called once per row. Handles quotes and CRLF. */
export function createCsvParser(delimiter, onRecord) {
  let field = "";
  let record = [];
  let inQuotes = false;
  let pendingQuote = false;
  const delim = delimiter.charCodeAt(0);
  const endField = () => { record.push(field); field = ""; };
  const endRecord = () => {
    endField();
    if (!(record.length === 1 && record[0] === "")) onRecord(record);
    record = [];
  };
  return {
    push(chunk) {
      let start = 0;
      for (let i = 0; i < chunk.length; i++) {
        const c = chunk.charCodeAt(i);
        if (pendingQuote) {
          pendingQuote = false;
          if (c === 34) { field += '"'; start = i + 1; continue; }
          inQuotes = false;
          start = i;
        }
        if (inQuotes) {
          if (c === 34) { field += chunk.slice(start, i); pendingQuote = true; start = i + 1; }
          continue;
        }
        if (c === 34 && field === "" && start === i) { inQuotes = true; start = i + 1; continue; }
        if (c === delim) { field += chunk.slice(start, i); endField(); start = i + 1; }
        else if (c === 10) { field += chunk.slice(start, i); if (field.endsWith("\r")) field = field.slice(0, -1); endRecord(); start = i + 1; }
      }
      field += chunk.slice(start);
    },
    end() {
      if (pendingQuote) inQuotes = false;
      if (field !== "" || record.length) { if (field.endsWith("\r")) field = field.slice(0, -1); endRecord(); }
    },
  };
}

function sniffDelimiter(lines) {
  let best = ",";
  let bestScore = -1;
  for (const d of DELIMITERS) {
    const counts = lines.slice(0, 20).map((l) => l.split(d).length - 1);
    const consistent = counts.every((c) => c === counts[0]);
    const score = counts[0] > 0 ? counts[0] * (consistent ? 2 : 1) : -1;
    if (score > bestScore) { best = d; bestScore = score; }
  }
  return best;
}

const isNumber = (v) => v !== "" && v != null && !(v instanceof Date) && Number.isFinite(Number(v));

/** Guess each column's role from sample rows: timestamp, sensor (numeric), on-off (0/1), or text. */
export function detectColumns(header, rows) {
  return header.map((name, index) => {
    const values = rows.map((r) => r[index]).filter((v) => v !== "" && v != null);
    const unnamedIndex = /^(unnamed[: _]*\d*|index|)$/i.test(String(name).trim());
    const timestamp = detectTimestampFormat(values.slice(0, 200));
    const numeric = values.length && values.filter(isNumber).length / values.length >= 0.98;
    const binary = numeric && values.every((v) => Number(v) === 0 || Number(v) === 1);
    let role = "ignore";
    if (timestamp && !(numeric && !timestamp.kind.startsWith("epoch"))) role = "timestamp";
    else if (binary) role = "digital";
    else if (numeric && !unnamedIndex) role = "analog";
    return { name: String(name).trim() || `column_${index + 1}`, index, role, timestamp, numeric, binary };
  });
}

/** Read the first ~1 MB of a CSV file: delimiter, header and preview rows. */
export async function sniffCsv(blob) {
  const text = await blob.slice(0, SNIFF_BYTES).text();
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) throw new Error("The file has fewer than two non-empty lines, so there is nothing to analyse.");
  const delimiter = sniffDelimiter(lines);
  const records = [];
  const parser = createCsvParser(delimiter, (r) => { if (records.length <= PREVIEW_ROWS) records.push(r); });
  parser.push(text.slice(0, text.lastIndexOf("\n") + 1));
  parser.end();
  const headerLooksNumeric = records[0].every(isNumber);
  const header = headerLooksNumeric ? records[0].map((_, i) => `column_${i + 1}`) : records[0];
  const rows = headerLooksNumeric ? records : records.slice(1);
  return { delimiter, header, rows, hasHeader: !headerLooksNumeric };
}

/** Growable typed array. */
function growable(Type) {
  let data = new Type(1 << 16);
  let length = 0;
  return {
    push(v) {
      if (length === data.length) { const next = new Type(data.length * 2); next.set(data); data = next; }
      data[length++] = v;
    },
    done: () => data.slice(0, length),
  };
}

/** Builds typed columns from records according to the mapping. */
export function createColumnBuilder(mapping) {
  const time = growable(Float64Array);
  const analog = mapping.analog.map(() => growable(Float32Array));
  const digital = mapping.digital.map(() => growable(Float32Array));
  let rowsRead = 0;
  let badTimestamps = 0;
  const parse = parserForKind(mapping.timestampKind);
  return {
    add(record) {
      rowsRead++;
      const t = parse(record[mapping.timestampIndex]);
      if (!Number.isFinite(t)) { badTimestamps++; return; }
      time.push(t);
      mapping.analog.forEach((c, i) => { const v = record[c.index]; analog[i].push(v === "" || v == null ? NaN : Number(v)); });
      mapping.digital.forEach((c, i) => { const v = record[c.index]; digital[i].push(v === "" || v == null ? NaN : Number(v)); });
    },
    finish() {
      return {
        time: time.done(),
        analog: Object.fromEntries(mapping.analog.map((c, i) => [c.name, analog[i].done()])),
        digital: Object.fromEntries(mapping.digital.map((c, i) => [c.name, digital[i].done()])),
        rowsRead,
        badTimestamps,
      };
    },
  };
}

/** Stream a whole CSV file into typed columns, reporting progress as a fraction of bytes read. */
export async function readCsvColumns(blob, { delimiter, hasHeader }, mapping, onProgress = () => {}) {
  const builder = createColumnBuilder(mapping);
  let first = hasHeader;
  const parser = createCsvParser(delimiter, (record) => { if (first) { first = false; return; } builder.add(record); });
  const reader = blob.stream().pipeThrough(new TextDecoderStream()).getReader();
  let bytes = 0;
  let lastReport = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    parser.push(value);
    bytes += value.length;
    if (bytes - lastReport > 4_000_000) { lastReport = bytes; onProgress(Math.min(0.99, bytes / blob.size)); }
  }
  parser.end();
  onProgress(1);
  return builder.finish();
}

/** Typed columns from an in-memory table (spreadsheets and PDF tables). */
export function rowsToColumns(rows, mapping) {
  const builder = createColumnBuilder(mapping);
  for (const row of rows) builder.add(row);
  return builder.finish();
}
