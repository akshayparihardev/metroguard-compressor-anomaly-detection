// Turns an uploaded file into a table the analyse page can preview and map. CSV and text files are only
// sniffed here (the worker streams them in full); spreadsheets and PDF tables are read whole on this thread
// with libraries that load only when such a file is chosen.

import { detectColumns, sniffCsv } from "./analysis/table-reader.js";

export const ACCEPTED = ".csv,.tsv,.txt,.xlsx,.xls,.pdf";
const DETECT_ROWS = 400;
const CELL_GAP_EM = 0.9;
const SAME_WORD_EM = 0.1;
const SAME_LINE_EM = 0.5;

export function fileKind(name) {
  const extension = name.toLowerCase().split(".").pop();
  if (["csv", "tsv", "txt"].includes(extension)) return "csv";
  if (["xlsx", "xls"].includes(extension)) return "excel";
  if (extension === "pdf") return "pdf";
  return null;
}

const looksLikeData = (row) => row.filter((v) => v !== "" && v != null).every((v) => v instanceof Date || Number.isFinite(Number(v)));

/** Unique, non-empty column names. */
function columnNames(raw) {
  const seen = new Map();
  return raw.map((value, i) => {
    const base = String(value ?? "").trim() || `column_${i + 1}`;
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count ? `${base}_${count + 1}` : base;
  });
}

/** Unique names plus each column's detected role. Detection sees the raw header, so unnamed index columns are recognised. */
function describeColumns(rawHeader, rows) {
  const header = columnNames(rawHeader);
  const columns = detectColumns(rawHeader, rows.slice(0, DETECT_ROWS)).map((column, i) => ({ ...column, name: header[i] }));
  return { header, columns };
}

/** Header and data rows from records that may or may not start with a header row. */
function fromRecords(records) {
  const width = Math.max(...records.map((r) => r.length));
  const padded = records.map((r) => (r.length < width ? [...r, ...new Array(width - r.length).fill("")] : r));
  const hasHeader = !looksLikeData(padded[0]);
  const headerKey = JSON.stringify(padded[0]);
  const rows = hasHeader ? padded.slice(1).filter((r) => JSON.stringify(r) !== headerKey) : padded;
  return { ...describeColumns(hasHeader ? padded[0] : padded[0].map(() => ""), rows), rows };
}

let sheetJs;
function loadSheetJs() {
  sheetJs ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = new URL("./vendor/xlsx.mini.min.js", import.meta.url).href;
    script.onload = () => resolve(window.XLSX);
    script.onerror = () => reject(new Error("The spreadsheet reader could not be loaded. Check the connection and try again."));
    document.head.append(script);
  });
  return sheetJs;
}

let pdfJs;
function loadPdfJs() {
  pdfJs ??= import("./vendor/pdf.min.mjs").then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = new URL("./vendor/pdf.worker.min.mjs", import.meta.url).href;
    return lib;
  });
  return pdfJs;
}

/** Group a page's text pieces into lines (by height on the page) and cells (by horizontal gaps). */
function pageLines(items) {
  const pieces = items
    .filter((item) => item.str && item.str.trim() !== "")
    .map((item) => ({ text: item.str.trim(), x: item.transform[4], y: item.transform[5], width: item.width, size: Math.abs(item.transform[3]) || item.height || 10 }))
    .sort((a, b) => b.y - a.y || a.x - b.x);
  const lines = [];
  for (const piece of pieces) {
    const line = lines.at(-1);
    if (line && Math.abs(line.y - piece.y) <= piece.size * SAME_LINE_EM) line.pieces.push(piece);
    else lines.push({ y: piece.y, pieces: [piece] });
  }
  return lines.map(({ pieces: linePieces }) => {
    linePieces.sort((a, b) => a.x - b.x);
    const cells = [];
    let end = -Infinity;
    for (const piece of linePieces) {
      const gap = piece.x - end;
      if (cells.length && gap < piece.size * CELL_GAP_EM) cells[cells.length - 1] += (gap > piece.size * SAME_WORD_EM ? " " : "") + piece.text;
      else cells.push(piece.text);
      end = piece.x + piece.width;
    }
    return cells;
  });
}

async function readPdf(file, onStatus) {
  const lib = await loadPdfJs();
  const doc = await lib.getDocument({ data: new Uint8Array(await file.arrayBuffer()), isEvalSupported: false }).promise;
  const lines = [];
  for (let p = 1; p <= doc.numPages; p++) {
    onStatus(`Reading page ${p} of ${doc.numPages}`);
    const page = await doc.getPage(p);
    lines.push(...pageLines((await page.getTextContent()).items));
    page.cleanup();
  }
  const pages = doc.numPages;
  await doc.destroy();
  const counts = new Map();
  for (const cells of lines) if (cells.length >= 2) counts.set(cells.length, (counts.get(cells.length) ?? 0) + 1);
  if (!counts.size) throw new Error("No table with two or more columns was found. Only PDFs with selectable text can be read, not scanned images.");
  const width = [...counts.entries()].sort((a, b) => b[1] - a[1] || b[0] - a[0])[0][0];
  const records = lines.filter((cells) => cells.length === width);
  return { ...fromRecords(records), pages, skippedLines: lines.length - records.length };
}

async function readExcel(file, onStatus) {
  onStatus("Loading the spreadsheet reader");
  const XLSX = await loadSheetJs();
  onStatus("Reading the workbook");
  const workbook = XLSX.read(await file.arrayBuffer(), { type: "array", cellDates: true, dense: true });
  const sheet = (name) => {
    const records = XLSX.utils.sheet_to_json(workbook.Sheets[name], { header: 1, raw: true, defval: "", blankrows: false });
    if (records.length < 2) throw new Error(`The sheet "${name}" has fewer than two rows.`);
    return { ...fromRecords(records), sheet: name };
  };
  const looksLikeSeries = (table) => table.columns.some((c) => c.role === "timestamp") && table.columns.some((c) => c.role === "analog");
  let first = null;
  for (const name of workbook.SheetNames) {
    let table;
    try {
      table = sheet(name);
    } catch {
      continue;
    }
    first ??= table;
    if (looksLikeSeries(table)) { first = table; break; }
  }
  if (!first) throw new Error("No sheet in the workbook has two or more rows.");
  return { sheets: workbook.SheetNames, chooseSheet: sheet, ...first };
}

/**
 * Read a chosen file into { kind, header, rows, columns, ... }. For CSV, `rows` holds only the preview and
 * `csv` carries what the worker needs to stream the rest.
 */
export async function readUploadedFile(file, onStatus = () => {}) {
  const kind = fileKind(file.name);
  if (!kind) throw new Error("Choose a CSV, TSV or text file, an Excel workbook (.xlsx or .xls) or a PDF.");
  if (kind === "csv") {
    onStatus("Reading the first rows");
    const sniff = await sniffCsv(file);
    return { kind, ...describeColumns(sniff.header, sniff.rows), rows: sniff.rows, csv: { delimiter: sniff.delimiter, hasHeader: sniff.hasHeader } };
  }
  const table = kind === "excel" ? await readExcel(file, onStatus) : await readPdf(file, onStatus);
  if (!table.rows.length) throw new Error("No data rows were found under the header.");
  return { kind, ...table };
}

/** The same workbook with another sheet chosen. */
export const withSheet = (table, name) => ({ ...table, ...table.chooseSheet(name) });
