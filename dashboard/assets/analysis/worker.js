// Runs the analysis off the main thread so the page stays responsive. The page posts one "run" message and
// receives "progress", then "done" (the dashboard bundle) or "error".

import { runPipeline } from "./pipeline.js";
import { readCsvColumns, rowsToColumns } from "./table-reader.js";

const READ_SHARE = 0.12;

self.onmessage = async ({ data }) => {
  const { source, mapping, meta, options } = data;
  const progress = (stage, fraction, message) => self.postMessage({ type: "progress", stage, fraction, message });
  try {
    progress("read", 0, "Reading the file");
    const columns = source.kind === "csv"
      ? await readCsvColumns(source.file, source, mapping, (f) => progress("read", READ_SHARE * f, "Reading the file"))
      : rowsToColumns(source.rows, mapping);
    const notes = columns.badTimestamps
      ? [`${columns.badTimestamps.toLocaleString("en-GB")} of ${columns.rowsRead.toLocaleString("en-GB")} rows had a timestamp that could not be read and were skipped.`]
      : [];
    const bundle = await runPipeline(columns, { ...meta, rowsRead: columns.rowsRead, notes }, options,
      (stage, fraction, message) => progress(stage, READ_SHARE + (1 - READ_SHARE) * fraction, message));
    self.postMessage({ type: "done", bundle });
  } catch (error) {
    self.postMessage({ type: "error", message: String(error?.message ?? error) });
  }
};
