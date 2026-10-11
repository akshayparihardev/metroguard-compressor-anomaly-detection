import { lineChart } from "./charts.js";
import { compact, el, fmt, footer, initShell, isUpload, loadJSON, plainTable, showError } from "./app.js";

const RBM_COLOURS = { 1: "--brand", 5: "--alert", 10: "--ink-2" };
const asBool = (v) => String(v).toLowerCase() === "true";
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function heat(value) {
  const v = Number(value);
  return el("span", { class: "heat-cell num", style: `background:color-mix(in srgb, var(--brand) ${Math.round(v * 75)}%, transparent)` }, v.toFixed(3));
}

try {
  const [context, audit, training] = await Promise.all([loadJSON("context"), loadJSON("audit"), loadJSON("training")]);
  const upload = isUpload(context);
  const nWindows = context.windows.length;
  const { wrap } = initShell(
    "data.html",
    "Data and training",
    `What the ${upload ? "uploaded" : "raw"} file contains, where it is incomplete, and how the RBM and the VAE behaved while training.`,
  );
  const days = audit.span.split(" ")[0];
  const spike = training.vae.train_loss.reduce((best, v, i) => (v > best.v ? { v, i } : best), { v: -Infinity, i: 0 });
  const kValues = Object.keys(training.rbm).sort((a, b) => a - b);
  const rbmEpochs = training.rbm[kValues[0]].epoch;
  const bestEpoch = training.vae_best_epoch;
  const stateColumn = upload ? audit.state_column : "Motor_current";
  const amps = upload ? "" : " A";

  wrap.append(...compact(
    el("div", { class: "facts reveal" },
      el("div", { class: "fact" }, el("b", {}, fmt.int(audit.rows)), el("span", {}, upload ? `rows, ${audit.columns_used} of ${audit.columns} columns used` : `readings, ${audit.columns - 1} columns after dropping the index`)),
      el("div", { class: "fact" }, el("b", {}, `${audit.cadence_seconds} s`), el("span", {}, "usual spacing in the file")),
      el("div", { class: "fact" }, el("b", {}, `${days} days`), el("span", {}, `${fmt.date(Date.parse(audit.first_timestamp.replace(" ", "T") + "Z") / 60000)} to ${fmt.date(Date.parse(audit.last_timestamp.replace(" ", "T") + "Z") / 60000)}`)),
      el("div", { class: "fact" }, el("b", {}, `${fmt.num(audit.days_without_readings, 1)} days`), el("span", {}, "inside gaps with no readings")),
      el("div", { class: "fact" }, el("b", {}, `${fmt.num(audit.frozen_percent, 2)}%`), el("span", {}, `of readings in frozen stretches (${fmt.int(audit.frozen_readings)})`)),
    ),
    el("p", {}, upload
      ? `The most common spacing between readings is ${audit.cadence_seconds} s, and the analysis used ${fmt.windowLength(context.window_seconds)} windows. Timestamps were read as ${context.source.timestamp_note.toLowerCase()} and used exactly as written, with no timezone conversion; reported windows are assumed to use the same clock.`
      : `The source documentation states two different sampling rates (1 Hz in one place, 0.1 Hz in another). The file itself shows one reading every ${audit.cadence_seconds} s, with 9 to 13 s jitter. Timestamps carry no timezone; this project assumes the timestamps and the failure reports share the same local time.`),
    upload && context.source.notes.length ? el("div", { class: "note" }, context.source.notes.map((n) => el("p", {}, n))) : null,

    el("h2", {}, "Gaps and frozen stretches"),
    el("div", { class: "grid-2" },
      el("div", {}, plainTable({
        caption: "How the time steps are classified",
        note: "A gap is a step above twice the usual spacing; a long gap is at least one hour.",
        columns: [{ label: "Class", key: "class" }, { label: "Count", align: "right", render: (r) => fmt.int(r.count) }, { label: "Share of steps", align: "right", render: (r) => `${fmt.num(r["percent of deltas"], 3)}%` }],
        rows: audit.gap_classes,
      })),
      el("div", {}, plainTable({
        caption: "Ten longest gaps",
        columns: [{ label: "From", render: (r) => String(r.gap_start) }, { label: "To", render: (r) => String(r.gap_end) }, { label: "Hours", align: "right", render: (r) => fmt.num(r.duration_hours, 1) }],
        rows: audit.longest_gaps,
      })),
    ),
    el("p", {}, upload
      ? (audit.frozen_runs.length
          ? `In ${audit.frozen_runs.length} stretch${audit.frozen_runs.length === 1 ? "" : "es"} of an hour or more, every sensor repeats exactly the same value. That usually means stuck logging rather than machine behaviour, so windows overlapping these stretches are left out of scoring and evaluation.`
          : "No stretch of an hour or more was found in which every sensor repeats exactly the same value.")
      : "In seven stretches every analog sensor repeats exactly the same value for hours. That looks like stuck logging rather than machine behaviour, so windows overlapping these stretches are left out of scoring and evaluation."),
    audit.frozen_runs.length
      ? plainTable({
          caption: "Frozen-value stretches",
          columns: [{ label: "From", render: (r) => String(r.start) }, { label: "To", render: (r) => String(r.end) }, { label: "Readings", align: "right", render: (r) => fmt.int(r.readings) }, { label: "Hours", align: "right", render: (r) => fmt.num(r.duration_hours, 1) }],
          rows: audit.frozen_runs,
        })
      : null,

    nWindows ? el("h2", {}, upload ? `The ${plural(nWindows, "reported window")}` : "The four reported windows") : null,
    nWindows ? plainTable({
      caption: "Coverage of each reported window in the file",
      note: upload ? "The windows you entered, inclusive at both ends." : "The windows are company-reported intervals, not row-by-row labels. Inclusive at both ends.",
      columns: [
        { label: "Event", render: (r) => el("b", {}, String(r.event)) },
        { label: "Start", render: (r) => String(r.start) },
        { label: "End", render: (r) => String(r.end) },
        { label: "Hours", align: "right", render: (r) => fmt.num(r.duration_hours, 1) },
        { label: "Readings", align: "right", render: (r) => fmt.int(r.readings) },
        { label: "Coverage", align: "right", render: (r) => `${r.coverage_percent}%` },
        { label: "Longest silent stretch", align: "right", render: (r) => fmt.seconds(r.largest_gap_seconds) },
        { label: "Fully covered", render: (r) => (asBool(r.fully_covered) ? el("span", { class: "tag ok" }, "yes") : el("span", { class: "tag warn" }, "gaps inside")) },
      ],
      rows: audit.event_coverage,
    }) : null,

    audit.digital.length ? el("h2", {}, "Digital signals") : null,
    audit.digital.length
      ? el("p", {}, upload
          ? `${plural(audit.digital.length, "on/off signal")}. A signal that is almost always on or almost always off gives a model little to learn from.`
          : "Eight on/off signals, all strictly 0 or 1. In the training month several are almost constant, which matters for what a model can learn from them.")
      : null,
    audit.digital.length
      ? el("div", { class: "grid-2" },
          el("div", {}, plainTable({
            caption: "Share of readings with the signal on",
            columns: [
              { label: "Signal", render: (r) => el("b", {}, r.column) },
              { label: "Active", align: "right", render: (r) => el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.active_fraction, 2)), el("span", { class: "meter brand" }, el("i", { style: `width:${r.active_fraction * 100}%` }))) },
              { label: "Almost constant", render: (r) => (asBool(r.near_constant) ? el("span", { class: "tag warn" }, "yes") : el("span", { class: "muted" }, "no")) },
            ],
            rows: audit.digital_signals,
          })),
        )
      : null,
    audit.digital.length
      ? plainTable({
          caption: "Share on, by month",
          note: upload ? `The test period starts ${context.source.test_start.slice(0, 10)}.` : "February is the training month.",
          columns: [{ label: "Month", render: (r) => el("b", {}, String(r.month)) }, ...audit.digital.map((name) => ({ label: name.replace("_", " "), align: "right", render: (r) => heat(r[name]) }))],
          rows: audit.digital_by_month,
        })
      : null,

    stateColumn ? el("h2", {}, upload ? `Operating states of ${stateColumn}` : "Motor current") : null,
    stateColumn
      ? el("p", {}, upload
          ? `The Markov chain follows ${stateColumn} through ${plural(audit.motor_states.length, "operating state")}, with boundaries found from the training period alone. The table applies the same rule to the full record.`
          : "The motor current falls into three clear operating states, found from the training month alone. The 7 A and 9 A states sometimes mentioned for this compressor are not supported by this file.")
      : el("p", { class: "muted" }, "No sensor showed two or more distinct operating states in the training period, so the Markov chain was not fitted."),
    stateColumn
      ? plainTable({
          caption: "Operating states (provisional rule, full record)",
          columns: [
            { label: "State", render: (r) => el("b", {}, String(r.state)) },
            { label: upload ? "Typical value" : "Typical current", align: "right", render: (r) => `${fmt.num(r.peak_amps, 2)}${amps}` },
            { label: "Range", render: (r) => `${r.lower_cut_amps == null ? "below" : fmt.num(r.lower_cut_amps, 2)} to ${r.upper_cut_amps == null ? "above" : fmt.num(r.upper_cut_amps, 2)}${amps}`.replace("below to", "up to").replace("to above", "and above") },
            { label: "Share of readings", align: "right", render: (r) => el("span", { class: "cell-bar" }, el("span", { class: "num" }, fmt.pct(r.share_of_readings)), el("span", { class: "meter brand" }, el("i", { style: `width:${r.share_of_readings * 100}%` }))) },
          ],
          rows: audit.motor_states,
        })
      : null,

    el("h2", {}, "Training curves"),
    el("h3", {}, "RBM, trained with contrastive divergence"),
    el("p", {}, upload
      ? `Reconstruction error and mean free energy for k = ${kValues.join(", ")}, ${rbmEpochs.length} epochs. Solid lines are the training windows, dashed lines the held-out validation windows. Curves still falling at the last epoch mean the RBMs could have trained longer.`
      : `Reconstruction error and mean free energy for k = 1, 5 and 10 Gibbs steps, ${rbmEpochs.length} epochs. Solid lines are the training windows, dashed lines the held-out validation windows. The curves were still falling at the last epoch, so the RBMs are under-trained.`),
    el("div", { class: "grid-2" }, el("div", { id: "rbm-error" }), el("div", { id: "rbm-energy" })),
    el("h3", {}, "VAE"),
    el("p", {}, upload
      ? `The best validation loss came at epoch ${bestEpoch} of ${training.vae.epoch.length}, and the weights from that epoch were kept for scoring.`
      : `The best validation loss came at epoch ${bestEpoch} of ${training.vae.epoch.length}; the early-stopping rule never triggered. Training loss peaked at ${fmt.int(spike.v)} in epoch ${training.vae.epoch[spike.i]} and recovered. The cause is a near-constant digital signal that is almost always off in the training month, so its rare activations become extreme values after standardising.`),
    el("div", { id: "vae-loss" }),
    footer(context),
  ));

  const rbmSeries = (metric) =>
    Object.entries(training.rbm).flatMap(([k, h]) => [
      { label: `k = ${k}, training`, xs: h.epoch, ys: h[`fit_${metric}`], colour: RBM_COLOURS[k] },
      { label: `k = ${k}, validation`, xs: h.epoch, ys: h[`validation_${metric}`], colour: RBM_COLOURS[k], dash: [5, 4] },
    ]);
  lineChart(document.getElementById("rbm-error"), { title: "Reconstruction error", subtitle: "squared error per visible unit", series: rbmSeries("reconstruction_error"), xLabel: "epoch", yLabel: "error" });
  lineChart(document.getElementById("rbm-energy"), { title: "Mean free energy", subtitle: "lower means the data looks more typical to the model", series: rbmSeries("mean_free_energy"), xLabel: "epoch", yLabel: "free energy" });
  lineChart(document.getElementById("vae-loss"), {
    title: "VAE loss per window", subtitle: "squared error plus KL term, logarithmic axis",
    series: [
      { label: "training", xs: training.vae.epoch, ys: training.vae.train_loss, colour: "--brand" },
      { label: "validation", xs: training.vae.epoch, ys: training.vae.validation_loss, colour: "--alert", dash: [5, 4] },
    ],
    logY: true, xLabel: "epoch", yLabel: "loss", vline: { x: bestEpoch, label: `best epoch ${bestEpoch}` },
  });
} catch (error) {
  initShell("data.html", "Data and training");
  showError(error);
}
