import { el, fmt, footer, initShell, loadJSON, plainTable, showError } from "./app.js";

try {
  const [context, metrics, audit] = await Promise.all([loadJSON("context"), loadJSON("metrics"), loadJSON("audit")]);
  const { wrap } = initShell(
    "claims.html",
    "Claims and limits",
    "A prototype built on four reported events from one compressor. This page says what that supports, what it does not, and which words the project uses on purpose.",
  );
  const primary = metrics.rows.filter((r) => r.is_primary_score && r.percentile === context.primary_percentile);
  const by = Object.fromEntries(primary.map((r) => [r.model, r]));
  const rbm = primary.filter((r) => r.model.startsWith("rbm_k"));
  const rbmLow = Math.min(...rbm.map((r) => r.alert_window_fraction));
  const total = context.runtimes_seconds.run_all_steps + context.runtimes_seconds.step01_audit_raw_data;

  const list = (items) => el("ul", { class: "list" }, items.map((t) => el("li", {}, t)));
  wrap.append(
    el("div", { class: "split" },
      el("section", {},
        el("h3", {}, "What the evidence supports"),
        list([
          `The pipeline is reproducible. With the fixed seed (${context.seed}) a full re-run reproduced every threshold and metric exactly; all steps after the audit take about ${fmt.seconds(context.runtimes_seconds.run_all_steps)}.`,
          "Thresholds were fitted on a held-out part of the training month and written to disk before each model scored any test window.",
          "Nothing was fitted on test data: scalers, bin edges, state boundaries and network structure all come from the training windows.",
          `The Markov chain flags only ${fmt.pct(by.markov_state_model.alert_window_fraction, 2)} of test windows, so its detections of ${by.markov_state_model.events_detected} of 4 reported windows carry more weight than those of the other models.`,
          "Evaluation is at event level against the four reported windows, with unconfirmed alerts counted separately.",
        ]),
      ),
      el("section", {},
        el("h3", {}, "What it does not support"),
        list([
          el("span", {}, el("strong", {}, "Failure prediction. "), `Delays near −${context.lead_tolerance_hours} h are clipped to the start of the lead range. No model was tested on whether it warns before a failure.`),
          el("span", {}, el("strong", {}, "Causes. "), "Bayesian-network arrows are learned dependencies. Some directions flipped between identical runs."),
          el("span", {}, el("strong", {}, "That the RBM or VAE beat the baseline. "), `They did not: the baseline flags ${fmt.pct(by.zscore_baseline.alert_window_fraction)} of windows, the RBMs ${fmt.pct(rbmLow)} or more, the VAE ${fmt.pct(by.vae.alert_window_fraction)}, with no more reported windows detected.`),
          el("span", {}, el("strong", {}, "That detection is easy to win. "), `The detection ranges cover ${fmt.pct(metrics.detection_range_share)} of the test period, so frequent alerts overlap them by chance.`),
          el("span", {}, el("strong", {}, "General conclusions. "), "Four events, one machine, one training month."),
        ]),
      ),
    ),

    el("h2", {}, "Words used on purpose"),
    plainTable({
      columns: [
        { label: "Use", render: (r) => el("b", {}, r[0]) },
        { label: "Instead of", render: (r) => el("span", { class: "muted" }, r[1]) },
        { label: "Why", render: (r) => r[2] },
      ],
      rows: [
        ["Unconfirmed alert", "False alarm", "The reported windows are four listed intervals. An alert outside them is not known to be wrong."],
        ["Reported window", "Failure label", "Rows inside a window are not all faulty, and rows outside are not guaranteed normal."],
        ["Learned dependency", "Cause", "The network shows which sensor states vary together."],
        ["Unusual", "Faulty", "A score says a window differs from the training month, not why."],
        ["Candidate training period", "Fault-free month", "The first month holds no reported window, but nothing proves it was healthy."],
      ],
    }),

    el("h2", {}, "Limits of the data and the setup"),
    list([
      `The record has ${fmt.num(audit.days_without_readings, 1)} days with no readings, and ${fmt.pct(audit.frozen_percent / 100, 2)} of readings sit in frozen stretches that were excluded from scoring.`,
      "The training month is not representative of later months: the high motor-current state, the digital signals and the pressure signals all behave differently afterwards. Most models therefore flag far more windows than the 1% they were calibrated to.",
      "The RBM and the VAE were still improving when training stopped at the configured limits, and no tuning was done after seeing test results.",
      "The timezone of the timestamps and of the failure reports is not stated; both are assumed to be local time. UCI's own notes contain an unresolved date inconsistency for event 2.",
      "A dashboard like this one is a view of saved results; it does not run any model, and it cannot say anything about compressors other than the one recorded.",
    ]),

    el("h2", {}, "Source and reproduction"),
    el("p", {}, "Data: MetroPT-3 Dataset, UCI Machine Learning Repository, ", el("a", { href: "https://doi.org/10.24432/C5VW3R", rel: "noopener" }, "doi.org/10.24432/C5VW3R"), ", licensed CC BY 4.0. The raw file is not included here; download it from the ", el("a", { href: "https://archive.ics.uci.edu/dataset/791/metropt+3+dataset", rel: "noopener" }, "UCI page"), "."),
    el("p", {}, `Running the audit and the seven later steps took about ${fmt.seconds(total)} on the author's machine. The repository's README lists the commands, and its decision log records every choice with the numbers behind it.`),
    footer(context),
  );
} catch (error) {
  initShell("claims.html", "Claims and limits");
  showError(error);
}
