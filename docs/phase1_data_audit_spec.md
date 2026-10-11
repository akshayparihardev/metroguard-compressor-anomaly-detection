# MetroGuard — Step 1 Data Audit Specification

## Project

**Title:** MetroGuard: Real-World Anomaly Detection in Metro Air-Compressor Telemetry Using Bayesian Networks, RBMs and VAEs

**Today's objective:** Audit the raw MetroPT-3 dataset and produce evidence that will guide modelling decisions. Do not train models or implement anomaly detection in step 1. Do not claim that any failure can already be detected.

## Context and scope

MetroPT-3 is a real multivariate time-series dataset collected from analogue and digital sensors on a metro train's Air Production Unit (air compressor). The UCI release lists 1,516,948 observations and 15 signals; the CSV is approximately 208 MB. UCI metadata has inconsistent sampling-rate descriptions (1 Hz in one field and 0.1 Hz in another), so derive the actual timestamp cadence from the downloaded file rather than assuming either value.

- Official source: https://archive.ics.uci.edu/dataset/791/metropt+3+dataset
- DOI: https://doi.org/10.24432/C5VW3R
- License: CC BY 4.0; retain attribution in the project report and README.

The company-provided failure intervals listed by UCI are:

| Event | Start | End | Reported issue | UCI maintenance note (as published) |
|---|---|---|---|---|
| 1 | 2020-04-18 00:00:00 | 2020-04-18 23:59:00 | Air leak | — |
| 2 | 2020-05-29 23:30:00 | 2020-05-30 06:00:00 | Air leak | "Maintenance on 30Apr at 12:00" |
| 3 | 2020-06-05 10:00:00 | 2020-06-07 14:30:00 | Air leak | "Maintenance on 8Jun at 16:00" |
| 4 | 2020-07-15 14:30:00 | 2020-07-15 19:00:00 | Air leak | "Maintenance on 16Jul at 00:00" |

Treat these as reported event windows, not dense row-by-row ground-truth labels. Record any metadata ambiguity; do not silently resolve inconsistencies or infer causes. Known ambiguities to note in the report:

- UCI numbers two events as "#1"; this spec renumbers them 1–4 for convenience only.
- The maintenance note for event 2 ("30Apr") precedes the event itself (29–30 May); it may be a typo for 30 May. Do not correct it silently — report it.
- The timezone of the timestamps and of the failure reports is not stated. Assume both use the same local time, and state this assumption explicitly.

## Step 1 tasks

### 1. Set up the project skeleton

Create the repository layout, naming rules, `.gitignore`, `pyproject.toml`, `requirements.txt` and `config.yaml` exactly as described in `docs/project_spec.md` sections 2, 3 and 5. For step 1 only these files need real content: `pipeline/step01_audit_raw_data.py`, `src/metroguard/config_loader.py`, `src/metroguard/raw_data_loader.py`, `src/metroguard/failure_windows.py`, `src/metroguard/plotting.py`. Do not create empty placeholder modules for later steps.

Keep the raw CSV out of Git. Use Python with pandas, numpy, matplotlib and pyarrow. Prefer straightforward, reproducible code over extra frameworks.

### 2. Obtain and load the official data

Download `MetroPT3(AirCompressor).csv` from the official UCI dataset page above and place it in `data/raw/`. Do not download a transformed copy from an unofficial mirror. Make the script fail with a helpful message if the raw file is missing; do not silently generate or substitute synthetic data.

Load the CSV without modifying the original file. Save a typed parquet cache to `data/processed/` so later steps never re-read the CSV. Print and save:

- Actual file path and file size.
- Shape, column names and dtypes.
- First and last five rows.
- Parsed timestamp minimum/maximum and total time span.
- Number of rows, duplicate rows, duplicate timestamps, missing values per column, and non-monotonic timestamps before sorting.
- Numeric summary statistics for sensor columns.

Identify the timestamp column by its actual header. Parse it as datetime. Preserve the original input; make a separate working dataframe for analysis. If the file contains an unnamed index column (e.g. `Unnamed: 0`), report it and exclude it from sensor analysis.

### 3. Audit timestamp cadence and gaps

Sort the working dataframe by timestamp for time-series analysis, while reporting whether the original rows were already ordered. Compute consecutive timestamp differences and report:

- Minimum, median, mean and mode of positive time differences.
- Counts/percentages for common time-difference values.
- Number of zero, negative and unusually large intervals.
- A defensible gap definition, explicitly stated in the report. For example, compare each interval with the dominant/median cadence; do not declare every non-median interval a data error.
- A histogram or bar chart of timestamp deltas, excluding invalid/non-positive deltas from the plot but reporting them separately.
- A simple timeline or daily row-count plot showing data coverage and longer gaps.

Do not resample, interpolate, forward-fill or downsample the primary dataframe in step 1. If a small diagnostic sample is needed for plotting speed, clearly label it as display-only.

### 4. Audit the digital signals

The eight digital signals (COMP, DV_eletric/DV_electric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulses — use the actual headers) are expected to be binary. Later, these may become RBM visible units, so verify rather than assume:

- Unique values and their counts for each digital column.
- Whether each column is strictly binary (0/1). Report any non-binary values exactly as found.
- The fraction of time each signal is active (value 1), overall and per month.
- Flag any digital column that is almost always constant (e.g. active < 0.5% or > 99.5% of the time), since it would carry little information for modelling.

Do not recode, threshold or drop any column in step 1; only report.

### 5. Inspect the four reported failure windows

Use the exact event windows in the table above. For each interval:

- Count observations within the interval and state whether the raw data covers the full interval.
- Plot the available analogue signals over time, with each reported event window visibly marked. At minimum include Motor_current, TP2, TP3, H1, DV_pressure, Reservoirs, and Oil_temperature if those exact columns exist.
- Create focused plots around each failure interval (with a suitable context window before and after it) so the plot remains readable.
- Note visible deviations descriptively only. Do not conclude that a visual change proves a failure or that a specific sensor caused it.
- Clearly distinguish missing event-window coverage from a model failure; no model exists yet.

If a column is absent or named differently, inspect the actual headers and map it transparently; do not silently substitute a different signal.

### 6. Check the planned chronological split

UCI recommends using the first month for training and the rest for testing. Without training anything:

- Report the exact date range and row count of the candidate training period (the first calendar month present in the data, and alternatively the first 30 days from the first timestamp).
- Confirm that no reported failure window overlaps the candidate training period.
- Report the row count of the remaining test period and how many of its rows fall inside the reported failure windows.

This only documents the split; do not create train/test files or fit any scaler in step 1.

### 7. Investigate motor-current operating states

Inspect the actual Motor_current column if present:

- Plot a histogram/density-style distribution and a time-series sample.
- Report descriptive statistics, common rounded values, unique-value count (or a useful approximation if needed), and quantiles.
- Explore whether distinct operating regimes are visible. The often-mentioned approximate 0 A / 4 A / 7 A / 9 A regimes are hypotheses to test, not labels to hard-code.
- If the data supports candidate states, document a transparent provisional rule or cluster summary and show examples. Do not finalize a state discretisation or train the Markov chain in step 1.

### 8. Produce a concise audit report

Write `reports/data_audit_report.md` with:

1. Dataset source, DOI, license and attribution.
2. File/schema summary and a brief column inventory separating analogue and digital signals where supported by source metadata.
3. Timestamp audit and observed cadence; explicitly mention the UCI metadata's conflicting 1 Hz / 0.1 Hz descriptions and state what the file itself shows.
4. Missing/duplicate/order/gap findings.
5. Digital-signal audit findings (binary or not, activity rates, near-constant columns).
6. The four event windows, data coverage for each, and links/relative paths to generated plots.
7. Candidate chronological split: date ranges, row counts, and confirmation that training contains no reported failure window.
8. Motor-current distribution findings and whether distinct candidate operating states are supported.
9. Data quality caveats (including the metadata ambiguities listed above), unanswered questions, and concrete recommendations for step 2.
10. A clear statement: no model was trained and no detection-performance claim is made in this audit.

Save plots under `reports/figures/data_audit/` with descriptive filenames, for example `timestamp_delta_distribution.png`, `daily_row_counts.png`, `digital_signal_activity.png`, `motor_current_histogram.png`, `motor_current_timeseries.png`, and one plot per event window. Use readable axes, units where known, titles and legends. Do not use misleading y-axis truncation without disclosure.

## Guardrails

- This is an exploration-only task. Do not implement or train the Bayesian Network, RBM, VAE, GAN, NADE, MADE, a baseline detector, or a Markov state-transition model today.
- Do not select anomaly-score thresholds using failure windows. When models are introduced later, thresholds must be selected using training data only (for example, a training-normal 99th percentile if justified), then frozen before test evaluation.
- Preserve chronological evaluation later; do not randomly shuffle time-series rows into train and test sets.
- Do not treat all rows inside a reported failure interval as equally and perfectly labelled ground truth.
- Do not claim causation from sensor associations.
- Never claim to have run code or inspected the dataset unless execution actually succeeded. If download or parsing fails, report the exact blocker and leave a reproducible next step.
- Keep the work small enough to complete in one day. Avoid a UI, dashboard, API, Docker, or elaborate package architecture at this stage.

## Acceptance criteria

Step 1 is complete only when:

- [ ] The official raw CSV is available locally and is not committed to Git.
- [ ] `python pipeline/step01_audit_raw_data.py` runs reproducibly on the raw CSV.
- [ ] A concise audit report with real computed values exists at `reports/data_audit_report.md`.
- [ ] Timestamp cadence and gaps are measured rather than assumed.
- [ ] Every digital signal's value set and activity rate is reported.
- [ ] All four reported event windows have coverage counts and readable plots (including explicit notes if any are not covered by the file).
- [ ] The candidate training period is documented and confirmed free of reported failure windows.
- [ ] Motor-current distribution is plotted and candidate states are treated as a question, not an assumption.
- [ ] Figures are saved under `reports/figures/data_audit/`, and the parquet cache exists in `data/processed/`.
- [ ] No model training or unsupported performance/causality claim was added.

## Suggested commands

Create a virtual environment using the method appropriate for the user's OS, activate it, then install the small dependency set. Install the pinned versions from `requirements.txt`, then `pip install -e .` so `src/metroguard` is importable.

Run:

```
python pipeline/step01_audit_raw_data.py
```

At the end, print a short terminal summary of generated outputs and the most important measured data-quality findings.
