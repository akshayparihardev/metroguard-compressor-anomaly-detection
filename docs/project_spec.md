# MetroGuard — Master Project Specification (End-to-End)

**Title:** MetroGuard: Real-World Anomaly Detection in Metro Air-Compressor Telemetry Using Bayesian Networks, RBMs and VAEs
**Course:** Deep Learning II (CDT7001), Semester VII. One-week prototype.
**Author:** Akshay Parihar

This file is the single source of truth for the whole project. `docs/phase1_data_audit_spec.md` is the detailed spec for step 1 and stays valid. `CLAUDE.md` (repo root) repeats the working rules for coding sessions. Read all three before writing any code.

---

## 0. Goal in one paragraph

Learn the normal behaviour of a real metro-train air compressor from sensor data (MetroPT-3, UCI), score how unusual each later period is with several probabilistic models (z-score baseline, RBM trained with Gibbs sampling + Contrastive Divergence, VAE, optional Markov state model), turn scores into alerts using thresholds fixed on training data only, give each alert probabilistic context with a Bayesian Network, and evaluate alerts against the four company-reported air-leak intervals. The deliverable is reproducible code, figures, a metrics table and an honest report.

**Scope lock.** Build exactly what this spec describes. No extra models, no API, no external AI services, no extra datasets, no hyperparameter searches beyond the values listed in `config.yaml`. The single permitted exception is the read-only static dashboard in `dashboard/` (step 10), approved by the author as a deliberate scope extension: it only displays files the pipeline already produced. If something outside this spec looks useful, write it under "Possible future work" in the decision log instead of building it.

## 1. Non-negotiable rules (apply in every step)

1. **Real data only.** Use the official `MetroPT3(AirCompressor).csv` in `data/raw/`. Never generate or substitute synthetic data. If the file is missing, stop with a clear message.
2. **Chronological split.** Never shuffle rows across train/test. Test data must not influence any fitted object (scaler, bins, mean/std, model weights, BN CPDs, thresholds, window choices).
3. **Thresholds from training data only.** Fit models on the *fit* part of the training period; compute thresholds on the held-out *calibration* tail of the training period (see step 2). Freeze thresholds before any test scoring. Never tune anything by looking at failure windows.
4. **Failure windows are reported intervals, not dense labels.** Evaluate at event level. Alerts outside reported windows are called **unconfirmed alerts**, never "false alarms".
5. **No causal claims.** The BN shows probabilistic association between sensor states, not causes.
6. **No prediction claims.** We evaluate anomaly detection. Detection delay may be negative (alert before the window) and that is reported as a finding, not a guarantee.
7. **Implement RBM training ourselves.** Gibbs sampling, CD-k and free energy are written by us (NumPy or PyTorch tensors). Do not use `sklearn.neural_network.BernoulliRBM` or any packaged RBM.
8. **Report, don't hide.** If a model underperforms the baseline, or a step is blocked, say so in the report with numbers.
9. **Never claim to have run something that did not run.** Every number in reports must come from an actual execution saved under `reports/tables/` or `outputs/`.
10. **Reproducibility.** Fixed random seeds, all tunable values in `config.yaml`, one command per step.

## 2. Repository structure and naming

### 2.1 Layout

```
DL 2 Project/                         # repo root: only what a reader needs first
├── README.md                         # what the project is, setup, how to run each step
├── CLAUDE.md                         # working rules for coding sessions
├── config.yaml                       # every tunable value
├── pyproject.toml                    # package metadata, tool config (ruff, pytest)
├── requirements.txt                  # pinned dependency versions
├── .gitignore
├── docs/                             # specs and explanations (written by us, not generated)
│   ├── project_spec.md               # this file
│   ├── phase1_data_audit_spec.md     # detailed spec for step 1
│   ├── claude_code_prompts.md        # copy-paste prompts for each step (for the human)
│   └── MetroGuard_Complete_Project_Explained_Updated.pdf
├── data/                             # git-ignored except .gitkeep files
│   ├── raw/                          # official CSV, never modified
│   └── processed/                    # cached parquet / npz built from raw
├── src/metroguard/                   # reusable library code, no top-level execution
│   ├── __init__.py
│   ├── config_loader.py              # reads config.yaml into a typed object
│   ├── step_run_tracking.py          # skip-if-up-to-date check + per-step runtime logging
│   ├── raw_data_loader.py            # reads the CSV once, parses timestamps, caches parquet
│   ├── failure_windows.py            # the four reported intervals + overlap helpers
│   ├── chronological_split.py        # fit / calib / test split by time
│   ├── feature_builder.py            # windowing, scaling, discretisation, binarisation
│   ├── zscore_baseline.py            # z-score anomaly scores
│   ├── restricted_boltzmann_machine.py  # RBM, Gibbs sampling, CD-k, free energy
│   ├── variational_autoencoder.py    # VAE model, training loop, reconstruction scores
│   ├── dashboard_export.py           # step 10: shapes saved results into dashboard JSON
│   ├── markov_state_model.py         # optional motor-current state transitions
│   ├── bayesian_network.py           # BN structure, CPDs, queries, alert context
│   ├── alert_thresholds.py           # threshold fitting on calib + freezing to disk
│   ├── alert_events.py               # score -> alert -> merged alert events
│   ├── event_evaluation.py           # event-level metrics against reported windows
│   └── plotting.py                   # all figure functions, consistent style
├── pipeline/                         # one runnable script per step, thin wrappers around src/
│   ├── step01_audit_raw_data.py
│   ├── step02_build_features.py
│   ├── step03_score_zscore_baseline.py
│   ├── step04_train_rbm.py
│   ├── step05_train_vae.py
│   ├── step06_fit_markov_states.py   # optional
│   ├── step07_build_bayesian_network.py
│   ├── step08_evaluate_alerts.py
│   ├── run_all_steps.py              # runs steps 2–8 in order, skipping up-to-date ones
│   └── step10_build_dashboard_data.py  # exports saved results to dashboard/data (step 10, extension)
├── tests/                            # pytest, fast (< 1 minute total), no raw CSV needed
│   ├── test_failure_windows.py
│   ├── test_restricted_boltzmann_machine.py
│   ├── test_chronological_split.py
│   ├── test_alert_thresholds.py
│   └── test_alert_events.py
├── dashboard/                        # read-only static site (Vercel), step 10 extension; data/ is generated
├── outputs/                          # machine artifacts, git-ignored, always regenerable
│   ├── models/                       # trained weights (.pt / .npz), fitted scaler, bin edges
│   ├── scores/                       # per-window anomaly scores per model (parquet)
│   └── logs/                         # per-step run logs incl. runtime
└── reports/                          # human-readable results, committed
    ├── data_audit_report.md          # step 1
    ├── decision_log.md               # every decision + reason, step by step
    ├── final_report.md               # step 9
    ├── tables/                       # thresholds.json, metrics.csv, metrics.md
    └── figures/                      # one subfolder per step: data_audit/, rbm/, vae/, evaluation/, bayesian_network/
```

### 2.2 Naming rules

- Every file and folder name says exactly what it contains or does. No `utils.py`, `helpers.py`, `misc/`, `temp/`, `new/`, `final_v2`, `test2.py`, `untitled`.
- Python: `snake_case` for files, functions and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants.
- Library modules in `src/metroguard/` are named after the thing they contain (`variational_autoencoder.py`). Pipeline scripts are named `stepNN_<verb>_<object>.py` after the action they perform (`step04_train_rbm.py`).
- Figures: `<subject>_<what_it_shows>.png`, e.g. `rbm_free_energy_vs_time_k5.png`, `event2_zoom_vae_reconstruction_error.png`.
- Tables: `<subject>.<ext>`, e.g. `metrics.csv`, `thresholds.json`.
- Use full names over abbreviations, except standard ones in the field: `rbm`, `vae`, `bn`, `cd_k`, `calib`.

## 3. Engineering practices

### 3.1 Clean code
- Library code lives in `src/metroguard/`; pipeline scripts only parse arguments, call library functions, save outputs and log a summary. No logic duplicated between scripts.
- Type hints and a one-line docstring on every public function. Small functions with one job.
- Paths via `pathlib.Path` and `config.yaml`; never hard-coded absolute paths. The project runs on Windows, so no shell-only commands in the pipeline.
- `logging` for progress messages; each step ends with a short summary of what it wrote.
- Format and lint with `ruff` (config in `pyproject.toml`). Remove dead code, unused imports and commented-out blocks before finishing a step.
- No notebooks as the source of truth. A notebook may exist only in `notebooks/` for exploration and is never imported or required to reproduce results.
- No duplicate copies of files anywhere in the repo.

### 3.2 Green (resource-efficient) computing
- Read the 208 MB CSV **once** in step 1/2, save a typed parquet cache to `data/processed/`, and have every later step read the cache.
- Use `float32` for features and model tensors; use categorical/bool dtypes for digital signals.
- Work on windowed data (step 2), never on the raw 1.5 M rows inside models.
- Each step skips work when its outputs are newer than its inputs and the relevant config values are unchanged; `--force` recomputes.
- Train only the configurations listed in `config.yaml` (e.g. RBM k = 1, 5, 10). No grid searches.
- Small models and early stopping on `fit_validation`. Use the GPU only when it actually speeds up training; the RBM may run on CPU.
- Log wall-clock runtime per step in `outputs/logs/`; mention total runtime in the final report.
- Never commit data, model weights or caches.

### 3.3 Version control
- `git init` at the start. Commit after each completed step with a message like `step04: train RBM with CD-k (k=1,5,10)`.
- `.gitignore`: `data/raw/*`, `data/processed/*` (keep `.gitkeep`), `outputs/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `*.pt`, `.ipynb_checkpoints/`.
- Pinned versions in `requirements.txt`.

## 4. Steps

Each step: implement → run → save outputs → add an entry to `reports/decision_log.md` (what was done, key numbers, decisions and why) → run `pytest` and `ruff` → commit → continue. **Checkpoint:** after step 1, stop and show the audit summary plus the proposed step 2 decisions, because preprocessing depends on what the audit finds. After approval, continue through the remaining steps, stopping only on a real blocker.

### Step 1 — Audit raw data (`step01_audit_raw_data.py`)
Follow `docs/phase1_data_audit_spec.md` exactly. No models. Outputs: `reports/data_audit_report.md`, `reports/figures/data_audit/`, and the parquet cache in `data/processed/`.

### Step 2 — Build features (`step02_build_features.py`)
**Split (by time):**
- `train` = candidate training period from the audit (default: first calendar month present). Confirm zero overlap with reported failure windows.
- Inside `train`: `fit` = first ~80% by time, `calib` = last ~20% by time. Models fit on `fit`; thresholds computed on `calib` (scores on data the model was not fitted on, so thresholds are not artificially low).
- Inside `fit`: the last ~10% by time is `fit_validation`, used only for early stopping and training curves. `calib` is never used for any training decision, so it stays clean for thresholds.
- `test` = everything after `train`.

**Windowing:** aggregate raw readings into fixed time windows (default 60 s; final value chosen from the audited cadence and written to `config.yaml` with the reason). Per window: mean of analog sensors, plus std for Motor_current and the main pressures if useful; fraction-active for digital signals. Drop or flag windows with too few raw readings (threshold in config); never interpolate across long gaps.

**Representations (all fitted on `fit` only):**
| Consumer | Representation |
|---|---|
| Baseline | analog window means, mean/std from `fit` |
| VAE | analog + digital features, standardised with a scaler fitted on `fit` |
| RBM | binary visible vector: each digital signal → 1 if active fraction ≥ 0.5; each analog feature → one-hot over quantile bins (default 3: low/normal/high, cut points from `fit`) |
| BN | the same discrete states as the RBM bins, as categorical variables |
| Markov | motor-current operating state per window (only if the audit supports distinct states) |

Save processed arrays with timestamps to `data/processed/`. Save scaler and bin edges to `outputs/models/`.

### Step 3 — Z-score baseline (`step03_score_zscore_baseline.py`)
Score per window = max over analog features of |z|, z computed with `fit` mean/std. Also save the mean |z| variant. Thresholds per the threshold rules below.

### Step 4 — RBM with Gibbs sampling and CD-k (`step04_train_rbm.py`)
- Bernoulli–Bernoulli RBM. Energy E(v,h) = −aᵀv − bᵀh − vᵀWh.
- Free energy F(v) = −aᵀv − Σⱼ log(1 + exp(bⱼ + (vᵀW)ⱼ)). Use a numerically stable softplus.
- p(h=1|v) = σ(b + Wᵀv), p(v=1|h) = σ(a + Wh).
- CD-k: positive phase from data, negative phase from k Gibbs steps starting at the data; update ΔW ∝ ⟨vhᵀ⟩_data − ⟨vhᵀ⟩_model, same for biases. Mini-batches, learning rate, momentum/weight decay in config.
- Train three models: k = 1, 5, 10 (same seed, epochs, hidden units).
- Track per epoch: reconstruction error and mean free energy on `fit` and `fit_validation`; plot.
- Anomaly score = free energy F(v). Higher = less typical.
- **Tests (`tests/test_restricted_boltzmann_machine.py`):** on a tiny RBM (e.g. 4 visible, 3 hidden) compare F(v) against brute-force −log Σₕ exp(−E(v,h)); check shapes and value ranges of conditional probabilities; one CD step changes weights in the expected direction on a toy dataset.
- Add a derivation note to the decision log: log-likelihood gradient = data term − model term, why the model term is intractable (partition function), why sampling is needed (Unit III), why Gibbs works for the RBM (bipartite graph ⇒ conditional independence, Unit II), and what CD-k approximates (Unit IV).

### Step 5 — VAE (`step05_train_vae.py`)
- PyTorch MLP encoder → (μ, log σ²) → reparameterisation → MLP decoder. Small (e.g. latent 4–8). Loss = reconstruction (MSE on standardised features) + β·KL, β = 1 default.
- Early stopping on `fit_validation` loss (never on `calib`, which is reserved for thresholds).
- Anomaly score = reconstruction error (primary); also save negative ELBO as a secondary score.
- Save loss curves.

### Step 6 — Markov state model (optional, `step06_fit_markov_states.py`)
Only if the audit shows distinct motor-current states. Estimate the transition matrix on `fit` with add-one smoothing. Score per transition = −log P(sₜ | sₜ₋₁); window score = max over transitions inside it. If skipped, record why in the decision log.

### Thresholds and alerts (`alert_thresholds.py`, `alert_events.py`)
- For every model/score: threshold = percentile of that score on `calib` (default 99th; also report 99.5th as a sensitivity row). Save all thresholds to `reports/tables/thresholds.json` **before** scoring `test`.
- Alert = window with score > threshold.
- Alert events: merge alerts separated by ≤ `merge_gap` (default 10 min); drop events shorter than `min_event_len` (default 1 window, i.e. keep all). Values fixed in config before evaluation.

### Step 7 — Bayesian Network (`step07_build_bayesian_network.py`)
- Variables: the discretised analog states + key digital signals (keep it readable, ~8–12 nodes).
- Structure: learn with pgmpy Hill-Climb + BIC on `fit`, then show the DAG; optionally constrain with simple domain edges documented in the decision log. CPDs via Bayesian estimation (BDeu) on `fit`.
- Syllabus demos (saved to the decision log + one DAG figure):
  - causal query (parent state → child probability) and evidential query (child state → parent probability);
  - d-separation checks and the three local-independence cases (node and parents, node and non-parents, node and descendants).
- Alert context: for 1–2 alert events from the best model, set the observed states during the event as evidence and report, per variable, how unlikely the observed state is under the BN (e.g. −log P(state | Markov blanket)) compared with its typical value on `calib`. Wording: "unusual / associated", never "caused".

### Step 8 — Evaluate alerts (`step08_evaluate_alerts.py`)
For each model (baseline, RBM k = 1/5/10, VAE, Markov if present), on `test` only:
| Metric | Definition |
|---|---|
| Events detected | reported window counted as detected if any alert event overlaps `[start − lead_tolerance, end]` (default lead_tolerance = 24 h, fixed in config before evaluation) |
| Detection delay | first alert time in that range minus window start (negative = before) |
| Unconfirmed alerts/day | alert events outside all `[start − lead_tolerance, end]` ranges, divided by test days |
| Alert-window fraction | share of test windows flagged |

Also: sensitivity table for threshold percentile 99 vs 99.5. Outputs: `reports/tables/metrics.csv`, `reports/tables/metrics.md`, figures in `reports/figures/evaluation/` (score vs time with threshold line and shaded failure windows, per-event zooms, CD-k comparison).

### Step 10 — Static dashboard (scope extension, `step10_build_dashboard_data.py`)
Read-only static site in `dashboard/` for Vercel. The build script copies or reshapes **only** files the pipeline already produced (`reports/tables`, `outputs/scores` downsampled to hourly maxima plus window-level series around the four events, `outputs/logs` runtimes) into `dashboard/data/`. No raw CSV, no model weights, no retraining. Every number on a page comes from that data. Pages: overview, metrics (99 vs 99.5), scores over time with failure windows and per-event zoom, Bayesian network, data and training evidence, and a claims-and-limits panel.

### Step 9 — Final report and README
`reports/final_report.md` sections:
1. Problem and motivation
2. Data (source, DOI, CC BY 4.0 attribution, audit findings, metadata issues)
3. Method: split, features, each model with its syllabus unit, thresholds, alert events, BN
4. Results: metrics table, figures, CD-k comparison, BN case study
5. Discussion: which approach worked best and why, where each failed
6. Limitations: 4 reported events only, windows are not dense labels, unconfirmed alerts, no causation, no prediction claim, single machine
7. Theory comparison (no implementation): GAN, NADE, MADE vs RBM/VAE (likelihood, sampling, training stability)
8. Syllabus mapping table (Units I–V)
9. Reproduction steps and total runtime

`README.md`: one-paragraph summary, repo layout, setup (venv + `pip install -r requirements.txt` + `pip install -e .`), dataset download instructions, one command per step, where outputs go, data attribution.

## 5. Configuration (defaults, all in `config.yaml`)

```yaml
seed: 42
paths:
  raw_csv: data/raw/MetroPT3(AirCompressor).csv
  processed_dir: data/processed
  models_dir: outputs/models
  scores_dir: outputs/scores
  logs_dir: outputs/logs
  tables_dir: reports/tables
  figures_dir: reports/figures
split: {train_period: first_calendar_month, calib_fraction: 0.2, fit_validation_fraction: 0.1}
windows: {size_seconds: 60, min_readings_fraction: 0.5}
discretise: {n_bins: 3, method: quantile}
rbm: {n_hidden: 32, k_values: [1, 5, 10], epochs: 30, batch_size: 256, learning_rate: 0.01, momentum: 0.5, weight_decay: 0.0001}
vae: {latent_dim: 6, hidden_sizes: [64, 32], epochs: 50, batch_size: 256, learning_rate: 0.001, beta: 1.0, early_stopping_patience: 5}
thresholds: {percentiles: [99, 99.5], primary_percentile: 99}
alerts: {merge_gap_minutes: 10, min_event_windows: 1}
evaluation: {lead_tolerance_hours: 24}
```
Change values only for a reason grounded in training data or the audit, and log the reason in the decision log.

## 6. Definition of done

- [ ] Repo matches the layout in 2.1; no stray, duplicate or temporary files; names follow 2.2.
- [ ] All pipeline steps run in order from a fresh clone plus the raw CSV.
- [ ] `pytest` passes (including the RBM brute-force free-energy test) and `ruff check` is clean.
- [ ] `reports/tables/thresholds.json` written before any test scoring (timestamp noted in the decision log).
- [ ] `reports/tables/metrics.csv` has a row per model, including baseline and all three CD-k variants.
- [ ] Figures exist for: data audit, training curves (RBM, VAE), scores vs time with windows, per-event zooms, CD-k comparison, BN DAG.
- [ ] `final_report.md` uses only numbers from saved outputs, includes limitations, the theory section and total runtime.
- [ ] No synthetic data, no causal or prediction claims, no "false alarm" wording.
- [ ] Git history has one commit per completed step.
