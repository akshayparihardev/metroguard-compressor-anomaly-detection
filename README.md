# MetroGuard

Anomaly detection on real metro air-compressor telemetry (MetroPT-3, UCI). Five approaches learn what
"normal" looks like from the first month of data and score how unusual later periods look: a z-score
baseline, an RBM trained with Gibbs sampling and contrastive divergence (written from scratch, no packaged
RBM), a VAE, a Markov chain over motor-current states, and a Bayesian network that explains alerts. Results
are evaluated at event level against four operator-reported air-leak intervals.

Course project for Deep Learning II (CDT7001). Exploratory: four reported events on one machine, and no
claim of failure prediction or causation.

## What it found

- Thresholds fitted on the training month flag only about 1% of that month, but **10% to 21% of later windows**
  for the baseline, RBM and VAE, because the training month is not representative of the rest of the record.
- All of those models "detect" the four reported windows, but the detection ranges cover only 4.2% of the test
  period, so that is weak evidence. **No learned model beat the z-score baseline.**
- The Markov chain is the most selective (0.44% of windows flagged) and detects three of four events.

Details and numbers: [`reports/final_report.md`](reports/final_report.md). Every decision and its reason:
[`reports/decision_log.md`](reports/decision_log.md). A read-only dashboard of the results lives in
[`dashboard/`](dashboard/README.md).

## Repository layout

```
config.yaml                every tunable value (seed, window size, model settings, thresholds)
src/metroguard/            library code, grouped by role
  data/                    loading, auditing, splitting and featurising the raw readings
  models/                  baseline, RBM, VAE, Markov chain, Bayesian network
  alerting/                thresholds, alert events, event-level evaluation
  reporting/               figures and the dashboard export
  config_loader.py  artifact_names.py  step_run_tracking.py
pipeline/                  one thin script per step (step01 ... step10) and run_all_steps.py
tests/                     pytest, mirrors src/metroguard (no raw data needed)
dashboard/                 read-only static site, deployed on Vercel
reports/                   audit report, decision log, final report, tables/, figures/
docs/                      project specification and explanations
data/  outputs/            raw and cached data, models, scores, logs (git-ignored, regenerable)
```

## Quick start

Requires Python 3.14. The RBM and VAE use the GPU when PyTorch can see one, otherwise the CPU.

```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
pip install -e .
```

Download `MetroPT3(AirCompressor).csv` from the
[UCI repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) into `data/raw/`, then:

```
python pipeline/step01_audit_raw_data.py
python pipeline/run_all_steps.py
```

Both skip work whose outputs are up to date; add `--force` to recompute. On the author's machine the audit
takes about 55 s and steps 2 to 8 about 2 minutes.

## Pipeline

| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `step01_audit_raw_data.py` | Audits the raw file (cadence, gaps, digital signals, event windows, motor states) | `reports/data_audit_report.md`, parquet cache |
| 2 | `step02_build_features.py` | 60 s windows, chronological split, fit-only scaler, bins and state cut points | `data/processed/`, `outputs/models/` |
| 3 | `step03_score_zscore_baseline.py` | Z-score baseline; thresholds frozen from calib | `outputs/scores/`, `reports/tables/thresholds.json` |
| 4 | `step04_train_rbm.py` | RBM with CD-k for k = 1, 5, 10 | scores, training curves |
| 5 | `step05_train_vae.py` | VAE with early stopping on a validation slice | scores, training curves |
| 6 | `step06_fit_markov_states.py` | Markov chain over motor-current states | scores, transition matrix |
| 7 | `step07_build_bayesian_network.py` | Learns the network, runs the syllabus demos, explains two alerts | `reports/tables/`, DAG figure |
| 8 | `step08_evaluate_alerts.py` | Event-level metrics and evaluation figures | `reports/tables/metrics.*` |
| 9 | (written report) | `reports/final_report.md` | |
| 10 | `step10_build_dashboard_data.py` | Exports saved results for the dashboard | `dashboard/data/` |

Library code for each step lives in `src/metroguard/`; the scripts only parse arguments, call it and save
outputs. Names of the files the steps pass to each other are defined once, in `artifact_names.py`.

## Development

```
pytest                              # fast, needs no raw data
ruff check . && ruff format --check .
```

- **Rules:** [`CLAUDE.md`](CLAUDE.md) holds the working rules (real data only, chronological split, nothing fitted
  on test data, thresholds frozen before test scoring, honest wording) and
  [`docs/project_spec.md`](docs/project_spec.md) is the single source of truth.
- **Reproducibility:** fixed seed in `config.yaml`; step 7 pins `PYTHONHASHSEED` because pgmpy's structure
  search breaks ties in set order.
- **Windows note:** older builds of numpy, scipy and torch are pinned because Windows Smart App Control blocked
  newer ones; `models/pgmpy_import_fallback.py` covers one blocked statsmodels import. See the decision log.

## Data and attribution

MetroPT-3 Dataset, UCI Machine Learning Repository, https://doi.org/10.24432/C5VW3R, licensed CC BY 4.0.
The raw file is not included in this repository.
