# MetroGuard

Anomaly detection in real metro air-compressor telemetry (MetroPT-3, UCI). Models learn "normal" behaviour
from the first month and score how unusual later periods look: a z-score baseline, an RBM trained with Gibbs
sampling and contrastive divergence (CD-k, implemented here, no packaged RBM), a VAE and a Markov state model.
Alerts use thresholds fixed on training data only, a Bayesian network gives probabilistic context for alerts,
and results are evaluated at event level against four operator-reported air-leak intervals. Deep Learning II
(CDT7001) course project; exploratory, with no failure-prediction and no causal claims.
Start with `reports/final_report.md`; the master specification is `docs/project_spec.md`.

## Repository layout

```
config.yaml           every tunable value          pyproject.toml, requirements.txt   packaging, pinned versions
src/metroguard/       reusable library code        pipeline/                          one script per step
tests/                pytest (no raw CSV needed)   docs/                              specs and explanations
data/raw/             official CSV (git-ignored)   data/processed/                    parquet/npz caches (git-ignored)
outputs/              models, scores, logs (git-ignored, regenerable)
reports/              audit report, decision log, final report, tables/, figures/
```

## Setup (Windows, Python 3.14)

```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
```

The CUDA build of torch is used when a GPU is visible (the RBM and VAE ran on an RTX 4050); otherwise the CPU
is used. Versions are pinned to ones that load on the development machine: Windows Smart App Control blocked
newer builds of numpy, scipy and torch (see `reports/decision_log.md`).

## Data

Download `MetroPT3(AirCompressor).csv` from https://archive.ics.uci.edu/dataset/791/metropt+3+dataset and place
it in `data/raw/`. The scripts stop with a clear message if it is missing; no substitute data is ever used.

Attribution (CC BY 4.0): MetroPT-3 Dataset, UCI Machine Learning Repository, https://doi.org/10.24432/C5VW3R.

## Run

```
python pipeline/step01_audit_raw_data.py            # audit, parquet cache, reports/data_audit_report.md
python pipeline/step02_build_features.py            # 60 s windows, chronological split, fit-only transforms
python pipeline/step03_score_zscore_baseline.py     # baseline scores, thresholds frozen from calib
python pipeline/step04_train_rbm.py                 # RBM, CD-k = 1, 5, 10 (GPU)
python pipeline/step05_train_vae.py                 # VAE with early stopping on fit_validation (GPU)
python pipeline/step06_fit_markov_states.py         # Markov chain over motor-current states
python pipeline/step07_build_bayesian_network.py    # Bayesian network, demonstrations, alert context
python pipeline/step08_evaluate_alerts.py           # event-level metrics and evaluation figures
python pipeline/run_all_steps.py                    # steps 2 to 8 in order
python pipeline/step10_build_dashboard_data.py      # export saved results for the dashboard (step 10)
```

Every step skips itself when its outputs are newer than its inputs and its configuration is unchanged; add
`--force` to recompute. Run time per step is logged in `outputs/logs/`.

## Where results go

| What | Where |
|---|---|
| Audit report and figures | `reports/data_audit_report.md`, `reports/figures/data_audit/` |
| Frozen thresholds | `reports/tables/thresholds.json` |
| Metrics | `reports/tables/metrics.csv`, `reports/tables/metrics.md` |
| Training curves, evaluation figures, network DAG | `reports/figures/rbm/`, `vae/`, `evaluation/`, `bayesian_network/` |
| Decisions and reasons, step by step | `reports/decision_log.md` |
| Final report | `reports/final_report.md` |
| Dashboard data (generated, committed so Vercel can serve it) | `dashboard/data/` |
| Trained models, scores, logs (not committed) | `outputs/models/`, `outputs/scores/`, `outputs/logs/` |

## Dashboard (read-only, static)

`dashboard/` is a plain HTML, CSS and JavaScript site that displays results the pipeline already saved. It
runs no model, reads no raw data and has no build step. `step10_build_dashboard_data.py` writes the small JSON
files in `dashboard/data/` (hourly score maxima, window-level scores around the four events, metrics,
training curves, audit tables, network structure); every number on a page comes from them.

Preview locally:

```
python -m http.server 8000 --directory dashboard
```

then open http://localhost:8000. Deploy on Vercel: push the repository to GitHub, import it in Vercel, set the
**Root Directory** to `dashboard`, framework preset **Other**, and leave the build and output settings empty
(`dashboard/vercel.json` already sets clean URLs and cache headers). Fonts (Schibsted Grotesk, JetBrains Mono)
are self-hosted under the SIL Open Font License; see `dashboard/assets/fonts/LICENSE.txt`.

## Tests and style

```
pytest
ruff check . && ruff format --check .
```
