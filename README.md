# MetroGuard

Anomaly detection in real metro air-compressor telemetry (MetroPT-3, UCI) using a z-score
baseline, an RBM trained with Gibbs sampling and CD-k, a VAE, an optional Markov state model,
and a Bayesian Network for alert context. Deep Learning II (CDT7001) course project.
The full specification is in `docs/project_spec.md`. This README is completed in step 9.

## Setup (Windows)

```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Data

Download `MetroPT3(AirCompressor).csv` from
https://archive.ics.uci.edu/dataset/791/metropt+3+dataset and place it in `data/raw/`.
Data: MetroPT-3 Dataset, UCI Machine Learning Repository, https://doi.org/10.24432/C5VW3R,
licensed CC BY 4.0.

## Run

```
python pipeline/step01_audit_raw_data.py          # add --force to recompute
```

Outputs: `reports/data_audit_report.md`, `reports/figures/data_audit/`,
`data/processed/` (parquet cache, git-ignored), `outputs/logs/` (run logs, git-ignored).
