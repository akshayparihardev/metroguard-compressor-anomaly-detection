# CLAUDE.md — working rules for MetroGuard

Read `docs/project_spec.md` (master spec) and `docs/phase1_data_audit_spec.md` (step 1) before writing code. They are the single source of truth. If this file and the spec ever disagree, the spec wins; say so instead of guessing.

## Stay on scope
- Build only what the spec describes, step by step, in order. No extra models, features, APIs, external AI services, extra datasets or hyperparameter searches. The only UI allowed is the static dashboard in `dashboard/` (spec step 10): it shows saved pipeline outputs and never trains, scores or reads the raw CSV. Its "Analyse your data" page (spec step 10b) runs the same pipeline in the visitor's browser on the visitor's own file; it never uploads the file and never changes the MetroGuard results.
- Ideas outside the spec go under "Possible future work" in `reports/decision_log.md`. Do not build them.
- Before changing any default in `config.yaml`, state the reason from the audit or training data and log it.

## Non-negotiable project rules
1. Real data only (`data/raw/MetroPT3(AirCompressor).csv`). Never synthetic or substitute data. Missing file → stop with a clear message.
2. Chronological split. Nothing fitted on test data: no scaler, bins, mean/std, weights, CPDs, thresholds or window choices.
3. Models fit on `fit`; thresholds from `calib`; thresholds saved to `reports/tables/thresholds.json` and frozen before any test scoring. Never tune using failure windows.
4. Evaluate at event level. Alerts outside reported windows are "unconfirmed alerts", never "false alarms".
5. No causal claims from the BN. No failure-prediction claims.
6. RBM, Gibbs sampling, CD-k and free energy are implemented by us. No packaged RBM.
7. Every number in a report comes from code that actually ran and saved its output. Never invent or estimate results.
8. If a model underperforms the baseline or a step is blocked, report it with numbers.

## Repo hygiene and naming
- Follow the layout and naming rules in spec sections 2.1 and 2.2 exactly. Names say what a file does or contains. No `utils.py`, `helpers.py`, `misc/`, `temp/`, `v2`, `final_final`, `test2`.
- Library code in `src/metroguard/`; `pipeline/stepNN_<verb>_<object>.py` scripts are thin wrappers.
- No duplicate files, no dead code, no commented-out blocks, no unused imports, no empty placeholder modules.
- Paths via `pathlib` and `config.yaml`. The machine is Windows: no shell-only commands in pipeline code.
- Generated artifacts go to `outputs/` (git-ignored) or `reports/` (committed, human-readable). Nothing generated in the repo root.

## Resource-efficient (green) computing
- Read the raw CSV once and cache parquet in `data/processed/`; later steps read the cache.
- float32 tensors, windowed data in models, early stopping, only the configs listed in `config.yaml`.
- Skip a step whose outputs are up to date; `--force` recomputes.
- Log runtime per step to `outputs/logs/`.

## How to work each step
1. Say in two or three lines what the step will do and which files it touches.
2. Implement, then run it for real.
3. Run `pytest` and `ruff check` (and `ruff format`). Fix failures.
4. Add an entry to `reports/decision_log.md`: what was done, key numbers, decisions and why.
5. Commit: `stepNN: <what was done>`.
6. Continue to the next step. Stop only at the step 1 checkpoint (show audit summary and proposed step 2 decisions, wait for approval) or on a real blocker.

## When unsure
Ask one specific question rather than guessing. Never silently change the spec, the rules or the metric definitions.
