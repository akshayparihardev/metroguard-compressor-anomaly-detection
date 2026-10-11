# MetroGuard decision log

Each entry records what was done, the key numbers, the decisions taken and why.
All numbers come from saved outputs (`reports/data_audit_report.md`, `outputs/logs/`).

## Step 01: audit raw data (2026-10-11)

**What was done.** `pipeline/step01_audit_raw_data.py` read the raw CSV once (218.3 MB,
1,516,948 rows x 17 columns), parsed timestamps, wrote a typed parquet cache
(`data/processed/metropt3_sensor_readings.parquet`: analog float32, digital bool) and wrote
`reports/data_audit_report.md` plus 10 figures in `reports/figures/data_audit/`. Runtime 55 s
(`outputs/logs/step01_audit_raw_data_manifest.json`). No model was trained.

**Key numbers.**
- Cadence measured from the file: mode 10 s (0.1 Hz), median 10 s. 88.2% of deltas are exactly
  10 s, 11.8% are 9–20 s jitter. UCI's "1 Hz" description does not match the file.
- 0 missing values, 0 duplicate rows, 0 duplicate timestamps, already in time order.
- 351 gaps (> 20 s), 160 of them at least 1 h; 37.9 days of the 213-day span have no readings.
- 7 frozen-value runs (all analog sensors unchanged for at least 60 min): 50,425 readings (3.32%).
- All 8 digital columns found by header and strictly 0/1. LPS is near-constant overall (0.34%).
- Event coverage: events 1, 2 and 4 are about 100%; event 3 is 91.6% (largest internal gap 3.8 h).
- Candidate training period (first calendar month): 214,850 rows, 1–28 Feb (no data on 29 Feb),
  no overlap with any reported window. Test: 1,302,098 rows, 29,954 of them in reported windows.
- Motor current: three candidate states, about 0.05 A (54.7%), 3.75 A (29.5%) and 5.95 A (15.8%).
  The 7 A and 9 A hypotheses are not supported.

**Decisions and reasons.**
- *Failure-window boundaries are inclusive* (`failure_windows.in_window_mask`). Published end
  times such as 23:59:00 read as "up to and including". This leaves 6 readings after 23:59:00 on
  18 Apr outside event 1; they are reported and the window is not extended.
- *Digital columns are identified by header name, then verified by value set*
  (`raw_data_loader.identify_column_roles`). The report also lists any unexpected 0/1 column
  (none were found).
- *Gap definition*: deltas above 2x the observed cadence (20 s) count as gaps; 9–20 s is
  normal jitter. Thresholds are in `config.yaml` under `audit`.
- *Unnamed index column* `Unnamed: 0` (0 to 15,169,470, step 10) is reported and left out of
  the cache. Its step of 10 suggests the file was subsampled from a 1 Hz source, which may
  explain the conflicting metadata. This is an inference, not a confirmed fact.
- *Motor-current candidate states* come from histogram peaks with merging of shallow valleys
  (`raw_data_audit.motor_current_candidate_states`). An earlier version of the rule produced a
  spurious split at 5.75 A / 5.95 A and a mislocated idle peak (0.25 A); it was corrected before
  this entry.
- *Plots break lines at gaps* (`plotting.break_lines_at_gaps`) so missing data is not drawn as a
  straight line.

**Additions beyond the original step 1 layout (please confirm at the checkpoint).**
- `src/metroguard/raw_data_audit.py`: the audit measurements and report rendering. Without it,
  this logic would sit inside the pipeline script, which spec 3.1 says must stay thin.
- `config.yaml` gained an `audit:` section (gap multiplier, long-gap hours, near-constant
  fraction, event context, display resampling, frozen-run minimum) and `paths.reports_dir`.
  Nothing was added to or changed in the existing defaults.
- The frozen-value run check is not named in the step 1 spec. It was added because the event 1
  plot showed about 15 h of perfectly constant readings, which directly affects windowing.

**Environment.** Python 3.14.6 (the only interpreter installed). Windows Smart App Control
(on) blocked recently released builds: numpy 2.5.4 (`mtrand` DLL) and torch 2.11
(`caffe2_nvrtc.dll`, both the venv copy and the system copy). Older, established builds load
fine. Pinned: numpy 2.4.6, pandas 2.3.3, pyarrow 23.0.1, matplotlib 3.10.8, pgmpy 1.1.2 and
torch 2.9.1+cu128. Verified: `torch.cuda.is_available()` is True on the NVIDIA GeForce RTX 4050
Laptop GPU, and a float32 matrix multiply ran on it. Smart App Control was not changed.
Models will use the GPU (CUDA) for the VAE and, where it speeds training up, the RBM.

## Step 02: build features (2026-10-11)

**What was done.** `pipeline/step02_build_features.py` (2.5 s, reads only the parquet cache) turned
1,516,948 readings into 60 s windows, split them chronologically and fitted every transformation
on `fit` windows only. Outputs in `data/processed/`: `window_features.parquet`,
`window_standardised_features.parquet` (VAE input), `window_discrete_states.parquet` (BN input),
`window_rbm_visible.npz` (38 binary visible units). Fitted objects in `outputs/models/`:
`feature_standardiser.json`, `discretisation_bin_edges.json`, `motor_current_state_cut_points.json`.

**Key numbers** (from `outputs/logs/step02_build_features.log`).
- Window 60 s = 6 expected readings; a window needs at least 3 (0.5 x 6) to be usable.
- Training period 2020-02-01 to 2020-02-28 (ends before 2020-03-01). `calib` starts 2020-02-23 12:54.

| split | windows | usable | too few readings | in frozen run |
|---|---|---|---|---|
| fit_train | 25,581 | 25,561 | 20 | 0 |
| fit_validation | 2,840 | 2,840 | 0 | 0 |
| calib | 7,104 | 7,100 | 4 | 0 |
| test | 217,195 | 206,857 | 196 | 10,146 |

- Features per window: 7 analog means, std of Motor_current/TP2/TP3, 8 digital active fractions (18).
- Fit-only constant features (scale set to 1): `Oil_level_active_fraction`,
  `Caudal_impulses_active_fraction`.
- Every analog feature got 3 quantile bins (no duplicate cut points, so no merging was needed).
  RBM visible units: 10 features x 3 bins + 8 digital = 38.
- Motor-current states from `fit` readings only: cut points 1.85 A and 4.35 A, 3 states. (The step 1
  all-months rule gave 2.55 A and 4.35 A; the lower cut point moved because February alone was used.)

**Decisions and reasons.**
- *Split boundaries use counts of usable windows*, not calendar time: with 37.9 days of gaps in
  the full file, 80/20 by calendar time would not give 80/20 of the data. `calib` = last 20% of usable
  training windows, `fit_validation` = last 10% of the remaining `fit`. Models train on `fit_train`;
  early stopping uses `fit_validation`; thresholds use `calib` only.
- *Scaler, bin edges and motor-state cut points are fitted on `fit_train` + `fit_validation`*, never on
  `calib` or `test`. `fit_validation` otherwise only influences early stopping, so including it in
  these statistics is harmless; this reads "fitted on `fit`" in the spec as both fit parts.
- *Unusable windows stay in the table* with flags (`in_frozen_run`, `too_few_readings`, `usable`)
  instead of being deleted, so nothing is hidden. Models and evaluation must filter on `usable`.
- *Frozen-run windows are excluded from scoring* (10,146 test windows, 4.7%). Reason: all sensors
  repeat identical values for hours there, which looks like stuck logging, not machine behaviour.
- *No interpolation* across gaps: windows exist only where readings exist.
- *Motor state per window* uses the window-mean motor current. Step 6 may use raw readings for
  transitions inside windows, as the spec says; decided there.
- *Added `tests/test_feature_builder.py`* (not in the spec's layout list) to test windowing,
  fit-only bins and the standardiser. 17 tests pass in total.

## Step 03: z-score baseline (2026-10-11)

**What was done.** `pipeline/step03_score_zscore_baseline.py` (0.3 s) scored every window with the max
(and mean) over the 7 analog window means of |z|, using mean/std from `fit` windows only
(`outputs/models/zscore_reference.json`). Scores for 252,720 windows (242,358 usable) are in
`outputs/scores/zscore_baseline_scores.parquet`.

**Key numbers** (`reports/tables/thresholds.json`, 7,100 usable calib windows).

| score | 99th percentile | 99.5th percentile |
|---|---|---|
| max_abs_z (primary) | 4.705 | 6.334 |
| mean_abs_z | 2.129 | 2.434 |

**Decisions and reasons.**
- *Thresholds are written before the model's test windows are scored.* The script scores the
  non-test windows, computes the calib percentiles, writes them to `thresholds.json` (entry stamped
  `frozen_at` 2026-10-11T11:17:26), and only then scores the test windows.
  Later models follow the same order within their own step, so each model's thresholds are frozen
  before that model scores test. A single global "all thresholds before any test scoring" is not
  possible when models are trained in later steps.
- *Alert rule*: score strictly above the threshold (`alert_thresholds.alert_mask`).
- *The baseline uses only the 7 analog window means*, as the spec says; the std and digital
  features are for the VAE and RBM.
- Tests added in `tests/test_alert_thresholds.py` (22 tests pass in total).

## Step 04: RBM with Gibbs sampling and CD-k (2026-10-11)

**What was done.** `src/metroguard/restricted_boltzmann_machine.py` implements the Bernoulli RBM,
Gibbs sampling, CD-k and free energy ourselves with PyTorch tensors (no packaged RBM).
`pipeline/step04_train_rbm.py` trained three models (k = 1, 5, 10; same seed 42, 32 hidden units,
30 epochs, batch 256, lr 0.01, momentum 0.5, weight decay 1e-4) on `fit_train` (25,561 windows) with
38 visible units. It ran on the **GPU (CUDA, NVIDIA GeForce RTX 4050 Laptop GPU)**; total step
time 16.2 s. Outputs: `outputs/models/rbm_k{1,5,10}.pt`, `outputs/scores/rbm_scores.parquet`,
`reports/tables/rbm_training_history.csv`, `reports/figures/rbm/rbm_training_curves_cd_k.png`.

**Key numbers** (epoch 30, from `rbm_training_history.csv`).

| k | fit recon. error | fit mean F | validation recon. error | validation mean F |
|---|---|---|---|---|
| 1 | 0.0610 | -55.38 | 0.0632 | -55.07 |
| 5 | 0.0608 | -56.01 | 0.0631 | -55.70 |
| 10 | 0.0609 | -56.03 | 0.0630 | -55.75 |

Frozen thresholds on 7,100 calib windows (free energy, `reports/tables/thresholds.json`, stamped
2026-10-11 11:19:21 / :26 / :32, each before that model scored test):
99th percentile -46.88 (k=1), -47.51 (k=5), -47.57 (k=10); 99.5th -45.99, -46.83, -46.80.

**Findings reported, not hidden.**
- *The RBMs are not converged.* Reconstruction error (0.180 to 0.061) and mean free energy were
  still falling steadily at epoch 30 for all three models, and the validation curves track the fit
  curves closely (no overfitting). With the configured 30 epochs and lr 0.01 the models are
  under-trained. `config.yaml` was not changed, because the spec forbids tuning beyond its listed values;
  training longer is listed under possible future work.
- *k = 1, 5 and 10 are almost indistinguishable* in these curves (differences in the third decimal
  of reconstruction error and about 0.7 in mean free energy). This is a result of this setup, not
  evidence about CD-k in general.
- *Negative free energies are expected*: F(v) = -a^T v - sum softplus(.) is not normalised, so only
  differences between windows matter. Higher = less typical.

**Tests** (`tests/test_restricted_boltzmann_machine.py`, 7 tests, 29 pass in total): F(v) equals
the brute-force -log sum_h exp(-E(v,h)) on a 4-visible, 3-hidden RBM for all 16 visible vectors;
free energy stays finite for large weights; conditional probabilities have the right shapes and lie
in (0, 1); samples are binary; the CD gradient has the expected sign and size on a zero-initialised
toy RBM; CD-k training lowers the free energy of the training data and raises that of an unseen
pattern.

**Decisions and reasons.**
- *Negative phase*: k Gibbs steps starting at the data, sampling intermediate visible vectors and
  using probabilities at the last step (a standard variance-reducing choice); positive and negative
  hidden statistics use probabilities. Weight decay applies to W only.
- *Frozen-before-test order* as in step 3: non-test windows are scored, calib thresholds frozen, then
  test windows scored.
- *No early stopping for the RBM*: the spec asks for early stopping only for the VAE; the RBM trains
  the configured epochs and `fit_validation` is tracked for the curves.

**Derivation note (Units II to IV).**
- *Log-likelihood gradient.* With p(v) = (1/Z) sum_h exp(-E(v,h)) and Z = sum_{v,h} exp(-E(v,h)),
  d log p(v) / dW_ij = <v_i h_j>_data - <v_i h_j>_model. The data term is easy: given v, the hidden
  units are conditionally independent, so <v_i h_j>_data uses p(h_j = 1 | v). The model term is the
  expectation under the model's joint distribution.
- *Why the model term is intractable.* It needs the partition function Z, a sum over all
  2^(n_visible + n_hidden) configurations. For 38 visible and 32 hidden units that is 2^70, so it cannot
  be computed exactly.
- *Why sampling is needed (Unit III).* An expectation we cannot sum can be estimated from samples
  drawn from the distribution, here the model's own distribution, without ever computing Z.
- *Why Gibbs sampling works for an RBM (Unit II).* The RBM is a bipartite undirected graph (no
  visible-visible or hidden-hidden edges), so given the visible layer all hidden units are conditionally
  independent, and vice versa. A whole layer can be sampled at once from p(h | v) = sigmoid(b + W^T v)
  and p(v | h) = sigmoid(a + W h), and alternating the two gives a Markov chain whose stationary
  distribution is the RBM's joint distribution.
- *What CD-k approximates (Unit IV).* Running that chain to convergence for every update is too slow.
  CD-k starts the chain at the data and stops after k steps, replacing the model expectation with the
  statistics of the k-step reconstruction. The result is a biased but cheap, low-variance estimate of
  the gradient, which in practice moves the weights in a useful direction. Larger k lowers the bias
  at higher cost.

## Possible future work

- Train the RBM for more epochs (its curves had not flattened at epoch 30), and compare k on
  converged models. Not done now because the spec fixes epochs at 30 and forbids tuning.

- None yet.
