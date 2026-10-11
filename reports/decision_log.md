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
- `src/metroguard/data/raw_data_audit.py`: the audit measurements and report rendering. Without it,
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
- *Added `tests/data/test_feature_builder.py`* (not in the spec's layout list) to test windowing,
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
- Tests added in `tests/alerting/test_alert_thresholds.py` (22 tests pass in total).

## Step 04: RBM with Gibbs sampling and CD-k (2026-10-11)

**What was done.** `src/metroguard/models/restricted_boltzmann_machine.py` implements the Bernoulli RBM,
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

**Tests** (`tests/models/test_restricted_boltzmann_machine.py`, 7 tests, 29 pass in total): F(v) equals
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

## Step 05: VAE (2026-10-11)

**What was done.** `src/metroguard/models/variational_autoencoder.py` (MLP encoder 18 -> 64 -> 32 -> (mu, log
sigma^2) with latent 6, reparameterisation, mirrored decoder) and `pipeline/step05_train_vae.py` trained on
`fit_train` (25,561 windows, 18 standardised features) on the **GPU (CUDA, RTX 4050)**, 44 s. Adam,
lr 0.001, batch 256, beta 1.0, up to 50 epochs, early stopping patience 5 on `fit_validation` loss
(2,840 windows); `calib` was never used for any training decision. Outputs:
`outputs/models/vae.pt`, `outputs/scores/vae_scores.parquet`, `reports/tables/vae_training_history.csv`,
`reports/figures/vae/vae_training_curves.png`.

**Key numbers** (`vae_training_history.csv`, `thresholds.json`).
- All 50 epochs ran; best epoch 47 (validation loss 2.779 = reconstruction 0.912 + KL 1.866). Early
  stopping did not trigger, so the model was still improving at the epoch cap.
- Frozen calib thresholds (7,100 windows, stamped 2026-10-11 11:23:00, before test scoring):
  reconstruction_error 99th 0.5737, 99.5th 0.7783; negative_elbo 99th 16.94, 99.5th 26.07.

**Findings reported, not hidden.**
- *One-epoch training spike.* Training loss jumped to 160,060 at epoch 16 and recovered. Cause: in
  February `LPS_active_fraction` is almost always 0 (std about 0.0005), so its rare activations
  standardise to z up to 168.5 on `fit_train` (and up to 674.1 on `test`, which is why the figure uses a
  log axis). A few such windows dominate a summed squared error. This is a property of the data plus the
  spec's standardise-on-`fit` rule; the preprocessing was not changed.
- *Consequence for evaluation*: any later window with `LPS` active will have a very large VAE score
  and will very likely raise an alert. Step 8 will show whether this inflates unconfirmed alerts.
- *Not converged*: best epoch 47 of 50.

**Decisions and reasons.**
- *Loss* = per-window sum of squared errors over the 18 standardised features + beta x KL(q(z|x) || N(0, I)),
  averaged over the batch. The summed form (a Gaussian likelihood with fixed variance) keeps the KL term
  from dominating; a per-feature mean would shrink the reconstruction term by 18x.
- *Scores use z = mean of q(z | x)* (no sampling), so they are deterministic. Primary: mean squared
  reconstruction error per feature. Secondary: negative ELBO = summed squared error + KL.
- *Validation loss uses z = mean* for the same reason; training uses sampled z.
- *Best weights are restored* after early stopping, so the saved model is the best epoch.
- No VAE test file: the spec's test list has none for the VAE.

## Step 06: Markov state model (2026-10-11)

**Was it supported?** Yes. The step 1 audit found three distinct motor-current states (about 0.05, 3.75
and 5.95 A), and step 2 re-derived the cut points from `fit` readings only: 1.85 A and 4.35 A.

**What was done.** `src/metroguard/models/markov_state_model.py` and `pipeline/step06_fit_markov_states.py`
(0.6 s, CPU/NumPy; no GPU needed) assigned a state to every raw reading, counted state-to-state
steps on readings inside usable `fit` windows, and estimated the transition matrix with add-one
smoothing. Score per transition = -log P(s_t | s_{t-1}); window score = max over the transitions that
end in the window. Outputs: `outputs/models/markov_transition_matrix.json`,
`outputs/scores/markov_scores.parquet`.

**Key numbers** (`outputs/logs/step06_fit_markov_states.log`).
- Transition counts on fit readings (rows = from, columns = to; states 0 = idle, 1 = about 3.75 A,
  2 = about 5.95 A): [[121738, 4, 908], [936, 37769, 15], [1, 926, 9521]].
- Smoothed matrix diagonal: 0.9926, 0.9754, 0.9111. Rarest observed steps: idle to 3.75 A (4 times) and
  5.95 A to idle (1 time).
- Frozen threshold (7,100 calib windows, 2026-10-11 11:24:02): 99th = 99.5th = 4.905
  (= -log 0.00741, the 0 -> 2 transition).
- 44 of 252,720 windows have no valid transition (NaN score, never alerted).

**Findings and decisions.**
- *The 99th and 99.5th percentile thresholds are identical* because the score takes only a few discrete
  values. With the spec's rule "alert = score strictly above the threshold", only transitions rarer than
  0 -> 2 raise an alert (idle to 3.75 A, 3.75 A to 5.95 A, 5.95 A to idle). The 99.5 sensitivity row
  for this model will therefore equal the 99 row.
- *A transition belongs to the window that contains the later reading*, so a step across a window
  boundary is scored in the new window.
- *Steps across gaps are not transitions*: a step counts only if the two readings are at most 2 x the
  observed cadence (20 s) apart. This matches the audit's gap definition and avoids inventing
  transitions across missing data.
- *Per-reading states* are used here (not the per-window state from step 2), as the spec says
  "max over transitions inside it". The `motor_current_state` column from step 2 is not used by this step.

## Step 07: Bayesian network (2026-10-11)

**What was done.** `src/metroguard/models/bayesian_network.py` and `pipeline/step07_build_bayesian_network.py`
(2.1 s, CPU) learned a discrete BN with pgmpy 1.1.2 on 28,401 usable `fit` windows: Hill-Climb search with
the BIC score (`bic-d`, max in-degree 3), then CPDs by Bayesian estimation (BDeu, equivalent sample size 10).
Nodes (11, from `config.yaml` `bayesian_network.nodes`): the 7 analog window-mean states (3 quantile bins)
and the binarised COMP, DV_eletric, Towers, MPG. Outputs: `reports/tables/bayesian_network_edges.csv`,
`reports/tables/bayesian_network_demonstrations.md`, `reports/tables/bayesian_network_alert_context.md`,
`reports/figures/bayesian_network/bayesian_network_dag.png`.

**Key numbers.**
- 18 learned edges. Strongest dependency (mutual information 1.085 nats): `Reservoirs_mean -> TP3_mean`.
  The causal-direction and evidential queries show these two are near-deterministic copies of each other,
  which is expected for two pressure sensors on the same circuit. The direction of this edge flipped
  between two otherwise identical runs (see "Reproducibility" below), which is a concrete reminder that
  learned edge direction says nothing about what causes what.
- d-separation (see the demonstrations file): chain DV_eletric_active / MPG_active / COMP_active is connected
  (True) and becomes blocked once MPG_active is observed (False); fork COMP_active / MPG_active /
  Towers_active is connected (True) and blocked (False) given MPG_active; collider DV_pressure_mean /
  Motor_current_mean / TP3_mean is blocked (False) and becomes connected (True) once Motor_current_mean is
  observed. The generator prefers patterns where evidence changes the answer, because this DAG is dense and
  other paths often keep two nodes connected.
- Local independence: node vs a parent is dependent, node vs a non-parent non-descendant given its parents
  is independent (d-separated), node vs a descendant is dependent, as the theory predicts.

**Alert context.** Model used: `zscore_baseline` (max_abs_z, 99th percentile), chosen because among the
primary-score models it detected the most reported events on test (4 of 4, ties broken by fewest
unconfirmed alerts per day). Explained events: the alert events inside reported windows 1 and 2.
Share of event windows above the calib 95th percentile of -log P(state | Markov blanket), against 5%
expected by chance (flag: more than 10%):
- Window 1: `Towers_active` 42.7% and `MPG_active` 57.3% flagged; `COMP_active` and all analog variables not
  flagged.
- Window 2: `Towers_active` 39.3% and `MPG_active` 60.0% flagged; `COMP_active` and all analog variables not
  flagged.
Absolute surprisals are small (thousandths of a nat for most variables), so this is a statement about
association relative to calib, not a strong effect. The flagged variables are digital compressor signals;
the analog states were not unusual given their blankets. Wording is "unusual / associated", never "caused".

**Decisions and reasons.**
- *Node choice*: the 4 digital signals that are near-constant in February (Oil_level, Caudal_impulses,
  Pressure_switch, LPS) and the 3 within-window std features are left out to keep the BN readable and
  because constant nodes carry no learnable dependency (approved at the step 1 checkpoint).
- *Unusual flag*: a variable is unusual if the share of event windows above its calib 95th percentile
  exceeds 2 x 5%. A first version compared the event mean with the calib 95th percentile; that was
  corrected because it flagged variables whose event mean was below the calib mean and was trivially
  satisfied where calib surprisal was almost zero.
- *Which model's events are explained* was chosen by test detections. This is a case-study selection
  only: no threshold, model or setting was changed because of it.
- *Order*: `alert_events.py` and `event_evaluation.py` were written during step 7 (earlier than step 8)
  because the "best model" for the case study needs event-level detection counts. Step 8 reuses them.
- `pgmpy` 1.1.2 needs `statsmodels.api`, whose time-series DLLs Windows Smart App Control blocks here.
  `src/metroguard/models/pgmpy_import_fallback.py` registers a minimal stand-in (only the three regression classes
  pgmpy asks for) if, and only if, the real import fails. The discrete BN code paths never use it. Also
  pinned older, loadable builds: scipy 1.17.1 (1.18.1 was blocked) and statsmodels 0.14.6.
- pgmpy API drift handled: `pgmpy.causal_discovery.HillClimbSearch` with `return_type="dag"`, categorical
  columns (integer columns are treated as continuous), and `DiscreteBayesianEstimator` for BDeu.
- New `config.yaml` keys (no existing default changed): the `bayesian_network` section.
- *Reproducibility*: a first full re-run produced the same scores and metrics but different edge directions
  for some node pairs. pgmpy's hill climb breaks ties between equally scoring (Markov-equivalent) edge
  directions in set-iteration order, which depends on Python's randomised string hash; with some hash seeds
  the search was also very slow (one run was stopped after several minutes). `rerun_with_fixed_hash_seed()`
  in `step_run_tracking.py` now re-runs step 7 with `PYTHONHASHSEED=0`; two consecutive runs give a
  byte-identical `bayesian_network_edges.csv`. Hidden consequence: the particular edge directions shown are
  one member of an equivalence class, not a finding.

## Step 08: evaluate alerts (2026-10-11)

**What was done.** `pipeline/step08_evaluate_alerts.py` (10 s) built alert events (alerts merged when separated
by at most 10 min, every event kept) from the frozen thresholds and evaluated every model and score on
**usable test windows only** (206,857 windows over 184.0 calendar days) against the four reported windows
with a 24 h lead tolerance, both fixed in `config.yaml` before evaluation. Outputs:
`reports/tables/metrics.csv` (one row per model, score and percentile), `reports/tables/metrics.md`,
and 11 figures in `reports/figures/evaluation/` (score vs time per model, one zoom per event with all models,
`rbm_cd_k_comparison.png`).

**Key numbers** (`metrics.csv`, primary scores).

| model | percentile | alert-window fraction | events detected | unconfirmed alerts/day |
|---|---|---|---|---|
| zscore_baseline (max_abs_z) | 99 / 99.5 | 0.104 / 0.064 | 4 / 4 | 25.85 / 9.32 |
| rbm_k1 | 99 / 99.5 | 0.199 / 0.171 | 4 / 4 | 29.97 / 30.02 |
| rbm_k5 | 99 / 99.5 | 0.203 / 0.172 | 4 / 4 | 29.82 / 32.38 |
| rbm_k10 | 99 / 99.5 | 0.208 / 0.171 | 4 / 4 | 28.59 / 32.36 |
| vae (reconstruction_error) | 99 / 99.5 | 0.214 / 0.139 | 4 / 4 | 31.02 / 28.26 |
| markov_state_model | 99 / 99.5 | 0.0044 / 0.0044 | 3 / 3 (events 2, 3, 4) | 3.65 / 3.65 |

Secondary scores at p99: baseline mean_abs_z 0.112 flagged, 39.29 unconfirmed/day; VAE negative_elbo 0.164
flagged, 36.35 unconfirmed/day; both detect 4 of 4.

**What these results do and do not show (reported, not hidden).**
- *Alert rates are far above the calibration rate.* Thresholds were set at the 99th percentile of calib
  (about 1% flagged), yet 10% to 21% of test windows are flagged for the baseline, RBMs and VAE. February is
  not representative of later months (see the step 1 notes: different mix of motor-current states,
  different digital activity). These models mostly detect "different from February", which produces
  roughly 25 to 40 unconfirmed alert events per day.
- *"4 of 4 events detected" is weak evidence.* The four detection ranges cover only 7.6 of 184.0 test days
  (4.2%), so a model flagging 10% to 21% of windows overlaps them almost by chance. Only the Markov model, with
  a 0.44% alert-window fraction, gives a more selective result (3 of 4, event 1 missed).
- *Detection delays near -24 h are an artefact*: the delay is clipped to the start of the lead range when an
  alert event was already running. For event 4 the -17 h delay matches the start of the data after a gap
  (about 21:30 on 14 July), so it also does not indicate early warning. No prediction claim is made.
- *The baseline is not beaten.* The z-score baseline has the lowest alert rate among the baseline, RBM and
  VAE models at both percentiles, and RBM and VAE did not detect more events than it (all 4 of 4). The RBM
  and VAE are no more selective than the baseline here.
- *CD-k*: k = 1, 5 and 10 give almost identical alert rates (0.199 to 0.208) and identical detections; the
  models were also under-trained (step 4), so this says little about CD-k in general.
- *Percentile sensitivity*: moving from the 99th to the 99.5th percentile lowers the alert fraction for the
  baseline (0.104 to 0.064) and VAE (0.214 to 0.139) but not meaningfully for the RBMs (about 0.20 to 0.17);
  unconfirmed alerts per day do not fall for the RBMs. The Markov model's two thresholds are equal (step 6).
- Frozen-run windows (10,146 in test) and windows with too few readings are excluded, so alerts or
  detections cannot come from stuck logging.

**Decisions and reasons.**
- *Delay* uses the first alert event overlapping the range: max(event start, range start) minus window
  start; negative means before the reported start.
- *Unconfirmed alerts per day* = alert events not overlapping any detection range divided by the 184.0
  calendar days of the test period (not only the days with data; 37.9 days of the file are gaps).
- `metrics.md` states the detection-range share and how to read it, computed in code.
- Evidence that thresholds were frozen before test scoring (`frozen_at` timestamps versus score-file
  modification times) is printed at the end of `metrics.md`.

## Step 09: final report, README and definition-of-done check (2026-10-11)

**What was done.** Wrote `reports/final_report.md` (nine sections, numbers taken only from saved outputs),
completed `README.md` and added `pipeline/run_all_steps.py` (steps 2 to 8 in order, each skipping itself
when up to date). A full forced run of steps 2 to 8 finished in 119.6 s (step 1: 55.0 s; about 174.6 s in
total); it reproduced `metrics.csv` and every threshold exactly.

**Definition of done (spec section 6).**
- Layout matches 2.1, names follow 2.2, no stray or duplicate files: **passed, with documented additions**
  not in the 2.1 tree: `src/metroguard/data/raw_data_audit.py`, `src/metroguard/models/pgmpy_import_fallback.py`,
  `tests/data/test_feature_builder.py`, and extra tables in `reports/tables/` (training histories and Bayesian
  network tables). No forbidden names (`utils`, `helpers`, `misc`, `temp`, `v2`, ...) in tracked files.
- All steps run in order from the raw CSV: **passed on this checkout** (`run_all_steps.py --force`
  after step 1). **Not tested** from a literal fresh clone with a new virtual environment.
- `pytest` (35 tests, including the RBM brute-force free-energy test) and `ruff check` / `ruff format`:
  **passed**.
- `thresholds.json` written before test scoring, timestamps noted: **passed per model** (each model's
  `frozen_at` precedes its own test scoring; evidence table at the end of `metrics.md`). A single global
  "all thresholds before any test scoring" is not possible when models are trained in later steps (see
  step 03).
- `metrics.csv` has a row per model including the baseline and all three CD-k variants: **passed**.
- Figures for data audit, RBM and VAE training curves, scores vs time with windows, per-event zooms, CD-k
  comparison and the Bayesian-network DAG: **passed**.
- `final_report.md` uses only saved numbers and includes limitations, the theory section and runtime:
  **passed**. Unit numbers I and V in the syllabus table are left for the author, because the spec does not
  name them.
- No synthetic data, no causal or prediction claims, no "false alarm" wording: **passed** (the phrase appears
  only in sentences that rule it out).
- One commit per completed step: **mostly**. Steps 1 and 7 have a second commit (torch pin and environment
  record; hash-seed reproducibility fix), and the step 1 environment note was written after the step 1
  commit.

**Honest summary of results.** No learned model beat the simple z-score baseline; all models flagged far more
test windows than the 1% they were calibrated to, because February is not representative of later months;
"4 of 4 events detected" is weak evidence because the detection ranges cover only 4.2% of the test period; the
Markov model was the most selective (3 of 4, 3.65 unconfirmed alerts per day).

## Step 10: read-only static dashboard (deliberate scope extension, 2026-10-11)

**Why this is outside the original scope.** `docs/project_spec.md` and `CLAUDE.md` locked the scope against
dashboards and UIs. The author explicitly approved a read-only static dashboard for deployment on Vercel. Both
files were amended to allow a `dashboard/` folder only; nothing else in the scope lock changed. The dashboard
shows saved outputs. It runs no model, scores nothing, reads no raw CSV and no model weights.

**What was built.**
- `src/metroguard/reporting/dashboard_export.py` and `pipeline/step10_build_dashboard_data.py` (thin wrapper, skips when up
  to date) turn `reports/tables`, `outputs/scores` (hourly maxima, plus window-level series around each reported
  window) and `outputs/logs` runtimes into 10 JSON files in `dashboard/data/` (about 1.4 MB; the four event files
  load only when opened). The audit and Bayesian-network tables are parsed from the generated markdown reports; the
  parser fails loudly if a report's format changes, so a number can never silently go blank.
- `dashboard/` (no framework, no build tool, no runtime dependencies): six pages (Overview, Metrics, Scores over
  time, Bayesian network, Data and training, Claims and limits), shared `assets/styles.css`, `app.js`, `charts.js`
  (canvas charts), `linemap.js`, one script per page, self-hosted fonts, and `vercel.json`.
- `tests/reporting/test_dashboard_export.py` (8 tests, 43 in total pass) checks the parsers, the hourly downsampling and
  gap handling, the epoch conversion, strict JSON (no NaN) and the layered network layout.

**Design decisions and reasons** (frontend skills applied: `frontend-design`, `impeccable` Operate mode).
- *Signature element*: the 213-day record drawn as a metro line (data gaps dashed, reported windows as stations,
  the end of the training month as a bar) with a ribbon of how much of each day the baseline spends above its
  threshold. It makes the central finding visible at a glance: the models flag far more windows after the
  training month than they were calibrated to.
- *Everything else is a plain data tool*: left rail (top bar on mobile), standard tables, segmented controls,
  toggle chips, canvas charts with hover tooltip, drag-to-zoom and double-click reset. Light and dark themes
  follow the OS and can be toggled. One brand colour (teal), red for reported windows, orange for alerts.
- *No numbers are typed into the pages.* Sentences are assembled from the JSON, so the site cannot drift from the
  pipeline. The dashboard repeats the report's honest framing: detection numbers are shown beside alert rates,
  "unconfirmed alerts" instead of "false alarms", no prediction or causal claims, edge directions flagged as
  arbitrary.
- *Layered network layout* (parents left of children) replaced a spring layout that piled nodes together.
- *No figure PNGs were duplicated* into `dashboard/`; charts are drawn from data. Only the generated JSON is
  committed so that Vercel can serve it without the pipeline outputs.
- Quality checks run: text contrast at least 4.5:1 in both themes (the orange used only for graphics measures
  4.29:1 against the 3:1 needed), no horizontal overflow at 375 px, keyboard focus rings, skip link, reduced-motion
  respected, interactions exercised in a browser (tooltip, zoom, reset, node selection, sorting, toggles).

**UI upgrade pass (skills from `Claude Frontend Skills/`).** The five top-ranked skills (`frontend-design`,
`design-taste-frontend`, `ui-ux-pro-max`, `web-design-guidelines`, `impeccable`) were all already present, so
nothing was installed. Used: `frontend-design` and `impeccable` (direction, quality floor), `emil-design-eng`
(motion rules) and the `ui-ux-pro-max` database (chart, colour and typography guidance for an anomaly dashboard;
its script reads local CSV files only and was reviewed before running). `design-taste-frontend` was not applied
because it excludes dashboards.
- *Dark theme* is now an OLED ladder: page `#000000`, rail `#050506`, surfaces `#0b0c0e` / `#141518` / `#1c1d21`,
  borders `#25262b`, brighter teal accent. Every text pair measures at least 5.98:1. The theme is set on `<html>`
  before first paint (no flash), follows the OS until the visitor chooses, and a theme change cross-fades with the
  View Transitions API (skipped under reduced motion). A `theme-color` meta tag follows the theme.
- *Motion* (custom ease-out `cubic-bezier(0.23, 1, 0.32, 1)`, all under 300 ms except the one-off chart reveal):
  press feedback (`scale(0.97)`) on every control, a sliding thumb on segmented controls that updates in place so
  keyboard focus is kept, bars that grow in, pills that pop in sequence, smooth cross-fade between pages, a
  skeleton while data loads. Nothing animates on keyboard-driven or high-frequency actions.
- *Charts*: the line draws in left to right when the range changes (650 ms, ease-out-expo), hover dots on every
  panel, diamond markers for single alert windows (shape, not just colour), a text summary per panel ("946 of 2,005
  scored windows above threshold"), event labels on the red bands, monospace axis numerals, and keyboard inspection
  (arrow keys, Shift for bigger steps, Home/End, Escape). This follows the database's accessibility advice for
  anomaly charts: mark anomalies by shape and text, and make point details reachable without a mouse.
- Re-ran the responsive sweep after the changes: 102 of 102 combinations passed.

**Responsive check.** A scripted sweep loaded all six pages at 17 viewport sizes from 280 x 600 to 3440 x 1440
(including short landscape phones and tablets) and flagged any horizontal page overflow, element outside the
viewport, chart wider than the viewport, empty page or error banner. 102 of 102 combinations passed after one
fix (the range control overflowed on 320 to 360 px phones). Touch screens also get larger tap targets and
tap-to-inspect on the charts; very large screens get a wider content column and slightly larger type.

**Not done.** No screen-reader pass beyond landmarks and labels, no automated visual-regression tests, and the
site was previewed locally but not yet deployed (the repository has no remote).

## Repository clean-up (2026-10-11)

**Goal.** Leave the repository easy to open and understand, as a senior maintainer would keep it. A scripted
audit came first (tracked files by folder, unused definitions, duplicated constants), then the changes below.

**What changed and why.**
- *Grouped the library by role.* Nineteen modules sat in one flat folder. They now live in four packages:
  `data/` (loading, audit, split, features), `models/` (baseline, RBM, VAE, Markov chain, Bayesian network),
  `alerting/` (thresholds, alert events, evaluation) and `reporting/` (figures, dashboard export), with the three
  foundation modules (`config_loader`, `artifact_names`, `step_run_tracking`) at the package root. Tests mirror the
  same grouping. Files were moved with `git mv`, so their history is kept, and the imports were rewritten by script.
  This amends the flat layout in spec section 2.1 at the author's request ("group similar files"); the spec layout
  now matches the repository.
- *One source of truth for file names.* `window_features.parquet` alone was declared in five scripts, and the four
  score files in three. They are now defined once in `src/metroguard/artifact_names.py`, so a writer and a reader
  can no longer drift apart.
- *Removed* one dead constant (`EXPECTED_DIGITAL_COUNT`) and all 41 `# fmt: skip` markers; the formatter now
  decides layout everywhere.
- *Tooling*: `.gitattributes` (LF in the repository, binary and generated files marked), `.editorconfig`,
  `requirements.txt` split into runtime and `requirements-dev.txt`, and a tighter `.gitignore` (`.claude/`,
  `.vercel/`, logs, OS files).
- *Docs*: the README was rewritten (result summary, layout, quick start, pipeline table with the library code behind
  each step), `dashboard/README.md` was added, and module paths in the spec, reports and prompts were updated.

**Verification.** 43 tests pass and `ruff check` / `ruff format --check` are clean. The whole pipeline (audit,
steps 2 to 8, dashboard export) was re-run from scratch after the refactor: `metrics.csv` and every threshold value
are identical to the committed ones, and four of the five dashboard data files are byte-identical; only run
timestamps and runtimes differ, so the committed timestamped files were kept.

**Left for the author to decide** (not changed without asking): a `LICENSE` for the code (the data is CC BY 4.0),
whether `docs/claude_code_prompts.md` and the explainer PDF belong in a public repository, and whether to add a
GitHub Actions workflow (tests and ruff on every push).

## Dashboard deployment and default theme (2026-10-11)

- Deployed on Vercel with **Root Directory `dashboard`** and the **Other** preset (no build). With the repository
  root as the Vercel root, the whole repository was served and `/` returned 404; setting the root fixed both.
  Live at https://metroguard-compressor-anomaly-detec.vercel.app; all six pages and the data files return 200.
- The site now **opens in the light theme for every visitor**, regardless of the operating system setting. The dark
  theme is opt-in through the sidebar button, the button label names the theme it switches to, and the choice is
  remembered across pages and visits. Checked with the browser emulating a dark operating system.

## Analyse your data: the pipeline in the browser (second scope extension, 2026-10-11)

**Why this is outside the original scope.** The author asked that a visitor be able to supply their own dataset and
get the same analysis, with the MetroGuard results unchanged when no file is given. Choices confirmed by the author:
the computation runs **in the visitor's browser** (no server, the file is never uploaded), **any time-series file**
is accepted with a column-mapping step, and the formats are **CSV, Excel and PDF tables**. `docs/project_spec.md`
(scope lock, layout, new step 10b) and `CLAUDE.md` were amended for this page only; nothing else in the scope lock
changed, and `dashboard/data/` and `reports/` are untouched by it.

**What was built.**
- `dashboard/assets/analysis/` re-implements steps 1 to 8 in plain JavaScript (no dependencies), module by module
  against the Python code: `table-reader.js` (streaming quote-aware CSV parser, delimiter sniffing, column-role
  detection), `timestamps.js` (ISO, day-first, month-first, Unix seconds and milliseconds, spreadsheet dates, read
  as naive wall-clock time), `preparation.js` (cadence audit, frozen runs, windows, the chronological split with the
  same rounding as Python, fit-only scaler and quantile bins), `rbm.js` (Gibbs sampling, CD-k, free energy),
  `vae.js` (dense layers, manual backpropagation, Adam, early stopping that restores the best epoch), `markov.js`,
  `bayesian-network.js` (hill climb with BIC, tabu list and the same legal-move rules as pgmpy, BDeu parameters,
  d-separation, Markov-blanket surprisal), `alerting.js`, `bundle.js` (writes the same JSON documents as step 10)
  and `pipeline.js`, which runs them in order. A seeded generator (Mulberry32) replaces NumPy's, so a run is
  repeatable but not bit-identical to Python.
- `worker.js` runs the analysis in a module Web Worker; `analyse.html` and `page-analyse.js` handle file choice,
  a preview with one role selector per column, the timestamp format, the PDF confirmation, settings, optional
  reported windows, progress with cancel, and the summary. `file-import.js` reads the files; `saved-analysis.js`
  keeps the latest result in IndexedDB.
- `app.js` now loads either the MetroGuard JSON or the saved analysis, shows a banner with "Back to MetroGuard
  results", and adds the page to the navigation. All six pages were generalised from four fixed events to any
  number (including none: detection columns and delay tables are then hidden, and every alert is unconfirmed), and
  their MetroGuard-specific sentences are kept for the MetroGuard data only; uploaded data gets neutral wording.
- SheetJS 0.20.3 and pdf.js 4.10.38 (Apache 2.0, licences in `dashboard/assets/vendor/`) are vendored and loaded
  only when an Excel or PDF file is chosen. No build step was added.
- 13 Node unit tests in `dashboard/tests/engine.test.mjs` (`node --test`): CSV parsing across chunk boundaries,
  timestamp detection, column roles, numpy-compatible quantiles and rounding, the split, frozen runs and windows,
  the RBM free energy against brute force, CD-k lowering the free energy, VAE training, Markov states, structure
  recovery and d-separation, alert merging and delay, and an end-to-end run that checks the bundle and that every
  model's thresholds were frozen. `.vercelignore` keeps the tests out of the deployment.

**Rules kept.** Only the visitor's own file is analysed and nothing synthetic is generated. The split is by time;
scaler, bins, state cut points, network structure, CPDs and thresholds come from the fit or calib windows; each
model's thresholds are recorded with a timestamp before it scores a test window; evaluation is at event level and
alerts outside the reported windows are "unconfirmed". The pages make no causal or prediction claims. Visitors
cannot tune hyperparameters: they choose only what depends on their data (column roles, timestamp format, training
period, window length, Markov state column, which of k = 1, 5, 10 to train, and reported windows).

**Parity on the real MetroPT-3 file** (all 1,516,948 rows, defaults, the four reported windows), 99th percentile:

| Model | Python flagged | Browser flagged | Python / browser detected | Python / browser unconfirmed per day |
|---|---:|---:|:---:|---:|
| Z-score baseline | 10.40% | 10.40% | 4 / 4 | 25.85 / 25.85 |
| RBM, k = 1 | 19.88% | 21.09% | 4 / 4 | 29.97 / 27.89 |
| RBM, k = 5 | 20.25% | 20.29% | 4 / 4 | 29.82 / 29.64 |
| RBM, k = 10 | 20.79% | 20.24% | 4 / 4 | 28.59 / 29.61 |
| VAE | 21.38% | 21.55% | 4 / 4 | 31.02 / 28.57 |
| Markov chain | 0.44% | 0.44% | 3 / 3 (events 2, 3, 4) | 3.65 / 3.65 |

The audit (cadence 10 s, 37.9 days without readings, 50,425 frozen readings in 7 runs), the split, the 206,857
usable test windows, the z-score thresholds (4.7052) and the Markov threshold (4.9048) are identical. The RBM and
VAE differ slightly because their random initialisation and batch order come from a different generator (the VAE
stopped at epoch 41 with the best at 36; Python ran 50 with the best at 47); none of the conclusions changes.
The Bayesian-network search reaches exactly pgmpy's score: BIC −140,851.6 with 18 edges. Fifteen edges are the
same as Python's; in the other three `COMP_active` stands where Python has `MPG_active`, and the two signals are
identical in all 28,401 fit windows, so both graphs score the same. Getting there took two changes. A single
hill climb is path-dependent (pgmpy's own logic, confirmed in its source), so the search runs from 10 seeded node
orders and keeps the best BIC. And because moves such as X→Y and Y→X have equal scores up to the last bit, which
differs between JavaScript engines (Node found −140,866.0 with 19 edges, Chromium −140,851.6 with 18, from the
same code), near-equal moves are now treated as ties and the first in order wins. Node and Chromium then learn
the same graph with the same 10 restart scores.
The whole run took 2 min 49 s in Chromium on the author's laptop (CPU only; RBM 117 s, VAE 38 s), and the saved
result is 1.9 MB.

**Checked in a browser.** The full CSV above; an Excel workbook whose first sheet is notes (the page now opens the
first sheet that looks like a time series); a 32-page PDF table (1,863 rows recovered exactly, 32 title lines left
out, repeated headers dropped) run without reported windows; switching back to the MetroGuard results; all seven
pages in both modes with no error banner and no `null`, `NaN` or `undefined` in the text; and a responsive sweep
of all seven pages showing an uploaded analysis at seven widths from 280 to 1440 px (49 of 49 passed: no horizontal
page overflow, nothing outside the viewport, no error banner).

**Bugs found and fixed while testing.** The CSV parser duplicated quoted fields (found by the unit tests); the
latent-space sample stopped at 8,469 points instead of 12,000; the Markov chain was skipped on hourly data because
auto mode only tried `Motor_current` (it now falls back to the sensor with the most levels); without reported
windows the network case study picked a model that raised no alert (it now picks the most selective model that
did); optional sections passed to `append()` printed the text "null" (a `compact()` helper filters them); the
network search depended on the JavaScript engine (see above).

**Limits.** Very large files are bounded by browser memory (the 218 MB CSV worked). Excel files hold at most
1,048,576 rows. PDF tables are rebuilt from text positions, so scans are not supported and the preview must be
confirmed. A short file can fail the split with a clear message ("too few windows"). The JavaScript models run on
the CPU, so they are slower than the GPU pipeline.

## Possible future work

- Clip or robustly scale near-constant features (such as `LPS_active_fraction`) before the VAE, and train
  the VAE longer. Not done now because it is outside the spec's fixed preprocessing and epochs.
- Train the RBM for more epochs (its curves had not flattened at epoch 30), and compare k on
  converged models. Not done now because the spec fixes epochs at 30 and forbids tuning.
