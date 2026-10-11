# MetroGuard: Anomaly Detection in Metro Air-Compressor Telemetry

Deep Learning II (CDT7001), Semester VII. Author: Akshay Parihar.
Every number in this report comes from a saved output: `reports/data_audit_report.md`,
`reports/decision_log.md`, `reports/tables/` (`metrics.csv`, `metrics.md`, `thresholds.json`, training
histories, Bayesian-network tables) and `outputs/logs/`. This is a one-week exploratory prototype: **no
claim is made that failures can be predicted**, and the Bayesian network shows association, not cause.

## 1. Problem and motivation

A metro train's air-production unit (compressor) produces sensor readings every few seconds. A maintenance
team cannot inspect millions of readings. The question studied here: can probabilistic models learn what
"normal" looks like from an early period, so that later periods that look unusual can be flagged, and can a
Bayesian network say which sensor states look unusual during a flag? The flags are checked against four
air-leak intervals reported by the operator. The goal is a screening aid that narrows down where to look,
not a predictor of failures.

## 2. Data

- **Source:** MetroPT-3 (Air Compressor), UCI Machine Learning Repository,
  https://archive.ics.uci.edu/dataset/791/metropt+3+dataset, DOI https://doi.org/10.24432/C5VW3R,
  licence **CC BY 4.0** (attribution retained here and in the README).
- **Content:** 1,516,948 rows, 2020-02-01 00:00:00 to 2020-09-01 03:59:50, 7 analog signals (TP2, TP3, H1,
  DV_pressure, Reservoirs, Oil_temperature, Motor_current) and 8 digital signals, no missing values, no
  duplicate timestamps, already in time order (audit section 4).
- **Cadence:** the file shows one reading every **10 s** (88.2% of intervals), not the 1 Hz one UCI field
  states; UCI's metadata gives conflicting rates (1 Hz and 0.1 Hz).
- **Gaps:** 351 gaps longer than 20 s, 160 of them at least 1 hour; 37.9 of the 213 days have no readings.
  Seven stretches (50,425 readings, 3.32%) have every analog sensor frozen at the same value for 6 to 62 hours,
  which looks like stuck logging; windows overlapping them are excluded from scoring.
- **Reported events** (treated as intervals, not dense labels): event 1 18 Apr (air leak), event 2 29-30 May,
  event 3 5-7 Jun, event 4 15 Jul. Event 3 has 91.6% reading coverage; the others are about 100%.
- **Metadata issues recorded, not resolved:** UCI numbers two events "#1"; event 2's maintenance note says
  "30Apr", before the event itself (possibly a typo); the timezone is not stated (timestamps and reports
  are assumed to share local time); 29 Feb has no data.

## 3. Method

**Split (chronological, nothing shuffled).** Train = February (1-28 Feb, no overlap with any reported
window). Within it, by usable 60 s window: `fit_train` 25,561, `fit_validation` 2,840 (early stopping only),
`calib` 7,100 (thresholds only). `test` = everything after February: 206,857 usable windows.

**Windows and features.** 60 s windows (6 readings; at least 3 required). Per window: 7 analog means, the
std of Motor_current, TP2 and TP3, and the active fraction of each digital signal. Standardiser, quantile
bin edges (3 bins) and motor-current state cut points are fitted on `fit` windows only.

**Models** (each fitted on `fit`; thresholds from `calib`):

| Model | Score | Syllabus link |
|---|---|---|
| z-score baseline | max over 7 analog means of \|z\| (and mean \|z\|) | baseline |
| RBM, Gibbs sampling + CD-k (k = 1, 5, 10), implemented by us in PyTorch on the GPU | free energy F(v) | Units II-IV (see the derivation in the decision log) |
| VAE (18-64-32-6, beta = 1, PyTorch, GPU) | reconstruction error; negative ELBO as secondary | generative models |
| Markov chain over 3 motor-current states (add-one smoothing) | max transition surprisal in the window | Markov models |

**Thresholds and alerts.** Threshold = 99th percentile of the score on `calib` (99.5th as a sensitivity row),
written to `reports/tables/thresholds.json` before that model scored test (timestamps in `metrics.md`).
Alert = score strictly above the threshold; alerts closer than 10 minutes are merged into events.

**Bayesian network.** 11 nodes (7 analog states plus COMP, DV_eletric, Towers, MPG), structure by Hill-Climb
with BIC on `fit` windows, CDPs by BDeu estimation; 18 edges. Used for syllabus demonstrations and to give
context to two alert events.

**Evaluation.** Event level, on usable test windows only. A reported window counts as detected if an alert
event overlaps [start - 24 h, end]. Alerts outside all such ranges are **unconfirmed alerts**.

## 4. Results

### 4.1 Metrics (usable test windows, 184.0 calendar days; `reports/tables/metrics.md`)

Primary scores at the 99th percentile threshold (the 99.5th percentile row is in the table below it):

| model | alert-window fraction | events detected | unconfirmed alerts/day |
|---|---|---|---|
| z-score baseline (max \|z\|) | 0.104 | 4 of 4 | 25.85 |
| RBM k = 1 | 0.199 | 4 of 4 | 29.97 |
| RBM k = 5 | 0.203 | 4 of 4 | 29.82 |
| RBM k = 10 | 0.208 | 4 of 4 | 28.59 |
| VAE (reconstruction error) | 0.214 | 4 of 4 | 31.02 |
| Markov state model | 0.0044 | 3 of 4 (events 2, 3, 4) | 3.65 |

At the 99.5th percentile: baseline 0.064 / 4 of 4 / 9.32; RBM k = 1, 5, 10: 0.171, 0.172, 0.171 / 4 of 4 /
30.02, 32.38, 32.36; VAE 0.139 / 4 of 4 / 28.26; Markov 0.0044 / 3 of 4 / 3.65 (its two thresholds are equal
because the score takes few discrete values).

### 4.2 How to read these numbers

- Thresholds were set to flag about 1% of `calib`, but 10% to 21% of test windows are flagged by the baseline,
  RBM and VAE. February is not representative of later months, so these models mostly respond to "different
  from February". That produces about 25 to 40 unconfirmed alert events per day.
- The four detection ranges cover only 7.6 of 184.0 test days (4.2%). A model flagging 10% to 21% of windows
  overlaps them largely by chance, so "4 of 4" is weak evidence. Only the Markov model, with a 0.44% alert
  fraction, is selective, and it misses event 1.
- Detection delays near -24 h are an artefact of clipping to the start of the lead range, not early warning
  (for event 4, -17 h matches when data resumes after a gap). Delays per model are in `metrics.md`.

### 4.3 Training and CD-k comparison

- **RBM** (30 epochs, GPU): fit reconstruction error fell from 0.180 to about 0.061 for all three k, and the
  curves had not flattened at epoch 30, so the models are under-trained. k = 1, 5 and 10 are almost
  indistinguishable (fit reconstruction 0.0610, 0.0608, 0.0609; `reports/figures/rbm/rbm_training_curves_cd_k.png`,
  `reports/figures/evaluation/rbm_cd_k_comparison.png`).
- **VAE** (50 epochs, GPU): best epoch 47 (validation loss 2.779); early stopping did not trigger. A one-epoch
  training spike at epoch 16 is explained by `LPS_active_fraction`, nearly always 0 in February, which
  standardises to z up to 168 (`reports/figures/vae/vae_training_curves.png`).

### 4.4 Bayesian network case study (`reports/tables/bayesian_network_*.md`)

- 18 edges learned; the strongest dependency, `Reservoirs_mean -> TP3_mean` (1.085 nats), is two pressure
  sensors that behave almost as copies. The direction of some edges is not identifiable (they flipped
  between two runs before the hash seed was fixed), a reminder that edges are not causes.
- Demonstrations: a chain, a fork and a collider show the expected d-separation behaviour (observing the middle
  node blocks a chain and a fork and opens a collider), and the three local-independence cases behave as theory
  predicts.
- Alert context for the baseline's alert events in reported windows 1 and 2: `Towers_active` and `MPG_active`
  were unusual given their Markov blankets (more than twice the share above the calib 95th percentile
  expected by chance); the analog variables were not. The effect sizes are small. This describes association
  with the alert, not a cause.
- Figures: `reports/figures/bayesian_network/bayesian_network_dag.png`; scores over time per model, one zoom
  per event and the CD-k comparison are in `reports/figures/evaluation/`.

## 5. Discussion

- **Which approach worked best?** On this evidence none of the learned models beat the simple z-score baseline.
  The baseline had the lowest alert rate of the baseline/RBM/VAE group at both percentiles (0.104 vs 0.199-0.214
  at p99), and the RBM and VAE detected no more events. The Markov model was the most selective (3.65
  unconfirmed alerts/day), but it uses one signal and missed one event.
- **Why did the models struggle?** The training month is short and different from the rest of the record: the
  high motor-current state is 6.1% of readings in February against 10.6% to 27.6% later, and four digital signals
  are nearly constant in February. Distribution shift dominates the scores. The RBM and VAE were also not
  converged at the configured epoch counts.
- **Where each failed.** Baseline: many unconfirmed alerts. RBM: under-trained, k makes no visible difference.
  VAE: rare activations of a near-constant signal produce huge scores. Markov: one signal only, event 1 missed.
- **What was not tuned.** No setting was changed after seeing test results. Only the spec's fixed values were
  used; ideas for follow-up are in the decision log under "Possible future work".

## 6. Limitations

- Only four reported events, on one machine; results are exploratory and cannot support general conclusions.
- Reported windows are not dense labels: rows inside are not uniformly faulty, rows outside are not guaranteed
  normal. Large sustained high-DV_pressure periods outside the windows may be unreported events or normal use.
- Alerts outside reported windows are **unconfirmed alerts**; they are not known to be false.
- No causal claim (Bayesian network edges are learned dependencies, direction is partly arbitrary) and no
  prediction claim (delays are not early warnings).
- Timezone unknown; 37.9 days missing; 3.32% of readings are frozen values (excluded from scoring, but the
  excluded windows include part of event 1's context).
- Training period of one month, RBM and VAE not converged, CD-k not compared on converged models.
- `calib` was never used for training decisions (early stopping used `fit_validation`).

## 7. Theory comparison (no implementation)

| | Likelihood | Sampling | Training stability |
|---|---|---|---|
| RBM (implemented) | unnormalised; the partition function makes exact likelihood intractable | Gibbs sampling, alternating layers; chains can mix slowly | CD-k is a biased but cheap gradient; sensitive to learning rate |
| VAE (implemented) | lower bound (ELBO), not exact | one pass: sample z, decode | stable with Adam; risk of ignoring the latent (posterior collapse) |
| GAN | implicit; no likelihood available | one pass through the generator | unstable: min-max game, mode collapse, no convergence guarantee |
| NADE | exact, tractable (autoregressive factorisation) | sequential, one variable at a time | stable maximum-likelihood training; depends on variable ordering |
| MADE | exact, tractable (masked autoencoder gives an autoregressive model) | sequential, one variable at a time | stable; depends on ordering and masks |

For anomaly scoring this matters: exact-likelihood models (NADE, MADE) give a principled score, the RBM's free
energy is an unnormalised surrogate, the VAE's score is a bound or a reconstruction error, and GANs offer no
likelihood-based score.

## 8. Syllabus mapping

| Unit | Where it appears |
|---|---|
| II | Bipartite RBM, conditional independence, why Gibbs sampling works (decision log, step 04) |
| III | Why sampling is needed for the RBM's intractable model term (decision log, step 04) |
| IV | CD-k as an approximation to the log-likelihood gradient (step 04) |
| Bayesian networks | Hill-Climb + BIC structure learning, BDeu CPDs, d-separation, local independence, causal-direction and evidential queries, Markov-blanket alert context (step 07) |
| VAE / Markov model | Steps 05 and 06 |
| I, V | The unit numbers for the remaining topics are not given in the project spec; fill from the course outline |

## 9. Reproduction and runtime

```
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
python pipeline/step01_audit_raw_data.py
python pipeline/run_all_steps.py
```

Place `MetroPT3(AirCompressor).csv` in `data/raw/` first. Runtime on this machine (RTX 4050 laptop GPU) from
`outputs/logs/*_manifest.json`: step 1 54.98 s; steps 2-8 together 119.63 s (step 2 1.81 s, step 3 0.59 s,
step 4 29.17 s, step 5 39.10 s, step 6 0.67 s, step 7 2.27 s, step 8 6.99 s); about 174.6 s in total.
Software is pinned in `requirements.txt` (older builds of numpy, scipy and torch were used because Windows
Smart App Control blocked newer ones; see the decision log).
