# Claude Code prompts for MetroGuard

Copy-paste prompts, one per step, in order. Rules live in `CLAUDE.md`, so prompts only say what is specific to the step.

**How to use**
- One step per prompt. Do not combine steps.
- After each step, run the **Teach me** prompt and the **Review** prompt before moving on.
- If something is wrong, use a **Fix** prompt; do not start the next step on a broken one.
- Start a new Claude Code session every 2–3 steps to keep context clean. The first message of a new session is always the **Session start** prompt.

---

## Session start (use at the beginning of every new session)

```
Read CLAUDE.md, docs/project_spec.md and reports/decision_log.md (if it exists).
Tell me in 5 lines or fewer: which steps are done, what the last commit was,
and which step is next. Do not write code yet.
```

---

## Step 1 — Audit raw data

```
Implement step 1 exactly as in docs/phase1_data_audit_spec.md and docs/project_spec.md
sections 2, 3 and 5. First show me a short plan: files you will create, how you will
parse timestamps, how you detect gaps, how you identify the 8 digital columns from the
real headers, and whether failure-window boundaries are inclusive or exclusive and why.
Wait for my approval of the plan.
```

After approval:

```
Plan approved. Implement step 1, run it on the real CSV, run pytest and ruff,
commit, and then stop at the step 1 checkpoint. Show me:
1. Key audit numbers (rows, time span, actual cadence, gaps, duplicates, missing values).
2. Digital-signal findings (binary or not, near-constant columns).
3. Coverage of each of the 4 failure windows.
4. Candidate train period dates and confirmation of zero overlap with failure windows.
5. Motor-current findings: are distinct operating states visible?
6. Your proposed step 2 decisions with a reason for each: window size, number of bins,
   fit/calib/test date ranges, which features to keep, whether step 6 (Markov) is supported.
Do not start step 2.
```

## Step 1 → 2 approval

Edit the decisions if you disagree, then send:

```
Step 2 decisions approved: [paste the final list here].
Write them into config.yaml and reports/decision_log.md with their reasons, commit,
then implement step 2 (docs/project_spec.md, Step 2). Verify and report:
- row counts and date ranges of fit, calib and test;
- that the scaler, bin edges and mean/std were fitted on fit only (show how the code ensures it);
- shapes and dtypes of each saved representation;
- number of windows dropped for too few readings.
Run pytest and ruff, commit, and stop.
```

## Step 3 — Z-score baseline

```
Implement step 3 (z-score baseline) and the shared threshold and alert-event code
(alert_thresholds.py, alert_events.py) from docs/project_spec.md.
Compute thresholds from calib only and save them to reports/tables/thresholds.json
before scoring test. Add tests for threshold fitting and alert-event merging using
small hand-made arrays. Report the thresholds and the number of alert events in calib
and in test. Do not evaluate against failure windows yet. Run pytest and ruff, commit, stop.
```

## Step 4 — RBM with Gibbs sampling and CD-k

```
Implement step 4 (RBM) from docs/project_spec.md. Write the RBM ourselves: energy,
free energy (stable softplus), p(h|v), p(v|h), Gibbs step, CD-k update.
First write tests/models/test_restricted_boltzmann_machine.py with the brute-force free-energy
check on a 4-visible, 3-hidden RBM and make it pass before training on real data.
Then train k = 1, 5, 10 with the config values, save weights to outputs/models/,
training-curve figures to reports/figures/rbm/, free-energy scores for calib and test
to outputs/scores/, and add the k values' thresholds to thresholds.json before test scoring.
Add the derivation note (gradient = data term − model term, partition function,
why Gibbs works on a bipartite graph, what CD-k approximates) to the decision log.
Report: training time per k, final reconstruction error and mean free energy on fit vs calib.
Run pytest and ruff, commit, stop.
```

## Step 5 — VAE

```
Implement step 5 (VAE) from docs/project_spec.md in PyTorch. Use the GPU if available.
Early stopping on calib loss. Save the model, loss curves (reports/figures/vae/),
reconstruction-error and negative-ELBO scores for calib and test, and the thresholds
(before test scoring). Report: epochs run, best calib loss, training time, and whether
calib loss diverged from fit loss. Run pytest and ruff, commit, stop.
```

## Step 6 — Markov state model (only if step 1 supported it)

```
Implement step 6 from docs/project_spec.md only if the decision log says motor-current
states are supported. Show the state definition used, the transition matrix (fit only,
add-one smoothing) as a table and a heatmap, then score calib and test and save
thresholds before test scoring. If step 6 is not supported, write the reason in the
decision log and skip. Run pytest and ruff, commit, stop.
```

## Step 7 — Bayesian Network

```
Implement step 7 from docs/project_spec.md with pgmpy. Use the discretised states from
step 2, fit only. Keep it to about 8–12 nodes. Produce:
1. The learned DAG as a figure (reports/figures/bayesian_network/).
2. One causal query and one evidential query, with the numbers.
3. d-separation checks and one example for each of the three local-independence cases
   (node and parents, node and non-parents, node and descendants).
4. Alert context for 1–2 alert events from the step 3–5 model with the lowest calib
   alert rate, using "unusual / associated" wording only.
Write all of this to the decision log. Run pytest and ruff, commit, stop.
```

## Step 8 — Evaluate alerts

```
Implement step 8 from docs/project_spec.md. Before computing anything, confirm that
reports/tables/thresholds.json was written before test scoring and print its timestamp
and the lead_tolerance value from config. Then compute, for every model on test only:
events detected, detection delay per event, unconfirmed alerts per day, alert-window
fraction, and the 99 vs 99.5 sensitivity rows. Save metrics.csv and metrics.md, the
score-vs-time figures with shaded failure windows, per-event zooms and the CD-k
comparison. Show me the metrics table. Do not change any threshold or config value
after seeing results. Run pytest and ruff, commit, stop.
```

## Step 9 — Final report and README

```
Write reports/final_report.md and README.md as in docs/project_spec.md, Step 9.
Use only numbers from reports/tables/ and outputs/; for every number, the source file
must exist. Include limitations, the GAN/NADE/MADE theory section, the syllabus mapping
table and total runtime. Then go through the Definition of done checklist and mark each
item pass or fail with evidence. Run pytest and ruff, commit, stop.
```

---

## After every step

### Teach me (for viva preparation)

```
Explain the step you just finished as if teaching me for a viva:
concept → intuition → how our code does it (point to file and function names) →
what the numbers we got mean → 3 likely viva questions with short answers.
Do not change any files.
```

### Review (best in a fresh session, so it does not grade its own work)

```
Act as a strict reviewer. Do not change any files. Check step [N] against
docs/project_spec.md and CLAUDE.md:
1. Does anything use test data or failure windows to fit, tune or choose a value?
2. Does any number in reports/decision_log.md lack a saved output that produced it?
3. Any naming, layout or hygiene violation (spec 2.1, 2.2, 3)?
4. Any scope creep beyond the spec?
5. Any bug or numerical risk in the code you can point to by file and line?
List findings ranked by severity, each with the exact fix.
```

---

## Fix prompts

### Something is wrong

```
This is wrong: [describe what you saw, paste the error or the number].
Find the cause, explain it in 2–3 lines, fix it with the smallest change,
re-run the affected step, and show me the before and after.
Do not change anything unrelated.
```

### Claude Code went off-scope

```
Stop. [file/feature] is outside docs/project_spec.md. Remove it, restore the
spec layout, and move the idea to "Possible future work" in the decision log.
Show me git status afterwards.
```

### Results look too good or too bad

```
Before we trust these results, check for leakage: confirm that scaler, bins, model
weights and thresholds were fitted only on fit or calib, that test windows were never
used for any choice, and that failure-window timestamps are aligned with the data's
timestamps. Show the code lines that prove each point.
```
