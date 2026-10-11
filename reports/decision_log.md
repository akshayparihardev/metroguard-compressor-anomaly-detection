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

**Environment.** Python 3.14.6 (the only interpreter installed). Windows Application Control
blocked loading torch's `caffe2_nvrtc.dll` (both the CUDA 12.8 wheel in `.venv` and the existing
system install) and numpy 2.5.4's `mtrand` DLL. numpy 2.4.6 / pandas 2.3.3 / pyarrow 23.0.1 /
matplotlib 3.10.8 / pgmpy 1.1.2 load fine and are pinned. A Python 3.12 venv would not help,
because the block is an OS policy, not a Python version issue. **torch is blocked for step 5
until this is resolved.**

## Possible future work

- None yet.
