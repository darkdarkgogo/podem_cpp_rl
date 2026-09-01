# Historical Reward Monitor

This is a retrospective export, not a new training run or live instrumentation.
The training environment and source logs/checkpoints were not modified.

## Open TensorBoard

From the `PODEM` directory in PowerShell:

```powershell
$export = Get-Content artifacts/reward_monitor/latest_export.json -Raw | ConvertFrom-Json
& 'artifacts/reward_monitor/.venv/Scripts/python.exe' -m tensorboard.main --logdir $export.logdir --host 127.0.0.1 --port 6006 --load_fast false --samples_per_plugin scalars=10000 --reload_interval 30
```

Open <http://127.0.0.1:6006>. Use the Step axis and smoothing 0 initially.
Do not set `--samples_per_plugin scalars=0`: this backend returns no scalar samples.
TensorBoard 2.17.1 is installed only in `.venv` with access to existing training
dependencies through system site-packages. Its scalar dashboard does not require TensorFlow.

Recommended tag filter:

```text
^(reward/rolling100_mixed_tasks|sweep/reward_mean_mixed_tasks|same_200_hard_train_faults/reward_mean|validation/detected_of_500)$
```

For circuit-specific hard-fault trends, filter `same_hard_by_circuit/`.
The latest export also contains `reward_overview.png`, `reward_by_circuit.png`,
`episode_metrics.csv`, `summary.json`, and `provenance.json`.

## Interpretation

SmartATPG-GAE does not yet show convincing stable convergence in this run:

| Metric | Hard sweep 1 | Hard sweep 2 | Hard sweep 3 |
| --- | ---: | ---: | ---: |
| SmartATPG mean return, all 600 training faults | 85.75 | 86.08 | 86.54 |
| SmartATPG mean return, same 200 hard training faults | 82.75 | 82.52 | 75.88 |
| SmartATPG detected, same 200 hard training faults | 178 | 178 | 171 |
| DeepGate mean return, same 200 hard training faults | 76.93 | 76.93 | 76.93 |

The overall mean looks like a plateau but masks regression on the hard subset.
For example, SmartATPG's same 20 hard c6288 faults have mean returns of
107.73, 107.36, and 60.01. A flat DeepGate curve alone does not establish optimality.

The fixed 500-fault validation set detected 497 faults at SmartATPG's selected
medium-sweep-1 checkpoint and 495 at the final checkpoint. Validation reward was
not recorded and cannot be recovered from aggregate detection/search counts.

## Definitions and Limits

- Each run has 4,200 complete-fault episodes, including 199 zero-decision episodes;
  optimizer diagnostics cover 4,001 updates.
- The primary reward curves use external episode return before fixed `/100`
  scaling, excluding the weighted RND bonus. Combined return and RND are separate tags.
- Step means completed training faults, not actor decisions or elapsed time.
  Episode timestamps were unavailable; event wall times are export timestamps.
- Mixed-task rolling means describe the training stream, not a controlled
  convergence test. Curriculum boundaries occur after episodes 800 and 2,400.
- The same-hard cohort contains 20 identical fault IDs per circuit, 200 total,
  across the three hard sweeps. These are training returns collected with evolving
  policies and exploration, not held-out fixed-policy evaluation. Cohort summaries
  are plotted at sweep completion.
- There is one seed and only three hard sweeps. The data do not establish robust
  convergence, generalization, or a universal advantage of either encoder.
- Reward trends and the native full-circuit benchmark measure different things;
  a lower training return does not invalidate the final checkpoint's measured
  search-time improvement.

## Re-export and Checks

```powershell
& 'artifacts/reward_monitor/.venv/Scripts/python.exe' artifacts/reward_monitor/export_reward_curves.py
```

The exporter verifies manifest/checkpoint consistency, unique committed episodes,
all expected fault IDs, every raw external reward after event-file reload, and
source SHA-256 hashes before/after export. It keeps the last attempt for duplicate
stage/sweep/circuit/fault keys and requires full expected coverage after skipping
malformed JSON log lines. See `provenance.json` for source hashes and definitions.

Original experiment sources: `artifacts/paper_v7_gae_20260831` (DeepGate-GAE) and
`artifacts/paper_v8_smartatpg` (SmartATPG-GAE). Future training runs still need
explicit live TensorBoard logging if live monitoring is desired.
