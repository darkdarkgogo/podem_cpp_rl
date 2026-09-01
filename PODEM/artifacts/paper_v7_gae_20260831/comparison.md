# GAE vs V6: One-Seed Experiment

Same ten-circuit fault split, seed 2026, BC 20 epochs, curriculum sweeps 2/2/3.
CPU single-thread training; native evaluation uses seed 14 and backtrack limit 500.
Each model/circuit gets 1 warmup and 5 interleaved measured runs; times are medians.
ATPG time is the native stage interval. On this Windows build, CRT clock() measures elapsed time, not process CPU time.
Timing reference: [Microsoft CRT documentation](https://learn.microsoft.com/en-us/cpp/c-runtime-library/time-management?view=msvc-170).
This compares the new pipeline, including gamma/Advantage normalization/Critic initialization changes, not GAE alone.
Full-circuit evaluation includes training/validation faults; it is not an unseen-circuit generalization test.

Old selected checkpoint: medium_sweep_1, validation score [-497, 3, 0.9720533964928707, 1.0431815416066397, 25218].
New selected checkpoint: behavior_cloning, validation score [-496, 4, 1.0, 1.0, 27474].

IMPORTANT: The new pipeline fell back to behavior cloning; no post-GAE/PPO checkpoint beat it on the selection score. The new selected benchmark columns describe this fallback, not the final PPO policy.

## Detection and Aborts

| Circuit | Heuristic detected | V6 detected | New selected detected | Heuristic abort | V6 abort | New selected abort |
|---|---:|---:|---:|---:|---:|---:|
| c432 | 782 | 782 | 782 | 4 | 5 | 4 |
| c499 | 751 | 766 | 751 | 13 | 8 | 13 |
| c1355 | 2702 | 2702 | 2702 | 8 | 8 | 8 |
| c1908 | 3560 | 3552 | 3560 | 76 | 84 | 76 |
| c2670 | 5010 | 5006 | 5010 | 41 | 45 | 41 |
| c3540 | 6201 | 6164 | 6201 | 66 | 77 | 66 |
| c5315 | 10454 | 10454 | 10454 | 13 | 13 | 13 |
| c7552 | 14914 | 14942 | 14914 | 120 | 84 | 120 |
| c6288 | 12508 | 12508 | 12508 | 2 | 2 | 2 |
| s38417_scan | 76430 | 76431 | 76430 | 52 | 52 | 52 |

## ATPG Elapsed Time

| Circuit | Heuristic s | V6 s | New selected s | New vs V6 | New vs heuristic |
|---|---:|---:|---:|---:|---:|
| c432 | 0.0090 | 0.0160 | 0.0120 | -25.0% | +33.3% |
| c499 | 0.0450 | 0.0370 | 0.0520 | +40.5% | +15.6% |
| c1355 | 0.0740 | 0.0850 | 0.1610 | +89.4% | +117.6% |
| c1908 | 0.3840 | 0.4550 | 0.4150 | -8.8% | +8.1% |
| c2670 | 0.3010 | 0.3550 | 0.3210 | -9.6% | +6.6% |
| c3540 | 0.9470 | 1.1020 | 1.0240 | -7.1% | +8.1% |
| c5315 | 0.2780 | 0.3040 | 0.2930 | -3.6% | +5.4% |
| c7552 | 2.3500 | 1.7500 | 2.4790 | +41.7% | +5.5% |
| c6288 | 0.0950 | 0.1090 | 0.1070 | -1.8% | +12.6% |
| s38417_scan | 10.8280 | 11.0680 | 11.5000 | +3.9% | +6.2% |

Aggregate (sum of per-circuit medians for time):

```json
{
  "heuristic": {
    "detected": 133312,
    "total_faults": 135344,
    "equivalent_detected": 61122,
    "equivalent_faults": 62292,
    "aborted": 395,
    "backtracks": 238872,
    "backtrace_steps": 3906240,
    "atpg_seconds": 15.311,
    "wall_seconds": 16.464415199999834
  },
  "old_v6": {
    "detected": 133307,
    "total_faults": 135344,
    "equivalent_detected": 61136,
    "equivalent_faults": 62292,
    "aborted": 378,
    "backtracks": 220808,
    "backtrace_steps": 3834293,
    "atpg_seconds": 15.280999999999999,
    "wall_seconds": 18.493760499999993
  },
  "gae": {
    "detected": 133312,
    "total_faults": 135344,
    "equivalent_detected": 61122,
    "equivalent_faults": 62292,
    "aborted": 395,
    "backtracks": 238872,
    "backtrace_steps": 3906240,
    "atpg_seconds": 16.364,
    "wall_seconds": 19.56766700000007
  }
}
```

Detected counts are uncollapsed weighted fault counts; aborted counts are collapsed fault attempts.
Keep coverage and runtime separate when deciding whether a model is better.

## New Pipeline Validation History

| Checkpoint | Detected / 500 | Aborted | Mean backtrack ratio | Mean backtrace ratio |
|---|---:|---:|---:|---:|
| behavior_cloning | 496 | 4 | 1.0000 | 1.0000 |
| easy_sweep_1 | 496 | 4 | 19.6714 | 1.1523 |
| easy_sweep_2 | 494 | 6 | 2.3107 | 1.1285 |
| medium_sweep_1 | 494 | 6 | 2.3107 | 1.1285 |
| medium_sweep_2 | 494 | 6 | 2.3107 | 1.1285 |
| hard_sweep_1 | 494 | 6 | 2.3107 | 1.1285 |
| hard_sweep_2 | 494 | 6 | 2.3107 | 1.1285 |
| hard_sweep_3 | 494 | 6 | 2.3107 | 1.1285 |
