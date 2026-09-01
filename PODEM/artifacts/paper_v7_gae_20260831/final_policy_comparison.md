# Actual Final GAE Policy vs Old V6

Actual final GAE actor, not the validation-selected BC fallback. Old V6 and the heuristic are remeasured in this same paired run.
Same circuits, seed 14, backtrack limit 500. One warmup plus five measured interleaved runs; elapsed times are medians.
One training seed; same-circuit evaluation includes training/validation faults. This is not a GAE-only ablation.

| Circuit | V6 detected | Final GAE detected | V6 abort | Final GAE abort | V6 ATPG s | Final GAE ATPG s | Time change |
|---|---:|---:|---:|---:|---:|---:|---:|
| c432 | 782 | 781 | 5 | 6 | 0.0160 | 0.0160 | +0.0% |
| c499 | 766 | 766 | 8 | 8 | 0.0370 | 0.0380 | +2.7% |
| c1355 | 2702 | 2702 | 8 | 8 | 0.0850 | 0.0450 | -47.1% |
| c1908 | 3552 | 3560 | 84 | 77 | 0.4550 | 0.4250 | -6.6% |
| c2670 | 5006 | 5010 | 45 | 41 | 0.3550 | 0.3070 | -13.5% |
| c3540 | 6164 | 6194 | 77 | 65 | 1.1020 | 0.9460 | -14.2% |
| c5315 | 10454 | 10451 | 13 | 14 | 0.3040 | 0.2980 | -2.0% |
| c7552 | 14942 | 14936 | 84 | 112 | 1.7500 | 2.3080 | +31.9% |
| c6288 | 12508 | 12508 | 2 | 2 | 0.1090 | 0.1080 | -0.9% |
| s38417_scan | 76431 | 76429 | 52 | 57 | 11.0680 | 10.2430 | -7.5% |

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
  "gae_final": {
    "detected": 133337,
    "total_faults": 135344,
    "equivalent_detected": 61145,
    "equivalent_faults": 62292,
    "aborted": 390,
    "backtracks": 227855,
    "backtrace_steps": 3798786,
    "atpg_seconds": 14.733999999999998,
    "wall_seconds": 18.144197000000034
  }
}
```

Detected counts are uncollapsed weighted fault counts; aborted counts are collapsed fault attempts.
