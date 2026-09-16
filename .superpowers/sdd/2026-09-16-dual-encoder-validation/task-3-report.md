# Task 3 Report: Dual Validation Comparison with One SCOAP Baseline

## Status

Implemented the Task 3 comparison workflow with strict V8 run validation,
one resumable SCOAP validation baseline, and atomic JSON/CSV reports.

The agreed JSON schema contains exactly four `comparisons` groups (two models
times two rounds). Each group contains ordered `rows=[TOTAL, circuit...]`.
`direct_comparisons` contains exactly two round groups with the same ordered
row scopes. CSV flattens every model-vs-SCOAP and GAT-minus-MEAN row.

## RED

Command:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Key output:

```text
ModuleNotFoundError: No module named 'compare_smartatpg_validation'
Ran 1 test in 0.001s
FAILED (errors=1)
```

## GREEN

Focused command:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Key output:

```text
Ran 20 tests in 0.107s
OK
```

Regression command:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg PODEM.tests.test_smartatpg_training
```

Key output:

```text
Ran 45 tests in 0.261s
OK
```

Additional checks:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m py_compile PODEM/scripts/compare_smartatpg_validation.py PODEM/tests/test_linux_smartatpg.py
git diff --check
```

Both completed successfully with no output.

## Modified Files

- Created `PODEM/scripts/compare_smartatpg_validation.py`.
- Modified `PODEM/tests/test_linux_smartatpg.py`.
- Created this report.

## Implementation and Self-Review

- V8 identity keys are exact; format, encoder, 2/8/1/200 protocol, manifest
  hash, runtime validation catalog hash, circuit order, and rounds 1/2 are
  checked before SCOAP evaluation.
- Zero or multiple `is_best=True` rows are rejected.
- `ScoapValidationEvaluator` selects `heuristic_action` through native
  `backtrace_rl`, forces SCOAP on, and uses the same `_evaluate_fault` event
  callback path as learned validation.
- The complete runtime fault order is evaluated only once. The cache stores
  identity, per-fault records, and summary; resume recomputes and validates the
  summary before reuse.
- JSON and CSV retain total/per-circuit raw counts, work, return, timing,
  coverage, SCOAP deltas/reductions, and same-round GAT-minus-MEAN values.
- Reports use temporary files followed by replacement.
- No dual-GPU orchestrator was added.

## Concerns

The existing Task 2 training code marks every newly improved round
`is_best=True` without clearing an earlier best row. If round 2 improves over
round 1, its emitted metrics may therefore contain two best rows. Task 3
intentionally rejects that file because the brief explicitly requires a unique
best row. This cross-task producer/consumer mismatch was reported to the parent
task and was not changed here because Task 2 is outside this task's scope.

No reviewer or sub-agent was used, as required by the task brief.

## Review Fix Round 1/5: Complete and Consistent Model Metrics

The Important review finding was reproduced before implementation. New tests
mutated an otherwise valid model round in four ways: one circuit episode short
of its runtime catalog, a TOTAL additive metric inconsistent with circuit rows,
an invalid derived fault coverage, and an invalid derived mean.

RED command:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg.ValidationComparisonTests.test_rejects_incomplete_or_inconsistent_round_metrics
```

RED key output:

```text
Ran 1 test in 0.059s
FAILED (failures=4)
AssertionError: ValueError not raised
```

The comparison loader now validates both model runs after resolving the
runtime catalog and before starting or reusing SCOAP. Every circuit episode
count must match its catalog, TOTAL episodes must match the complete validation
order, all additive raw totals/counts/work/time must equal the circuit-row sum,
and coverage plus backtracks/backtrace/return means must match their raw totals.

GREEN commands:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg.ValidationComparisonTests.test_rejects_incomplete_or_inconsistent_round_metrics
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

GREEN key output:

```text
Ran 1 test in 0.046s
OK
Ran 21 tests in 0.191s
OK
```

`py_compile` and `git diff --check` also completed successfully with no
output. The two Minor review notes were intentionally left unchanged for later
rounds, as directed. No sub-agent was used.
