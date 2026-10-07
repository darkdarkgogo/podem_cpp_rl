# Default SmartATPG Validation Run Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python scripts/validate_smartatpg.py` use the project-local `artifacts/smartatpg_dual` run without requiring arguments.

**Architecture:** Define a project-root-anchored `DEFAULT_RUN_DIR` beside the existing `ROOT` constant. Use it as argparse's default while preserving explicit `--run-dir` overrides and all existing validation behavior.

**Tech Stack:** Python 3, argparse, pathlib, unittest/pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-default-validation-run-directory-design.md`

## Global Constraints

- The default must be `PODEM/artifacts/smartatpg_dual` regardless of the current working directory.
- Explicit `--run-dir PATH` must continue to override the default.
- No training, checkpoint, dataset, output, seed, or validation-circuit formats may change.

---

### Task 1: Default run directory CLI behavior

**Files:**
- Modify: `PODEM/python/rl_podem/validation.py:8-10,325-334`
- Test: `PODEM/tests/test_linux_smartatpg.py`
- Modify: `PODEM/RL_GUIDE.md:23-31`

**Interfaces:**
- Produces: `DEFAULT_RUN_DIR: pathlib.Path` equal to `ROOT / "artifacts" / "smartatpg_dual"`.
- Preserves: `main(argv=None)` and `run_fresh_validation(run_dir, dataset_root=None, output_dir=None, seed=2026)`.

- [ ] **Step 1: Add failing CLI tests**

Add tests that patch `validation.run_fresh_validation`, call `validation.main([])`, and assert the first positional argument is `validation.ROOT / "artifacts" / "smartatpg_dual"`. Add a second test calling `validation.main(["--run-dir", "chosen-run"])` and assert `Path("chosen-run")` overrides the default.

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_linux_smartatpg.py -k "main_uses_default_run_dir or main_allows_run_dir_override" -q -p no:cacheprovider`

Expected: FAIL because `--run-dir` is currently required.

- [ ] **Step 3: Implement the default**

Add:

```python
DEFAULT_RUN_DIR = ROOT / "artifacts" / "smartatpg_dual"
```

Change the parser declaration to:

```python
parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
```

- [ ] **Step 4: Update documentation**

Make `python3 scripts/validate_smartatpg.py` the primary example and state that `--run-dir` remains available to select another run.

- [ ] **Step 5: Run focused validation**

Run: `python -m pytest tests/test_linux_smartatpg.py -q -p no:cacheprovider`

Expected: PASS or environment-only skips.

- [ ] **Step 6: Commit the implementation**

Commit only `validation.py`, `test_linux_smartatpg.py`, and `RL_GUIDE.md` with message `feat: default SmartATPG validation run directory`.
