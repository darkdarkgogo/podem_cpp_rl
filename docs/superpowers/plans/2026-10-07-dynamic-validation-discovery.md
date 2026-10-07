# Dynamic SmartATPG Validation Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SmartATPG validation discover every `.bench` circuit in the validation directory without a fixed name list.

**Architecture:** Keep discovery centralized in `rl_podem.data_split`. Return a deterministic filename-sorted tuple, reject empty datasets and unexpected entries, and leave checkpoint/training formats unchanged so existing trained runs remain usable.

**Tech Stack:** Python 3, `pathlib`, `unittest`, pytest-compatible tests.

**Spec:** `docs/superpowers/specs/2026-10-07-dynamic-validation-discovery-design.md`

## Global Constraints

- Existing trained checkpoints must remain valid.
- Validation ordering must be deterministic by filename.
- The validation directory must contain at least one `.bench` file and no other entries.
- Training manifests and model formats must not change.

---

### Task 1: Dynamic validation dataset discovery

**Files:**
- Modify: `PODEM/python/rl_podem/data_split.py:202-255`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: `dataset_root: PathLike`.
- Produces: `discover_validation_dataset(dataset_root) -> tuple[Path, ...]`, ordered by filename.

- [ ] **Step 1: Write failing discovery tests**

Import `discover_validation_dataset` and add tests that create arbitrary `.bench` filenames, assert sorted discovery without passing expected names, and assert an empty validation directory raises `ValueError` containing `at least one BENCH`.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_smartatpg_training.py -k "dataset_discovery or empty_validation" -q`

Expected: arbitrary-name discovery fails against the fixed six-name contract and the empty-directory error does not match.

- [ ] **Step 3: Implement minimal dynamic discovery**

Change the public interface to:

```python
def discover_validation_dataset(dataset_root):
    ...
    if not validation_paths:
        raise ValueError(
            "SmartATPG validation split must contain at least one BENCH circuit"
        )
    ...
    return validation_paths
```

Remove `expected_validation_names` from `discover_dataset` and `discover_mean_dataset`, and call `discover_validation_dataset(dataset_root)` directly. Preserve sorting, missing-directory handling, and unexpected-entry rejection.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_smartatpg_training.py -k "dataset_discovery or empty_validation or mean_dataset" -q`

Expected: PASS.

### Task 2: Documentation and real-dataset verification

**Files:**
- Modify: `PODEM/RL_GUIDE.md:33`
- Verify: `PODEM/data/validation/*.bench`

**Interfaces:**
- Consumes: dynamic discovery behavior from Task 1.
- Produces: documentation describing runtime discovery of all validation circuits.

- [ ] **Step 1: Update user documentation**

Replace the fixed-six wording with a statement that validation discovers every `.bench` file at runtime in deterministic filename order.

- [ ] **Step 2: Verify the real dataset inventory**

Run a Python one-liner importing `discover_validation_dataset` and print the discovered stems.

Expected exactly, in order:

```text
aes_core b12_C b15_C b17_C b20_C b21_C b22_C des_perf wb_conmax
```

- [ ] **Step 3: Run the complete relevant test module**

Run: `python -m pytest tests/test_smartatpg_training.py -q`

Expected: PASS (or an explicitly reported environment-only skip).

- [ ] **Step 4: Check the final diff**

Run: `git diff --check` and inspect only the planned files. Confirm no checkpoint format, training manifest format, or unrelated user files changed.
