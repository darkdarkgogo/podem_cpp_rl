# SmartATPG Validation Module Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove every dependency from `rl_podem.validation` to `rl_podem.training` and leave validation logic outside the training worker.

**Architecture:** Add a torch-free `validation_core.py` for validation algorithms and an `artifact_io.py` for shared checkpoint identity and atomic JSON utilities. Keep `training.py` focused on training and `validation.py` focused on fresh validation orchestration, with both importing only neutral shared modules.

**Tech Stack:** Python 3.9, PyTorch, pybind11 `cpp_podem`, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-09-24-smartatpg-validation-module-boundary-design.md`

## Global Constraints

- Preserve `SMARTATPG_INFERENCE_ROUND_V1` and all serialized output schemas.
- Preserve validation catalog order, native call arguments, summary fields, and best-round ordering.
- Do not add compatibility re-exports to `training.py`.
- `validation_core.py` must not import PyTorch or `training.py`.
- Public CLI arguments and artifact paths remain unchanged.

---

### Task 1: Establish shared artifact contracts

**Files:**
- Create: `PODEM/python/rl_podem/artifact_io.py`
- Modify: `PODEM/python/rl_podem/training.py`
- Modify: `PODEM/python/rl_podem/validation.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Produces: `INFERENCE_CHECKPOINT_FORMAT: str`.
- Produces: `manifest_hash(path) -> str`.
- Produces: `atomic_json(path, value) -> None`.
- Produces: `atomic_json_lines(path, records) -> None`.

- [ ] **Step 1: Write the failing ownership test.** Import the four symbols from `rl_podem.artifact_io`; assert the format string is unchanged, a known file has the expected SHA-256 hash, and JSON/JSONL outputs end in one newline.
- [ ] **Step 2: Verify failure.** Run `python -m pytest PODEM/tests/test_linux_smartatpg.py -q`; expect collection to fail because `artifact_io` does not exist.
- [ ] **Step 3: Implement the neutral module.** Move the constant and JSON/hash functions without changing serialization. Keep `_atomic_torch_save` in `training.py`, and update training/validation imports.
- [ ] **Step 4: Verify success.** Run the Step 2 command and expect all tests to pass.
- [ ] **Step 5: Commit.** Commit the module, import migrations, and ownership tests as `refactor: share SmartATPG artifact contracts`.

### Task 2: Extract validation core

**Files:**
- Create: `PODEM/python/rl_podem/validation_core.py`
- Modify: `PODEM/python/rl_podem/training.py`
- Modify: `PODEM/python/rl_podem/validation.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`
- Modify: `PODEM/tests/test_smartatpg.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Produces: `_native_validation_batch(item, fault_ids, embedding_path, actor_path, records_path, seed, reward_scheme) -> list[dict]`.
- Produces: `_load_validation_catalogs(manifest, circuits) -> list[dict]`.
- Produces: `_validation_catalog_hash(circuits) -> str` and `_validation_order(circuits) -> list[tuple[str, str]]`.
- Produces: `_evaluate_fault(evaluator, item, fault_id, backtrack_limit, seed, reward_scheme) -> dict`.
- Produces: `_summarize_fault_records(records) -> dict` and `_summarize_validation(records, circuits, round_number) -> dict`.
- Produces: `validation_score(summary, round_number) -> tuple`.

- [ ] **Step 1: Add dependency-boundary tests.** Assert that `validation.py` contains no `from .training import`, `training.py` contains no `validation_core`, and none of the validation helper definitions remain in `training.py`. Update behavior-test imports to target `validation_core`.
- [ ] **Step 2: Verify failure.** Run the three SmartATPG test modules together and confirm the boundary assertion fails before extraction.
- [ ] **Step 3: Move the complete validation cluster.** Move all interfaces above plus `_catalog_fault_ids` to `validation_core.py`. Import constants, catalog operations, native path conversion, and reward helpers directly from their owning modules. Do not import `training.py` or PyTorch.
- [ ] **Step 4: Migrate orchestration and tests.** Make `validation.py` import validation-domain helpers from `validation_core.py`; move test imports while leaving training-state tests with training.
- [ ] **Step 5: Verify success.** Run `python -m pytest PODEM/tests/test_linux_smartatpg.py PODEM/tests/test_smartatpg_training.py PODEM/tests/test_smartatpg.py -q`; expect all selected tests to pass except known optional-fixture skips.
- [ ] **Step 6: Commit.** Commit as `refactor: isolate SmartATPG validation core`.

### Task 3: Verify behavior and documentation

**Files:**
- Modify: `PODEM/RL_GUIDE.md`
- Modify: `PODEM/docs/SMARTATPG_11D_使用说明.md`
- Test: `PODEM/tests/`

**Interfaces:**
- Consumes: the dependency graph established by Tasks 1 and 2.
- Produces: documented internal ownership and a reviewed, test-clean commit range.

- [ ] **Step 1: Document the internal boundary.** State that `training.py` owns no validation helpers, `validation.py` orchestrates fresh validation, and `validation_core.py` owns catalog/native/summary/selection logic.
- [ ] **Step 2: Compile and run the complete suite.** Compile the four affected Python modules, rebuild only if native sources changed, and run `python -m pytest PODEM/tests -q --junitxml=.codex-test-results.xml`; require zero failures and zero errors.
- [ ] **Step 3: Inspect dependencies.** Use `rg` to confirm no validation-to-training import and no validation helper definitions in `training.py`; run `git diff --check` and inspect `git status --short`.
- [ ] **Step 4: Commit documentation.** Commit as `docs: describe SmartATPG validation ownership`.
- [ ] **Step 5: Request review.** Review the complete implementation range against the spec, fix every Critical or Important finding, rerun the complete suite, and leave the worktree clean.
