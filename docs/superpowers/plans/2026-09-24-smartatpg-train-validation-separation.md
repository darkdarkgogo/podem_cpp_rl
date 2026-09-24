# SmartATPG Train/Validation Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate SmartATPG training from validation and produce one fresh SCOAP/GAT/Mean comparison per validation run.

**Architecture:** Two thin public scripts call reusable package modules. Training launches isolated GAT and Mean workers and persists complete per-round inference state; validation recreates embeddings from those states, uses existing native batch APIs, selects best rounds, and writes the three-way table.

**Tech Stack:** Python 3, PyTorch, pybind11 C++ ATPG extension, `unittest`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-09-24-smartatpg-train-validation-separation-design.md`

## Global Constraints

- Backtrack limit remains exactly 100.
- Seed defaults to 2026.
- GAT-GRU and Mean training retain their existing reward schemes and hyperparameters.
- Validation is fresh-only and never consumes prior validation output.
- Comparison runtime is the sum of native per-fault `atpg_seconds` only.
- Model/embedding/circuit setup stays outside that timer; actor forward inside ATPG remains included.

---

### Task 1: Package data and training implementation

**Files:**
- Create: `PODEM/python/rl_podem/data_split.py`
- Create: `PODEM/python/rl_podem/smartatpg_portable.py`
- Create: `PODEM/python/rl_podem/training.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Produces: `rl_podem.data_split.prepare_training_data(...)` and `worker_main(argv=None)`.
- Produces: per-round `inference_round_NN.pth` plus `model_round_NN.txt`.

- [ ] Move reusable preparation/portable logic into importable package modules and convert script-local imports to relative package imports.
- [ ] Write tests proving the worker does not resolve validation graphs or call native validation.
- [ ] Save a CPU-cloned inference payload after each round and assert its manifest/encoder/round identity.
- [ ] End each round by incrementing training state directly; remove the validation phase and validation resume state.
- [ ] Run `python -m pytest tests/test_smartatpg_training.py -q` and make it pass.

### Task 2: Dual-model training entry point

**Files:**
- Rewrite: `PODEM/scripts/train_smartatpg.py`
- Create/modify tests: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Consumes: `python -m rl_podem.training MANIFEST OUTPUT --encoder ...`.
- Produces: `training/gat`, `training/mean`, logs, manifests, and `train_summary.json`.

- [ ] Write failing CLI tests for the documented `--dataset-root`, `--output-dir`, `--gat-gpu`, `--mean-gpu`, and `--seed` interface.
- [ ] Adapt the existing dual launcher orchestration into the new train entry point while removing the automatic comparison command.
- [ ] Assert that successful training requires all round/final artifacts and never requires validation metrics/best model.
- [ ] Run the focused CLI tests and make them pass.

### Task 3: Independent validation and three-way table

**Files:**
- Create: `PODEM/python/rl_podem/validation.py`
- Create: `PODEM/scripts/validate_smartatpg.py`
- Modify: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Consumes: dual run directory, both manifests, and per-round inference checkpoints.
- Produces: `validation_three_way_comparison.csv`, matching JSON, `model_selection.json`, and three detailed JSONL files.

- [ ] Write failing tests for round evaluation, best-round selection, fresh SCOAP execution, and one native batch call per circuit/method.
- [ ] Load each checkpoint's `policy_old`, generate embeddings outside the native timer, and call `run_native_validation` once per circuit.
- [ ] Call `run_native_scoap_validation` fresh once per circuit without reading a cache.
- [ ] Build six circuit rows plus TOTAL, including core and derived columns; compute TOTAL coverage from summed counts.
- [ ] Validate finite nonnegative metrics and identical ordered catalogs before writing outputs atomically.
- [ ] Run the focused validation tests and make them pass.

### Task 4: Timing-boundary verification

**Files:**
- Modify only if required: `PODEM/src/python_bindings.cpp`
- Modify: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Confirms: per-fault `atpg_seconds` starts after setup and includes policy decisions within `ATPG::test()`.

- [ ] Add a source/behavior regression test that rejects setup-inclusive wall/native timing in comparison summaries.
- [ ] Confirm both native validation paths call `policy->start_run()` after input/level/fault setup and immediately before `atpg.test()`.
- [ ] Rebuild the extension only if C++ changes are necessary and run native validation tests.

### Task 5: Remove superseded experiment entry points and update docs

**Files:**
- Delete superseded SmartATPG launch/comparison/preparation scripts after imports have moved.
- Modify: `README.md`, `PODEM/RL_GUIDE.md`, shell helpers, and affected tests.

**Interfaces:**
- Leaves: `scripts/train_smartatpg.py` and `scripts/validate_smartatpg.py` as the two formal SmartATPG experiment commands.

- [ ] Replace documentation and shell command references with the two new entry points.
- [ ] Keep generic conversion, dataset generation, plotting, and native build tools available.
- [ ] Remove tests of deleted launchers and replace them with package/public-entry coverage.
- [ ] Run the complete test suite with `python -m pytest -q`.
- [ ] Inspect `git diff --check` and `git status --short` before handoff.

