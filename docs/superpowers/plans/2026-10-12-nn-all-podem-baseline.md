# NN-All PODEM Baseline Implementation Plan

> **For agentic workers:** Implement the tasks below in order, checking each box after its test cycle. The current user requested inline execution in this chat; do not dispatch subagents.

**Goal:** Add a separately trained NN-All PODEM baseline to the existing stuck-at-fault comparison.

**Architecture:** Python computes static 11D candidate features, collects path outcomes, trains a 11→60→1 classifier and exports a versioned native artifact. A focused C++ decision policy scores every unresolved fanin during PODEM backtrace. The existing validation orchestration adds an opt-in four-way run using the same fault catalog and runtime definition.

**Tech Stack:** C++11, pybind11, Python 3.9+, PyTorch, unittest/pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-nn-all-podem-baseline-design.md`

## Global Constraints

- Train only on `PODEM/data/train`; evaluate only on `PODEM/data/validation`.
- Default validation seed is 2026 and backtrack limit is 100.
- Native policy must score at every applicable multi-fanin backtrace level; no level-6 gate.
- Output name is `nn_all_podem`; do not claim bit-for-bit Atalanta reproduction.
- Preserve the existing three-way validation contract when NN-All is omitted.

---

### Task 1: Static 11D feature contract

**Files:**
- Create: `PODEM/python/rl_podem/nn_all_features.py`
- Test: `PODEM/tests/test_nn_all.py`

**Interfaces:**
- `load_nn_all_features(path: Path) -> NnAllFeatures`: maps each BENCH wire name to depth, CC0, CC1 and COP probability and maps objective name to gate type.
- `candidate_vector(features: NnAllFeatures, objective: str, candidate: str) -> tuple[float, ...]`: returns the exact 11D vector.
- `write_native_features(path: Path, features: NnAllFeatures) -> None`: writes a versioned circuit-hash-bound gate table consumed by C++.

- [ ] **Step 1: Write tests for a two-input AND, NAND, OR, NOR and NOT fixture.** Assert PI COP=0.5, gate recurrence, CC/depth, the seven one-hot positions and exact vector length/order.
- [ ] **Step 2: Run `python -m pytest tests/test_nn_all.py -q`; confirm the new feature imports fail.**
- [ ] **Step 3: Implement the parser atop `load_circuit_graph`, with `COP[PI]=0.5`, `AND=∏p`, `NAND=1-∏p`, `OR=1-∏(1-p)`, `NOR=∏(1-p)`, `NOT=1-p`, and `candidate_vector=(level,cc0,cc1,cop,*one_hot[parent_type])`.** Reject undefined candidates, nonfinite values and unsupported gate types.
- [ ] **Step 4: Run `python -m pytest tests/test_nn_all.py -q`; confirm the feature and export tests pass. Commit the task.**

### Task 2: Training sample collector and classifier

**Files:**
- Create: `PODEM/python/rl_podem/nn_all_training.py`
- Create: `PODEM/scripts/train_nn_all.py`
- Modify: `PODEM/tests/test_nn_all.py`

**Interfaces:**
- `PathSampleCollector`: Python `run_stuck_at` callbacks storing each selected candidate by decision sequence; `on_backtrack` marks its sequence negative, and `episode_end` marks remaining decisions positive only on detection. Aborted unresolved decisions are excluded.
- `run_training(dataset_root, output_dir, *, seed=2026, faults_per_circuit=30, epochs=20) -> dict`: deterministic circuit split, fault selection, fitting, checkpoint selection and native model export.
- Native model text begins `NN_ALL_PODEM_MODEL_V1`, then `feature_schema`, `hidden_dim 60`, `circuit_manifest_sha256`, followed by `w1`, `b1`, `w2`, `b2` arrays with declared sizes.

- [ ] **Step 1: Test that backtracked sequence yields label 0, surviving sequence on a detected fault yields 1, and aborted remaining sequences have no label. Test that a train/holdout split is by whole circuit.**
- [ ] **Step 2: Run the collector tests and confirm failure due to absent module.**
- [ ] **Step 3: Implement deterministic fault sampling from `catalog_stuck_at`, event attribution, binary cross-entropy training of `torch.nn.Sequential(Linear(11,60), Sigmoid(), Linear(60,1), Sigmoid())`, holdout selection and metadata export. Reject empty or single-class partitions.**
- [ ] **Step 4: Run collector/model tests and a bounded fixture command with `--max-circuits` for smoke. Confirm model and summary files load. Commit the task.**

### Task 3: Native NN-All decision policy

**Files:**
- Create: `PODEM/src/nn_all_policy.h`
- Create: `PODEM/src/nn_all_policy.cpp`
- Modify: `PODEM/setup.py`
- Modify: `PODEM/src/python_bindings.cpp`
- Test: `PODEM/tests/test_nn_all.py`

**Interfaces:**
- `smartatpg::NnAllPolicy : DecisionPolicy` consumes the model and feature table, returns the highest scoring unmasked candidate, supports only `BACKTRACE`, and caches score by `(objective_id,candidate_id)` within one circuit.
- `cpp_podem.run_native_nn_all_validation(circuit_path, feature_path, model_path, backtrack_limit, seed, fault_ids, circuit_name) -> list[dict]` uses `NativeValidationPolicy` and returns the existing native per-fault record fields.
- `cpp_podem.score_nn_all_candidate(model_path, vector11) -> float` checks Python/native inference parity.

- [ ] **Step 1: Test sigmoid score parity on a tiny known model, masked candidate selection, tie order, non-level-6 invocation, ordered fault IDs, and errors for malformed artifacts.**
- [ ] **Step 2: Run native tests and confirm the new bindings are absent.**
- [ ] **Step 3: Implement strict model/feature readers and scoring in `nn_all_policy.cpp`. Use `DecisionRequest.objective_id` and `candidate_ids`; ignore masked actions. Add sources and bindings. The native runner sets `backtrace_rl`, `backtrack_limit=100`, SCOAP on for unchanged search behavior, `backtrace_lock=false`, and `set_drop_detected_faults(false)`.**
- [ ] **Step 4: Rebuild with `python -m pip install -e .`, rerun native tests, verify that missing model raises before search, and commit the task.**

### Task 4: Four-way validation and documentation

**Files:**
- Modify: `PODEM/python/rl_podem/validation.py`
- Modify: `PODEM/python/rl_podem/validation_tables.py`
- Modify: `PODEM/scripts/validate_smartatpg.py`
- Modify: `PODEM/tools/plot_final_comparison.py`
- Modify: `PODEM/RL_GUIDE.md`
- Modify: `PODEM/tests/test_nn_all.py`
- Modify: `PODEM/tests/test_final_comparison_plots.py`

**Interfaces:**
- `run_fresh_validation(..., nn_all_run_dir=None)` emits existing three-way files when omitted; when supplied, loads the NN-All native model, evaluates the same `circuits[*].episode_fault_ids`, and emits four-way files and `detailed/nn_all_podem_records.jsonl`.
- `build_four_way_row(circuit, scoap, gat, mean, nn_all)` follows existing column conventions and adds counts for all methods.

- [ ] **Step 1: Test opt-in CLI parsing, identical fault order, TOTAL arithmetic, runtime sum, model identity metadata and preservation of three-way output when omitted.**
- [ ] **Step 2: Run validation tests and confirm failure for missing four-way interface.**
- [ ] **Step 3: Add `--nn-all-run-dir`, native evaluation and four-way output. Keep model loading and feature export outside `atpg_seconds`; validate every record and summary before writing. Make the plotter accept `nn_all_podem` in four-way input with its own color and the same SCOAP denominator. Document training and validation commands plus the published-source discrepancy.**
- [ ] **Step 4: Run focused tests and existing `test_smartatpg.py`, `test_smartatpg_training.py`, `test_final_comparison_plots.py`. Perform one bounded end-to-end NN-All train/validate smoke on a fixture, inspect outputs, and commit.**

## Final review

- [ ] Confirm every source citation and experimental label says PODEM port.
- [ ] Check `git diff --check`, test results, output schemas and working-tree status.
- [ ] Report the exact tests run, smoke data scope and any difference from full-scale training.
