# SCOAP Validation Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the final comparison SCOAP baseline with one circuit initialization per validation circuit while preserving all existing fault-level results and output schemas.

**Architecture:** Add a name-free native heuristic decision policy and feed it through the existing `NativeValidationPolicy` recorder. Expose a dedicated pybind function, then make the comparison script call it once per circuit, with an explicit cache-bypass flag and circuit wall-time logging.

**Tech Stack:** C++11, pybind11, Python 3, unittest/pytest

**Spec:** `docs/superpowers/specs/2026-09-22-scoap-validation-batch-design.md`

## Global Constraints

- Preserve validation fault order and the comparison JSON/CSV schema.
- Preserve `backtrack_limit=100`, SCOAP heuristic actions, reward formulas, and detected-fault dropping disabled.
- Reset the C random seed at the start of every fault.
- Do not add checkpoint or resume behavior.
- Do not change GAT or MEAN training behavior.

---

### Task 1: Native SCOAP validation entry point

**Files:**
- Modify: `src/python_bindings.cpp:176-425`
- Modify: `src/python_bindings.cpp:569-583`
- Test: `tests/test_smartatpg.py:721-815`

**Interfaces:**
- Consumes: `NativeValidationPolicy`, `ATPG::retain_faults(const vector<string>&)`, and `DecisionRequest::heuristic_action`.
- Produces: `cpp_podem.run_native_scoap_validation(circuit_path, backtrack_limit, seed, fault_ids, reward_scheme, circuit_name) -> list[dict]`.

- [ ] **Step 1: Write a failing native equivalence test**

Add a test that computes reference records with `ScoapValidationEvaluator` and `_evaluate_fault`, calls the missing native batch entry point for the same ordered fault IDs under both reward schemes, and compares fault ID, outcome, backtracks, backtrace steps, and return. Also assert non-negative time and rejection of empty fault IDs, an unknown reward scheme, and a non-100 backtrack limit.

- [ ] **Step 2: Run the focused native test and verify it fails**

Run:

```powershell
python -m pytest tests/test_smartatpg.py -k native_scoap_validation -q
```

Expected: failure because `cpp_podem.run_native_scoap_validation` is not exported.

- [ ] **Step 3: Add the native heuristic policy and configurable progress label**

Add:

```cpp
class NativeHeuristicPolicy : public smartatpg::DecisionPolicy {
public:
  bool needs_gate_names() const override { return false; }
  int select(const smartatpg::DecisionRequest &request) override {
    return request.heuristic_action;
  }
};
```

Extend `NativeValidationPolicy` with a `progress_label` constructor argument defaulting to `"NATIVE_VALIDATE"`. Store it and emit:

```cpp
std::fprintf(stdout, "%s circuit=%s completed=%zu/%zu\n",
             progress_label_.c_str(), circuit_name_.c_str(),
             records_.size(), total_faults_);
```

Keep the existing `std::srand(seed_)` in `on_episode_start()`.

- [ ] **Step 4: Implement and export `run_native_scoap_validation`**

Validate the arguments, construct `NativeHeuristicPolicy`, wrap it in `NativeValidationPolicy` with progress label `SCOAP_VALIDATE`, initialize one ATPG instance, retain all requested faults, disable fault dropping, run `test()` once, and convert validated ordered records to Python dictionaries using the same fields as `run_native_validation`.

Export it from `PYBIND11_MODULE` with arguments `circuit_path`, `backtrack_limit`, `seed`, `fault_ids`, `reward_scheme`, and optional `circuit_name`.

- [ ] **Step 5: Rebuild and run the focused native test**

Run:

```powershell
python -m pip install -e .
python -m pytest tests/test_smartatpg.py -k "native_scoap_validation or native_validation_matches" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the native entry point**

```powershell
git add src/python_bindings.cpp tests/test_smartatpg.py
git commit -m "feat: batch native SCOAP validation per circuit"
```

### Task 2: Comparison batching and cache bypass

**Files:**
- Modify: `scripts/compare_smartatpg_validation.py:108-143`
- Modify: `scripts/compare_smartatpg_validation.py:381-421`
- Modify: `scripts/compare_smartatpg_validation.py:440-613`
- Test: `tests/test_linux_smartatpg.py:180-317`

**Interfaces:**
- Consumes: `cpp_podem.run_native_scoap_validation(...)` from Task 1.
- Produces: `_evaluate_scoap_batch(item, fault_ids, backtrack_limit, seed, reward_scheme) -> list[dict]`, `_load_or_run_scoap(..., force=False)`, and CLI option `--force-scoap`.

- [ ] **Step 1: Replace the old evaluator test with failing batch tests**

Mock `cpp_podem.run_native_scoap_validation` and assert one call receives all fault IDs for a circuit. Verify normalization produces the existing keys:

```python
{
    "circuit", "fault_id", "outcome", "detected", "redundant",
    "aborted", "backtracks", "backtrace_steps", "return",
    "test_vectors", "atpg_seconds",
}
```

Add assertions for wrong result count, out-of-order results, non-finite return/time, missing native symbol, one native call per circuit, normal cache reuse, forced cache bypass, and CLI forwarding.

- [ ] **Step 2: Run the focused comparison tests and verify failure**

Run:

```powershell
python -m pytest tests/test_linux_smartatpg.py -k "scoap or comparison" -q
```

Expected: failures because batching and `force_scoap` are not implemented.

- [ ] **Step 3: Implement `_evaluate_scoap_batch`**

Import `cpp_podem`, check for the new symbol, time the entire call with `time.perf_counter()`, normalize each native row, validate count/order/finiteness, and print:

```python
print(
    f"SCOAP_CIRCUIT_DONE circuit={item['name']} "
    f"faults={len(records)} wall_s={wall_seconds:.3f}",
    flush=True,
)
```

- [ ] **Step 4: Batch `_load_or_run_scoap` by circuit**

Change the signature to `def _load_or_run_scoap(output_dir, identity, circuits, seed, force=False)`. Reuse a valid cache only when `force` is false. Otherwise iterate `circuits`, call `_evaluate_scoap_batch` with `item["episode_fault_ids"]`, extend `records`, and preserve the existing payload schema and atomic write.

- [ ] **Step 5: Thread the force option through the public API and CLI**

Add `force_scoap=False` to `build_validation_comparison`, pass it to `_load_or_run_scoap`, add `parser.add_argument("--force-scoap", action="store_true")`, and pass `args.force_scoap` from `main()`.

- [ ] **Step 6: Run focused comparison tests**

Run:

```powershell
python -m pytest tests/test_linux_smartatpg.py -k "scoap or comparison" -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Python integration**

```powershell
git add scripts/compare_smartatpg_validation.py tests/test_linux_smartatpg.py
git commit -m "refactor: run SCOAP comparison by circuit batch"
```

### Task 3: Regression verification

**Files:**
- Verify: `src/python_bindings.cpp`
- Verify: `scripts/compare_smartatpg_validation.py`
- Verify: `tests/test_smartatpg.py`
- Verify: `tests/test_linux_smartatpg.py`

**Interfaces:**
- Consumes: all interfaces from Tasks 1 and 2.
- Produces: a tested implementation with a clean working tree.

- [ ] **Step 1: Run the complete SmartATPG test modules**

```powershell
python -m pytest tests/test_smartatpg.py tests/test_smartatpg_training.py tests/test_linux_smartatpg.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run the full test suite**

```powershell
python -m pytest tests -q
```

Expected: all tests pass, apart from explicitly documented environment-only skips.

- [ ] **Step 3: Inspect the final diff and working tree**

```powershell
git diff HEAD~2 --check
git status --short
```

Expected: no whitespace errors and no untracked build artifacts.

- [ ] **Step 4: Request code review and resolve findings**

Provide the reviewer with the spec, plan, base SHA, and head SHA. Fix all critical and important findings, then rerun the affected focused and full tests.
