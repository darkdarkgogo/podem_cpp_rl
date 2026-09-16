# Dual-Encoder Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train GAT-GRU on GPU 0 and fanin-MEAN on GPU 1 against the same 1024/6 data split, validate both models after each of two rounds, run SCOAP once on the same validation faults, and emit JSON/CSV comparisons without running ISCAS benchmark.

**Architecture:** Extend the existing versioned trainer to instantiate either existing PPO agent and to emit self-describing per-round validation metrics. Add a pure comparison/report module that validates both runs, evaluates the fixed SCOAP baseline once, and writes atomic JSON/CSV reports. Add a Linux orchestrator that prepares data once, runs both trainers concurrently on isolated CUDA devices, then invokes the comparison module.

**Tech Stack:** Python 3.9, PyTorch 1.11, C++/pybind11 `cpp_podem`, `unittest`, Bash, JSON, CSV.

**Spec:** `docs/superpowers/specs/2026-09-16-dual-encoder-validation-design.md`

## Global Constraints

- Training data is exactly the 1024 circuits in `data/train`.
- Validation data is exactly `b12_C`, `b15_C`, `b17_C`, `b20_C`, `b21_C`, and `b22_C`.
- Both models use V8: `normal_rounds=2`, `faults_per_update=8`, `k_epochs=1`, `backtrack_limit=200`.
- GAT-GRU defaults to physical GPU 0 and fanin-MEAN defaults to physical GPU 1.
- The two trainers share the manifest, fault order, validation catalog, training seed, and validation seed.
- SCOAP runs exactly once on the full validation fault catalog and is reused as the baseline for both rounds.
- New dual training does not create a benchmark bundle or invoke any ISCAS benchmark script.
- Existing single GAT launcher, V6/V7 manifests, and V12 models retain their current behavior.
- All new reports are written through a temporary file followed by atomic replacement.

---

### Task 1: Enable fanin-MEAN in the versioned trainer

**Files:**
- Modify: `PODEM/scripts/train_smartatpg.py:29-43`
- Modify: `PODEM/tests/test_smartatpg_training.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: existing `SmartATPGPPOAgent` from `rl_podem.smartatpg` and `GATGRUSmartATPGPPOAgent` from `rl_podem.gat_gru`.
- Produces: `AGENT_TYPES = {"fanin_mean": SmartATPGPPOAgent, "level_gat_gru": GATGRUSmartATPGPPOAgent}`; the existing `--encoder` option accepts either key.

- [ ] **Step 1: Write a failing encoder-registry test**

Add to `PODEM/tests/test_smartatpg_training.py`:

```python
from train_smartatpg import AGENT_TYPES
from rl_podem.gat_gru import GATGRUSmartATPGPPOAgent
from rl_podem.smartatpg import SmartATPGPPOAgent


def test_trainer_exposes_both_encoder_agents(self):
    self.assertEqual(set(AGENT_TYPES), {"fanin_mean", "level_gat_gru"})
    self.assertIs(AGENT_TYPES["fanin_mean"], SmartATPGPPOAgent)
    self.assertIs(AGENT_TYPES["level_gat_gru"], GATGRUSmartATPGPPOAgent)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_smartatpg_training.SmartATPGTrainingTests.test_trainer_exposes_both_encoder_agents
```

Expected: failure because `fanin_mean` is absent from `AGENT_TYPES`.

- [ ] **Step 3: Register the existing MEAN agent**

In `PODEM/scripts/train_smartatpg.py`, import and register the agent:

```python
from rl_podem.smartatpg import SmartATPGPPOAgent

AGENT_TYPES = {
    "fanin_mean": SmartATPGPPOAgent,
    "level_gat_gru": GATGRUSmartATPGPPOAgent,
}
```

Do not change the default encoder; the existing single-model launcher must still default explicitly to `level_gat_gru`.

- [ ] **Step 4: Run trainer and artifact compatibility tests**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_smartatpg_training PODEM.tests.test_smartatpg.SmartATPGTests.test_v13_records_two_round_batch8_epoch1_protocol
```

Expected: all tests pass, including V13 export/loading for the MEAN policy.

- [ ] **Step 5: Commit the trainer extension**

```powershell
git add PODEM/scripts/train_smartatpg.py PODEM/tests/test_smartatpg_training.py
git commit -m "feat: enable mean SmartATPG training"
```

---

### Task 2: Emit comparable per-circuit validation metrics and run identity

**Files:**
- Modify: `PODEM/src/python_bindings.cpp:1-180`
- Modify: `PODEM/scripts/train_smartatpg.py:190-234, 475-565, 704-778`
- Modify: `PODEM/tests/test_smartatpg_training.py`
- Modify: `PODEM/tests/test_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Produces: `_summarize_fault_records(records: list[dict]) -> dict` with count/outcome/work/time fields.
- Produces: `_summarize_validation(records, circuits, round_number) -> dict` with existing top-level totals plus `circuits: list[dict]` in manifest order.
- Produces: `<training-dir>/validation_identity.json` containing format, manifest hash, encoder variant, training protocol, validation catalog hash, and ordered validation circuit names.
- Produces: native `run_stuck_at(...)` summaries gain `atpg_seconds`, measured only around `atpg.test()`.
- Produces: each fault record gains `redundant`, `aborted`, `test_vectors`, and `atpg_seconds`.

- [ ] **Step 1: Add failing tests for outcome classification and per-circuit totals**

Add tests using two circuits and three synthetic records:

```python
def test_validation_summary_preserves_each_circuit_and_all_outcomes(self):
    circuits = [
        {"name": "a", "episode_fault_ids": ["a0", "a1"]},
        {"name": "b", "episode_fault_ids": ["b0"]},
    ]
    records = [
        {"circuit": "a", "fault_id": "a0", "outcome": 1,
         "detected": 1, "redundant": 0, "aborted": 0,
         "backtracks": 2, "backtrace_steps": 3, "return": 100.0,
         "test_vectors": 1, "atpg_seconds": 0.1},
        {"circuit": "a", "fault_id": "a1", "outcome": 0,
         "detected": 0, "redundant": 1, "aborted": 0,
         "backtracks": 4, "backtrace_steps": 5, "return": -100.0,
         "test_vectors": 0, "atpg_seconds": 0.2},
        {"circuit": "b", "fault_id": "b0", "outcome": 2,
         "detected": 0, "redundant": 0, "aborted": 1,
         "backtracks": 6, "backtrace_steps": 7, "return": -100.0,
         "test_vectors": 0, "atpg_seconds": 0.3},
    ]
    result = _summarize_validation(records, circuits, 1)
    self.assertEqual(result["detected_faults"], 1)
    self.assertEqual(result["redundant_faults"], 1)
    self.assertEqual(result["aborted_faults"], 1)
    self.assertEqual([row["circuit"] for row in result["circuits"]], ["a", "b"])
    self.assertEqual(result["circuits"][0]["test_vectors"], 1)
    self.assertAlmostEqual(result["atpg_seconds"], 0.6)
```

Also add a test that the identity payload has exactly the declared protocol and catalog hash:

```python
identity = _validation_identity(config, [
    {"name": "b12_C"}, {"name": "b15_C"},
])
self.assertEqual(identity["encoder_variant"], "fanin_mean")
self.assertEqual(identity["validation_circuits"], ["b12_C", "b15_C"])
self.assertEqual(identity["faults_per_update"], 8)
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_smartatpg_training
```

Expected: failures for missing per-circuit fields and missing `_validation_identity`.

- [ ] **Step 3: Expose native ATPG time and classify each validation fault**

In `PODEM/src/python_bindings.cpp`, include `<chrono>`, measure immediately before and after `atpg.test()`, and add the elapsed value to the returned summary:

```cpp
const auto atpg_started = std::chrono::steady_clock::now();
atpg.test();
const double atpg_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - atpg_started).count();
// after policy->summary()
result["atpg_seconds"] = atpg_seconds;
```

Keep input parsing, levelization, fault-list generation, and Python orchestration outside this interval. Add a native-bridge test that asserts `run_stuck_at(...)["atpg_seconds"] >= 0.0`.

Return these additional fields from `_evaluate_fault`, using `summary["atpg_seconds"]`:

```python
outcome = int(terminal["outcome"])
return {
    # existing keys remain unchanged
    "detected": int(outcome == 1),
    "redundant": int(outcome == 0),
    "aborted": int(outcome not in (0, 1)),
    "test_vectors": int(outcome == 1),
    "atpg_seconds": float(summary["atpg_seconds"]),
}
```

`test_vectors` is one for a detected validation fault and zero otherwise because validation runs one requested fault per native invocation with fault dropping disabled.

- [ ] **Step 4: Add a reusable record summarizer**

Implement:

```python
def _summarize_fault_records(records):
    count = len(records)
    totals = {
        "episodes": count,
        "detected_faults": sum(int(item["detected"]) for item in records),
        "redundant_faults": sum(int(item["redundant"]) for item in records),
        "aborted_faults": sum(int(item["aborted"]) for item in records),
        "backtracks_total": sum(int(item["backtracks"]) for item in records),
        "backtrace_steps_total": sum(int(item["backtrace_steps"]) for item in records),
        "return_total": sum(float(item["return"]) for item in records),
        "test_vectors": sum(int(item["test_vectors"]) for item in records),
        "atpg_seconds": sum(float(item["atpg_seconds"]) for item in records),
    }
    divisor = max(1, count)
    totals.update(
        fault_coverage=totals["detected_faults"] / divisor,
        backtracks_mean=totals["backtracks_total"] / divisor,
        backtrace_steps_mean=totals["backtrace_steps_total"] / divisor,
        return_mean=totals["return_total"] / divisor,
    )
    return totals
```

Update `_summarize_validation` to preserve its exact-order coverage check, call this helper for all records, and add one row per circuit under `circuits`.

- [ ] **Step 5: Write validation identity atomically**

Implement `_validation_identity(config, validation_circuits)` returning:

```python
{
    "format": "SMARTATPG_VALIDATION_IDENTITY_V1",
    "manifest_hash": config["manifest_hash"],
    "encoder_variant": config["encoder_variant"],
    "normal_rounds": config["rounds"],
    "faults_per_update": config["faults_per_update"],
    "k_epochs": config["k_epochs"],
    "backtrack_limit": config["backtrack_limit"],
    "validation_catalog_hash": config["validation_catalog_hash"],
    "validation_circuits": [item["name"] for item in validation_circuits],
}
```

Write it to `validation_identity.json` after config construction and before checkpoint loading. On resume, require an existing identity file to equal the newly computed payload; otherwise fail rather than overwrite it.

- [ ] **Step 6: Run training tests**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_smartatpg_training
```

Expected: all tests pass; existing score ordering remains unchanged because the score continues to read the existing aggregate fields.

- [ ] **Step 7: Commit comparable validation output**

```powershell
git add PODEM/src/python_bindings.cpp PODEM/scripts/train_smartatpg.py PODEM/tests/test_smartatpg_training.py PODEM/tests/test_smartatpg.py
git commit -m "feat: record comparable validation metrics"
```

---

### Task 3: Compare two model runs with one SCOAP validation baseline

**Files:**
- Create: `PODEM/scripts/compare_smartatpg_validation.py`
- Modify: `PODEM/tests/test_linux_smartatpg.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Produces: `ScoapValidationEvaluator.run(circuit_path, *, backtrack_limit, seed, fault_ids, use_scoap, event_callback) -> dict` compatible with `train_smartatpg._evaluate_fault`.
- Produces: `build_validation_comparison(manifest_path, gat_dir, mean_dir, output_dir, seed=2026) -> dict`.
- Produces: `scoap_validation.json`, `validation_comparison.json`, and `validation_comparison.csv` in `output_dir`.

- [ ] **Step 1: Add failing protocol and report tests**

Extend the scripts-inventory assertion to include `compare_smartatpg_validation.py`. Add a temporary-directory test that writes matching GAT/MEAN identities and two-round metrics, patches SCOAP evaluation, and asserts:

```python
result = build_validation_comparison(manifest, gat_dir, mean_dir, output)
self.assertEqual(result["format"], "SMARTATPG_DUAL_VALIDATION_COMPARISON_V1")
self.assertEqual(result["models"]["smartatpg_gat_gru"]["best_round"], 1)
self.assertEqual(result["models"]["smartatpg_mean"]["best_round"], 2)
self.assertEqual(len(result["comparisons"]), 4)
self.assertTrue((output / "scoap_validation.json").is_file())
self.assertTrue((output / "validation_comparison.csv").is_file())
```

Add rejection tests for mismatched `manifest_hash`, `validation_catalog_hash`, missing round 2, wrong encoder variants, and a protocol other than 2/8/1/200.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Expected: import failure for the new comparison module.

- [ ] **Step 3: Implement strict run loading**

In `compare_smartatpg_validation.py`, implement:

```python
EXPECTED_ENCODERS = {
    "smartatpg_gat_gru": "level_gat_gru",
    "smartatpg_mean": "fanin_mean",
}


def _load_run(name, directory):
    identity = json.loads((directory / "validation_identity.json").read_text("utf-8"))
    rounds = json.loads((directory / "validation_metrics.json").read_text("utf-8"))
    if identity["encoder_variant"] != EXPECTED_ENCODERS[name]:
        raise ValueError(f"Wrong encoder variant for {name}")
    if (identity["normal_rounds"], identity["faults_per_update"],
            identity["k_epochs"], identity["backtrack_limit"]) != (2, 8, 1, 200):
        raise ValueError(f"Wrong V8 training protocol for {name}")
    if [item.get("round") for item in rounds] != [1, 2]:
        raise ValueError(f"Incomplete validation rounds for {name}")
    return identity, rounds
```

Validate exact identity equality for manifest hash, validation catalog hash, validation circuit order, and protocol fields before evaluating SCOAP.

- [ ] **Step 4: Implement a heuristic evaluator using the same native callback path**

`ScoapValidationEvaluator.decision_callback` returns `int(request["heuristic_action"])`. Its `run` method calls `cpp_podem.run_stuck_at` with `rl_mode="backtrace_rl"`, `use_scoap=True`, the requested single `fault_ids` list, and the supplied event callback. This makes `_evaluate_fault` collect exactly the same terminal fields used for learned models.

Resolve the manifest with `_resolve_circuit_records`, generate the runtime catalog with `_load_validation_catalogs`, and call `_evaluate_fault` for each `(circuit, fault_id)` in `_validation_order`. Summarize the records with `_summarize_validation(..., round_number=0)` and write `scoap_validation.json` atomically. If a matching complete SCOAP file already exists, validate and reuse it on resume.

- [ ] **Step 5: Build comparison rows and atomic reports**

For each model and each round, create one total row plus one row per circuit. Each row contains the raw metrics and deltas:

```python
{
    "scope": "total",  # or "circuit"
    "circuit": "TOTAL",
    "round": 1,
    "model": "smartatpg_gat_gru",
    "fault_coverage": model_row["fault_coverage"],
    "fault_coverage_delta": model_row["fault_coverage"] - baseline["fault_coverage"],
    "backtracks_reduction_percent": percentage_reduction(
        baseline["backtracks_total"], model_row["backtracks_total"]
    ),
    "backtrace_steps_reduction_percent": percentage_reduction(
        baseline["backtrace_steps_total"], model_row["backtrace_steps_total"]
    ),
    "atpg_seconds_reduction_percent": percentage_reduction(
        baseline["atpg_seconds"], model_row["atpg_seconds"]
    ),
}
```

Include all raw detected/redundant/aborted/test-vector/work/time fields in both JSON and CSV. Add direct GAT-minus-MEAN values for each matching scope/round under `direct_comparisons`. Determine each model's `best_round` from the unique metrics row with `is_best=True`; reject zero or multiple best rows.

- [ ] **Step 6: Run comparison tests**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Expected: all comparison and existing launcher tests pass.

- [ ] **Step 7: Commit validation comparison reporting**

```powershell
git add PODEM/scripts/compare_smartatpg_validation.py PODEM/tests/test_linux_smartatpg.py
git commit -m "feat: compare dual validation runs with SCOAP"
```

---

### Task 4: Add the dual-GPU Linux orchestrator

**Files:**
- Create: `PODEM/scripts/run_dual_smartatpg_training_linux.py`
- Create: `PODEM/train_dual_smartatpg_linux.sh`
- Modify: `PODEM/tests/test_linux_smartatpg.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`

**Interfaces:**
- Produces: `_run_parallel_training(jobs: list[dict]) -> dict[str, float]`, where each job supplies name, command, log path, environment, and output prefix.
- Produces: CLI options `--gat-gpu` (default 0), `--mean-gpu` (default 1), common seed/profile/round/backtrack/output options, and separate optional continuation checkpoints.
- Consumes: `compare_smartatpg_validation.py MANIFEST GAT_DIR MEAN_DIR OUTPUT_DIR --seed SEED`.

- [ ] **Step 1: Write failing command-construction and GPU tests**

Add a launcher test with mocked preparation, parallel training, and comparison execution. Assert the two training commands contain:

```python
self.assertEqual(gpu_by_encoder, {
    "level_gat_gru": "0",
    "fanin_mean": "1",
})
for command in train_commands:
    self.assertEqual(command[command.index("--rounds") + 1], "2")
    self.assertEqual(command[command.index("--k-epochs") + 1], "1")
```

Assert preparation is invoked once, both train commands reference the same resolved `training_manifest.json`, comparison is invoked once after both model files exist, and no command references `prepare_smartatpg_benchmark.py`, `benchmark_smartatpg.py`, or `benchmark_bundle`.

Add validation tests for fewer than two visible CUDA devices, equal GPU IDs, negative IDs, and a non-Linux platform.

- [ ] **Step 2: Write a failing fail-fast process test**

Use two fake process objects and synchronization events. Make the GAT job return code 9 while the MEAN job remains active, then assert `_run_parallel_training` calls `terminate()` on the MEAN process and raises `SystemExit(9)`. Also test two zero return codes produce timings for both names.

- [ ] **Step 3: Run focused launcher tests and verify they fail**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Expected: import failure for `run_dual_smartatpg_training_linux`.

- [ ] **Step 4: Implement preparation and isolated commands**

The orchestrator must:

1. validate Linux, PyTorch, the C++ extension, two distinct in-range GPU IDs, rounds 2, and backtrack limit 200;
2. run `prepare_smartatpg_training.py` once with `--resume`;
3. construct both `train_smartatpg.py` commands with the same manifest and seed;
4. set `CUDA_VISIBLE_DEVICES` to the selected physical GPU separately for each process;
5. give each child `PYTHONPATH` containing `PODEM/python` and `PODEM/scripts`;
6. write `training_run_metadata.json` before child launch and finalize it only after comparison succeeds.

Use output directories `smartatpg_gat_gru` and `smartatpg_mean`, and logs `train_gat_gru.log` and `train_mean.log`.

- [ ] **Step 5: Implement concurrent fail-fast execution**

Use `ThreadPoolExecutor(max_workers=2)` around the existing streaming `_tee_command`. Register each `Popen` through its `on_start` callback under a lock. Prefix console lines with `[GAT] ` or `[MEAN] `. When a future returns a nonzero code, call `terminate()` on every other registered process whose `poll()` is `None`, wait for both futures, and raise `SystemExit` with the failing code. Record elapsed seconds per model only when both succeed.

- [ ] **Step 6: Invoke comparison without benchmark packaging**

After verifying both directories contain `model_best.txt`, `model_latest.txt`, `validation_identity.json`, and `validation_metrics.json`, run:

```text
python -u scripts/compare_smartatpg_validation.py \
  preparation/training_manifest.json \
  smartatpg_gat_gru \
  smartatpg_mean \
  OUTPUT_DIR \
  --seed 2026
```

Write the final metadata fields `timings`, `validation_comparison`, and `finished_at`. Do not create a `benchmark_bundle` field.

- [ ] **Step 7: Add the Bash entry point**

Create `PODEM/train_dual_smartatpg_linux.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

exec "${PYTHON:-python}" -u scripts/run_dual_smartatpg_training_linux.py \
  --output-dir artifacts/smartatpg_dual_top30_hard_2rounds_batch8_bt200 \
  --dataset-root data \
  --rounds 2 \
  --backtrack-limit 200 \
  --seed 2026 \
  --gat-gpu 0 \
  --mean-gpu 1 \
  "$@"
```

- [ ] **Step 8: Run launcher tests**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest PODEM.tests.test_linux_smartatpg
```

Expected: all tests pass.

- [ ] **Step 9: Commit the dual launcher**

```powershell
git add PODEM/scripts/run_dual_smartatpg_training_linux.py PODEM/train_dual_smartatpg_linux.sh PODEM/tests/test_linux_smartatpg.py
git commit -m "feat: launch dual SmartATPG training"
```

---

### Task 5: Full compatibility verification and handoff

**Files:**
- Modify only if a test exposes a defect in the files changed by Tasks 1-4.
- Test: `PODEM/tests/test_smartatpg_training.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`
- Test: selected cases in `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Consumes: all new trainer, report, and launcher interfaces.
- Produces: a clean working tree containing tested dual-model training with no regression to V12/V13 loaders or legacy manifests.

- [ ] **Step 1: Compile all changed Python modules**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m py_compile `
  PODEM/scripts/train_smartatpg.py `
  PODEM/scripts/compare_smartatpg_validation.py `
  PODEM/scripts/run_dual_smartatpg_training_linux.py
```

Expected: exit code 0.

- [ ] **Step 2: Rebuild the native extension with the timing field**

Run from `PODEM`:

```powershell
cmd.exe /d /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && "C:\Users\acer\.conda\envs\d2l\python.exe" setup.py build_ext --inplace'
```

Expected: exit code 0 and an updated `python/cpp_podem.cp39-win_amd64.pyd`.

- [ ] **Step 3: Run the relevant test suites**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -m unittest `
  PODEM.tests.test_smartatpg_training `
  PODEM.tests.test_linux_smartatpg `
  PODEM.tests.test_smartatpg.SmartATPGTests.test_v12_contains_fanin_mean_encoder_and_portable_inference_matches_torch `
  PODEM.tests.test_smartatpg.SmartATPGTests.test_v13_records_two_round_batch8_epoch1_protocol `
  PODEM.tests.test_smartatpg.SmartATPGTests.test_deferred_trainer_collects_eight_faults_for_one_update `
  PODEM.tests.test_smartatpg.SmartATPGTests.test_deferred_empty_fault_does_not_reward_previous_trajectory
```

Expected: all selected tests pass.

- [ ] **Step 4: Check patch hygiene and exact scope**

Run:

```powershell
git diff --check
git status --short
rg -n "benchmark_bundle|prepare_smartatpg_benchmark|benchmark_smartatpg" `
  PODEM/scripts/run_dual_smartatpg_training_linux.py `
  PODEM/train_dual_smartatpg_linux.sh
```

Expected: no whitespace errors; the final `rg` has no matches; only intended files are modified.

- [ ] **Step 5: Request an independent code review**

Review `474a9c5..HEAD` against the approved spec. The reviewer must explicitly inspect shared-fault identity, PPO batch boundaries for MEAN, two-process failure behavior, one-time SCOAP execution, JSON/CSV consistency, and absence of ISCAS benchmark invocation.

- [ ] **Step 6: Fix important review findings and rerun Steps 1-4**

For every Critical or Important finding, add a regression test before changing implementation. Amend the affected task commit or add one focused fix commit.

- [ ] **Step 7: Commit any final test-only or review fixes**

```powershell
git add PODEM docs
git commit -m "test: verify dual SmartATPG validation workflow"
```

Skip this commit when the working tree is already clean.
