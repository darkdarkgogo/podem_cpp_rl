# SmartATPG GAT Epoch-4 Learning Rates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change only `level_gat_gru` training to four PPO epochs with Actor learning rate `0.0003` and Critic learning rate `0.001`, while preserving all `fanin_mean` epoch-1 behavior and rejecting old GAT artifacts.

**Architecture:** Centralize the encoder-specific training protocol in the preparation module and consume it from training and launch scripts. Split portable Actor formats by encoder: V13 remains Mean epoch-1, while a new V14 format represents GAT epoch-4; the loader rejects every other encoder/format pairing. Bump only the GAT agent checkpoint format.

**Tech Stack:** Python 3, PyTorch, `unittest`, pybind11-backed SmartATPG integration.

**Spec:** `docs/superpowers/specs/2026-09-21-smartatpg-epoch4-learning-rates-design.md`

## Global Constraints

- GAT: `normal_rounds=2`, `faults_per_update=8`, `k_epochs=4`, `actor_lr=0.0003`, `critic_lr=0.001`.
- Mean: `normal_rounds=2`, `faults_per_update=8`, `k_epochs=1`, `actor_lr=0.001`, `critic_lr=0.01`.
- Old GAT V12/V13 models, epoch-1 manifests, and old GAT checkpoints must be rejected.
- Existing Mean V12/V13 models and epoch-1 training behavior must remain supported.
- Do not add KL, clip-fraction, or experiment-comparison metrics.

---

### Task 1: Encoder-specific training protocol

**Files:**
- Modify: `PODEM/scripts/prepare_smartatpg_training.py`
- Modify: `PODEM/scripts/train_smartatpg.py`
- Modify: `PODEM/scripts/run_smartatpg_training_linux.py`
- Modify: `PODEM/scripts/run_dual_smartatpg_training_linux.py`
- Modify: `PODEM/python/rl_podem/gat_gru.py`
- Test: `PODEM/tests/test_smartatpg_training.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Consumes: encoder variants `level_gat_gru` and `fanin_mean`.
- Produces: `training_hyperparameters(encoder_variant) -> dict[str, int | float]` with `k_epochs`, `actor_lr`, and `critic_lr`.

- [ ] **Step 1: Add failing protocol tests**

Assert the exact per-encoder dictionaries:

```python
self.assertEqual(training_hyperparameters("level_gat_gru"), {
    "k_epochs": 4, "actor_lr": 0.0003, "critic_lr": 0.001,
})
self.assertEqual(training_hyperparameters("fanin_mean"), {
    "k_epochs": 1, "actor_lr": 0.001, "critic_lr": 0.01,
})
```

Also assert the single and dual Linux launch commands pass `--k-epochs 4` only for GAT and `--k-epochs 1` for Mean.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -S -m unittest PODEM.tests.test_smartatpg_training PODEM.tests.test_linux_smartatpg -v
```

Expected: failures show the current shared epoch-1 protocol and old learning rates.

- [ ] **Step 3: Implement encoder-specific constants and agent construction**

Add the central resolver:

```python
TRAINING_HYPERPARAMETERS = {
    "level_gat_gru": {"k_epochs": 4, "actor_lr": 0.0003, "critic_lr": 0.001},
    "fanin_mean": {"k_epochs": 1, "actor_lr": 0.001, "critic_lr": 0.01},
}

def training_hyperparameters(encoder_variant):
    try:
        return dict(TRAINING_HYPERPARAMETERS[encoder_variant])
    except KeyError as error:
        raise ValueError(f"Unsupported SmartATPG encoder: {encoder_variant}") from error
```

Use the resolver for manifest creation/reuse, CLI validation, agent construction, checkpoint configuration, and Linux launch commands. Override only `GATGRUSmartATPGPPOAgent` defaults to `0.0003` and `0.001`, and bump its `TRAINING_FORMAT` version so old GAT checkpoints fail strict loading.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Task 1 command and require all tests to pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add PODEM/scripts PODEM/python/rl_podem/gat_gru.py PODEM/tests
git commit -m "feat: tune PPO epochs and learning rates for GAT"
```

### Task 2: Split GAT and Mean Actor protocols

**Files:**
- Modify: `PODEM/python/rl_podem/cpp_bridge.py`
- Modify: `PODEM/scripts/smartatpg_portable.py`
- Modify: `PODEM/scripts/benchmark_smartatpg.py`
- Modify: `PODEM/scripts/prepare_smartatpg_benchmark.py`
- Modify: `PODEM/python/rl_podem/smartatpg_artifacts.py`
- Test: `PODEM/tests/test_smartatpg.py`
- Test: `PODEM/tests/test_linux_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: Actor metadata containing `encoder_variant`, `faults_per_update`, and `k_epochs`.
- Produces: Mean format `SMARTATPG_MODEL_V13_BATCH8_EPOCH1` and GAT format `SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4`.

- [ ] **Step 1: Add failing format and rejection tests**

Test all allowed pairs:

```python
("fanin_mean", "SMARTATPG_MODEL_V13_BATCH8_EPOCH1", 1)
("level_gat_gru", "SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4", 4)
```

Add negative tests that reject GAT V12/V13 and reject Mean V14. Keep a positive regression test for Mean V12/V13.

- [ ] **Step 2: Run format tests and verify failure**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -S -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_linux_smartatpg -v
```

Expected: failures show the exporter and portable loader still hard-code V13 epoch-1.

- [ ] **Step 3: Implement strict encoder/format pairing**

Define:

```python
MEAN_MODEL_FORMAT = "SMARTATPG_MODEL_V13_BATCH8_EPOCH1"
GAT_MODEL_FORMAT = "SMARTATPG_MODEL_V14_GAT_BATCH8_EPOCH4"
```

Select the output header from `encoder_variant`, validate the matching K value, and make the portable/benchmark loaders reject mismatched formats. Benchmark protocol validation must derive expected K from its encoder rather than use a shared literal.

- [ ] **Step 4: Run format tests and verify pass**

Run the Task 2 command and require all tests to pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add PODEM/python/rl_podem PODEM/scripts PODEM/tests
git commit -m "feat: version the GAT epoch-4 actor protocol"
```

### Task 3: Documentation and full regression

**Files:**
- Modify: `PODEM/docs/SMARTATPG_11D_使用说明.md`
- Modify: active protocol references under `PODEM/scripts` and current tests where names still claim GAT epoch-1.

**Interfaces:**
- Consumes: the encoder-specific protocol and model format constants from Tasks 1 and 2.
- Produces: current documentation that describes GAT epoch-4 and Mean epoch-1 without rewriting historical design records.

- [ ] **Step 1: Update active documentation and identifiers**

Document the exact GAT and Mean hyperparameters, the V14/V13 split, and the intentional rejection of old GAT artifacts. Rename current GAT test methods and output directory defaults that contain `epoch1`; leave Mean-specific epoch-1 names intact.

- [ ] **Step 2: Run syntax and diff checks**

Run:

```powershell
git diff --check
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -S -m compileall -q PODEM/python PODEM/scripts
```

Expected: both commands exit successfully with no output indicating errors.

- [ ] **Step 3: Run SmartATPG regression tests**

Run:

```powershell
& 'C:\Users\acer\.conda\envs\d2l\python.exe' -S -m unittest PODEM.tests.test_smartatpg PODEM.tests.test_smartatpg_training PODEM.tests.test_linux_smartatpg -v
```

Expected: all tests pass; no full two-round dataset training is required.

- [ ] **Step 4: Audit active epoch literals**

Run:

```powershell
rg -n "PPO_EPOCHS_PER_UPDATE = 1|level_gat_gru.*k_epochs.*1|V13_BATCH8_EPOCH1" PODEM/python PODEM/scripts PODEM/docs/SMARTATPG_11D_使用说明.md
```

Expected: remaining V13/epoch-1 references are explicitly Mean-only or negative compatibility checks; no active GAT path uses K=1.

- [ ] **Step 5: Commit Task 3**

```powershell
git add PODEM/docs PODEM/scripts PODEM/tests docs/superpowers/plans
git commit -m "docs: describe encoder-specific SmartATPG PPO protocol"
```
