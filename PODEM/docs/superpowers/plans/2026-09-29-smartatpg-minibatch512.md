# SmartATPG PPO Minibatch 512 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the fixed SmartATPG PPO transition minibatch from 128 to 512 while preserving the four-fault, four-epoch GAE protocol.

**Architecture:** The numerical default changes in shared training configuration and PPO construction. Because minibatch size is part of compatibility metadata, Python export, portable parsing, C++ parsing, and both training/model format identifiers advance together.

**Tech Stack:** Python 3, PyTorch, unittest, C++14 native actor reader, Git

**Spec:** `PODEM/docs/superpowers/specs/2026-09-29-smartatpg-minibatch512-design.md`

## Global Constraints

- `minibatch_size` is fixed at 512 for both encoders.
- Four-fault rollouts, K=4, GAE, rewards, learning rates, metric weighting, and one RND update per rollout do not change.
- Short final minibatches are retained.
- Previous MINIBATCH128 checkpoint and model formats are rejected.

---

### Task 1: Change training defaults and batching regression tests

**Files:**
- Modify: `PODEM/python/rl_podem/data_split.py`
- Modify: `PODEM/python/rl_podem/ppo.py`
- Modify: `PODEM/python/rl_podem/smartatpg.py`
- Modify: `PODEM/tests/test_smartatpg.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Produces: `training_hyperparameters(variant)["minibatch_size"] == 512`.
- Produces: SmartATPG and base PPO constructor defaults of 512.

- [ ] **Step 1: Update tests to expect minibatch 512 and 1200-transition slicing**

```python
self.assertEqual(agent.minibatch_size, 512)
self.assertEqual(
    [len(batch) for batch in evaluated_batches],
    [512, 512, 176] * 4,
)
```

- [ ] **Step 2: Run focused tests and verify the old 128 protocol fails**

Run: `python -m unittest tests.test_smartatpg_training.SmartATPGPreparationTests.test_training_hyperparameters_are_encoder_specific tests.test_smartatpg.SmartATPGTests.test_ppo_reshuffles_and_uses_transition_minibatches -v`

Expected: FAIL on 128 defaults and batching.

- [ ] **Step 3: Change fixed defaults to 512**

```python
"minibatch_size": 512
```

Change the base PPO default and SmartATPG `setdefault` to 512 as well.

- [ ] **Step 4: Run focused batching/configuration tests**

Run: `python -m unittest tests.test_smartatpg_training.SmartATPGPreparationTests tests.test_smartatpg.SmartATPGTests.test_ppo_reshuffles_and_uses_transition_minibatches tests.test_smartatpg.SmartATPGTests.test_update_metrics_weight_short_minibatches_by_transition_count -v`

Expected: PASS.

### Task 2: Advance checkpoint and exported-model protocols

**Files:**
- Modify: `PODEM/python/rl_podem/smartatpg.py`
- Modify: `PODEM/python/rl_podem/gat_gru.py`
- Modify: `PODEM/python/rl_podem/cpp_bridge.py`
- Modify: `PODEM/python/rl_podem/smartatpg_portable.py`
- Modify: `PODEM/src/rl_policy.cpp`
- Modify: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Produces: V8 training checkpoint formats containing `BATCH4_MINIBATCH512_EPOCH4`.
- Produces: new Mean V18 and GAT V19 exported model headers containing `MINIBATCH512`.
- Consumes: required metadata `minibatch_size=512`.

- [ ] **Step 1: Update format and compatibility tests**

Require the new V18/V19 headers and assert `model.minibatch_size == 512`. Convert stale-format fixtures to the previous MINIBATCH128 headers.

- [ ] **Step 2: Advance Python formats and validators**

```python
MEAN_MODEL_FORMAT = "SMARTATPG_MODEL_V18_MEAN_GAE_BATCH4_MINIBATCH512_EPOCH4"
GAT_MODEL_FORMAT = "SMARTATPG_MODEL_V19_GAT_DEPTHNORM_GAE_BATCH4_MINIBATCH512_EPOCH4"
```

Portable and export validation require the numeric value 512.

- [ ] **Step 3: Advance native C++ formats and metadata validation**

The native reader accepts only V18/V19 for the current batched formats and requires `minibatch_size == 512`.

- [ ] **Step 4: Rebuild the native extension and run protocol tests**

Run: `python setup.py build_ext --inplace`

Run: `python -m unittest tests.test_smartatpg.SmartATPGTests.test_encoder_specific_actor_formats_and_training_protocols tests.test_smartatpg.SmartATPGTests.test_gat_gru_dimensions_gradients_and_portable_parity -v`

Expected: PASS.

### Task 3: Update current documentation and perform focused verification

**Files:**
- Modify: `PODEM/RL_GUIDE.md`
- Modify: `PODEM/docs/SMARTATPG_11D_使用说明.md`

**Interfaces:**
- Produces: user documentation matching the fixed 512 protocol.

- [ ] **Step 1: Replace active-protocol descriptions of 128 with 512**

Document short final minibatches, unchanged four-fault/K4 behavior, and rejection of previous MINIBATCH128 artifacts.

- [ ] **Step 2: Run only the affected test modules**

Run: `python -m unittest tests.test_smartatpg tests.test_smartatpg_training -v`

Expected: PASS, except existing environment-dependent skips.

- [ ] **Step 3: Run necessary static checks**

Run: `python -m py_compile` for changed Python modules.

Run: `git diff --check`

Expected: PASS.
