# SmartATPG PPO Four-Fault Minibatch Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train both SmartATPG encoders with four-complete-fault rollouts, four PPO epochs, and independently shuffled 128-transition minibatches while preserving the approved reward and GAE semantics.

**Architecture:** Rollout collection remains owned by `training.py`, target construction and minibatch optimization remain owned by the base PPO agent, and graph-aware indexed evaluation remains owned by `SmartATPGPPOAgent`. Training/model protocol metadata carries all three batching parameters so Python, portable, and native consumers reject stale artifacts.

**Tech Stack:** Python 3, PyTorch, unittest, C++17 native actor reader, Git

**Spec:** `PODEM/docs/superpowers/specs/2026-09-28-smartatpg-ppo-minibatch-rollout-design.md`

## Global Constraints

- A rollout contains four complete faults and never ends inside a fault.
- Both encoders use `k_epochs=4` and `minibatch_size=128`.
- GAE and rollout-wide Advantage normalization run exactly once before shuffling.
- Every epoch uses a fresh `torch.randperm`; the final short minibatch is retained.
- GAT reward remains `cubic_backtrack_depthnorm_v2` with `-10/H_c`; Mean reward remains `legacy_pi_exponential`.
- RND keeps one full-rollout optimizer step per PPO update.
- Update metrics cover all epochs and are weighted by transition count; final-epoch metrics are also exposed separately.

---

### Task 1: Fix the training and artifact protocol constants

**Files:**
- Modify: `PODEM/python/rl_podem/data_split.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`
- Modify: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Produces: `FAULTS_PER_UPDATE = 4` and `TRAINING_HYPERPARAMETERS[variant]["minibatch_size"] == 128`.
- Produces: `training_hyperparameters("fanin_mean")["k_epochs"] == 4`.

- [ ] **Step 1: Change protocol assertions to the new constants**

```python
self.assertEqual(FAULTS_PER_UPDATE, 4)
self.assertEqual(training_hyperparameters("fanin_mean")["k_epochs"], 4)
self.assertEqual(training_hyperparameters("fanin_mean")["minibatch_size"], 128)
self.assertEqual(training_hyperparameters("level_gat_gru")["minibatch_size"], 128)
```

- [ ] **Step 2: Run the focused tests and verify they fail on the old values**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: FAIL where the suite still observes eight faults or Mean K=1.

- [ ] **Step 3: Update the shared protocol constants**

```python
FAULTS_PER_UPDATE = 4
TRAINING_HYPERPARAMETERS = {
    "level_gat_gru": {
        **ADVANTAGE_HYPERPARAMETERS,
        "k_epochs": 4,
        "minibatch_size": 128,
        "actor_lr": 0.0003,
        "critic_lr": 0.001,
    },
    "fanin_mean": {
        **ADVANTAGE_HYPERPARAMETERS,
        "k_epochs": 4,
        "minibatch_size": 128,
        "actor_lr": 0.001,
        "critic_lr": 0.01,
    },
}
```

- [ ] **Step 4: Run the focused protocol tests**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: protocol constant tests PASS; tests for not-yet-updated consumers may still fail.

- [ ] **Step 5: Commit the protocol constants**

```bash
git add PODEM/python/rl_podem/data_split.py PODEM/tests/test_smartatpg_training.py PODEM/tests/test_smartatpg.py
git commit -m "feat: define four-fault PPO minibatch protocol"
```

### Task 2: Implement indexed transition minibatch PPO

**Files:**
- Modify: `PODEM/python/rl_podem/ppo.py`
- Modify: `PODEM/python/rl_podem/smartatpg.py`
- Modify: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Consumes: `minibatch_size: int` constructor hyperparameter.
- Produces: `_evaluate_rollout(indices)` returning evaluations in the exact supplied index order.
- Produces: update metrics `minibatches_per_epoch`, `optimizer_steps`, `rollout_transitions`, all-epoch weighted losses, `approx_kl`, `clip_fraction`, and corresponding `*_last_epoch` metrics.

- [ ] **Step 1: Add failing tests for minibatch slicing, shuffle count, target reuse, weighted metrics, and indexed graph evaluation**

```python
with patch("rl_podem.ppo.torch.randperm", wraps=torch.randperm) as shuffle:
    metrics = agent.update()
self.assertEqual(shuffle.call_count, agent.k_epochs)
self.assertEqual(metrics["minibatches_per_epoch"], 3)
self.assertEqual(metrics["optimizer_steps"], 12)
self.assertEqual(metrics["rollout_transitions"], 350)
```

Use a recording test agent with 350 synthetic steps to assert index batches have sizes `128, 128, 94` in each epoch and every index appears once per epoch. Patch `full_fault_targets` and assert it is called once.

- [ ] **Step 2: Run the focused PPO tests and verify failure**

Run: `python -m unittest tests.test_smartatpg.SmartATPGTests -v`

Expected: FAIL because the current update evaluates the full rollout and performs one optimizer step per epoch.

- [ ] **Step 3: Add and validate the minibatch hyperparameter**

```python
self.minibatch_size = int(minibatch_size)
if self.minibatch_size <= 0:
    raise ValueError("minibatch_size must be positive")
```

Include `minibatch_size` in `hyperparameters()` so full checkpoint resume validation rejects stale training state.

- [ ] **Step 4: Make base and graph-aware rollout evaluation indexed**

```python
def _evaluate_rollout(self, indices):
    return [self.policy.evaluate_step(self.buffer.steps[index]) for index in indices]
```

For SmartATPG, build differentiable embeddings only for distinct graphs represented by `indices`, preserve supplied order, and never cache these embeddings across optimizer steps.

- [ ] **Step 5: Replace full-batch PPO with per-epoch minibatches**

```python
for epoch in range(self.k_epochs):
    permutation = torch.randperm(step_count).tolist()
    for start in range(0, step_count, self.minibatch_size):
        indices = permutation[start:start + self.minibatch_size]
        evaluated = self._evaluate_rollout(indices)
        # calculate selected losses, backward, optimizer.step()
```

Construct targets before this loop. Keep the existing single full-rollout RND update after it.

- [ ] **Step 6: Add transition-weighted all-epoch and final-epoch aggregation**

```python
weighted_total[name] += float(batch_metric) * batch_size
metrics[name] = weighted_total[name] / (self.k_epochs * step_count)
metrics[f"{name}_last_epoch"] = last_epoch_total[name] / step_count
```

Use `approx_kl = ((ratio - 1) - log_ratio).mean()` and `clip_fraction = ((ratio - 1).abs() > eps_clip).float().mean()` for each minibatch.

- [ ] **Step 7: Run the focused PPO tests**

Run: `python -m unittest tests.test_smartatpg.SmartATPGTests -v`

Expected: PASS.

- [ ] **Step 8: Commit indexed minibatch PPO**

```bash
git add PODEM/python/rl_podem/ppo.py PODEM/python/rl_podem/smartatpg.py PODEM/tests/test_smartatpg.py
git commit -m "feat: optimize PPO with shuffled transition minibatches"
```

### Task 3: Wire training configuration, fault counts, and observability

**Files:**
- Modify: `PODEM/python/rl_podem/training.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: `hyperparameters["minibatch_size"]`.
- Produces: checkpoint/config field `minibatch_size` and update log/TensorBoard counters.
- Produces: `agent.update(rollout_faults=batch_faults)` so update metrics report the actual final partial-rollout size.

- [ ] **Step 1: Add failing tests for boundaries `[4, 8, 10]`, config propagation, and final partial-rollout metrics**

```python
boundaries = [i for i in range(1, 11) if _fault_update_boundary(i, 10, 4)]
self.assertEqual(boundaries, [4, 8, 10])
```

- [ ] **Step 2: Run training tests and verify failure**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: FAIL on old fault boundaries and missing minibatch configuration.

- [ ] **Step 3: Pass and persist the protocol fields**

```python
agent = AGENT_TYPES[args.encoder](
    graphs,
    hidden_dim=32,
    lr_actor=hyperparameters["actor_lr"],
    lr_critic=hyperparameters["critic_lr"],
    rnd_beta=args.rnd_beta,
    k_epochs=args.k_epochs,
    gamma=hyperparameters["gamma"],
    advantage_method=hyperparameters["advantage_method"],
    gae_lambda=hyperparameters["gae_lambda"],
    normalize_returns=hyperparameters["normalize_returns"],
    normalize_advantages=hyperparameters["normalize_advantages"],
    return_scale=hyperparameters["return_scale"],
    minibatch_size=hyperparameters["minibatch_size"],
)
config["minibatch_size"] = hyperparameters["minibatch_size"]
```

Include `minibatch_size` in `_training_protocol` and pass the actual `batch_faults` into `agent.update`.

- [ ] **Step 4: Expand TensorBoard and console update reporting**

Log the weighted PPO metrics, final-epoch metrics, `rollout_faults`, `rollout_transitions`, `minibatches_per_epoch`, `optimizer_steps`, and `epochs`.

- [ ] **Step 5: Run training tests**

Run: `python -m unittest tests.test_smartatpg_training -v`

Expected: PASS.

- [ ] **Step 6: Commit training integration**

```bash
git add PODEM/python/rl_podem/training.py PODEM/tests/test_smartatpg_training.py
git commit -m "feat: integrate four-fault minibatch training"
```

### Task 4: Version Python, portable, and native model protocols

**Files:**
- Modify: `PODEM/python/rl_podem/cpp_bridge.py`
- Modify: `PODEM/python/rl_podem/smartatpg_portable.py`
- Modify: `PODEM/src/rl_policy.cpp`
- Modify: `PODEM/tests/test_smartatpg.py`
- Modify: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: exported metadata fields `faults_per_update=4`, `minibatch_size=128`, `k_epochs=4`.
- Produces: new encoder-specific format headers accepted consistently by Python and C++ readers.

- [ ] **Step 1: Change artifact tests to require new headers and minibatch metadata**

```python
self.assertEqual(model.faults_per_update, 4)
self.assertEqual(model.minibatch_size, 128)
self.assertEqual(model.k_epochs, 4)
```

Add stale-header and mismatched-field rejection cases.

- [ ] **Step 2: Run artifact tests and verify failure**

Run: `python -m unittest tests.test_smartatpg tests.test_smartatpg_training -v`

Expected: FAIL on old headers and missing `minibatch_size` parsing.

- [ ] **Step 3: Advance formats and validate all batching fields in Python**

Use unambiguous new GAT and Mean format names containing `BATCH4_MINIBATCH128_EPOCH4`. Add `minibatch_size` to the exported protocol keys and `PortableModel`.

- [ ] **Step 4: Synchronize the native reader**

Require the new encoder-specific headers, then parse and validate:

```cpp
faults_per_update == 4
minibatch_size == 128
k_epochs == 4
```

- [ ] **Step 5: Run Python and native protocol tests**

Run: `python -m unittest tests.test_smartatpg tests.test_smartatpg_training -v`

Expected: PASS.

- [ ] **Step 6: Commit protocol versioning**

```bash
git add PODEM/python/rl_podem/cpp_bridge.py PODEM/python/rl_podem/smartatpg_portable.py PODEM/src/rl_policy.cpp PODEM/tests/test_smartatpg.py PODEM/tests/test_smartatpg_training.py
git commit -m "feat: version four-fault minibatch model protocol"
```

### Task 5: Full verification and documentation consistency

**Files:**
- Modify: `PODEM/RL_GUIDE.md`
- Modify: `PODEM/docs/SMARTATPG_11D_使用说明.md`

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: user-facing commands and protocol descriptions consistent with the implementation.

- [ ] **Step 1: Search for stale protocol identities**

Run: `rg -n "BATCH8|faults_per_update.?8|FAULTS_PER_UPDATE = 8|k_epochs.?1|128" PODEM`

Expected: no stale current-protocol claims outside explicit legacy-rejection fixtures or historical design documents.

- [ ] **Step 2: Update current user documentation where it describes the active protocol**

Document four faults per rollout, 128 transitions per minibatch, four epochs for both encoders, final partial minibatches, and old checkpoint incompatibility.

- [ ] **Step 3: Run the complete Python suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS.

- [ ] **Step 4: Run syntax and diff checks**

Run: `python -m compileall python scripts`

Run: `git diff --check`

Expected: both commands PASS.

- [ ] **Step 5: Commit final documentation or verification fixes**

```bash
git add PODEM/RL_GUIDE.md PODEM/docs/SMARTATPG_11D_使用说明.md
git commit -m "docs: describe SmartATPG minibatch training"
```
