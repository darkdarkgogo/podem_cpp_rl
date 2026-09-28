# SmartATPG Depth-Normalized Reward and GAE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply `-10/H_c` to GAT backtrace steps and enable the approved GAE protocol for both SmartATPG encoders across training, validation, and exported artifacts.

**Architecture:** Centralize circuit depth and step-reward calculation in `smartatpg_rewards.py`, pass the resulting scalar into native validation, and retain the existing Mean reward path. Reuse the existing detached GAE implementation by configuring both encoder agents at construction time, then version every GAT inference contract that validates the reward protocol.

**Tech Stack:** Python 3, PyTorch, unittest/pytest, C++17, pybind11, CMake

**Spec:** `docs/superpowers/specs/2026-09-28-smartatpg-depthnorm-gae-design.md`

## Global Constraints

- GAT uses `H_c=max(1,max(graph.levels))` and `r_step=-10/H_c` without clipping or comparison experiments.
- Mean keeps `legacy_pi_exponential`, including the fixed `-0.1` step reward.
- Both encoders use GAE with `gamma=0.99`, `gae_lambda=0.97`, `normalize_returns=False`, `normalize_advantages=True`, and `return_scale=100.0`.
- Existing cubic backtrack, RND, terminal rewards, PPO epoch counts, and learning rates remain unchanged.
- Native validation consumes the Python-computed step reward so it cannot use a different depth convention.

---

### Task 1: Central reward helper and trainer behavior

**Files:**
- Modify: `python/rl_podem/smartatpg_rewards.py`
- Modify: `python/rl_podem/cpp_bridge.py`
- Test: `tests/test_smartatpg.py`

**Interfaces:**
- Produces: `smartatpg_circuit_depth(levels: Iterable[int]) -> int`
- Produces: `smartatpg_backtrace_step_reward(reward_scheme: str, circuit_depth: int) -> float`
- Consumes: `CircuitGraph.levels`

- [ ] **Step 1: Write failing helper and trainer tests**

```python
self.assertEqual(smartatpg_circuit_depth((0,)), 1)
self.assertEqual(smartatpg_circuit_depth((0, 50)), 50)
self.assertEqual(smartatpg_backtrace_step_reward(GAT_REWARD_SCHEME, 50), -0.2)
self.assertEqual(smartatpg_backtrace_step_reward(GAT_REWARD_SCHEME, 100), -0.1)
self.assertEqual(smartatpg_backtrace_step_reward(GAT_REWARD_SCHEME, 200), -0.05)
self.assertEqual(smartatpg_backtrace_step_reward(MEAN_REWARD_SCHEME, 50), -0.1)
```

Assert that a GAT trainer records `circuit_depth`, `backtrace_step_reward`, and the depth-normalized component sum while a Mean trainer remains at `-0.1`.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_smartatpg.py -k "step_reward or encoder_specific_reward_events" -q`

Expected: failures for missing helpers or the old fixed `-0.1` GAT reward.

- [ ] **Step 3: Implement the reward helpers and trainer wiring**

Set `GAT_REWARD_SCHEME = "cubic_backtrack_depthnorm_v2"`. Validate that depth is an integer greater than zero in the step helper. In `CppPodemBacktraceV2Trainer.__init__`, calculate the graph depth once, derive the scheme-specific step reward, and use it for each attributed `backtrace_step`. Add both values to episode metrics.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_smartatpg.py -k "reward or encoder_specific" -q`

Expected: all selected Python reward tests pass.

### Task 2: Enable GAE in the training protocol

**Files:**
- Modify: `python/rl_podem/data_split.py`
- Modify: `python/rl_podem/training.py`
- Test: `tests/test_smartatpg_training.py`
- Test: `tests/test_smartatpg.py`

**Interfaces:**
- Produces: encoder hyperparameter dictionaries containing `advantage_method`, `gamma`, `gae_lambda`, `normalize_returns`, `normalize_advantages`, and `return_scale`
- Consumes: `BacktracePPOAgentV2.__init__` existing keyword interface

- [ ] **Step 1: Extend failing hyperparameter and construction tests**

Assert that both encoder entries include:

```python
{
    "gamma": 0.99,
    "advantage_method": "gae",
    "gae_lambda": 0.97,
    "normalize_returns": False,
    "normalize_advantages": True,
    "return_scale": 100.0,
}
```

Keep the existing encoder-specific `k_epochs`, Actor LR, and Critic LR assertions. Add a multi-terminal target test proving that a terminal at an interior batch position resets GAE continuation.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `python -m pytest tests/test_smartatpg_training.py -k hyperparameter -q`

Expected: the current dictionaries omit the GAE keys.

- [ ] **Step 3: Pass the approved protocol into both agent constructors**

Extend `TRAINING_HYPERPARAMETERS` and pass the six GAE keys from `training.py` to `AGENT_TYPES[args.encoder](...)`. Include them in the saved training configuration so resume checks and run metadata expose the active Advantage protocol.

- [ ] **Step 4: Verify GAE and checkpoint rejection tests**

Run: `python -m pytest tests/test_advantages.py tests/test_smartatpg_training.py -q`

Expected: GAE tests, configuration tests, and old-hyperparameter rejection tests pass.

### Task 3: Python and native validation parity

**Files:**
- Modify: `python/rl_podem/validation.py`
- Modify: `python/rl_podem/validation_core.py`
- Modify: `src/python_bindings.cpp`
- Test: `tests/test_smartatpg.py`
- Test: `tests/test_smartatpg_validation.py`
- Test: `tests/test_linux_smartatpg.py`

**Interfaces:**
- Changes: `_native_validation_batch(..., reward_scheme, backtrace_step_reward, ...)`
- Changes: `cpp_podem.run_native_validation(..., reward_scheme, backtrace_step_reward, journal_path="", circuit_name="")`
- Changes: `NativeValidationPolicy(..., reward_scheme, backtrace_step_reward, ...)`

- [ ] **Step 1: Add failing Python/native parity and API contract tests**

Update mock call assertions to require the scalar step reward. For GAT, compute it from the loaded validation `CircuitGraph`; for Mean, require exactly `-0.1`. Extend the native reward test to compare a GAT episode against the Python event-derived return.

- [ ] **Step 2: Run validation tests and verify signature failures**

Run: `python -m pytest tests/test_smartatpg_validation.py tests/test_linux_smartatpg.py -q`

Expected: native-call assertions fail until the new parameter is propagated.

- [ ] **Step 3: Implement Python validation reward parity**

Make `_evaluate_fault` obtain the scheme-specific reward from the already-loaded graph and replace `PAPER_REWARD["non_pi"]` for GAT. Pass the same scalar through `_native_validation_batch`.

- [ ] **Step 4: Implement native validation reward parity**

Add a finite, negative `backtrace_step_reward` constructor argument to `NativeValidationPolicy`; use it in `on_backtrace_step`. Update both native entry points and pybind declarations. Keep SCOAP/Mean behavior at `-0.1`, and update the expected GAT scheme string.

- [ ] **Step 5: Build and run parity tests**

Run: `cmake --build build --config Release`

Run: `python -m pytest tests/test_smartatpg.py tests/test_smartatpg_validation.py tests/test_linux_smartatpg.py -q`

Expected: Python and native validation returns agree and invalid reward parameters are rejected.

### Task 4: Exported model protocol and full regression

**Files:**
- Modify: `python/rl_podem/cpp_bridge.py`
- Modify: `python/rl_podem/smartatpg_portable.py`
- Modify: `src/rl_policy.cpp`
- Modify: affected tests under `tests/`

**Interfaces:**
- Produces: a new GAT model format string paired only with `cubic_backtrack_depthnorm_v2`
- Preserves: Mean model format and `legacy_pi_exponential`

- [ ] **Step 1: Update failing artifact contract tests**

Change expected GAT scheme and model-format constants. Add rejection cases for the former GAT scheme and for a new-format model carrying the wrong scheme.

- [ ] **Step 2: Run artifact tests and verify they fail**

Run: `python -m pytest tests/test_smartatpg.py tests/test_smartatpg_training.py -k "model or artifact or protocol or reward_scheme" -q`

Expected: old portable and C++ readers reject the new format until updated.

- [ ] **Step 3: Version the exported GAT contract**

Update the shared Python exporter/portable reader and the C++ actor loader to accept only the new GAT format/reward pairing. Do not change Mean format acceptance.

- [ ] **Step 4: Run formatting, Python tests, and native tests**

Run: `git diff --check`

Run: `python -m pytest tests/test_advantages.py tests/test_smartatpg.py tests/test_smartatpg_training.py tests/test_smartatpg_validation.py tests/test_linux_smartatpg.py -q`

Run: `ctest --test-dir build -C Release --output-on-failure`

Expected: all commands pass.

- [ ] **Step 5: Review the completed diff**

Compare the implementation with `docs/superpowers/specs/2026-09-28-smartatpg-depthnorm-gae-design.md`, inspect every reward-protocol literal, and request a focused code review before reporting completion.

