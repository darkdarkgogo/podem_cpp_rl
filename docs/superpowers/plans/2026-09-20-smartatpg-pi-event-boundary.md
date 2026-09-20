# SmartATPG PI Reward Event Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop initialization-only PODEM assignments from producing SmartATPG PI rewards and safely ignore PI events that have no PPO step.

**Architecture:** Fix the event source by deleting the initialization `notify_pi_result()` call while keeping the normal backtrace notification. Add a narrow guard in the Python training bridge so reward calculation occurs only after resolving a valid step index.

**Tech Stack:** C++14/MSVC pybind11 extension, Python 3.9, PyTorch in the `d2l` environment, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-20-smartatpg-pi-event-boundary-design.md`

## Global Constraints

- MEAN reward remains `10 - 7.5 * exp(0.07 * (B + P))` and continues to require `pi_visits > 0`.
- GAT reward, PODEM backtrack semantics, fault selection, and the two PI flip increments remain unchanged.
- Formal SmartATPG backtrack limit remains exactly 100.
- Existing non-finite warning/error diagnostics remain unchanged.
- Do not add optional episode logging or run full training/validation.

---

### Task 1: Correct the C++ PI Reward Event Boundary

**Files:**
- Modify: `PODEM/tests/test_smartatpg.py`
- Modify: `PODEM/src/podem.cpp:54-64`

**Interfaces:**
- Consumes: `cpp_podem.run_stuck_at(..., event_callback=...)` and the existing `pi_not_done` event dictionary.
- Produces: no `pi_not_done` event with `decision_sequence == 0` or `pi_visits == 0` during initialization.

- [ ] **Step 1: Add a failing native event regression test**

Add a test using the small existing `BENCH`, a decision callback returning action 0, and an event callback that records emitted events:

```python
def test_initial_mandatory_implication_emits_no_pi_reward_event(self):
    import cpp_podem
    events = []
    cpp_podem.run_stuck_at(
        str(self.path), lambda request: 0, events.append,
        100, 14, None, True, "backtrace_rl", "", True,
    )
    invalid = [
        event for event in events
        if event["event"] == "pi_not_done"
        and (
            int(event["decision_sequence"]) == 0
            or int(event["pi_visits"]) == 0
        )
    ]
    self.assertEqual(invalid, [])
```

- [ ] **Step 2: Run the regression against the current extension**

Run the single test with the `d2l` interpreter and explicit `PYTHONPATH`. Expected before the fix: FAIL with at least one initialization `pi_not_done` event containing sequence 0 or P=0.

- [ ] **Step 3: Remove the initialization notification**

Delete only this line from the `set_uniquely_implied_value(fault)` `case TRUE` branch:

```cpp
notify_pi_result(find_test);
```

Do not change the normal `notify_pi_result(detected_this_assignment)` call inside the `wpi` path.

- [ ] **Step 4: Rebuild the extension and rerun the regression**

Use the existing MSVC developer environment and:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -S setup.py build_ext --inplace
```

Expected: the focused regression passes.

- [ ] **Step 5: Commit the event-boundary fix**

```powershell
git add -- PODEM/src/podem.cpp PODEM/tests/test_smartatpg.py
git commit -m "fix: exclude PODEM initialization from PI rewards"
```

### Task 2: Guard Python Reward Attribution

**Files:**
- Modify: `PODEM/tests/test_smartatpg.py`
- Modify: `PODEM/python/rl_podem/cpp_bridge.py:474-488`

**Interfaces:**
- Consumes: `CppPodemBacktraceV2Trainer.sequence_to_step` and `agent.buffer.steps`.
- Produces: `pi_not_done` returns without computing reward unless its sequence maps to an in-range step.

- [ ] **Step 1: Add a failing unmapped-event unit test**

Create a MEAN trainer, start an episode, send a `pi_not_done` event with sequence 0 and P=0, then assert it does not raise and does not change episode reward:

```python
def test_mean_ignores_pi_event_without_policy_step(self):
    trainer = CppPodemBacktraceV2Trainer(
        self.graph, agent=self.agent(), auto_update=False,
        reward_scheme=MEAN_REWARD_SCHEME,
    )
    trainer.event_callback({"event": "episode_start"})
    trainer.event_callback({
        "event": "pi_not_done", "decision_sequence": 0,
        "backtracks": 0, "pi_visits": 0,
    })
    self.assertEqual(trainer._episode_extrinsic_reward, 0.0)
    self.assertEqual(trainer._legacy_pi_reward_sum, 0.0)
```

- [ ] **Step 2: Run the focused unit test**

Run only this test in `d2l`. Expected before the guard: FAIL with `SmartATPG reward counters are out of range.`

- [ ] **Step 3: Move validation before reward calculation**

Change the handler to:

```python
step_idx = self.sequence_to_step.get(int(event["decision_sequence"]))
if step_idx is None or not 0 <= step_idx < len(self.agent.buffer.steps):
    return
reward = smartatpg_pi_reward(
    int(event["backtracks"]),
    int(event["pi_visits"]),
    self.reward_alpha,
    self.reward_beta,
)
self.agent.add_reward_to_step(step_idx, reward)
self._episode_extrinsic_reward += reward
self._legacy_pi_reward_sum += reward
```

- [ ] **Step 4: Run the guard and valid-reward tests**

Run the new unmapped-event test plus `test_encoder_specific_reward_events`. Expected: both PASS; the valid event still produces the original reward.

- [ ] **Step 5: Commit the Python guard**

```powershell
git add -- PODEM/python/rl_podem/cpp_bridge.py PODEM/tests/test_smartatpg.py
git commit -m "fix: guard unmapped SmartATPG PI rewards"
```

### Task 3: Focused Final Verification

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: rebuilt `cpp_podem`, C++ event-boundary fix, and Python attribution guard.
- Produces: one successful small MEAN run with `backtrack_limit=100`.

- [ ] **Step 1: Run the three focused tests together**

Run the native event regression, unmapped-event unit test, and valid reward event test in one `unittest` command. Expected: three PASS.

- [ ] **Step 2: Run one small MEAN smoke**

Run the existing small-BENCH trainer through `cpp_podem` with `backtrack_limit=100` and a small fault subset. Expected: normal completion with no `SmartATPG reward counters are out of range` exception.

- [ ] **Step 3: Inspect only the required invariants**

Run `git diff --check`, `git status --short --untracked-files=no`, and targeted searches confirming:

- one `notify_pi_result` call remains in `PODEM/src/podem.cpp`, in the normal `wpi` path;
- three `++rl_pending_pi_assignments` sites remain;
- `backtrack_limit != 100` validation and existing non-finite tags remain.
