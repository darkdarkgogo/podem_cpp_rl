# Model-Specific Backtrace Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable backtrace-path lock whose automatic default is enabled for GAT and disabled for MEAN.

**Architecture:** Resolve the user-facing `auto/on/off` value in Python from the encoder variant, then pass one Boolean through the Python/C++ bridge into the existing ATPG instance. The C++ backtrace routine conditionally uses the existing `BacktraceLock`; all other PODEM, masking, and reward behavior remains unchanged.

**Tech Stack:** C++11/pybind11, Python 3, argparse, unittest/pytest

**Spec:** `docs/superpowers/specs/2026-10-08-model-specific-backtrace-lock-design.md`

## Global Constraints

- The CLI parameter is `--backtrace-lock {auto,on,off}` and defaults to `auto`.
- `auto` resolves to on for `level_gat_gru` and off for `fanin_mean`.
- `on` and `off` override either encoder.
- Do not change action masks, rewards, PI assignment, or PODEM backtrack behavior.
- Run only focused tests required for this feature.

---

### Task 1: Resolve and propagate the Python configuration

**Files:**
- Modify: `PODEM/python/rl_podem/cpp_bridge.py`
- Modify: `PODEM/python/rl_podem/training.py`
- Modify: `PODEM/python/rl_podem/validation.py`
- Modify: `PODEM/python/rl_podem/validation_core.py`
- Modify: `PODEM/scripts/train_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Produces: `resolve_backtrace_lock(mode: str, encoder_variant: str) -> bool`.
- Produces: `CppPodemBacktraceV2Trainer(..., backtrace_lock: bool)` and forwards it from `run()`.
- Consumes later: appended `backtrace_lock: bool` argument on `cpp_podem.run_stuck_at` and optional override on `cpp_podem.run_native_validation`.

- [ ] **Step 1: Add focused failing tests**

Test all six auto/override combinations and assert that trainer/validation bridge calls receive the resolved Boolean.

- [ ] **Step 2: Run the focused Python tests and confirm failure**

Run: `python -m pytest tests/test_smartatpg.py tests/test_smartatpg_training.py -q`

Expected: failures identify the missing resolver, CLI option, or forwarded argument.

- [ ] **Step 3: Implement the resolver and CLI propagation**

Add choices `("auto", "on", "off")`; reject unknown encoders/modes; store the resolved Boolean in training config; pass the original mode from the dual launcher to each encoder-specific training subprocess; resolve validation per model encoder and forward the Boolean to the native batch call.

- [ ] **Step 4: Run the same focused Python tests**

Expected: PASS.

### Task 2: Make the C++ lock conditional

**Files:**
- Modify: `PODEM/src/atpg.h`
- Modify: `PODEM/src/runtime_config.cpp`
- Modify: `PODEM/src/podem.cpp`
- Modify: `PODEM/src/python_bindings.cpp`
- Test: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Produces: `ATPG::set_backtrace_lock(bool enabled)`.
- Extends: `cpp_podem.run_stuck_at(..., use_scoap=False, backtrace_lock=True)`.
- Extends: `cpp_podem.run_native_validation(..., circuit_name="", backtrace_lock=None)`, where `None` derives GAT=true and MEAN=false from the loaded actor.

- [ ] **Step 1: Extend the native regression test**

Keep the existing lock-on assertion that one decision sequence owns multiple backtrace steps. Add a lock-off run asserting each revisited decision receives a fresh sequence.

- [ ] **Step 2: Run the focused native regression and confirm failure**

Run: `python -m pytest tests/test_smartatpg.py -k backtrace_lock -q`

Expected: failure because the binding does not accept the new control yet.

- [ ] **Step 3: Implement the C++ Boolean switch**

Store the setting on `ATPG`; when false, bypass all `BacktraceLock` reads and writes and call `choose_policy_candidate` on every eligible revisit. Append compatible binding arguments so existing direct `run_stuck_at` callers retain lock-on behavior. For native actor validation, derive the default from `encoder_variant()` when no override is supplied.

- [ ] **Step 4: Rebuild and run the focused behavior test**

Run: `python -m pip install -e .`

Run: `python -m pytest tests/test_smartatpg.py -k backtrace_lock -q`

Expected: PASS.

### Task 3: Verify the requested defaults and document the option

**Files:**
- Modify: `PODEM/RL_GUIDE.md`
- Test: `PODEM/tests/test_smartatpg.py`
- Test: `PODEM/tests/test_smartatpg_training.py`

**Interfaces:**
- Consumes: the resolver, CLI option, and native switch from Tasks 1 and 2.
- Produces: concise user documentation for automatic defaults and explicit overrides.

- [ ] **Step 1: Document one command-line paragraph**

State that `auto` is GAT on / MEAN off and that `on` or `off` forces the same behavior for both models in that command.

- [ ] **Step 2: Run only the focused final checks**

Run: `python -m pytest tests/test_smartatpg.py -k "backtrace_lock or resolve_backtrace" -q`

Run: `python -m pytest tests/test_smartatpg_training.py -k backtrace_lock -q`

Expected: PASS.

- [ ] **Step 3: Check the edited diff**

Run: `git diff --check`

Expected: no whitespace errors.
