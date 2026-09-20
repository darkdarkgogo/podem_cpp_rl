# SmartATPG PI Visit and Non-finite Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align SmartATPG PI visit counting with Agent-visible PI/PPI visits and add first-failure native validation diagnostics without changing rewards or search behavior.

**Architecture:** Keep the existing pending/cumulative visit flow and remove only the mandatory backward-implication increment. Keep native reward calculation in `NativeValidationPolicy::on_pi_not_done()`, but split it into named intermediates so warnings and non-finite errors can report the exact numerical state.

**Tech Stack:** C++14/MSVC extension via pybind11, Python `unittest`, setuptools editable build, PyTorch in the `d2l` conda environment.

**Spec:** `docs/superpowers/specs/2026-09-20-smartatpg-pi-visit-nonfinite-design.md`

## Global Constraints

- MEAN reward remains `10 - 7.5 * exp(0.07 * (B + P))`.
- GAT cubic reward, PODEM backtrack semantics, fault selection, and search flow remain unchanged.
- Formal SmartATPG backtrack limit remains exactly 100.
- Do not clip, normalize, saturate, or clamp the exponential reward.
- Run only focused tests, one extension rebuild, and one small `d2l` smoke; do not run full training or benchmarks.

---

### Task 1: Exclude Mandatory Implication from PI Visits

**Files:**
- Modify: `PODEM/tests/native_logic_conversion.cpp`
- Modify: `PODEM/tests/test_native_logic_conversion.py`
- Modify: `PODEM/src/podem.cpp:937-955`

**Interfaces:**
- Consumes: existing `ATPGScoapTestAccess` friend access to private ATPG state and methods.
- Produces: `pi-visit-semantics` native harness operation reporting mandatory and direct-backtrace pending counts.

- [ ] **Step 1: Add a failing native semantics test**

Add an `ATPGScoapTestAccess::pi_visit_semantics()` helper that loads `INPUT(a)`, `INPUT(b)`, `y=AND(a,b)`, enables an RL episode, calls `backward_imply(y, 1)`, records pending visits, clears state, calls `find_pi_assignment(a, 1)`, and returns `mandatory_count:direct_count`. Add the harness operation `pi-visit-semantics`, and assert from Python that its output is `0:1`.

- [ ] **Step 2: Run the focused test and verify the existing code fails**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -S -m unittest tests.test_native_logic_conversion.NativeLogicConversionTests.test_pi_visit_semantics -v
```

with `PYTHONPATH` containing the repository, `PODEM/python`, `PODEM/scripts`, and the `d2l` site-packages directory. Expected before the fix: failure because mandatory backward implication reports a nonzero count.

- [ ] **Step 3: Remove only the mandatory increment**

In the input branch of `ATPG::backward_imply()`, delete:

```cpp
if (rl_podem_episode_active)
  ++rl_pending_pi_assignments;
```

Do not change the increment in `find_pi_assignment()` or either PI flip branch.

- [ ] **Step 4: Run the focused native semantics test**

Run the command from Step 2. Expected: PASS with `0:1`.

- [ ] **Step 5: Commit the semantic fix**

```powershell
git add -- PODEM/src/podem.cpp PODEM/tests/native_logic_conversion.cpp PODEM/tests/test_native_logic_conversion.py
git commit -m "fix: align SmartATPG PI visit semantics"
```

### Task 2: Add Native Non-finite Reward Diagnostics

**Files:**
- Modify: `PODEM/src/python_bindings.cpp:242-255`
- Modify: `PODEM/tests/test_smartatpg.py`

**Interfaces:**
- Consumes: `NativeValidationPolicy::on_pi_not_done(sequence, backtracks, pi_visits)` and `current_fault_id_`.
- Produces: `SMARTATPG_REWARD_WARN` and `SMARTATPG_NONFINITE` stderr records with the approved fields.

- [ ] **Step 1: Add a focused source-contract test**

Add a test that reads `PODEM/src/python_bindings.cpp` and asserts the native validation reward path contains the two log tags, threshold `exponent >= 600.0`, finite checks for both `step_reward` and `reward_after`, and all required field labels: `fault`, `seq`, `B`, `P`, `BplusP`, `exponent`, `reward_before`, `step_reward`, and `reward_after`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -S -m unittest tests.test_smartatpg.SmartATPGTests.test_native_reward_diagnostic_contract -v
```

Expected before implementation: FAIL because diagnostic tags and named intermediate calculations are absent.

- [ ] **Step 3: Implement the diagnostic path**

Replace the compact reward update with named `double` intermediates. Print and flush the warning at exponent 600 or above. Before assigning `reward_`, print and flush the non-finite record and throw `std::runtime_error("Non-finite SmartATPG PI reward; see SMARTATPG_NONFINITE log")` if either the step or cumulative result is non-finite. Keep the episode-end finite check unchanged.

- [ ] **Step 4: Run focused diagnostics and existing native parity test**

Run the new contract test and:

```powershell
C:\Users\acer\.conda\envs\d2l\python.exe -S -m unittest tests.test_smartatpg.SmartATPGTests.test_native_validation_matches_python_policy_without_callbacks -v
```

Expected: both PASS; the parity test confirms normal finite rewards remain unchanged.

- [ ] **Step 5: Commit the diagnostic change**

```powershell
git add -- PODEM/src/python_bindings.cpp PODEM/tests/test_smartatpg.py
git commit -m "chore: add SmartATPG nonfinite reward diagnostics"
```

### Task 3: Rebuild and Smoke Test

**Files:**
- No source changes expected.

**Interfaces:**
- Consumes: the two implementation commits and the existing editable `cpp_podem` package.
- Produces: a rebuilt extension and one successful mapped-catalog/native ATPG smoke in `d2l`.

- [ ] **Step 1: Rebuild the extension**

From `PODEM`, run the `d2l` interpreter with `setup.py build_ext --inplace` so the current source is compiled without reinstalling unrelated dependencies.

- [ ] **Step 2: Run the two focused tests together**

Run only the PI visit semantics test, diagnostic contract test, and existing native reward parity test. Expected: PASS.

- [ ] **Step 3: Run one small smoke**

Use the rebuilt extension from the `d2l` environment to catalog or profile a small BENCH fault with `backtrack_limit=100`; assert the call returns normally and produces at least one fault/result.

- [ ] **Step 4: Inspect the final diff**

Run `git diff --check`, `git status --short`, and a targeted `rg` confirming exactly three remaining `++rl_pending_pi_assignments` sites: normal PI reach and the two PI flips. Do not run a repository-wide test suite.
