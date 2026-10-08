# Model-Specific Backtrace Lock Design

## Goal

Make SmartATPG backtrace path locking configurable while changing the default behavior by encoder: GAT keeps the current lock and MEAN makes a fresh RL choice when an unresolved decision gate is revisited.

## Interface

Training and validation commands accept `--backtrace-lock {auto,on,off}` with `auto` as the default.

- `auto`: enable the lock for `level_gat_gru`; disable it for `fanin_mean`.
- `on`: force the lock on for either encoder.
- `off`: force the lock off for either encoder.

The resolved Boolean is passed through the Python bridge to the C++ PODEM engine. Direct Python bindings expose the same Boolean control while preserving existing callers with a compatible default.

## Engine Behavior

When locking is enabled, the existing `BacktraceLock` behavior remains unchanged: revisiting the same unresolved objective within one PODEM path reuses the previous RL-selected input until the path generation changes.

When locking is disabled, backtrace does not read or write `BacktraceLock`; every visit to an eligible decision gate invokes the policy again. Action masks, rewards, PI assignment, and the PODEM backtrack algorithm are unchanged.

## Training and Validation

Both training and validation resolve `auto` from the selected encoder before starting a run. Native validation derives its automatic default from the actor artifact's encoder variant. The resolved mode is included wherever the run configuration or training protocol must remain reproducible.

## Required Tests

- Parameter resolution: GAT defaults on, MEAN defaults off, and explicit `on`/`off` overrides both.
- Native behavior: locking on reuses an unfinished choice; locking off requests a new decision when the same unfinished objective is revisited.
- Existing GAT behavior remains covered by the current backtrace-lock regression test.

Testing is limited to the focused SmartATPG unit tests and the necessary native build-backed test when the extension is available.
