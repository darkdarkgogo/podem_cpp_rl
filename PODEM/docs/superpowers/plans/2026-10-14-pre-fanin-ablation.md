# Pre-fanin GAT Ablation Implementation Plan

**Goal:** Run the three existing ablation variables on the original two-output GAT-GRU Actor using existing full results.

**Architecture:** Keep commit `7d92ab8` as the base. Add a no-objective-value model variant, a native cache switch, and separate training and validation entry points. Read the existing full records and checkpoint; reuse that checkpoint for no-cache validation.

**Tech Stack:** Python, PyTorch, pybind11, C++.

**Spec:** `PODEM/docs/superpowers/specs/2026-10-14-pre-fanin-ablation-design.md`

## Global Constraints

- Work only in the `experiment/pre-fanin-ablation` worktree.
- Preserve the 11-dimensional objective embedding and two-output Actor; never add a candidate fanin embedding.
- Use the same prepared GAT manifest, seed, reward, backtrack limit, and two-round protocol for trained groups.
- Do not start full training or validation during implementation.

---

### Task 1: No-objective-value variant

- [x] Add the 11-input GAT policy and agent variant with two Actor outputs.
- [x] Admit its variant and dimensions in checkpoint metadata, export, portable inference, and native inference.
- [x] Verify policy and artifact round trips with focused tests.

### Task 2: Native cache control

- [x] Add an optional logits-cache flag to native validation, preserving the current default.
- [x] Thread it through Python validation and verify the full/no-cache Actor identity.

### Task 3: Experiment orchestration

- [x] Add a training command for no-lock and no-objective-value groups using the existing full manifest.
- [x] Add a validation command for all four groups and an identity-checked comparison report.
- [x] Document commands and run focused syntax, unit, and native build checks.
