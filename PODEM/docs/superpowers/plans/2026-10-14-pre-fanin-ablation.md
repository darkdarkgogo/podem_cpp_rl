# Pre-fanin GAT Ablation Implementation Plan

**Goal:** Run the three existing ablation variables on the original two-output GAT-GRU Actor.

**Architecture:** Keep commit `7d92ab8` as the base. Add a no-objective-value model variant, a native cache switch, and separate training and validation entry points. Reuse the full checkpoint for no-cache validation.

**Tech Stack:** Python, PyTorch, pybind11, C++.

**Spec:** `PODEM/docs/superpowers/specs/2026-10-14-pre-fanin-ablation-design.md`

## Global Constraints

- Work only in the `experiment/pre-fanin-ablation` worktree.
- Preserve the 11-dimensional objective embedding and two-output Actor; never add a candidate fanin embedding.
- Use the same prepared GAT manifest, seed, reward, backtrack limit, and two-round protocol for trained groups.
- Do not start full training or validation during implementation.

---

### Task 1: No-objective-value variant

- [ ] Add the 11-input GAT policy and agent variant with two Actor outputs.
- [ ] Admit its variant and dimensions in checkpoint metadata, export, portable inference, and native inference.
- [ ] Verify policy and artifact round trips with focused tests.

### Task 2: Native cache control

- [ ] Add an optional logits-cache flag to native validation, preserving the current default.
- [ ] Thread it through Python validation and verify the full/no-cache Actor identity.

### Task 3: Experiment orchestration

- [ ] Add a training command for full, no-lock, and no-objective-value groups using one manifest.
- [ ] Add a validation command for all four groups and an identity-checked comparison report.
- [ ] Document commands and run focused syntax, unit, and native build checks.
