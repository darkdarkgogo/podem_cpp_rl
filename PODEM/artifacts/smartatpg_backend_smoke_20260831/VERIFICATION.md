# SmartATPG Backend Verification

Date: 2026-08-31. This is correctness verification, not a quality benchmark.

## Implemented

- Switchable SmartATPG and fixed DeepGate curriculum backends.
- Structural 14-column inputs, two 64-unit mean-GraphSAGE layers, graph pooling,
  and an 80-column graph-context/current-gate/mask policy descriptor.
- Joint BC/PPO encoder training, unchanged MC/GAE targets and potential rewards.
- Snapshot-paired descriptor/actor export and backward-compatible native readers.

## Checks

- 36 unit tests passed, including the existing 23 full-fault GAE tests.
- `verify_curriculum_v4.py` passed existing curriculum/reward/actor parity checks.
- `verify_full_fault_gae.py` passed both DeepGate MC and GAE paths, including
  checkpoint restore and exact simulated-interruption recovery.
- `verify_smartatpg.py` passed c432/c499 for both MC and GAE: two BC epochs,
  12 PPO updates per method, 100 Python/native logit pairs per method, and two
  standalone native circuit runs per method. Live-graph and exported-descriptor
  fault-run summaries matched. Interrupted resume was bitwise exact on CPU.
- SmartATPG smoke did not import the DeepGate extractor or consume `.emb` inputs.
- Existing DeepGate actor inference matched the old `atpg_rl_v4.exe` on 11
  reported c432 circuit/fault counter fields with the same inputs and seed.
- CLI export of c1355, absent from the two-circuit smoke training, produced 631
  descriptors and passed native snapshot/circuit validation without fine-tuning.
- All ten existing curriculum circuits passed feature extraction, including
  s38417_scan with 24,983 wires.
- MSVC built both the pybind extension and `build/atpg_rl_smartatpg.exe`.
  Legacy compiler warnings remain; no new compile failure remained.
- `compileall` and `git diff --check` passed.
- Review regressions cover checkpoint/JSON sidecar path collisions, accidental
  PPO-as-DeepGate checkpoints, Windows Unicode artifact paths and duplicate wire
  names. A real DeepGate encoder checkpoint still exports successfully.

Detailed smoke metadata is in `results.json`; per-method training, interrupted
resume, and standalone outputs are in the adjacent `.log` files. Smoke model
weights are test artifacts and are not a recommended production model.

After review hardening, the full two-circuit smoke was rerun successfully in
`../smartatpg_backend_final_20260831/`. Both methods again passed 12 updates,
100 logit pairs, two native runs and bitwise-exact interrupted recovery.
Independent review found no remaining blockers in the scoped fixes.

## Ready for Full Training

`../paper_v8_smartatpg/training_manifest.json` preserves the original ten-circuit
fault/teacher split and has no DeepGate embedding dependencies. No full
SmartATPG training campaign has been run, and there is no evidence here of better
coverage, speed, or circuit generalization. Graph defaults and normalization are
project choices implementing the documented SmartATPG-style design.
