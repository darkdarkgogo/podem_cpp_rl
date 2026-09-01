# SmartATPG Backend Implementation Plan

The user approved the written design and implementation on 2026-08-31.
The writing-plans skill is not installed; this is the executable local plan.

1. Add deterministic binary-BENCH features and structural SCOAP, with a small
   two-layer PyTorch GraphSAGE module. Test hand-calculated costs, invalid input,
   directed aggregation, node reindexing, and gradients.
2. Add a SmartATPG policy/agent using immutable graph references. Reuse the
   existing PPO target/loss implementation through an epoch-evaluation hook.
   Include encoder gradients in BC and PPO, and invalidate no-grad caches after
   parameter loads/updates. Keep DeepGate behavior unchanged.
3. Generalize the training bridge and curriculum BC state construction. Add
   backend-aware manifests, source-manifest preparation, checkpoint validation,
   and CLI routing. Preserve existing reward/fault semantics.
4. Add snapshot-paired descriptors and actor export. Extend Python/C++ readers
   and native backend validation without changing V1/V2 actor arithmetic.
5. Run unit tests, existing GAE/curriculum regressions, two-circuit BC/MC/GAE and
   resume smoke tests, and native parity/negative cases. Compile both extension
   and standalone executable. Request independent code review and fix findings.
6. Document commands, fresh-model requirements, verification results, and any
   residual limitations. Do not claim performance gains from smoke tests.

Do not revert or include unrelated existing GAE changes in a new commit.

## Review Lessons

- Backend switches must resolve existing dataset metadata before any resume
  writes; never reinterpret an old dataset using a new CLI default.
- Rollout state must own its masks, while trained graph representations must
  be recomputed under the current policy for every PPO epoch.
- Export validation must include generated sidecars and snapshot directories,
  not only the explicitly named output files.
- Validate encoder checkpoint identity strictly. Permissive state loading can
  otherwise turn a type mistake into apparently valid random embeddings.
- Test native artifact readers with Unicode paths and duplicate identity tables
  as well as numerically valid descriptors.
