# Lazy V2 Actor Cache Design

## Goal

Reduce native Actor V2 startup time without changing the trained network,
PODEM decisions, model file format, or Python training behavior. Replace eager
whole-circuit V2 logit precomputation with compute-on-first-use caching.

## Scope

- Change only native C++ inference for `SMARTATPG_ACTOR_V2`.
- Preserve Actor V1 eager encoded-gate caches and all propagation modes.
- Preserve the V2 actor topology, exported weights, binary-gate requirement,
  unknown-input filtering, and PODEM path locking.
- Continue loading and validating the complete circuit-specific embedding file
  at startup. Lazy loading individual rows from the text file is out of scope.

## Data Layout

During `NativeActorPolicy` construction, copy each gate's 64-dimensional
embedding into a contiguous array indexed by the existing numeric gate ID. This
removes string lookup from the decision path. Allocate the existing four-float
logit slots per gate, but leave them uninitialized until requested.

A validity array contains two entries per gate, one for each objective value.
The logical cache key is:

```text
gate_id * 2 + objective_value
```

The corresponding logit offset remains:

```text
gate_id * 4 + objective_value * 2
```

The policy is single-threaded, so cache population requires no mutex.

## Decision Flow

For a V2 backtrace decision:

1. Validate the objective gate ID, objective value, and two-candidate contract.
2. Check the validity entry for `(objective_gate_id, objective_value)`.
3. On a miss, run the unchanged V2 MLP once from the stored gate embedding,
   write both input logits, and mark the entry valid.
4. On a hit, skip the MLP and reuse the stored logits.
5. Return input 1 only when its logit is strictly greater; ties continue to
   select input 0.

Each key is therefore evaluated at most once per native process. The contiguous
numeric-ID embedding cache remains available for future misses; the temporary
name-keyed embedding table can be cleared after that cache is constructed.

## Compatibility And Failure Behavior

Actor V2 text files and fixed embedding files remain byte-compatible. Cache
population uses the same `ActorModel::backtrace_action_logits` computation as
the eager implementation, so actions must be identical. Existing dimension,
circuit-hash, gate-name, and objective-value validation remains in force.

Actor V1 follows its current path unchanged and still clears the source
embedding table after building its encoded backtrace and propagation caches.

## Verification

- Rebuild the pybind extension and native MSVC executable.
- Run `scripts/verify_paper_v3.py` to preserve reward, checkpoint, and
  Python/C++ actor parity coverage.
- Compare lazy V2 decisions and ATPG summaries against the existing eager-cache
  baseline for `c6288` and `s38417_scan`; coverage, vectors, backtrace steps,
  and backtracks must be identical.
- Repeat native wall-clock timing five times for `c6288` and three times for
  `s38417_scan` using the same best actor, embeddings, fault maps, and
  500-backtrack limit.
- Report startup/runtime improvement separately for each circuit. A performance
  regression or any behavioral difference blocks acceptance.
