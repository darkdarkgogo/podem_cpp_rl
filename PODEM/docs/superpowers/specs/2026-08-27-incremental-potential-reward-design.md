# Incremental Potential Reward Design

## Goal

Improve PPO credit assignment without changing the optimization target used by
the multicircuit curriculum. Backtrace and backtrack costs must be attributed to
the RL decision that caused them, while the total extrinsic reward for an
episode remains exactly equal to the existing baseline-relative terminal reward.

This change affects Python training only. C++ PODEM already emits decision
sequence IDs for backtrace and backtrack events. Native C++ actor inference,
actor export, embeddings, the circuit manifest, and PODEM search behavior remain
unchanged.

## Reward Potential

For heuristic baseline counts `B_h` and `T_h`, and current RL counts `B` and
`T`, define the search potential:

```text
Phi(B, T) =
    20 * clip((B_h - B) / max(B_h, 1), -2, 1)
  + 10 * clip((T_h - T) / max(T_h, 1), -2, 1)
```

The detection component remains `+100` for a detected fault and `-100`
otherwise. The existing episode objective is therefore:

```text
R_episode = detection + Phi(B_final, T_final)
```

On each backtrace or backtrack event, training computes the new potential and
assigns the delta to the decision sequence carried by the event:

```text
r_event = Phi(B_new, T_new) - Phi(B_old, T_old)
```

The delta is normally non-positive. Clipping behavior, zero-backtrack
baselines, and the relative weighting of backtracks and backtrace steps are
identical to the current reward.

## Exact Episode Equivalence

The trainer tracks all distributed potential deltas. At episode end it
recomputes the existing baseline-relative reward from the authoritative C++
final counters and adds only the residual:

```text
r_terminal = R_episode - sum(distributed potential deltas)
```

Consequently, the sum of extrinsic step rewards is exactly `R_episode`, with no
double counting. The residual also handles events that cannot be mapped to an
RL decision, such as search work before the first multi-candidate policy
decision or a C++ event without a usable sequence ID.

`gamma` remains `1.0`. The undiscounted sum preserves the baseline-relative
objective across long PODEM episodes, while moving potential deltas to their
causal steps makes per-step returns differ. RND intrinsic rewards and their
curriculum schedule remain unchanged.

## Trainer State And Events

At `episode_start`, the curriculum trainer resets current backtrack and
backtrace counters, distributed reward, and the sequence-to-step map, then loads
the fault's heuristic baseline.

For `backtrace_step`, it increments the local backtrace count, computes the
potential delta, and adds it to the mapped PPO step when a mapping exists.

For `backtrack`, it increments the local backtrack count and performs the same
delta attribution. `pi_not_done` remains ignored because its cost is already
represented by backtrace and backtrack counters.

At `episode_end`, authoritative C++ counters determine the final reward. Any
difference between event-derived counters and final counters is covered by the
terminal residual rather than silently changing the episode objective.

Training metrics record the distributed potential sum, terminal residual, and
counter mismatches so attribution coverage can be audited.

## Checkpoint Compatibility

Old curriculum checkpoints contain critic and optimizer state learned under the
terminal-only reward distribution. They must not resume under the new reward
semantics. The training configuration includes a reward-distribution version,
causing an explicit compatibility failure if an old checkpoint path is reused.

Fresh training uses a new checkpoint and actor output location. The existing
manifest, teacher samples, embedding artifacts, and old successful artifacts are
retained.

## Verification

Verification covers:

1. Potential values and deltas for improvement, regression, clipping, and a
   zero-backtrack heuristic baseline.
2. Telescoping event deltas plus terminal residual equal the existing terminal
   reward within floating-point tolerance.
3. Backtrace and backtrack events are assigned to their mapped PPO steps.
4. Missing or unmapped events are recovered by the terminal residual.
5. Existing teacher, behavior-cloning, checkpoint-resume, actor parity, and
   native build checks continue to pass.
6. A fresh smoke curriculum completes and reports finite losses before formal
   training starts.

## Acceptance Criteria

- Every episode preserves the existing total extrinsic reward to within
  `1e-9`.
- At least one mapped search-cost event changes a non-terminal PPO step reward
  in the integration verification.
- Old reward checkpoints cannot be resumed accidentally.
- Pure C++ inference artifacts and runtime behavior are unchanged.
- Formal training starts from behavior cloning in a new artifact directory.
