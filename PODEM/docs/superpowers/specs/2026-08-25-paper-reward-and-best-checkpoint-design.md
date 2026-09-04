# Paper Reward and Best Checkpoint Design

## Goal

Align V2 backtrace training with the SmartATPG reward function, raise the
backtrack limit enough to distinguish hard faults, and prevent the final sweep
from overwriting a better policy.

The change keeps gate embeddings fixed, trains only the V2 backtrace actor,
and leaves propagation RL disabled.

## Fixed Configuration

- Use a backtrack limit of 500 for profiling, training, validation, and final
  comparisons.
- Use the SmartATPG reward constants `alpha = 7.5` and `beta = 0.07`.
- Keep RND as a separately reported intrinsic exploration bonus. The paper does
  not publish a coefficient for combining RND with the environment reward, so
  `rnd_beta` remains configurable and defaults to the existing value of 0.05.
- Preserve existing V1/V2 checkpoints and actors. The new training manifest and
  checkpoint use V3 format identifiers because reward semantics and data splits
  are incompatible with previous V2 training state.

## Paper Reward Semantics

For one fault episode, the extrinsic reward is:

```text
-0.1                                             for each non-PI backtrace step
10 - 7.5 * exp(0.07 * (backtracks + pi_visits))  at a PI when the fault is not done
+100                                             when the fault is detected
-100                                             when the fault is undetected or aborted
```

The four cases are mutually exclusive at a transition. In particular, a PI
assignment that immediately detects the fault receives `+100`, not both the PI
reward and `+100`.

`backtracks` is the episode's cumulative PODEM backtrack count at the PI result.
`pi_visits` is the cumulative number of PI assignments simulated in the episode,
including assignments reached by normal backtrace and assignments flipped by
PODEM backtracking.

The old shaping terms are removed: no `-0.01` policy-decision penalty, no
standalone `-0.1` backtrack penalty, no `-0.05 * backtracks` terminal penalty,
and no `+1/-1` terminal reward.

The exponential PI reward and discounted returns are accumulated and normalized
in float64. Only normalized advantages and returns are converted to the policy's
float32 dtype. This avoids variance overflow at the 500-backtrack limit without
clipping or otherwise changing the paper formula.

## C++ Event Contract

The decision policy interface gains two training events:

- `backtrace_step`: emitted once for every gate-to-input traversal counted by
  `total_backtrace_steps`; it identifies the latest policy decision sequence.
- `pi_not_done`: emitted after a PI assignment has been simulated and the fault
  has not been detected; it contains cumulative `backtracks` and `pi_visits`.

The existing `episode_end` event supplies the terminal outcome. Detected faults
map to `+100`; redundant and backtrack-limit outcomes map to `-100`.

Native C++ inference policies ignore the additional events. Python training
attaches intermediate rewards to the latest PPO decision that led to the event.
Events before the first policy-controlled branch are counted for ATPG metrics
but do not create a synthetic PPO action.

## Hard-Fault Data Split

Preparation profiles faults with the limit of 500, sorts non-redundant faults by
descending backtracks, and takes the top 150 faults from each training circuit.
A seeded shuffle splits those faults into 100 training faults and 50 validation
faults. The sets are disjoint and recorded explicitly in the manifest as
`training_fault_ids` and `validation_fault_ids`.

The manifest records the profile, training, and validation backtrack limits so a
mismatched command fails early rather than silently changing the experiment.

## Validation and Model Selection

Validation runs before training as sweep 0 and after every complete sweep. It
uses:

- the fixed validation fault order;
- deterministic actor argmax rather than sampling;
- no PPO updates and no RND predictor updates;
- the same circuit, embedding, fault map, seed, and backtrack limit on every run.

Aggregate validation models are ordered lexicographically by:

1. more detected faults;
2. fewer aborted faults;
3. fewer backtracks;
4. fewer gate-to-input backtrace steps;
5. fewer policy decisions.

A strict improvement replaces the best model. Exact ties keep the earlier model
to avoid churn.

## Artifacts and Resume Behavior

Training produces:

- `actor_v2_best.txt`: deterministic validation winner for C++ inference;
- `actor_v2_latest.txt`: policy after the most recently completed circuit run;
- the resumable training checkpoint, containing latest PPO/RND state, progress,
  best policy weights, best score, best sweep, validation history, RNG state,
  manifest hash, reward constants, and backtrack limit.

Checkpoint and actor writes remain atomic through temporary-file replacement.
On resume, configuration and manifest hashes must match. A half-completed sweep
resumes training without validation; validation runs only after all circuit runs
for that sweep are complete. If an actor text file is missing, it is regenerated
from checkpoint state.

## Verification

Automated checks cover:

- each of the four paper reward branches and boundary values;
- no duplicate PI and terminal reward for a detected assignment;
- C++ event counts matching ATPG backtrace and PI counters;
- deterministic, disjoint 100/50 fault splits;
- deterministic validation with no parameter mutation;
- lexicographic best-score comparisons and tie handling;
- checkpoint interruption/resume and best/latest actor regeneration;
- V3 rejection of incompatible V2 checkpoints;
- Python/C++ V2 logit parity after export.

A smoke experiment uses small fault subsets and two sweeps to demonstrate that a
later degraded model updates `latest` but does not overwrite `best`. Full
training starts from a fresh V3 checkpoint after the smoke checks pass.
