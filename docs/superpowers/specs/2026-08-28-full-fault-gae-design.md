# Full-Fault GAE Design

## Goal

Add Generalized Advantage Estimation (GAE) to the V2 backtrace PPO agent while preserving Monte Carlo (MC) as a selectable baseline. A rollout remains one complete fault-solving episode; this change does not introduce fixed-length truncation or mid-fault policy updates.

## Training Semantics

The behavior-cloning stage remains unchanged. During PPO fine-tuning, `policy_old` collects every Actor decision for one fault. The trainer marks the final step terminal after the fault is detected or aborted, then performs one PPO update and clears the rollout buffer.

Integration correction: Actor behavior-cloning loss and schedule remain unchanged, but post-BC Critic initialization zeros only the output weight and sets output bias to 1. The former all-layer-zero initialization traps the Tanh Critic at a constant bias. Keeping its hidden features preserves the same initial value of 1 while permitting state-dependent learning. This is tested and recorded in curriculum checkpoint configuration for both MC and GAE.

The agent accepts:

- `advantage_method`: `"mc"` or `"gae"`; curriculum training defaults to `"gae"`.
- `gae_lambda`: defaults to `0.97` and must be in `[0, 1]`.
- `return_scale`: defaults to `100.0` in curriculum training and must be positive.
- `normalize_advantages`: defaults to `True`.

Both methods divide collected rewards by the same fixed `return_scale`. They do not standardize returns within an individual fault. This preserves reward signs and the value relationship between successful and failed faults while keeping numerical magnitudes manageable.

Implementation compatibility note: these defaults apply to the curriculum entry point. The low-level agent retains its legacy MC/return-normalization defaults for the existing paper-reward V3 entry point; curriculum training explicitly supplies the new settings. The existing curriculum already disabled return normalization and used a scale of 100 with gamma 1. The new CLI exposes gamma, defaulting to 0.99 as discussed, and allows the previous gamma 1 setting.

## MC Path

The MC path computes the complete discounted return for each step:

```text
G_t = r_t + gamma * G_(t+1)
A_raw_t = G_t - V_old(s_t)
value_target_t = G_t
```

The terminal continuation value is zero. This path provides a direct baseline for comparison with GAE under the same reward scaling and Actor Advantage normalization.

## GAE Path

For a complete terminal fault trajectory, the agent computes:

```text
delta_t = r_t + gamma * (1 - done_t) * V_old(s_(t+1)) - V_old(s_t)
A_raw_t = delta_t + gamma * lambda * (1 - done_t) * A_raw_(t+1)
value_target_t = A_raw_t + V_old(s_t)
```

The final transition has `done=True`, so its next-state value contribution is zero. Intermediate steps use the next stored old-policy value. No new state observation or mid-fault bootstrap callback is required.

With `gae_lambda=1`, terminal full-fault GAE should numerically match the MC Advantage, subject only to floating-point tolerance.

## Actor And Critic Inputs

The Actor uses a normalized copy of the raw Advantage:

```text
A_actor = (A_raw - mean(A_raw)) / (population_std(A_raw) + epsilon)
```

For a one-step rollout, normalization is skipped. The Critic never receives the normalized Actor Advantage; it minimizes MSE against the unnormalized `value_target`. This keeps the Actor gradient scale stable without changing the Critic's cross-fault value scale.

PPO clipping, entropy regularization, optimizer parameter groups, RND training, gradient clipping, and `policy_old` synchronization remain unchanged.

## Configuration And Checkpoints

The curriculum CLI exposes the method, gamma, lambda, reward scale, and Advantage-normalization switch. These values are recorded in the training configuration and agent checkpoint hyperparameters. Full training-state resume requires matching optimization semantics and fails clearly when loading a legacy return-normalized state into the new unnormalized mode. Old paper entry points may still resume their own legacy semantics. Actor-only V2 checkpoints remain loadable for inference or an explicit weights-only warm start.

Exported inference actors are unaffected because GAE changes training only.

## Metrics

Update metrics record the selected method and summary statistics for scaled rewards, raw Advantages, normalized Actor Advantages, and Critic targets. Curriculum run metrics continue to report `steps`, where `steps` is the number of Actor decisions collected across faults.

To make rollout length visible, curriculum summaries also report per-fault Actor-decision length statistics: minimum, mean, median, p90, maximum, and the number of zero-decision faults. These statistics are observational and do not alter update boundaries.

## Validation

Tests cover:

- Hand-computed MC returns on a terminal trajectory.
- Hand-computed GAE Advantages and value targets.
- Equality of full-fault `lambda=1` GAE and MC.
- Correct zero continuation value at the terminal step.
- Fixed reward scaling without per-fault return standardization.
- Actor-only Advantage normalization, including one-step rollouts.
- A post-BC Critic that starts at value 1 and can subsequently learn hidden weights.
- Checkpoint round trips, incompatible legacy training-state rejection, and actor-only checkpoint loading.
- Per-fault rollout-length summary statistics.

An end-to-end smoke run must complete behavior cloning, one curriculum PPO update in each method, checkpoint save/load, and deterministic evaluation without changing the C++ inference interface.

## Non-Goals

This change does not add fixed-length rollouts, mid-fault updates, parallel fault environments, recurrent state, or a richer Critic observation. Those can be evaluated separately after the full-fault MC/GAE comparison.
