# SmartATPG Depth-Normalized GAT Reward and GAE Design

## Goal

Remove the circuit-depth bias caused by the fixed GAT backtrace-step penalty and switch both SmartATPG encoders from Monte Carlo Advantage estimation to GAE without changing the Mean reward definition.

## Reward protocols

The GAT reward protocol becomes `cubic_backtrack_depthnorm_v2`. For a circuit graph, define:

```text
H_c = max(1, max(graph.levels))
r_step = -10 / H_c
```

Every attributed GAT `backtrace_step` receives that circuit-specific reward. GAT RND, the cubic backtrack penalty, and terminal rewards remain unchanged.

The Mean encoder keeps the `legacy_pi_exponential` protocol exactly as it is today: `backtrace_step=-0.1`, the existing PI exponential reward, and terminal rewards remain unchanged.

Training, Python validation, native validation, portable model validation, model export, and artifact metadata must agree on the protocol identity and reward value. Native validation receives the already-computed GAT step reward rather than independently deriving circuit depth.

## Advantage protocol

Both encoders use:

```text
advantage_method = "gae"
gamma = 0.99
gae_lambda = 0.97
normalize_returns = False
normalize_advantages = True
return_scale = 100.0
```

Rewards, including scaled RND reward, are divided by `return_scale` before target construction. GAE uses the values saved by `policy_old` during rollout, resets continuation at every terminal transition, and is calculated once before the PPO epoch loop. All PPO epochs reuse the same actor Advantages and critic value targets.

## Compatibility

The GAT reward scheme and exported GAT model format are versioned. A training checkpoint is resumable only when its reward scheme and complete PPO/RND hyperparameter dictionary match the current protocol. This rejects old GAT checkpoints because both reward and Advantage protocols changed, and old Mean checkpoints because the Advantage protocol changed.

Weights-only warm starts remain governed by the existing actor-only compatibility rules.

## Observability

Training metrics and configuration record the selected Advantage parameters. GAT episode metrics also record `circuit_depth` and `backtrace_step_reward` so the applied normalization is auditable.

## Acceptance criteria

- GAT step rewards are `-0.2`, `-0.1`, and `-0.05` for depths 50, 100, and 200.
- Depth is clamped to one for a zero-level circuit.
- Mean reward values remain bit-for-bit compatible with the current reward helper behavior.
- GAT cubic backtrack and terminal rewards do not change.
- Both encoder constructors receive the approved GAE parameters.
- GAE resets across multiple terminal transitions in a batched rollout and is not recomputed between PPO epochs.
- Python training, Python validation, and native validation produce matching GAT extrinsic returns.
- Portable/native model readers accept the new GAT protocol and reject stale protocol identities.
- Old training checkpoints are explicitly rejected by protocol or hyperparameter mismatch.

