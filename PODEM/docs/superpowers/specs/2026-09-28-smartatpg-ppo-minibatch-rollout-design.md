# SmartATPG PPO Four-Fault Rollout and Transition Minibatch Design

## Goal

Replace the current eight-fault full-batch PPO update with a four-complete-fault rollout followed by transition-level minibatch optimization. Both SmartATPG encoders use four PPO epochs, while the approved GAE and reward protocols remain unchanged.

## Fixed training protocol

Both `level_gat_gru` and `fanin_mean` use:

```text
episode = one complete fault
rollout = four complete faults
minibatch_size = 128 transitions
k_epochs = 4
gamma = 0.99
gae_lambda = 0.97
normalize_returns = False
normalize_advantages = True
return_scale = 100.0
drop_last = False
```

The round-level global shuffle of `(circuit, fault)` episodes remains unchanged. A PPO update occurs after each group of four completed faults. The final partial group at the end of a round is updated rather than discarded. A rollout must never stop in the middle of a fault.

The GAT reward remains `cubic_backtrack_depthnorm_v2`, including RND, cubic backtrack penalties, terminal rewards, and the already approved depth-normalized step reward `-10 / H_c`. The Mean reward remains `legacy_pi_exponential`, including its fixed `-0.1` backtrace-step reward. Reward ablation and learning-rate changes are outside this change.

## Target construction

GAE is calculated exactly once for the complete rollout and before any transition shuffle. It uses the `policy_old` values saved during collection and resets both bootstrapping and the backward recurrence at every fault-terminal transition. This produces one immutable Actor Advantage tensor and one immutable critic value-target tensor for the update.

Advantage normalization is performed over the complete rollout once. Advantages and value targets are not recomputed or renormalized per epoch or minibatch.

## PPO minibatch update

For a rollout containing `N` transitions, each PPO epoch generates a fresh `torch.randperm(N)`. Consecutive slices of at most 128 indices form the minibatches. The last short minibatch is retained, so every transition is used exactly once per epoch.

Each minibatch performs its own forward pass, loss calculation, gradient reset, backward pass, optional gradient clipping, and optimizer step. Therefore:

```text
minibatches_per_epoch = ceil(N / 128)
optimizer_steps = 4 * minibatches_per_epoch
```

Rollout evaluation accepts transition indices. It evaluates only the selected transitions and computes graph embeddings only for circuits represented in that minibatch. Embeddings with autograd history must not be cached across optimizer steps because policy parameters change after every step.

The RND predictor keeps its existing behavior: one full-rollout predictor update after PPO optimization. PPO minibatching does not multiply RND optimizer steps.

After all four PPO epochs finish, `policy_old` is synchronized from `policy`, the rollout buffer is cleared, and the update counter advances once.

## Metrics

For each minibatch metric, accumulate `metric * minibatch_transition_count`. The update-level mean divides the accumulated total by `k_epochs * N`, so differently sized minibatches contribute in proportion to their transition counts and all four epochs are represented.

The following metrics use that all-epoch transition-weighted aggregation:

- total loss
- policy loss
- value loss
- entropy
- ratio mean
- approximate KL divergence
- clip fraction

The same metrics are also recorded with a `last_epoch` suffix using only the fourth epoch, weighted by that epoch's `N` transitions. Existing rollout statistics derived from rewards, Advantages, and targets remain single-rollout statistics and are not multiplied by the epoch count.

Each update additionally reports:

```text
rollout_faults
rollout_transitions
minibatches_per_epoch
optimizer_steps
epochs
```

Training logs and TensorBoard expose the new update metrics and print the actual fault, transition, minibatch, epoch, and optimizer-step counts.

## Configuration and compatibility

`FAULTS_PER_UPDATE` changes from 8 to 4. `minibatch_size=128` becomes part of encoder training hyperparameters, runtime configuration, checkpoint hyperparameters, exported model metadata, portable parsing, and native parsing. Mean `k_epochs` changes from 1 to 4; GAT remains at 4.

Both GAT and Mean training/model protocol versions are advanced. Current readers accept only the new four-fault, minibatch-128, epoch-4 formats for current data-split training artifacts. Old eight-fault, Mean-K1, Monte Carlo, or otherwise incompatible training checkpoints cannot be resumed. Existing actor-only warm-start rules remain unchanged.

The training checkpoint remains valid only at an update boundary. With the new protocol, an intermediate checkpoint position must be divisible by four unless it is the end of the round.

## Components

- `python/rl_podem/data_split.py`: define four faults per update, minibatch size 128, and Mean/GAT K=4 metadata.
- `python/rl_podem/training.py`: pass and record minibatch configuration; log expanded update metrics.
- `python/rl_podem/ppo.py`: implement per-epoch shuffle, indexed minibatches, optimizer steps, and weighted metrics.
- `python/rl_podem/smartatpg.py`: evaluate only requested transition indices while preserving graph-encoder gradients.
- `python/rl_podem/gat_gru.py`: inherit the indexed evaluation path and preserve GAT K=4 defaults.
- `python/rl_podem/advantages.py`: retain the existing GAE implementation and add regression coverage for multiple terminals.
- `python/rl_podem/cpp_bridge.py`, `python/rl_podem/smartatpg_portable.py`, and `src/rl_policy.cpp`: synchronize exported and consumed protocol metadata and formats.
- `tests/test_smartatpg.py` and `tests/test_smartatpg_training.py`: cover PPO batching, target reuse, metrics, boundaries, and compatibility.

## Error handling

The agent rejects non-positive minibatch sizes. An update with an empty rollout remains a no-op. Indexed rollout evaluation rejects out-of-range or malformed indices rather than silently evaluating different transitions. Existing non-finite target and loss protections remain active.

Protocol readers reject missing or mismatched `faults_per_update`, `minibatch_size`, or `k_epochs` fields. Resume validation rejects checkpoints created at positions that are not valid four-fault boundaries.

## Acceptance criteria

- Updates occur after faults 4, 8, and so on; the fifth fault starts a new rollout.
- A round ending with fewer than four remaining faults still performs one update.
- GAE and Advantage normalization are called once per rollout before shuffling.
- Every PPO epoch calls a fresh `torch.randperm(N)`.
- `N=350` produces minibatches of 128, 128, and 94 transitions in each epoch.
- Each transition appears exactly once per epoch and four times per update.
- GAT and Mean both execute four PPO epochs.
- Multiple fault terminals reset GAE without cross-fault leakage.
- Indexed Mean and GAT evaluation updates their graph encoders and never reuses an autograd graph across optimizer steps.
- All-epoch metrics are transition-weighted across `4*N` sample appearances; `last_epoch` metrics use only the final `N`.
- `minibatches_per_epoch=ceil(N/128)` and `optimizer_steps=4*ceil(N/128)` are reported.
- Mean and GAT reward values remain unchanged from the currently approved protocols.
- Old eight-fault or Mean-K1 checkpoints and exported models are explicitly rejected.
- The Python test suite and native protocol tests pass.
