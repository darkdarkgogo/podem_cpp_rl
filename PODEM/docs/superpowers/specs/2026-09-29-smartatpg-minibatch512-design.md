# SmartATPG PPO Minibatch 512 Protocol Design

## Goal

Change the fixed SmartATPG PPO transition minibatch size from 128 to 512 without changing rollout boundaries, GAE semantics, PPO epoch count, rewards, or learning rates.

## Training protocol

Both `level_gat_gru` and `fanin_mean` use:

```text
episode = one complete fault
rollout = four complete faults
minibatch_size = 512 transitions
k_epochs = 4
gamma = 0.99
gae_lambda = 0.97
normalize_returns = False
normalize_advantages = True
return_scale = 100.0
drop_last = False
```

GAE and rollout-wide Advantage normalization continue to run exactly once before transition shuffling. Each PPO epoch generates a fresh permutation and divides it into slices of at most 512 transitions. The final short minibatch is retained. For example, `N=1200` produces minibatches of 512, 512, and 176 in each epoch.

Metrics remain transition-weighted across all minibatches and all four epochs. The formulas become:

```text
minibatches_per_epoch = ceil(N / 512)
optimizer_steps = 4 * minibatches_per_epoch
```

The RND predictor remains one full-rollout update per PPO update.

## Compatibility

`minibatch_size=512` remains a required field in manifests, training configuration, checkpoint hyperparameters, exported model metadata, portable parsing, and native parsing.

Both training checkpoint format identifiers advance from V7 to V8. Mean and GAT exported model headers advance to new versions that explicitly contain `MINIBATCH512`. Current readers reject the previous `MINIBATCH128` formats and any metadata whose minibatch size is not 512. Actor-only warm-start rules remain unchanged.

## Components

- `python/rl_podem/data_split.py`: set both encoder protocols to minibatch 512.
- `python/rl_podem/ppo.py`: change the generic PPO default to 512 while retaining positive-size validation and short-batch behavior.
- `python/rl_podem/smartatpg.py`: set the SmartATPG default to 512 and advance its training checkpoint format.
- `python/rl_podem/gat_gru.py`: advance the GAT training checkpoint format.
- `python/rl_podem/cpp_bridge.py`: export only the new Mean/GAT model formats and validate 512 metadata.
- `python/rl_podem/smartatpg_portable.py`: parse the new formats and require minibatch 512.
- `src/rl_policy.cpp`: accept the new formats and require `minibatch_size 512`.
- `tests/`: update fixed-protocol, batching, compatibility, portable, and native assertions.
- `RL_GUIDE.md` and `docs/SMARTATPG_11D_使用说明.md`: document the new fixed minibatch size.

## Acceptance criteria

- GAT and Mean training configurations both report `minibatch_size=512`.
- A 1200-transition rollout produces 512, 512, and 176 transition minibatches in every epoch.
- Every transition still appears exactly once per epoch.
- GAE and Advantage normalization are still calculated once before shuffling.
- The final minibatch is retained when it has fewer than 512 transitions.
- All-epoch and final-epoch metrics remain weighted by transition count.
- Four-fault rollout boundaries, K=4, rewards, learning rates, and RND update frequency do not change.
- New Python, portable, and native readers accept the MINIBATCH512 formats.
- Previous MINIBATCH128 checkpoints and exported models are rejected.
- Focused SmartATPG and training protocol tests pass.
