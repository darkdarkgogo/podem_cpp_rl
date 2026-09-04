# Paper-Style PPO and RND Training Design

## Objective

Train a new PPO actor from random initialization using fixed gate
embeddings, Random Network Distillation (RND), and 100 hard stuck-at faults
from each of `c6288.bench` and `s38417_scan.bench`. RND is training-only;
native C++ deployment continues to load only circuit embeddings and exported
actor tensors.

## Hard-Fault Selection

The C++ engine profiles faults with its original non-RL PODEM heuristic. Fault
dropping is disabled during profiling so every collapsed fault is attempted
independently. Each record contains the stable fault identifier, outcome, and
backtrack count.

Faults proven redundant are excluded. Remaining faults are sorted by
backtracks descending and then by fault identifier for deterministic ties.
The first 100 faults from each training circuit are saved in a JSON manifest.
Training runs only those identifiers and also disables cross-fault dropping so
one selected fault always corresponds to one PPO episode.

## RND

Each decision receives a stable RND observation built from fixed gate
features rather than the changing PPO encoder:

- backtrace: objective embedding, mean candidate embedding, mode `[1, 0]`;
- propagation: zero objective embedding, mean candidate embedding, mode
  `[0, 1]`.

A fixed random target MLP and trainable predictor MLP map this observation to a
32-dimensional feature. Their mean squared prediction error is normalized by
running statistics and contributes `rnd_beta * normalized_error` to the step
reward. The initial `rnd_beta` is `0.05`. The predictor is optimized only from
collected rollout observations.

## PPO and Checkpoints

The existing external rewards, PPO clipping, actor/critic architecture, and
sticky C++ path choices remain unchanged. A full training checkpoint stores
the current and old policies, PPO optimizer, RND target, RND predictor, RND
optimizer, running normalization state, update count, seed, and progress.
Actor export continues to write only the tensors consumed by
`NativeActorPolicy`.

## Training and Validation

Embeddings are generated for the exact two training BENCH files. PPO and RND
are initialized from scratch with
a fixed seed. Training alternates the two circuits and shuffles their selected
fault order reproducibly.

Smoke validation must prove fault filtering, profiling, RND parameter updates,
checkpoint resume, and unchanged actor export. Formal artifacts and logs are
written under a new `artifacts/paper_rnd/` directory so earlier models remain
untouched.
