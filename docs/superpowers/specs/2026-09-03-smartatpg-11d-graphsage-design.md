# SmartATPG 11-Dimensional GraphSAGE State

## Scope

Revise only the SmartATPG backend. DeepGate and the planned DeepGate2 migration
are out of scope for this change.

The SmartATPG policy must use a per-gate GraphSAGE embedding. It must not use a
pooled whole-circuit embedding, and it must not run behavior cloning (BC).

## Gate Features

The supported gate types are, in canonical one-hot order:

`PI, AND, NAND, OR, NOR, NOT, BUF`

`XOR` and `XNOR` are unsupported. The SmartATPG BENCH parser must reject them
with a clear error. Existing binary conversion remains responsible for lowering
unsupported logic before SmartATPG preprocessing.

Each gate has an 11-dimensional initial feature vector:

| Field | Width | Definition |
| --- | ---: | --- |
| Gate type | 7 | Canonical one-hot gate type |
| Level | 1 | Topological level, normalized within the circuit |
| Fanout | 1 | Number of driven input-pin occurrences, log-normalized |
| CC0 | 1 | Structural SCOAP 0-controllability, log-normalized |
| CC1 | 1 | Structural SCOAP 1-controllability, log-normalized |

SCOAP CO is not calculated or included. The feature schema identifier changes
so that old 14/80-dimensional artifacts cannot be loaded as this model.

## GraphSAGE Encoder

The encoder performs three incoming-neighbor mean-aggregation rounds. Every
round preserves the 11-dimensional node width and has an independent trainable
weight matrix and bias:

```text
h_v^0 = x_v
m_v^k = mean(h_u^(k-1) for u in fanins(v))
h_v^k = ReLU(W_k [h_v^(k-1) || m_v^k] + b_k), k = 1, 2, 3
```

For every round, the concatenated input width is 22 and the output width is 11.
An empty fanin neighborhood contributes an all-zero 11-dimensional mean.

The final gate embedding is `h_v^3`, with width 11. There is no graph mean
pooling, graph context, 64-dimensional hidden representation, or context cache.

`W_1`, `W_2`, and `W_3` are policy parameters. They are initialized with the
new SmartATPG model and updated by PPO gradients together with the Actor and
Critic. They are included in checkpoints and snapshot identity.

## Policy State And Mask

At a backtrace decision for gate `v`, the policy state is:

```text
state_v = [h_v^3 || action_mask]
```

The gate embedding is 11-dimensional. The binary action mask is a separate
2-dimensional state field, making the Actor/Critic state width 13. The mask is
also applied to action logits so an invalid input cannot be selected.

The mask is not part of the gate embedding and must not be stored in the static
gate-embedding table. Python training supplies it per decision. Native C++
inference appends the current two mask values to the selected gate's static
11-dimensional embedding before evaluating the policy.

The objective value embedding and existing two-logit backtrace action head are
retained. Forced moves remain solver-controlled.

## Training

SmartATPG skips behavior cloning unconditionally. A SmartATPG run starts PPO
from newly initialized GraphSAGE, Actor, and Critic parameters. DeepGate behavior
is unchanged in this scoped change.

The existing PPO reward, GAE/MC option, RND behavior, curriculum fault ordering,
checkpoint cadence, best-model selection, and resume semantics remain unchanged.
Resume accepts only checkpoints with the new feature schema, three-layer graph
configuration, 11-dimensional gate embedding, and 13-dimensional policy state.

## Artifacts And Native Inference

SmartATPG exports one 11-dimensional `h_v^3` row per circuit gate. Actor metadata
records the backend, feature schema, graph configuration, gate-embedding width,
policy-state width, and snapshot identity. Actor and embedding artifacts must
come from the same snapshot.

The native reader must distinguish the 11-dimensional embedding width from the
13-dimensional policy-state width. Legacy SmartATPG 80-dimensional descriptors
and their checkpoints are incompatible and must fail with an actionable error,
not be silently reinterpreted.

## Verification

Tests must cover:

- exact 11-column feature order and normalization;
- rejection of XOR and XNOR input gates;
- absence of CO from graph data and exported artifacts;
- three independent `Linear(22, 11)` GraphSAGE rounds;
- no graph pooling or shared circuit context;
- per-gate embeddings that change when up-to-three-hop fanin features change;
- mask exclusion from embedding files and inclusion in 13-dimensional policy state;
- PPO gradients and parameter updates for all three GraphSAGE layers;
- SmartATPG training that performs zero BC epochs regardless of the general BC default;
- checkpoint/resume rejection for the old SmartATPG schema;
- Python/native logit and selected-action parity for matching snapshots.

No full training campaign is required for implementation verification. A small
deterministic circuit and short PPO smoke run are sufficient.
