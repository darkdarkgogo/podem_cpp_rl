# Level-Wise GAT-GRU Comparison Design

## Objective

Add `SmartATPG-GAT-GRU-64` as a second trainable graph-policy variant and
compare it comprehensively with the existing 11-dimensional SmartATPG model.
The existing model remains the baseline and is not replaced.

The comparison must cover the same training circuits, fault selections,
curriculum order, reward function, PPO/RND settings, random seeds, validation
rules, and final 16-circuit native benchmark protocol. The encoder architecture
and the dimensions implied by that architecture are the intended differences.

## Model Architecture

Each gate keeps the existing 11-value input feature vector. A learned input
projection produces the initial hidden state:

```text
h0 = ReLU(Linear(11, 64)(x))
```

The graph encoder then performs exactly two complete level-wise sweeps:

1. Forward sweep from level 1 through the maximum level. For every gate at the
   current level, a single-head GAT aggregates its fanin hidden states. A
   forward GRU cell updates the gate from the aggregated message and its
   current hidden state.
2. Reverse sweep from the maximum level back to level 0. For every gate at the
   current level, a separate single-head GAT aggregates its fanout hidden
   states. A separate reverse GRU cell updates the forward result.

Nodes without neighbors in the active direction retain their current hidden
state. Updates within one level are simultaneous; the next level observes the
completed updates from the previous level. This makes one sweep propagate
information across the complete circuit depth rather than applying one global
edge update.

Forward and reverse parameters are independent and shared across levels. For
each direction:

- GAT projection weight: mathematical shape `64 x 64`; PyTorch linear weight
  shape `[64, 64]`.
- Attention vector: shape `[128]`, applied to the concatenation of transformed
  target and source states.
- GRU input size: 64.
- GRU hidden size: 64.

The final gate embedding is exactly 64-dimensional. The two action-mask values
are not part of the embedding. They are appended only when constructing a
decision descriptor, producing a 66-dimensional policy state for the actor.

## Attention And Update Semantics

For a directed neighbor edge `j -> i`, one attention head computes:

```text
z_i = W h_i
z_j = W h_j
e_ij = LeakyReLU(a^T [z_i || z_j])
alpha_ij = softmax_j(e_ij)
m_i = sum_j(alpha_ij * z_j)
h_i' = GRUCell(m_i, h_i)
```

The softmax is normalized independently over the active neighbors of each
target gate. The forward sweep uses original fanin edges. The reverse sweep
uses the same edges reversed so that fanout information moves toward primary
inputs. Self-loops are not added because the GRU hidden argument already
carries the gate's previous state.

## Software Boundaries

The new implementation is isolated from the baseline encoder:

- `smartatpg_features.py` continues to own deterministic circuit parsing,
  features, topology, and levels. It will expose the reverse adjacency and
  level groups needed by both Torch and portable inference.
- A dedicated GAT-GRU module owns the 64-dimensional encoder, policy, and PPO
  agent variant.
- The training command selects the encoder variant explicitly and writes to a
  separate output directory. Baseline checkpoint compatibility is preserved.
- Snapshot export records the encoder variant, graph configuration, embedding
  dimension, policy-state dimension, and all encoder tensors.
- Portable Python inference implements the same level-wise equations without a
  Torch runtime and exports fixed per-gate embeddings for native benchmarking.
- Native C++ actor loading accepts the new 64-dimensional embedding and
  66-dimensional policy state while preserving the existing 11/13 baseline.

The new variant remains in the SmartATPG family. It does not add another
external backend or vendor dependency.

## Artifact Compatibility

Existing baseline training checkpoints, V5 models, and V3 embedding tables
remain readable. The new encoder uses a new versioned model format and graph
configuration identifier so a baseline model cannot be paired accidentally
with 64-dimensional embeddings.

The model and embedding artifacts must validate:

- encoder variant and graph configuration;
- gate embedding dimension 64 and policy state dimension 66;
- circuit hash, tensor names, tensor shapes, finite values, and snapshot ID;
- exact pairing between actor parameters and exported embeddings.

Unsupported or mixed artifacts fail before ATPG starts. Writes remain atomic
through temporary files followed by replacement.

## Training And Comparison Protocol

Both variants train independently from random initialization. The comparison
uses identical prepared manifests and all shared hyperparameters. Each round
uses the same deterministic episode order for both variants. Validation and
best-checkpoint selection use the current score ordering unchanged.

The final report compares three modes:

- heuristic PODEM;
- baseline SmartATPG 11D fanin-mean GraphSAGE;
- SmartATPG-GAT-GRU-64.

For every circuit and aggregate totals, report detected, aborted and redundant
faults, backtracks, backtrace steps, generated vectors, and C++ ATPG interval
time. Also report model parameter count, training wall time, peak training
memory when available, graph preprocessing time, and embedding export time.
Embedding generation remains outside the C++ ATPG timing interval, matching
the existing benchmark definition.

## Error Handling

The graph loader rejects cycles, missing drivers, unsupported gate types, and
invalid level schedules. The GAT implementation rejects empty attention groups
where an update was expected, non-finite scores or hidden states, malformed
dimensions, and invalid action masks. Portable and Torch implementations must
fail on metadata or tensor-shape disagreement instead of guessing defaults.

## Tests And Acceptance Criteria

Unit tests must verify input projection dimensions, direction-specific
parameters, attention normalization, simultaneous same-level updates, forward
and reverse information flow, 64-dimensional embeddings, and 66-dimensional
decision descriptors. Gradient tests must show updates reaching the input
projection, both GAT directions, both GRU cells, and the actor.

Parity tests compare Torch and portable embeddings and actor logits within
documented floating-point tolerances. Native artifact tests verify both the
baseline and new formats, including rejection of cross-paired artifacts.

End-to-end acceptance requires both variants to train and export from the same
manifest, the three-way benchmark to complete on the configured circuit set,
all existing baseline tests to remain green, and no reintroduction of removed
external graph-encoder code or dependencies.
