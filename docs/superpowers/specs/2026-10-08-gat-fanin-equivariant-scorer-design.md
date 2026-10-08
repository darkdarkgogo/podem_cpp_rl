# GAT Fanin-Conditioned Backtrace Scorer Design

## Goal

Make `level_gat_gru` backtrace decisions depend on the identity and embedding of each candidate fanin, so reordering a gate's fanins reorders candidate scores without changing which logical branch is preferred. Keep `fanin_mean` behavior unchanged.

## Policy Architecture

The GAT-GRU graph encoder continues to produce one 11-dimensional embedding per gate. For every candidate fanin, the actor receives the concatenation of:

```text
[objective_gate_embedding (11), candidate_fanin_embedding (11), objective_value (1)]
```

A single shared MLP scores each candidate independently, with the current hidden width retained (`23 -> 32 -> 1`). The resulting candidate scores are logits for the existing categorical policy. The existing action mask removes candidates whose wires are already assigned. The critic remains conditioned on the objective gate and objective value, preserving its existing value-state meaning.

Because the same scorer is applied to every candidate, permuting the candidate order permutes the logits in the same way. The selected candidate therefore remains the same logical fanin, subject to floating-point tolerance in graph aggregation.

## Training and Inference Data Flow

Training selection obtains the objective embedding and both fanin embeddings from the current graph encoding, scores each pair, and samples from the masked categorical distribution. Rollout records retain the objective identity, objective value, candidate identities, action mask, selected candidate index, old log probability, and critic value. PPO reevaluation reconstructs the candidate embeddings from the same graph and applies the shared scorer to the stored candidate order.

The native C++ policy uses the embedding table's gate-id-indexed embeddings for both the objective and each candidate. It applies the exported shared scorer to each candidate, then selects the highest-scoring unmasked candidate. Mean-model inference and its actor path remain unchanged.

## Artifact Compatibility

The GAT actor tensor shape and semantics change from a 12-input, two-output actor to a 23-input, one-output shared scorer. Bump the GAT training/artifact protocol identity and update export and native shape validation accordingly. Reject old GAT actors with a clear incompatible-format error; they require retraining. Preserve existing `fanin_mean` formats and loading behavior.

## Acceptance Criteria

- For fixed graph embeddings and objective value, swapping two candidate embeddings swaps their scores within floating-point tolerance.
- Training action selection, PPO reevaluation, portable inference, and native inference use the same candidate-conditioned scoring and masking semantics.
- Only the GAT actor architecture and its protocol metadata change; the critic state, graph encoder, rewards, masks, and `fanin_mean` behavior remain unchanged.
- Old GAT artifacts fail with an explicit format or tensor-shape incompatibility message, while newly exported scorer artifacts load in portable and native inference.
- Existing workspace test suites are not run or expanded as part of implementation unless separately requested.
