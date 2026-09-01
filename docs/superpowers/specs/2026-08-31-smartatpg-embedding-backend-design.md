# Switchable SmartATPG and DeepGate Encoding

## Scope and Approval

The user approved replacing the encoding module with a trainable SmartATPG-style
graph encoder while retaining the fixed DeepGate backend behind a switch.
This document specifies that implementation, not a completed feature.

The reference is the workspace `smartATPG.pdf`, Sections 3.2.1 and 3.2.2.
The paper describes gate type, level, fanout, SCOAP features, GraphSAGE-style
aggregation, graph mean pooling, and a current-line state with an action mask.
Its layer widths, normalization details, and some implementation choices are not
fully specified. The explicit choices below are project defaults, not claims of
an exact reproduction of the authors' model or results.

Keep BC followed by PPO, the `mc/gae` switch, full-fault rollout boundaries,
incremental potential rewards, reward scaling, RND schedule, fault maps,
train/validation fault identities, and heuristic propagation unchanged.
Do not modify the existing uncommitted GAE work except for necessary integration.
Do not launch a full training campaign as part of implementation verification.

## Backend Contract

- `smartatpg`: train the graph encoder jointly with the existing BC/PPO policy.
  No DeepGate weights, embedding files, extractor imports, or PyG are required.
- `deepgate`: preserve the fixed precomputed embedding path and existing V1/V2
  inference behavior. Do not retrain or reinterpret historical embeddings.
- Expose `--embedding-backend smartatpg|deepgate` in the current curriculum
  preparation, training, and embedding export entry points. Native inference
  accepts the matching `-rl-embedding-backend` option.
- New preparation defaults to `smartatpg`. Training/export with no explicit
  flag use the backend recorded in their input manifest/checkpoint. Legacy
  artifacts without backend metadata mean `deepgate`, never `smartatpg`.
- An explicit flag conflicting with artifact metadata fails before a run.
  Switching backend starts a new model; it is not checkpoint conversion.

Older standalone training scripts may remain DeepGate-only and must say so.
The supported switchable training workflow is `train_curriculum.py`.

## Static Circuit Features

Add an isolated `rl_podem.smartatpg_features` module. Read the same normalized
binary BENCH consumed by C++ PODEM, with one graph node per named wire: primary
inputs and gate outputs, including synthetic decomposition wires. An OUTPUT
declaration marks its existing driver wire; do not create a duplicate PO node.
Node names are lookup keys, not learned categorical features.

Canonical gate type order is `PI, AND, NAND, OR, NOR, NOT, BUF, XOR, XNOR`.
Recognize `BUFF` as `BUF` and `EQV` as `XNOR` when reading supported netlists.
Feature schema `SMARTATPG_FEATURES_V1` has 14 columns in this order:

| Field | Columns | Definition |
| --- | --- | --- |
| Gate type | 9 | Canonical one-hot gate type |
| Level | 1 | PI level 0; otherwise one plus maximum fanin level |
| Fanout | 1 | Number of driven input-pin occurrences, excluding PO declarations |
| CC0, CC1, CO | 3 | Structural SCOAP costs, not signal probabilities |

Normalize level by `max(1, maximum_level)`. For fanout and each SCOAP column,
use `log1p(value) / max(1, log1p(maximum_finite_value))` within that circuit.
Clamp structural cost arithmetic to `1e9` to bound pathological growth. Keep
unobservable CO separate as infinity during calculation, exclude it from the
finite maximum, and encode it as 1. A zero column remains zero. All output
features must be finite float32. An empty finite set has maximum 0; a graph
without any declared output is invalid. These are deterministic input transforms,
unrelated to return normalization and requiring no fault outcomes or test labels.

Build topology using deterministic topological ordering and preserve input-pin
occurrences. Reject cycles, undefined drivers, conflicting declarations,
unsupported types, and non-binary logic gates with actionable diagnostics.
Preserve the C++ candidate order for action labels; graph indices must not
reorder actions. Check all callback/teacher wire names against the graph table.

### SCOAP Definition

Compute CC in forward topological order and CO in reverse order:

- PI: `CC0 = CC1 = 1`; each declared PO has `CO = 0`.
- AND: `CC0 = 1 + min(input CC0)`; `CC1 = 1 + sum(input CC1)`.
- OR: `CC0 = 1 + sum(input CC0)`; `CC1 = 1 + min(input CC1)`.
- NAND/NOR: swap the corresponding AND/OR output CC0 and CC1.
- NOT: swap input CC0/CC1 and add 1. BUF: preserve them and add 1.
- Binary XOR: `CC0 = 1 + min(a0+b0, a1+b1)` and
  `CC1 = 1 + min(a0+b1, a1+b0)`. XNOR swaps these two results.
- A branch CO is output CO plus 1 and the other inputs' sensitization costs:
  CC1 for AND/NAND, CC0 for OR/NOR, and `min(CC0, CC1)` for XOR/XNOR.
  Unary gates have no other-input term. A fanout stem takes the minimum over
  every branch and any directly observed PO connection.

These are structural SCOAP estimates, not exact reconvergence-aware costs.
Do not silently reuse `ATPG::calculate_scoap()` as it currently omits XOR/EQV
cases. The new extractor is independent of fault reordering and must not change
the legacy solver's SCOAP behavior.

## Graph Encoder and Policy State

Implement a small GraphSAGE mean encoder using ordinary PyTorch operations,
avoiding new compiled graph dependencies in the existing torch environment.
Initial defaults are two layers of width 64, ReLU, and full incoming-neighbor
aggregation on directed driver-to-consumer edges. Empty neighborhoods aggregate
to zero. Self information has its own concatenated block, not an added self edge.
Use no dropout, neighbor sampling, or batch-normalization running statistics.

```text
h_v^0 = x_v                                  # 14 node features
m_v^k = mean(h_u^(k-1) for u in fanins(v))
h_v^k = ReLU(W_k [h_v^(k-1) || m_v^k] + b_k)
g = mean_v(h_v^2)                            # 64 circuit-context features
z_v = [g || x_v || action_mask]              # 80 policy-input features
s_v = Tanh(Linear(80, 32)(z_v)) + Embedding(2, 32)(objective_value)
actor(s_v) -> 2 logits; critic(s_v) -> scalar
```

Follow the paper's explicit graph mean pooling plus current-line state.
Do not substitute a bare 14-feature MLP for the graph encoder, and do not add a
second concatenated node embedding in this version. The current objective value
remains an intentional project extension to the paper's listed state fields.
The actor/critic heads keep the existing 32-unit hidden layer and Tanh layout.
Do not add fault identity, D-frontier, candidate embeddings, or history features.

PODEM still enforces legal actions. In the existing binary integration, the
neural policy is called only when both inputs are available, so the supplied
mask is `[1, 1]`. Forced actions and dead ends remain solver-controlled and are
not newly added rollout steps. The mask is included for the state contract but
must not be advertised as newly informative dynamic state under this dispatch.

## BC and PPO Training

Keep graph data in an immutable circuit registry keyed by circuit hash. Separate
static inputs from trainable encoding: do not store detached learned embeddings
as the only state available to SmartATPG training.

BC samples store circuit and gate references, objective values, masks, and the
existing teacher targets/weights. Each optimizer step encodes each circuit
present in the minibatch once with gradients, gathers its states, and computes
the same weighted BC loss. The encoder is included in the BC optimizer; the
critic is not. Best-BC snapshots and resume state include the complete encoder.

During PPO rollout, freeze the full `policy_old`, including its graph encoder.
Compute its circuit context under `no_grad`, cache it by circuit hash and policy
revision, and record detached old log probabilities and values. Each step also
retains immutable circuit/gate references and its actual state mask.

Compute MC/GAE targets once from the stored rewards and old values. During each
PPO epoch, recompute graph context using the current trainable policy, once per
participating circuit, and use it for every step from that circuit. Aggregate
the loss, backpropagate once, then step the optimizer. Never reuse a detached
rollout encoding or an autograd graph from before an optimizer step. Actor and
critic losses both reach the shared encoder, whose learning rate is the actor
learning rate. Preserve the critic's separate head learning rate.

After updating, copy the entire policy to `policy_old` and invalidate all
policy-dependent caches. BC restoration, checkpoint loading, and validation
snapshot changes also invalidate them. Static graph features can stay cached.
No process-global cache may confuse different agents or revisions.

For SmartATPG RND, use the stable 14 current-gate input features plus the
two-value objective one-hot. Do not use a continuously changing trained graph
embedding as its observation, and do not backpropagate RND into the policy.
Keep the existing RND algorithm, coefficients, clipping and schedule. Record
this backend-specific observation schema in checkpoint metadata. DeepGate RND
retains its existing fixed-embedding observation unchanged.

## Data and Checkpoints

Use a new curriculum manifest version with backend-specific artifact validation.
SmartATPG preparation does not require an `.emb` file. Record circuit/fault-map
hashes, feature schema, graph configuration, and existing profile/teacher hashes.
Provide a source-manifest option to carry over existing fault splits and teacher
data into a new output directory without resampling, rerunning profiling, or
requiring the source DeepGate embeddings. Validate the reused non-embedding
artifacts. Never rewrite the original manifest or training artifacts.

New training checkpoints record backend, encoder configuration, feature and RND
schemas, all policy and optimizer states, RNG state, and existing training
configuration. Reject incompatible backend, graph schema, manifest or objective
configuration on resume, even when tensor dimensions happen to match.
Read existing DeepGate formats with their current defaults and validation.
Do not silently load a legacy actor into SmartATPG. Switching back to DeepGate
means selecting the corresponding manifest and checkpoint.

## Export and Native Inference

For SmartATPG, export circuit-specific static descriptors from a selected frozen
checkpoint: `[g || x_v || 1 || 1]` for every wire, 80 values per row. This is the
policy-input descriptor, not a claim that the GNN itself emits 80 dimensions.
Export the matching actor projection, objective embedding, and actor head.
The graph encoder remains in the Python checkpoint and is not run by C++.

Use `SMARTATPG_EMBEDDINGS_V2` and `SMARTATPG_ACTOR_V3` headers carrying explicit
backend/schema metadata and a common inference-snapshot identifier derived
from encoder and actor parameters/configuration. Store the circuit hash in
the descriptor artifact. Extend the native readers while preserving the legacy
embedding V1 and actor V1/V2 paths. Reject mismatched backend, schema, snapshot,
dimension, circuit, missing names, duplicate names and non-finite data.

Keep `-rl-emb` and `-rl-actor` as the native file arguments. The new optional
backend flag validates their metadata; omitting it resolves the artifact type.
Reuse the current actor arithmetic and lazy gate/objective-value logit cache.
The canonical mask in static descriptors is valid only because native policy
dispatch requires two available inputs. Keep and test that invariant; a future
change to dispatch must not reuse this cache for arbitrary masks.

Export best and latest snapshots with their own encoders. In particular, a BC
fallback actor must never be paired with final-PPO graph context. Write each
export as a complete snapshot and publish its manifest only after all files are
ready. Partial or mixed exports fail validation. Exporting a previously unseen
circuit runs the trained encoder without gradient updates or fault labels.

## Verification and Acceptance

- Hand-check gate features/SCOAP on PI, unary, AND/OR, inverted gates, XOR/XNOR,
  fanout, directly observed wires, dangling cones, repeated input pins and
  reconvergence. Test normalization, bounded costs and invalid netlists.
- Verify directed aggregation, empty fanins, pooling and permutation equivariance
  under node renaming/reindexing while preserving action-position semantics.
- Show finite, nonzero encoder gradients and changed weights after both BC and
  PPO. Show frozen rollout weights/log probabilities, correct target reuse,
  and refreshed current-policy encodings across PPO epochs and faults.
- Test cache invalidation for updates, resume, BC-best restoration and switching
  validation snapshots. No cross-agent cached context is allowed.
- Run the existing GAE, curriculum and native-policy regression tests with the
  DeepGate backend. Preserve historical actor logits and artifact readability.
- Compare Python live-graph logits, Python exported-descriptor logits and native
  logits for both objective values on multiple gates and circuits. Require
  `rtol=1e-4, atol=1e-5` numerical agreement and matching deterministic decisions
  on the parity fixtures, with documented first-index tie breaking. Include
  exact-tie fixtures, and report near-tie discrepancies rather than hiding them
  behind the numerical tolerance.
- Run short BC, MC and GAE smoke tests on at least two circuits, exercise resume
  and negative metadata cases, and verify operation without DeepGate imports or
  embedding files. Verify forced-action dispatch and native fault outcomes.
- Document both backend commands, artifact pairing, fresh-training requirements
  and separate graph-encoding/export versus ATPG timing. Smoke tests establish
  correctness, not improved coverage, speed or unseen-circuit generalization.

## Review Status

The user approved both the high-level approach and this written specification.
Implementation is covered by the adjacent implementation plan and the checks in
`PODEM/artifacts/smartatpg_backend_smoke_20260831/VERIFICATION.md` at repository
root. Correctness checks do not imply improved training or inference performance.
