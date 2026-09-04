# Static CO and 12D Graph Embeddings

The user approved adding static combinational SCOAP observability to the
graph input, without implementing propagation-gate decisions in this change.

## Feature Contract

Keep the first eleven columns unchanged. Append CO after CC1, producing
seven gate-type indicators and level, fanout, CC0, CC1, CO. Compute CC0/CC1
in forward topological order, then CO in reverse topological order with
PO boundary zero. Each AND/NAND side pin contributes CC1, each OR/NOR side
pin contributes CC0, and BUF/NOT adds one. Take the minimum across fanouts.
Enumerate side pins by position, including repeated connections. This is
the SCOAP independence approximation, not a proof of testability.

Preserve infinite raw CO for wires with no output path. Cap finite costs
at 10**9; normalize each cost column with the existing per-circuit log1p
rule using finite maxima, and map non-finite values to 1.

## Network and Artifact Contract

Both graph encoders output 12D. Fanin mean uses Linear(24,12); GAT-GRU uses
12x12 projections, 24 attention coefficients, and GRUCell(12,12), with one
forward and one reverse level sweep. SmartATPG Actor/Critic input is 12D;
agentATPG appends scalar object_val for 13D. Internal hidden width remains
32. Two mask bits are applied only after logits (state widths 14 and 15).

Use feature schema SMARTATPG_FEATURES_V3_12D_CO, model V8, embeddings V6,
and benchmark bundle V5. Preserve read-only V5/V6/V7 model inference with
11D graph features, but reject old checkpoints for new-architecture training.
Update graph configurations, RND schema, export validation, and C++ cache
input assembly together. Do not alter hard-detected selection, backtrack
limits, PPO failure updates, or the existing C++ heuristic SCOAP routine.

## Verification Plan

Check hand-calculated CO for every supported gate, multi-fanout minima,
output boundaries, disconnected logic, repeated pins and cost saturation.
Compare all feature columns between Torch and portable construction; verify
Actor inputs 12/13, graph/CO gradients, PPO updates, checkpoint round trips,
native scoring parity for both objective values, and legacy inference.
Compile the extension and run the test suite; full Linux training is deferred.
