# Binary Netlist and Object-Action Actor Design

## Goal

Replace per-candidate backtrace scoring with a paper-style backtrace actor that
produces both binary input actions in one forward pass. Normalize multi-input
benchmark gates into balanced two-input trees, preserve the original collapsed
stuck-at fault catalog, and make native inference a precomputed table lookup.

Propagation remains controlled by the original PODEM heuristic. Existing
circuits, V1 actor artifacts, and checkpoints remain available and are not
deleted or overwritten.

## Binary Netlist Normalization

The converter writes a new `<stem>_binary.bench` beside the original circuit.
The original `.bench` file remains unchanged.

Gates with more than two inputs are decomposed deterministically into balanced
trees. AND and OR use the same gate type at every level. NAND and NOR use AND
or OR for synthetic internal nodes and retain NAND or NOR at the final node, so
the original output inversion and output wire name are preserved. BUF and NOT
remain unchanged. The current simulator only defines XOR and EQV for exactly
two inputs, so the converter rejects wider XOR or EQV gates instead of silently
changing their semantics.

Synthetic names use a reserved prefix containing the original output name and
a deterministic sequence number. Pairing is stable and balanced, minimizing
the added logic depth and avoiding run-to-run action-order changes. The
converter verifies that every normalized logic gate has at most two inputs.

## Original Fault Preservation

Binary decomposition changes topology and would normally change fault
collapsing. Merely excluding synthetic gates is insufficient because an
original gate-input fault must be injected at the corresponding leaf branch of
the binary tree.

A versioned `<stem>_binary.faultmap` accompanies each normalized circuit. It
contains hashes of both source and normalized circuits and one record for each
collapsed fault generated from the original circuit:

- the original external fault ID;
- stuck-at type and original equivalent-fault count;
- the transformed target output wire for a GO fault;
- the transformed target node and input-wire name for a GI fault.

The original C++ fault generator is the source of truth for the catalog and
equivalence counts. A read-only export API exposes its descriptors to the
converter. When a valid fault map is supplied for a binary circuit, ATPG
bypasses normal fault-list generation and reconstructs exactly those mapped
faults. GI input indices are resolved from wire names after input rearrangement,
so level-based sorting cannot invalidate the map. Synthetic nodes participate
in simulation, implication, backtrace, and embedding extraction but do not add
new target faults.

Hash, target-name, duplicate-ID, and count mismatches fail before ATPG starts.
Without an explicit valid fault map, an ordinary circuit continues to use the
existing fault generator unchanged.

## Backtrace Actor V2

The V2 policy is backtrace-only. Its state consists of the current object-gate
embedding and the required objective value. With an embedding dimension `D`
and hidden dimension `H`, the network is:

```text
gate embedding [D]
  -> gate encoder [H]
  + objective-value embedding [H]
  -> backtrace hidden layer [H]
  -> two action logits [left, right]
```

The initial V2 configuration keeps the existing 64-dimensional DeepGate
embedding and uses `H=32`. Candidate embeddings and the propagation actor are
not part of V2. The critic consumes the same object/value state. RND remains a
training-only auxiliary task and observes the object embedding plus a two-value
one-hot objective representation.

The two output positions correspond to the deterministic input order after
`rearrange_gate_inputs()`. V2 checkpoints use a new format identifier and may
not resume V1 checkpoints. V2 native artifacts use
`SMARTATPG_ACTOR_V2`. The V1 loader and artifacts remain supported for existing
experiments; V2 artifacts reject propagation-only or both-RL execution.

## Mask and Path Lock

Action validity is enforced in C++, outside the neural network:

- `[1, 1]`: compare the two policy logits;
- `[1, 0]` or `[0, 1]`: choose the only unknown valid input without invoking
  policy selection;
- `[0, 0]`: report no backtrace continuation and let PODEM backtrack.

An existing path lock wins while its selected input remains unknown, its
objective value is unchanged, and no backtrack has occurred. Once that input
is correctly assigned, it becomes invalid in the mask. Gates such as AND with
an objective of one can then continue through the remaining unknown input. A
backtrack invalidates all path locks and recomputes masks from circuit values.

The hard mask is deliberately not a neural-network input. This keeps invalid
actions impossible while allowing static policy precomputation.

## Native Inference

After loading a V2 actor, C++ evaluates the actor once for every gate and both
objective values. It stores four floats per gate:

```text
logits[gate_id][objective_value][left_or_right]
```

The PODEM hot path performs only stable-ID lookup, mask handling, two float
loads, and one comparison. It performs no MLP operations, string lookup, tensor
construction, or dynamic allocation. Only backtrace logits are cached; no
propagation embedding or actor cache is created for V2.

Python training still runs the V2 actor normally so PPO can compute gradients.
Exported C++ logits must match Python evaluation within `1e-5`, and selected
actions must match exactly on all validation decisions.

## Interfaces

The normalization script accepts a source bench and writes the binary bench and
fault map without modifying the source. Paper-style preparation generates
binary forms before DeepGate extraction, because embeddings and circuit hashes
must correspond to the normalized graph.

Training V2 accepts only `backtrace_rl`. Native inference selects the actor
implementation from its artifact header. Existing V1 command lines continue to
work; V2 reports a clear error if requested with `propagate_rl` or `both_rl`.

## Validation

The implementation is accepted when all of the following pass:

1. Every generated binary circuit has fan-in at most two and retains the source
   primary-input and primary-output names.
2. Random-vector simulation produces identical primary outputs for each source
   and normalized circuit.
3. Mapped collapsed fault IDs, equivalent-fault counts, and total uncollapsed
   fault counts exactly match the source circuit.
4. Hard masks never select assigned inputs, path locks persist until completion,
   and backtracking invalidates locks.
5. Python and C++ V2 logits agree within `1e-5` with identical selected actions.
6. PPO plus RND completes a GPU training update on binary c6288 and
   s38417-scan faults.
7. Native V2 runs all binary benchmark circuits without propagation-policy
   decisions.
8. Baseline and V2 are compared on the same binary circuits using identical
   fault maps, seeds, and backtrack limits. Report wall time, internal CPU time,
   backtrace edges, backtracks, vectors, and coverage.
9. Original circuits and V1 models continue to load and produce their previous
   behavior.

## Deferred Optimizations

Binary actor and embedding formats, float16 or int8 quantization, SIMD kernels,
and parallel fault processing remain separate follow-up work. V2 table lookup
removes online MLP cost first, so these optimizations should be considered only
after profiling the new bottleneck.
