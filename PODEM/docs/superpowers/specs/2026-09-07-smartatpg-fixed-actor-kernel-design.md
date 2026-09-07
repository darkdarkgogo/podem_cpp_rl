# SmartATPG Fixed Actor Kernel Design

## Goal

Reduce native CPU inference overhead for the current SmartATPG V8 Actors while
preserving the existing logits and benchmark protocol.

## Compile optimization

Linux standalone and extension builds use `-O3 -march=native`. `-O3` enables
aggressive inlining, loop transformation, and auto-vectorization;
`-march=native` enables the instruction set of the benchmark machine. MSVC
keeps `/O2`, its supported optimizing build mode, so Windows remains buildable.
Fast-math is not enabled because reassociating floating-point reductions can
change an action when two logits are nearly equal.

## Fixed Actor kernels

The current trained models have direct Actors with shapes `12 -> 32 -> 2` for
fanin-mean and `13 -> 32 -> 2` for level-GAT-GRU. Native inference adds two
compile-time specialized kernels for these shapes. Their loop bounds are
constants, allowing the compiler to unroll or vectorize the dot products.

`NativeActorPolicy` uses fixed-capacity arrays for the maximum 13-value input
and 32-value hidden/state workspaces when a current model matches these shapes.
Models with other dimensions and legacy artifact versions continue to use the
existing dynamic buffers and generic Actor implementation.

The optimization does not batch independent PODEM decisions and does not move
inference to a GPU. Cache behavior remains unchanged: fanin-mean always runs
the Actor and level-GAT-GRU retains its logit cache.

## Validation

Build the extension and standalone executable, run the unit suite, and execute
both model variants on c432. For the same generated model artifacts and seed,
detected/aborted/redundant results and decision counts must remain stable. The
native timing counters provide the before/after performance observation.
