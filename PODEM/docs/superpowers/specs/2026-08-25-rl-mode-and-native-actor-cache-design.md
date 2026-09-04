# RL Mode Selection and Native Actor Cache Design

## Objective

Reduce native C++ RL inference overhead without changing exported actor
parameters or scoring semantics, and allow RL guidance to be enabled for
backtrace decisions, propagation decisions, or both. The default mode is
backtrace-only so D-frontier propagation follows the original PODEM heuristic.

## User Interface

The native executable accepts:

```text
-rl-mode backtrace_rl
-rl-mode propagate_rl
-rl-mode both_rl
```

`-rl-mode` defaults to `backtrace_rl`. It is valid only when `-rl-emb` and
`-rl-actor` are also provided. Unknown values fail before ATPG starts and print
the accepted values.

The pybind11 `run_stuck_at` function gains an `rl_mode` keyword argument with
the same accepted values and default. `CppPodemPPOTrainer.run` and the paper
training script expose the same option so training and deployment use matching
decision scopes.

Mode behavior is exact:

- `backtrace_rl`: RL selects among unknown fan-ins during backtrace. D-frontier
  propagation uses the original first-valid-gate heuristic and does not collect
  all propagation candidates.
- `propagate_rl`: backtrace uses the original PODEM heuristic. RL selects among
  valid D-frontier propagation gates.
- `both_rl`: RL controls both decision types, preserving the current combined
  behavior.

An episode remains active whenever either decision type is enabled. Backtrack
events are attributed to the latest RL decision that is valid for the selected
mode, so propagation-only training still receives a meaningful penalty.

## Stable Gate Identity

Each `WIRE` receives a stable integer RL gate ID when it is created. The ID is
local to one ATPG circuit instance and is independent of hash-table iteration
order and circuit levelling.

Python policies continue receiving gate names because callbacks use names to
locate PyTorch embedding tensors. Native policies receive gate IDs and do not
construct candidate-name vectors in the hot inference path. A decision request
can carry both representations, but ATPG populates names only when the active
policy declares that it needs them.

When native inference is enabled, ATPG supplies the complete ID-to-name table
to `NativeActorPolicy`. The policy validates that every circuit wire has an
embedding and builds indexed caches once. Missing or duplicate names remain
hard errors during initialization rather than failures in the search loop.

## Actor Cache

`NativeActorPolicy` precomputes two encoded vectors for every gate:

```text
backtrace cache[id]  = tanh(W_gate * embedding[id] + b_gate) + mode[0]
propagation cache[id] = tanh(W_gate * embedding[id] + b_gate) + mode[1]
```

This removes repeated 64-by-64 gate encoding from every decision. The original
Gate embeddings may be released after both caches are built because native
selection no longer consumes them.

Actor tensor names are resolved once during model loading. Dense layers keep
direct references to validated tensors instead of performing unordered-map
lookups by strings for every candidate.

The native scoring path uses reusable buffers sized during initialization for
the concatenated representation and hidden layer. It writes one candidate
score at a time and tracks the argmax directly, avoiding temporary candidate
embedding copies, pair vectors, hidden vectors, output vectors, and a logits
vector. Public vector-based scoring used by `score_actor` remains available for
compatibility and numerical tests.

The cache and reusable buffers belong to one policy instance. A policy instance
is used synchronously by one ATPG run, so no shared mutable global state or
locking is introduced.

## Compatibility

The actor artifact format remains `SMARTATPG_ACTOR_V1`; existing actor and
embedding files remain valid. PPO and RND checkpoint formats are unchanged.
RND remains training-only and is not added to C++ deployment.

For identical candidates and mode, optimized native logits must agree with the
existing reference implementation within floating-point tolerance, and argmax
selection must be identical. Cached inference therefore requires no retraining.

The Python callback path intentionally retains names and PyTorch behavior. Its
performance is not part of the native cache optimization because it is used for
training rather than deployment.

## Build Configuration

Native release builds must enable compiler optimization. MSVC builds use `/O2`
and the existing standard/exception flags; GCC or MinGW builds retain `-Ofast`.
The pybind extension also uses optimized compilation so training callback glue
does not accidentally use an unoptimized C++ PODEM engine.

## Validation

The implementation is accepted when all of the following pass:

1. Native and pybind builds complete with MSVC in the `d2l` environment.
2. Invalid mode values and mode-without-model arguments fail clearly.
3. `backtrace_rl` produces only backtrace callbacks and uses original
   propagation behavior.
4. `propagate_rl` produces only propagation callbacks and uses original
   backtrace behavior.
5. `both_rl` produces both callback types and remains compatible with the
   existing actor.
6. Omitting `rl_mode` is equivalent to explicitly selecting `backtrace_rl`.
7. Reference and cached actor scores agree within `1e-5`, with identical
   selected indices on representative candidate sets.
8. Existing fault filtering, path locks, PPO/RND checkpoint loading, actor
   export, and pure C++ inference smoke tests continue to pass.
9. The eight held-out combinational circuits plus `c6288` are benchmarked with
   the same seed, backtrack limit, warm-up, and three-run median procedure used
   previously. Results report end-to-end wall time and ATPG-only CPU time
   separately.

## Non-Goals

- Moving native actor inference to CUDA.
- Changing actor topology, gate embeddings, PPO, RND, or reward design.
- Retraining the current checkpoint as part of this optimization.
- Removing dynamic allocation from unrelated PODEM circuit and fault data
  structures.
