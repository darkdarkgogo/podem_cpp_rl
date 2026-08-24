# RL-Guided C++ PODEM Design

## Goal

Use the existing C++ stuck-at PODEM implementation as the only authoritative ATPG engine. During training, Python/PyTorch PPO selects backtrace and propagation candidates through pybind11 callbacks. During deployment, the same decisions are made by a dependency-free C++ actor using exported weights and precomputed DeepGate embeddings.

The Python PODEM implementation is not used for state transitions, implication, fault simulation, backtracking, or test detection.

## Scope

The first implementation targets `PODEM/src/podem.cpp`, specifically `ATPG::podem`, `ATPG::find_pi_assignment`, and `ATPG::find_propagate_gate` for stuck-at faults.

Transition-delay ATPG in `tdfatpg.cpp`, PODEM-X, fault simulation, test compression, and DeepGate model execution in C++ are outside the first implementation. Existing behavior remains the default when RL is disabled or artifacts are unavailable.

## Chosen Architecture

### Offline embedding generation

Python runs the trained DeepGate extractor once for each `.bench` circuit and writes a versioned embedding file. Each row is keyed by the C++ gate output wire name, not by parser order or an in-memory numeric ID.

The file contains:

- format version;
- circuit identifier and deterministic netlist fingerprint;
- embedding dimension and gate count;
- output wire name and embedding values.

The exporter rejects duplicate names. The C++ loader rejects missing names, dimension mismatches, non-finite values, and fingerprint mismatches. The fingerprint covers the exact netlist file, so name-to-structure mismatches cannot silently pass validation.

### Shared decision contract

C++ owns candidate discovery. A decision request contains:

- mode: `backtrace` or `propagation`;
- objective output wire name for backtrace;
- objective logic value;
- ordered candidate output wire names;
- decision sequence number;
- current fault identifier and current backtrack count for logging.

Only the candidate index is returned by the policy. C++ validates the index before applying it. The policy cannot directly modify wire values, the decision tree, or simulation state.

When a decision has zero candidates, C++ follows the existing failure path. With one candidate, C++ selects it without invoking the policy. With multiple candidates, C++ invokes the configured policy; if no policy is configured, the current level-based heuristic is used.

### Backtrace integration

`ATPG::find_pi_assignment` continues to compute inversion and the next objective value exactly as it does now. For multi-input AND, NAND, OR, and NOR gates, it gathers all unknown input wires in deterministic netlist order and asks the policy to choose one. NOT and BUF remain deterministic and do not create RL actions.

The objective embedding is the embedding associated with `object_wire`. Candidate embeddings are associated with candidate input wires. This preserves the existing actor's backtrace representation while letting the correct C++ algorithm perform the recursive backtrace.

### Propagation integration

`ATPG::find_propagate_gate` evaluates the same legality conditions as the current implementation: unknown output, marked path to a primary output, at least one D or D-bar input, and a valid X-path. Instead of returning the first legal gate encountered, it gathers all legal gates in deterministic order and asks the policy to choose one.

The chosen gate is then used by the unchanged objective-value logic in `ATPG::test_possible`.

### Training path

The pybind11 module exposes a training runner rather than a resumable Gym-style state machine. Python requests a circuit/fault run and supplies a decision callback. C++ executes the entire fault episode and synchronously calls Python whenever a multi-candidate decision is required.

This approach keeps the C++ search stack and decision tree intact. The callback receives immutable decision data and returns an integer action. C++ reports episode statistics and events needed for reward assignment:

- each decision;
- each PI-value flip/backtrack, attributed to the most recent relevant decision;
- terminal detected, redundant, aborted, or backtrack-limit outcome;
- number of decisions, simulations, and backtracks.

The initial reward schedule remains compatible with the Python implementation: a small per-decision penalty, a backtrack penalty, and terminal success/failure reward. Reward constants remain Python configuration so experiments do not require recompiling C++.

The binding releases the GIL while C++ runs and reacquires it only for the decision callback and event delivery. Python exceptions are converted into an episode error after C++ restores temporary PODEM state.

### Deployment path

Python exports the inference subset of `policy_old` to a versioned actor file:

- gate encoder weights and bias;
- two mode embedding vectors;
- backtrace actor weights and biases;
- propagation actor weights and biases;
- dimensions and format version.

The critic and optimizer state are not exported. The C++ implementation performs dense layers and `tanh` directly using `std::vector<float>`. It reproduces the Python formulas for gate encoding, backtrace pair scoring, and propagation mean-state scoring.

Deployment selects `argmax(logit)` for deterministic ATPG results. An optional seeded categorical mode may be added later, but is not part of the first implementation.

The CLI accepts explicit embedding and actor paths. If both are present and valid, RL inference is enabled. If either is omitted, the baseline heuristic is used. Invalid supplied artifacts are fatal configuration errors rather than silent fallback.

## Alternatives Considered

### Resumable `reset` and `step` environment

This offers a conventional Gym API but requires converting recursive backtrace and the local PODEM decision tree into persistent state. The algorithm-change risk is high, so it is not selected for the first implementation.

### ONNX Runtime or LibTorch deployment

This simplifies neural-network export but adds a large runtime and build dependency to a small C++11 project. The actor consists only of dense layers, mode embeddings, `tanh`, mean, and candidate scoring, so a small native implementation is preferred.

### Python subprocess or socket policy server

This avoids pybind11 but adds serialization overhead and process-lifecycle failure modes at every decision. It also does not meet the pure C++ deployment requirement.

## Artifact Compatibility

Training and deployment use the same embedding and actor formulas. Both artifact formats are versioned and include dimensions. Float values are exported with sufficient precision for round-trip reconstruction.

An exporter-side parity test runs representative backtrace and propagation candidate sets through PyTorch and the native C++ scorer. Export succeeds only when selected indices match and logits are within a documented tolerance.

Embedding files are circuit-specific. A new circuit requires one offline Python DeepGate preprocessing run before either training or pure C++ inference.

## Error Handling

- Missing candidate embeddings stop the episode and identify the wire name.
- An out-of-range callback action stops the episode without mutating the selected wire.
- Artifact version, dimension, fingerprint, or non-finite-value errors fail during loading.
- A Python callback exception is propagated with the decision sequence and fault identifier.
- Baseline mode never loads RL artifacts and must preserve existing output.

## Verification

1. Build and run the original C++ executable with RL disabled; compare fault coverage, pattern count, and backtrack totals against the pre-change baseline.
2. Unit-test embedding lookup, fingerprint validation, actor-file parsing, dense/tanh calculations, and invalid action handling.
3. Compare Python and C++ logits for both actors on fixed embeddings and candidate sets.
4. Run pybind11 episodes with a deterministic callback that reproduces the old heuristic; results must match the native baseline.
5. Train on a small circuit such as c17, save and export a checkpoint, then run pure C++ inference with the same circuit embedding.
6. Run repeated deterministic inference and verify identical selected actions and ATPG results.

## Success Criteria

- All PODEM state transitions and correctness decisions occur in C++.
- Python training receives decisions and outcomes from the C++ engine only.
- The same trained actor can be exported and evaluated in C++ without Python, PyTorch, DeepGate, ONNX Runtime, or LibTorch at ATPG runtime.
- A new circuit can be supported by generating its DeepGate embedding file once in Python.
- RL-disabled behavior remains compatible with the current C++ PODEM.
