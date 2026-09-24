# SmartATPG Validation Module Boundary Design

## Goal

Complete the internal separation between SmartATPG training and validation. The public workflows are already independent; this refactor removes the remaining dependency from `rl_podem.validation` to validation helpers hosted in `rl_podem.training`.

## Dependency rule

The final dependency graph is:

```text
training.py --------> shared contracts / I/O
validation.py ------> shared contracts / I/O
     |
     +-------------> validation_core.py
```

`validation.py` must not import `training.py`. `training.py` must not define validation execution, catalog, ordering, scoring, or summary functions.

## Module responsibilities

### `training.py`

Owns only the single-encoder training worker: manifest resolution for training circuits, PPO state and resume handling, episode ordering and batching, per-round checkpoint creation, and actor export.

It keeps training-only helpers such as `_episode_order`, `_fault_update_boundary`, `_save_state`, `_validate_resume`, and `_training_protocol`.

### `validation_core.py`

Provides validation-domain operations without importing `training.py` or importing PyTorch:

- `_native_validation_batch`
- `_catalog_fault_ids`
- `_load_validation_catalogs`
- `_validation_catalog_hash`
- `_validation_order`
- `_evaluate_fault`
- `_summarize_fault_records`
- `_summarize_validation`
- `validation_score`

It may depend on `data_split`, `cpp_bridge`, and `smartatpg_rewards`. Native `cpp_podem` remains lazily imported inside the functions that execute it.

### `validation.py`

Owns fresh validation orchestration: loading inference checkpoints, reconstructing embeddings, evaluating every round, selecting the best rounds, running the SCOAP baseline once, and writing the comparison artifacts. It imports validation-domain helpers from `validation_core.py`.

### `artifact_io.py`

Provides small shared artifact contracts and I/O functions needed by both workflows:

- `INFERENCE_CHECKPOINT_FORMAT`
- `manifest_hash(path)`
- `atomic_json(path, value)`
- `atomic_json_lines(path, records)`

The functions retain the current atomic temporary-file replacement behavior. Moving them must not change serialized JSON formatting or checkpoint identity validation.

The training module may keep `_atomic_torch_save` because validation does not write PyTorch checkpoints.

## Data and control flow

Training writes an inference checkpoint whose format identifier comes from `artifact_io.py`. It never imports `validation_core.py`.

Validation reads the same shared format identifier and manifest hash through `artifact_io.py`, builds the validation catalog through `validation_core.py`, and performs orchestration locally. No compatibility re-export is added to `training.py`; callers and tests must migrate to the owning module so a future dependency cannot silently return.

## Compatibility and behavior

This is a structural refactor only. It must preserve:

- checkpoint format strings and serialized payloads;
- ordered validation fault catalogs;
- native batch arguments and record order;
- validation summary fields and best-round lexicographic ordering;
- atomic JSON and JSONL output formatting;
- the two public CLI entry points and their arguments;
- the training and validation output directory layouts.

Private helper imports from tests are updated to their new owning modules. No deprecated aliases remain in `training.py`.

## Testing

Tests must prove both behavior and architecture:

1. `validation.py` does not import `training.py`.
2. `training.py` contains none of the validation helpers listed above and does not import `validation_core.py`.
3. Existing catalog, evaluator, summary, scoring, checkpoint, fresh-SCOAP, and six-circuit-plus-TOTAL tests continue to pass from their new modules.
4. Shared artifact I/O produces the same manifest hashes and atomic JSON/JSONL files.
5. The rebuilt native extension and complete test suite remain green.

## Non-goals

- No changes to PPO, reward schemes, graph encoders, native ATPG algorithms, timing boundaries, datasets, CLI arguments, or result schemas.
- No public API stabilization for underscore-prefixed helpers.
- No new generic framework beyond the two focused modules required to break the dependency.
