# Dynamic SmartATPG Validation Discovery

## Goal

Allow SmartATPG validation to evaluate every `.bench` circuit currently placed in `PODEM/data/validation`, including newly added circuits, without maintaining a fixed circuit-name list or retraining existing models.

## Scope

- Change validation discovery from an exact six-name contract to dynamic `.bench` discovery.
- Keep deterministic ordering by sorting paths by filename.
- Require at least one `.bench` file.
- Continue rejecting directories and non-`.bench` entries in the validation directory so accidental files cannot silently alter or disrupt an experiment.
- Keep training manifests, inference checkpoints, model formats, reward schemes, and model-loading identity checks unchanged. Existing trained checkpoints remain valid.
- Update validation documentation and focused tests to describe and verify the dynamic behavior.

## Data Flow

`validate_smartatpg.py` resolves the training run's dataset root, scans its `validation` directory at validation time, sorts every `.bench` file by filename, builds a fresh full fault catalog for each circuit, and evaluates every saved GAT-GRU round, every saved Mean round, and a fresh SCOAP baseline. The resulting comparison contains one row per discovered circuit plus `TOTAL`.

## Error Handling

- Missing validation directory: retain the current `FileNotFoundError`.
- No `.bench` circuits: raise a clear `ValueError`.
- Any directory or non-`.bench` entry: retain strict rejection with the unexpected entry names.
- Invalid BENCH contents: allow the existing graph/native parsers to report the specific parsing failure.

## Compatibility

Validation data is intentionally discovered independently from training. The trained checkpoint identity remains tied to the training manifest, not the validation filenames, so adding validation circuits does not require retraining or rewriting checkpoints. Historical validation outputs are not reused; validation is run fresh against the current directory contents.

The legacy `validation_circuit_count` training metadata remains unchanged for checkpoint-format compatibility. It is descriptive training-protocol metadata and is not used to limit runtime validation discovery.

## Tests

- Verify discovery accepts an arbitrary set of `.bench` files and returns filename-sorted paths.
- Verify an empty validation directory is rejected.
- Retain coverage for missing directories and unexpected non-BENCH entries.
- Run focused SmartATPG validation/data-split tests after the change.

## Non-goals

- Introducing a separate test split.
- Changing best-round selection behavior.
- Resuming partial validation runs.
- Retraining or modifying existing model artifacts.
