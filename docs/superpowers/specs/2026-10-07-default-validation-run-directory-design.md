# Default SmartATPG Validation Run Directory

## Goal

Allow SmartATPG validation to run with no command-line arguments by defaulting to the project's `PODEM/artifacts/smartatpg_dual` training run.

## Command-Line Behavior

- `python scripts/validate_smartatpg.py` uses `PODEM/artifacts/smartatpg_dual`.
- The default is anchored to the `PODEM` project root derived from the validation module location, so it does not depend on the caller's current working directory.
- `--run-dir PATH` remains supported and overrides the default.
- `--dataset-root`, `--output-dir`, and `--seed` retain their current optional behavior.

## Implementation

Define one module-level default run directory in `rl_podem.validation` from the existing `ROOT` path. Configure the argument parser with that path as the `--run-dir` default and remove the `required=True` constraint. Continue passing the parsed value through the existing `run_fresh_validation` interface without changing validation, checkpoint, or training formats.

## Errors

If the default run has not been trained or lacks `train_summary.json`, retain the existing missing-training-summary error. Explicit invalid paths receive the same error behavior.

## Tests and Documentation

- Test that calling `main([])` passes the project-root default to `run_fresh_validation`.
- Test that an explicit `--run-dir` overrides the default.
- Update `PODEM/RL_GUIDE.md` so the primary validation example has no arguments and documents the optional override.

## Non-goals

- Automatically selecting the newest run.
- Creating or training a missing default run.
- Changing dataset, output-directory, seed, model, checkpoint, or validation-circuit behavior.
