# Data-split SmartATPG training design

## Goal

Replace the current 16-circuit training workflow with a dataset-driven workflow:

- train on every BENCH circuit under `PODEM/data/train`;
- select the best model by evaluating every BENCH circuit under
  `PODEM/data/validation`;
- use a backtrack limit of 200 for heuristic profiling, PPO training, validation,
  and later benchmark execution;
- run exactly five normal PPO rounds and remove the additional failed-fault
  reinforcement phase;
- allow a current 11D BUF-free checkpoint to continue learning on a different
  circuit/fault manifest while retaining its complete learning state.

The graph feature and artifact contract remains
`SMARTATPG_FEATURES_V4_11D_CO_NO_BUF`. This change does not alter C++ PODEM's
internal fanout-stem representation.

## Dataset contract

The default dataset root is `PODEM/data` and contains:

- `train/`: exactly 1024 normalized combinational BENCH subcircuits;
- `validation/`: the six full validation circuits `b12_C`, `b15_C`, `b17_C`,
  `b20_C`, `b21_C`, and `b22_C`.

Both directories are enumerated by sorted filename. Every input must be a
regular `.bench` file, must have a unique stem, and must pass the 11D BUF-free
Python graph loader. Explicit `BUF`/`BUFF`, unsupported gates, malformed
drivers, and combinational cycles are rejected before profiling begins.

The preparation output is separate from the source dataset. It records source
paths, SHA-256 values, circuit hashes, graph statistics, profiling artifacts,
and fault lists. Changing, adding, removing, or renaming a dataset file makes an
existing manifest incompatible.

## Fault profiling and split semantics

Preparation uses traditional heuristic SCOAP PODEM with
`backtrack_limit=200` to enumerate and profile the complete collapsed fault
catalog of every circuit.

For a training circuit, the manifest retains every fault whose heuristic
profile has `outcome == 1`. No ranking, hard-fault truncation, or fixed
fault-count requirement is applied. A training circuit with zero detected
faults is rejected because it cannot produce an episode.

For a validation circuit, the manifest retains the entire enumerated fault
catalog, regardless of heuristic outcome. Detectable, redundant, and faults
that the heuristic did not resolve within 200 backtracks all remain validation
episodes. The heuristic outcome is profiling metadata only and never filters
the validation set.

Training and validation circuit identities and fault lists must remain
disjoint by directory and manifest section. Validation episodes never call an
optimizer update and never enter PPO or RND replay/state updates.

## Preparation and resumability

Preparation follows a fixed-manifest approach. It scans the dataset once and
writes one per-circuit profile file as each circuit completes. An atomic
preparation state records the dataset inventory and completed profiles, so an
interrupted 1030-circuit profiling run resumes without repeating valid work.

After all profiles pass validation, preparation atomically publishes the
training manifest. The manifest contains separate `train_circuits` and
`validation_circuits` sections rather than overloading one circuit list. Each
circuit record contains its BENCH path and hashes, complete profile path and
hash, and the selected episode fault IDs. Training records contain only
`outcome == 1` IDs; validation records contain every catalog ID.

Resume validates every reused source and profile hash. A changed source or
profile is an error; the user must choose a new preparation directory or
explicitly regenerate the manifest. Partial and finalized files are written by
temporary-file replacement.

## Five-round training flow

The Linux launcher prepares the dataset manifest, constructs graphs and native
trainers for all training circuits, and constructs read-only evaluators for all
validation circuits. It runs exactly five rounds with a backtrack limit of 200.

Each round contains every retained training fault exactly once. Episode order
is deterministic for a given seed and round, but is shuffled across
all training circuits so filename order does not form a curriculum. Existing
per-episode checkpointing is retained so an interrupted round resumes at the
next episode without replaying completed updates.

At the end of each complete round, the current policy is evaluated against
every validation fault exactly once with no parameter or RND-statistic update.
The best-model score is ordered as follows:

1. maximize validation detected-fault count (equivalently coverage because the
   validation catalog is fixed);
2. minimize total validation backtracks;
3. minimize total validation backtrace steps;
4. maximize total validation extrinsic return;
5. prefer the earlier round for an exact tie.

The latest checkpoint is saved independently from the best checkpoint. A run
that is interrupted during validation records validation progress and resumes
without training the next round or accidentally promoting a partial result.

The failed-fault reinforcement phase and its checkpoint, metrics, unresolved
fault list, CLI option, launcher step, and reinforced model output are removed
from this workflow. The final benchmark bundle uses the validation-selected
`model_best.txt`.

## Checkpoint modes

Two intentionally different modes are supported.

### Resume the same run

`--resume` requires the same manifest hash, output directory, five-round
configuration, backtrack limit, encoder variant, and seed. It restores all
model and learning state plus the exact round, episode, validation, and random
number generator progress.

### Continue on a different dataset

`--continue-from <checkpoint>` starts a new run and is mutually exclusive with
resuming existing progress. It accepts a current BUF-free 11D training or best
checkpoint and restores the complete agent learning state:

- graph encoder, Actor, Critic, and old-policy weights;
- PPO optimizer state;
- RND target/predictor, optimizer, and normalization statistics;
- model-side random state required for continuous learning.

It does not restore the source run's manifest binding, circuit trainers,
round/episode counters, validation progress, best score, or best checkpoint.
Those are initialized for the new manifest, and all best-model decisions are
made again on the new validation set. Tensor shapes, encoder variant, feature
schema, graph configuration, and current checkpoint format must match exactly;
old 12D or legacy artifact formats are rejected.

The new output directory records the source checkpoint path and SHA-256 so the
continuation lineage is auditable. An already nonempty output directory may
only be used through same-run `--resume`, never silently overwritten by
`--continue-from`.

## Linux entry points and artifacts

`train_smartatpg_linux.sh` defaults to:

- dataset root `data`;
- five rounds;
- backtrack limit 200;
- GAT-GRU encoder on physical GPU 0;
- a new output directory whose name identifies the 11D BUF-free data-split
  five-round protocol.

The launcher produces:

- `preparation/training_manifest.json` and resumable per-circuit profiles;
- `smartatpg_gat_gru/training_state.pth` for same-run resume;
- `smartatpg_gat_gru/best_training_state.pth` selected only by validation;
- `smartatpg_gat_gru/model_latest.txt` and `model_best.txt`;
- per-round training and validation metrics plus TensorBoard events;
- `benchmark_bundle/` containing only the validation-selected GAT-GRU model.

The exported actor becomes `SMARTATPG_MODEL_V12` and records the preparation
manifest hash, five-round protocol, and backtrack limit without embedding the
1024 circuit names in its header. The portable and native loaders accept only
this new model format. The unchanged 11D descriptor file remains
`SMARTATPG_EMBEDDINGS_V7`, paired to a model by snapshot hash. Manifest,
checkpoint, training-state, and benchmark-bundle format identifiers are also
incremented so the old 16-circuit, 50-fault, eight-round protocol cannot be
mistaken for this one.
Absolute paths are not used as identity: manifests store relocatable paths
relative to the preparation or dataset root plus content hashes, allowing the
repository and dataset to move from Windows to Linux.

## Error handling

The workflow fails explicitly when:

- either split is absent, empty, contains duplicate stems, or contains a
  non-BENCH entry selected for processing;
- any graph violates the 11D BUF-free input contract;
- profiling fails, returns duplicate fault IDs, or a training circuit has no
  `outcome == 1` faults;
- source/profile hashes change during resume;
- validation does not evaluate the full manifest catalog exactly once;
- same-run resume configuration differs;
- continuation checkpoint architecture or format is incompatible;
- `--resume` and `--continue-from` are requested together;
- continuation targets a nonempty output directory.

Errors name the split, circuit, source path, and operation so failures in a
large dataset can be located without inspecting the whole run.

## Verification

Tests must cover:

1. sorted discovery of all 1024 training and six validation BENCH files;
2. training fault selection includes every and only `outcome == 1` profile;
3. validation fault selection includes the entire catalog, including non-1
   outcomes;
4. the backtrack limit is 200 in profiling, training, validation, launcher, and
   benchmark metadata;
5. deterministic episode order contains each retained training fault once;
6. validation performs no optimizer/RND update and controls best-model
   selection;
7. exactly five normal rounds are allowed and no reinforcement phase runs;
8. preparation resumes per-circuit profiling and rejects changed hashes;
9. same-run resume restores exact progress;
10. cross-manifest continuation restores complete learning state while resetting
    progress and best-validation state;
11. incompatible legacy/12D checkpoints are rejected;
12. generated manifests and checkpoints remain usable after relocating the
    repository to a Linux path;
13. a small end-to-end fixture trains, validates, resumes, continues onto a
    different manifest, exports the best V12 model, and prepares a benchmark
    bundle.

Full production-data profiling and five-round GPU training are operational
runs, not unit tests, because they can contain a very large number of fault
episodes.
