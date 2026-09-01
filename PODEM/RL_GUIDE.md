# RL-Guided PODEM

The C++ PODEM implementation is the ATPG engine for both training and deployment. Python only runs DeepGate preprocessing and PPO optimization.

## Build the native executable

From `PODEM/src`, build with the existing Makefile or compile the sources including `rl_policy.cpp` and `rl_atpg.cpp`.

Run the original heuristic mode without RL options:

```text
atpg circuit.bench
```

Run native RL inference after exporting artifacts:

```text
atpg -fault-map circuit_binary.faultmap -rl-emb circuit_binary.emb -rl-actor actor_v2.txt -rl-mode backtrace_rl circuit_binary.bench
```

The embedding file is circuit-specific. The executable verifies its FNV-1a hash against the exact `.bench` file. `-rl-emb` and `-rl-actor` must be supplied together.
V2 actors support `backtrace_rl` only. They produce both input logits in one
forward pass. Native C++ stores embeddings by numeric gate ID and lazily computes
the two logits on the first use of each `(gate, objective value)` pair. Later
uses compare the cached values directly. The PODEM hot path still applies the
unknown-input filtering and honors the path lock. V1 actors remain loadable and
retain the legacy propagation modes.

The `.faultmap` maps selected source fault sites and equivalent-fault weights
onto the transformed binary netlist. For comment-declared XOR cells implemented
as private NAND/NOT networks, only the logical XOR output `GO-SA0/GO-SA1` faults
are retained; private `W/Z/X/Y` implementation faults are excluded. Its circuit
hash is verified before ATPG starts.

## Set up the Python training environment

Activate the `d2l` Conda environment and run all commands from this `PODEM`
directory:

```bat
call C:\ProgramData\Anaconda3\Scripts\activate.bat d2l
set DISTUTILS_USE_SDK=1
set MSSdk=1
python -m pip install -r python-requirements.txt
python -m pip install -e .
```

The editable install builds the `cpp_podem` pybind11 extension and installs the
`rl_podem` training package. It also keeps the vendored DeepGate extractor
discoverable under `vendor/deepgate_recgnn_extractor`, so no manual
`PYTHONPATH` configuration or sibling Python project is required.

On Windows, run the install command from an x64 Visual Studio Developer Command
Prompt. Microsoft Visual C++ 14 or newer is required. A POSIX-thread
MinGW toolchain can alternatively be selected with `--compiler=mingw32`; the
`win32`-thread MinGW variant cannot compile pybind11 because its standard
library lacks `std::mutex`. The two environment variables above make
setuptools use the compiler environment already initialized by the Developer
Command Prompt, including newer Visual Studio releases.

## Prepare and train the paper-reward V3 workflow

Convert any individual combinational benchmark without modifying its source:

```text
python scripts/convert_binary_bench.py sample_circuits/c432.bench
```

The converter builds balanced two-input AND/OR/NAND/NOR trees, filters expanded
XOR implementation-only faults, checks 256 random vectors for PO equivalence,
and verifies the mapped fault catalog through C++. Regenerate training manifests
and teacher profiles whenever a fault map changes.

Prepare the paper-style `c6288` and full-scan `s38417` data. This command
generates binary netlists, GPU DeepGate embeddings, profiles faults with a
500-backtrack limit, and deterministically splits the top 150 hard faults into
100 training faults and 50 validation faults per circuit:

```text
python scripts/prepare_paper_training.py artifacts/paper_v3 --count 100 --validation-count 50 --backtrack-limit 500 --deepgate-checkpoint artifacts/formal/deepgate_best.pth --device cuda:0
```

Train a fresh object-only PPO+RND model with the SmartATPG paper reward:

```text
python scripts/train_paper_rnd.py artifacts/paper_v3/training_manifest.json artifacts/paper_v3/training_state_v3.pth artifacts/paper_v3/actor_v2_best.txt --sweeps 5 --backtrack-limit 500
```

The extrinsic reward is `-0.1` per non-PI backtrace step,
`10 - 7.5 * exp(0.07 * (backtracks + PI visits))` for a non-terminal PI,
`+100` for detection, and `-100` for an undetected or aborted fault. RND remains
a separately scaled intrinsic exploration bonus.

Validation uses deterministic `argmax` without PPO or RND updates. The script
writes `actor_v2_best.txt` for deployment and `actor_v2_latest.txt` for resume
and diagnostics. Model selection first maximizes detected faults, then minimizes
aborted faults, backtracks, gate-to-input backtrace steps, and decisions. V3
training checkpoints intentionally reject V1/V2 checkpoints; the older actors
and checkpoints remain untouched. The V3 manifest binds the source circuit,
binary circuit, fault map, and embedding file by SHA256. Checkpoints also bind
all PPO/RND hyperparameters, while each training record stores separate
extrinsic, intrinsic, scaled-intrinsic, PPO-loss, and RND-loss metrics.

Run the automated reward, event, checkpoint, validation, and Python/C++ parity
checks after rebuilding the pybind extension:

```text
python scripts/verify_paper_v3.py
```

## Full-fault curriculum MC / GAE

The curriculum workflow first behavior-clones the heuristic teacher, then
updates PPO once per completed fault. There is no fixed-length rollout or
mid-fault update: one rollout contains that fault's Actor decisions. An aborted
fault is terminal for this training task, just like a detected fault.

### Linux SmartATPG 20-round training

Install a C++ compiler and Python development headers through the Linux
distribution, create the desired Python environment, then run from the `PODEM`
directory:

```bash
python -m pip install -r python-requirements.txt
python -m pip install -e .
python scripts/run_smartatpg_linux.py \
  --output-dir artifacts/smartatpg_linux_20rounds \
  --rounds 20 \
  --benchmark-repeats 5
```

The editable install compiles `cpp_podem` as a Linux extension. The launcher
uses SmartATPG graph descriptors and does not load DeepGate. It relocates the
historical manifest's Windows paths into the current checkout and verifies every
artifact hash before training. PyTorch uses `cuda:0` when CUDA is available and
otherwise uses CPU.

One round is one Easy sweep, one Medium sweep, and one Hard sweep. After each
round the frozen deterministic policy evaluates all 1,000 training faults and
all 500 validation faults, which this command reports as the test split:

```text
ROUND_EVAL round=1 train_mean_reward=... test_mean_reward=...
```

Re-running the same command resumes the checkpoint at the first incomplete
training unit or missing round evaluation. The run directory contains
`training_state.pth`, `actor_best.txt`, `actor_latest.txt`,
`round_metrics.json`, and `train.log`.

After round 20, the launcher builds a native executable and benchmarks heuristic
PODEM, RL best, and RL final with identical circuits, faults, seed, and backtrack
limit. One warm-up and five measured runs per model/circuit are used by default.
The `final_benchmark` directory contains `final_comparison.json`,
`final_comparison.csv`, and `FINAL_RESULTS.md`. Positive reduction percentages
mean RL used fewer steps or less time than heuristic; negative values mean RL
was worse. Offline SmartATPG graph/descriptor preprocessing is reported
separately and excluded from native ATPG time.

Use `--skip-benchmark` to stop after training and round evaluation. The final
benchmark can later be run directly:

```bash
python scripts/benchmark_smartatpg.py \
  artifacts/smartatpg_linux_20rounds/training_manifest.json \
  artifacts/smartatpg_linux_20rounds/training_state.pth \
  artifacts/smartatpg_linux_20rounds/native/atpg_rl_smartatpg \
  artifacts/smartatpg_linux_20rounds/final_benchmark \
  --repeats 5
```

Use an existing, hash-valid curriculum manifest and NEW output paths:

```text
python scripts/train_curriculum.py artifacts/paper_v6_xor_filtered/training_manifest.json artifacts/paper_v7_gae/training_state.pth artifacts/paper_v7_gae/actor_v2_best.txt --advantage-method gae --gamma 0.99 --gae-lambda 0.97 --return-scale 100 --log-rollouts
```

For an MC comparison with the same scaling and normalization settings:

```text
python scripts/train_curriculum.py artifacts/paper_v6_xor_filtered/training_manifest.json artifacts/paper_v7_mc/training_state.pth artifacts/paper_v7_mc/actor_v2_best.txt --advantage-method mc --gamma 0.99 --return-scale 100 --log-rollouts
```

Defaults are `--advantage-method gae`, `--gamma 0.99`, `--gae-lambda 0.97`,
`--return-scale 100`, and `--normalize-advantages`. Both curriculum methods
disable per-fault return standardization. All step rewards, including the RND
bonus, are divided by the same fixed scale; the relative reward weights remain
unchanged. MC fits discounted returns. GAE fits `raw_advantage + old_value`,
using zero continuation value at terminal transitions. With lambda equal to
one, full-fault GAE targets match MC targets.

Only the Actor's copy of Advantage is standardized using the population standard
deviation. The Critic target is not standardized. Single-decision rollouts skip
Advantage standardization; zero-decision faults are logged without a PPO update.
Use `--no-normalize-advantages` to disable Actor normalization for an ablation.
For the previous curriculum optimization settings, use
`--advantage-method mc --gamma 1 --no-normalize-advantages`.
The older paper-reward V3 entry point retains its legacy agent defaults.
The curriculum Critic now initializes only its output head to zero weight and
bias 1, preserving trainable hidden features. The previous all-zero initialization
could only learn a constant bias. Both new comparison modes share this fix, so
the legacy optimization flags alone do not reproduce the old initialization bug.

### Inspect rollout length

At startup, `PPO_CONFIG` prints the effective options and `rollout=full_fault`.
With `--log-rollouts`, each `ROLLOUT_RESULT` includes the circuit, fault ID,
`steps`, outcome, update status, and PPO metrics. `steps` counts Actor decisions,
not internal PODEM `backtrace_steps` or `backtracks`. Per-fault records are printed
after their curriculum work unit completes.

Every `TRAIN_RESULT.learning.rollout_steps` reports `count`, `min`, `mean`,
`median`, `p90`, `max`, and `zero_decision_faults`. These lengths include all
faults; p90 is the nearest-rank percentile. `learning.steps` remains the SUM
over the work unit, not the length of a single rollout. `raw_adv_*`,
`actor_adv_*`, and `value_target_*` distinguish the three learning signals;
summary `*_std_mean` fields are averages of within-rollout standard deviations.

### Checkpoints and verification

Resuming requires identical method, lambda, gamma, scale, normalization, reward
configuration, and manifest. Do not reuse a legacy training checkpoint path
when changing these settings. Existing inference actor exports are unchanged.
For an explicit weights-only warm start in Python, a fresh agent exposes
`load_actor_state_dict(state_dict)`: it imports the encoder and Actor weights,
leaves the new Critic intact, and does not restore old optimizer/RND state.

Run the mathematical/compatibility tests and the native integration smoke:

```text
python -m unittest discover -s tests -p test_full_fault_gae.py -v
python scripts/verify_full_fault_gae.py
```

The smoke uses the existing c432 curriculum data in an isolated temporary
directory. It exercises behavior cloning, both MC and GAE, checkpoint resume
(including bitwise comparison after a simulated mid-curriculum interruption),
rollout metrics, actor export, and deterministic evaluation. It does not
overwrite the original circuit outputs, training artifacts, or checkpoints.

## Switchable SmartATPG / DeepGate encoding

The current curriculum path supports `--embedding-backend smartatpg|deepgate`.
New preparation defaults to SmartATPG. Training/export infer the backend from
the manifest/checkpoint; legacy metadata means DeepGate. An explicit conflicting
flag fails. Changing the backend requires a fresh model, not reuse of the same
checkpoint. Existing V1/V2 actors and the earlier standalone training scripts
remain DeepGate-only; `train_curriculum.py` is the switchable training entry.

SmartATPG uses 14 static node features: nine gate-type one-hot entries, level,
fanout, and structural SCOAP CC0/CC1/CO. Two trainable 64-unit mean-GraphSAGE
layers aggregate incoming circuit edges, then mean-pool a 64-dimensional circuit
context. The policy receives `[graph_context, current_gate_features, mask]`,
80 values, projected to 32 dimensions and combined with the objective-value
embedding. The graph encoder learns during both BC and PPO. No DeepGate model,
DeepGate embeddings, PyG, or extra compiled graph library is needed.

This follows the paper's graph-context/state design, with documented project
defaults rather than an exact reproduction claim. The binary solver only calls
the policy when both inputs are available, so its mask is `[1,1]`; forced moves
remain solver-controlled. SmartATPG RND uses stable raw gate features plus the
objective one-hot, not the changing learned graph embedding. MC/GAE, full-fault
rollouts and potential reward semantics are unchanged.

### Prepare and train

Reuse the exact previous fault splits and teacher samples without reading any
DeepGate `.emb` files. This creates a new manifest, not a new random split:

```text
python scripts/prepare_curriculum_training.py artifacts/paper_v8_smartatpg --source-manifest artifacts/paper_v6_xor_filtered/training_manifest.json --embedding-backend smartatpg
python scripts/train_curriculum.py artifacts/paper_v8_smartatpg/training_manifest.json artifacts/paper_v8_smartatpg/training_state.pth artifacts/paper_v8_smartatpg/actor_best.txt --embedding-backend smartatpg --advantage-method gae --log-rollouts
```

For a DeepGate run, use the old DeepGate manifest and fresh output paths:

```text
python scripts/train_curriculum.py artifacts/paper_v6_xor_filtered/training_manifest.json artifacts/deepgate_comparison/training_state.pth artifacts/deepgate_comparison/actor_best.txt --embedding-backend deepgate --advantage-method gae
```

Completed preparation directories are protected. `--resume` checks their
original backend/configuration and reuses a valid manifest without rewriting
teacher data. A conflicting backend requires a new output directory.

### Export and native inference

At training completion, `actor_best.txt.json` and `actor_latest.txt.json` locate
complete snapshot-specific actor/descriptor pairs for each curriculum circuit.
Use the `actor` and `circuits.<name>.embeddings` paths from the SAME JSON.
Intermediate work-unit exports contain actor weights only; use explicit export
commands below to evaluate an interrupted checkpoint. Best may be the BC
fallback and must never be paired with latest-PPO descriptors.

To export a different circuit with no fine-tuning, choose the same checkpoint
and `--snapshot` for both files:

```text
python scripts/export_cpp_actor.py artifacts/paper_v8_smartatpg/training_state.pth artifacts/paper_v8_smartatpg/best_actor.txt --embedding-backend smartatpg --snapshot best
python scripts/export_cpp_embeddings.py sample_circuits/c432_binary.bench artifacts/paper_v8_smartatpg/training_state.pth artifacts/paper_v8_smartatpg/c432_best.emb --embedding-backend smartatpg --snapshot best
python scripts/build_native.py
build/atpg_rl_smartatpg.exe -bt 500 -seed 14 -fault-map sample_circuits/c432_binary.faultmap -rl-emb artifacts/paper_v8_smartatpg/c432_best.emb -rl-actor artifacts/paper_v8_smartatpg/best_actor.txt -rl-embedding-backend smartatpg sample_circuits/c432_binary.bench
```

Native SmartATPG inference uses exported static 80-value descriptors and the
existing lazy binary Actor cache, not a graph network on every decision. Backend,
schema, circuit, snapshot identity and dimensions are checked before inference.
Generated checkpoint, temporary, JSON and snapshot-directory paths are checked
for collisions before training writes. Native artifact readers support UTF-8
Windows paths and reject duplicate wire-name tables.
Legacy DeepGate inference still accepts its original files; optionally pass
`-rl-embedding-backend deepgate` to validate the selection. Report graph encoding
and export time separately from ATPG time when comparing performance.

For DeepGate **embedding** export, the checkpoint must be the original DeepGate
encoder checkpoint, such as `artifacts/formal/deepgate_best.pth`, not a PPO
checkpoint. The exporter now strictly checks the encoder weights and refuses
incompatible or partial state instead of silently using random initialization.

### Verification

```text
python setup.py build_ext --inplace
python -m unittest discover -s tests -v
python scripts/verify_curriculum_v4.py
python scripts/verify_full_fault_gae.py
python scripts/verify_smartatpg.py --output-dir artifacts/smartatpg_smoke_new
```

The SmartATPG smoke forces CPU and uses isolated copies of c432/c499 with one
fault per difficulty per split. It runs two BC epochs, short MC/GAE curricula,
exact interrupted-resume checks, live-graph/exported/native logit and fault-run
parity, and standalone backend rejection tests. It does not import DeepGate or
perform a full training campaign. Passing these checks does not establish better
coverage, wall time, or generalization.
