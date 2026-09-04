# Linux SmartATPG 20-Round Curriculum Training Design

## Objective

Extend the existing SmartATPG curriculum pipeline so it can be moved to Linux
and run for 20 complete curriculum rounds. Each round must train in the order
Easy, Medium, Hard, then deterministically evaluate the current policy on the
full training and validation/test fault splits and print their mean extrinsic
rewards. After all 20 rounds, run a fair native benchmark comparing the final
and best RL policies with the heuristic PODEM implementation.

The SmartATPG graph backend is the only embedding backend used by this
workflow.

## Definitions and Scale

One fault run is one episode and one variable-length rollout. An episode with
at least one Actor decision performs one PPO update, which reuses all decisions
from that episode for the configured eight PPO epochs.

One stage sweep trains every fault selected for that curriculum stage across
all ten circuits. One curriculum round consists of exactly one Easy sweep, one
Medium sweep, and one Hard sweep, in that order.

The current manifest contains 100 training and 50 validation faults per
circuit, stratified as 40/40/20 training faults and 20/20/10 validation faults
for Easy/Medium/Hard. The existing balanced stage selector therefore produces:

- Easy: 40 Easy faults per circuit, 400 episodes total.
- Medium: 40 Easy and 40 Medium faults per circuit, 800 episodes total.
- Hard: 20 Easy, 20 Medium, and 20 Hard faults per circuit, 600 episodes total.

Each round contains 1,800 training episodes. Twenty rounds contain 36,000
training episodes. The round index is also the selector sweep index, preserving
the current deterministic offset behavior: the Hard stage alternates the two
20-fault halves of the 40-fault Easy and Medium strata while including all 20
Hard faults.

Behavior cloning remains the existing 20-epoch initialization step. The new 20
rounds refer to curriculum PPO training after behavior cloning.

## Training Interface and Scheduling

`scripts/train_curriculum.py` gains a `--curriculum-rounds` positive integer
option. Supplying `--curriculum-rounds 20` selects round mode. The legacy
`--stage-sweeps` interface and its current behavior remain available when round
mode is not requested, so existing experiments and checkpoints are not silently
reinterpreted.

Round mode uses an outer loop over rounds 1 through 20 and an inner loop over
Easy, Medium, and Hard. Each stage retains the current exploration schedule:

- Easy: `rnd_beta=0.05`, `entropy_coef=0.01`.
- Medium: `rnd_beta=0.02`, `entropy_coef=0.005`.
- Hard: `rnd_beta=0.0`, `entropy_coef=0.001`.

The seed, GAE settings, reward scaling, advantage normalization, backtrack
limit, PPO epochs, and per-fault update behavior remain unchanged. Unit order is
deterministically shuffled from the configured seed, round, and stage so a
fresh run and a resumed run execute the same work.

## Deterministic Round Evaluation

Add a curriculum reward evaluator that uses the same SmartATPG graph policy and
baseline-relative reward as training but never writes to the rollout buffer,
computes gradients, updates PPO/RND, or samples actions. It always chooses the
valid action with maximum policy probability.

For each evaluated fault, the extrinsic reward is:

```text
detection reward + clipped backtrack improvement + clipped backtrace improvement
```

The detection reward is +100 for detection and -100 otherwise. Backtrack and
backtrace improvements use the fault's heuristic baseline and the existing
`REWARD_CONFIG`, including its clipping bounds and weights. RND intrinsic reward
is excluded.

After the Hard stage of every completed round, evaluate:

- all 1,000 training faults;
- all 500 validation faults, presented to the user as the test split.

The reported mean is the micro-average over faults: total extrinsic reward
divided by the number of evaluated faults. The output also retains total reward,
fault count, outcome/counter summaries, and per-circuit statistics. A zero-Actor-
decision fault still contributes its terminal and efficiency reward.

The terminal emits one concise human-readable line and one structured record:

```text
ROUND_EVAL round=1 train_mean_reward=... test_mean_reward=...
```

The structured records are stored as a JSON list rewritten through a temporary
file and atomic replacement after every round, and the same history is stored in
the checkpoint. Evaluation uses the same fixed seed for every round, making
changes across rounds attributable to policy changes rather than input ordering.

## Checkpoint and Model Selection

Round-mode checkpoints use an explicitly versioned format and include:

- model, optimizer, RND, and random-number-generator states;
- completed `(round, stage, circuit, difficulty)` training units;
- completed round evaluation records;
- the current latest policy and existing best-policy selection state;
- the manifest hash and complete round-mode configuration.

Checkpoint writes remain atomic. A repeated command resumes at the first
unfinished unit. If all training units for a round are complete but its
evaluation is missing, resume performs that evaluation exactly once before
continuing. A checkpoint whose manifest or training configuration differs is
rejected rather than partially resumed.

`actor_latest` represents the newest policy and is refreshed during training
and at each round boundary. `actor_best` retains the existing validation
selection order: maximize detected faults, then minimize aborted faults,
backtrack ratio, backtrace ratio, and decisions. Mean reward is reported and
stored but does not silently replace the established best-model criterion.

## Linux Launch and Portability

Provide a Linux-oriented SmartATPG launcher with configurable output directory,
round count, seed, and benchmark repeats. Its default training command requests
20 curriculum rounds, GAE, `gamma=0.99`, `gae_lambda=0.97`,
`return_scale=100`, normalized Actor advantages, 20 behavior-cloning epochs,
and batch size 256 for behavior cloning.

Historical manifests contain Windows absolute paths. The launcher creates a
run-local manifest whose circuit, fault-map, profile, and teacher paths point to
the corresponding files under the current Linux repository checkout. It does
not alter the historical manifest. Every relocated file is checked against the
existing SHA256 entry before training begins.

Linux setup uses `python -m pip install -e .`, which builds the `cpp_podem`
pybind11 extension as a native `.so` through the existing setuptools extension.
The launcher must reject a missing or unloadable extension with an actionable
setup message. SmartATPG training automatically uses `cuda:0` when PyTorch sees
CUDA and otherwise uses CPU. It must not reference or copy the Windows `.pyd`.

Console output is streamed live and mirrored to a run log. Output paths are
isolated under the requested run directory and include the portable manifest,
checkpoint, latest and best actors, round metrics, training log, final benchmark
artifacts, and run metadata.

## Final Heuristic Comparison

After round 20 and its evaluation finish successfully, automatically run a
native full-circuit benchmark for:

- heuristic PODEM;
- the round-20 `actor_latest` policy, reported as RL final;
- the validation-selected `actor_best` policy, reported as RL best.

All models use the same Linux native executable, ten circuits, complete fault
lists, mapped fault files, seed, `backtrack_limit=500`, and process environment.
Each model/circuit pair receives one unmeasured warm-up followed by five measured
runs by default. Model execution order is rotated, and timing summaries use the
median measured value.

The benchmark reports per-circuit and aggregate values for:

- detected and total faults;
- aborted and redundant faults;
- backtracks and backtrace steps;
- generated test vectors;
- native ATPG interval time;
- whole-process wall time.

For RL final and RL best, the report also gives percentage changes relative to
heuristic for ATPG time, wall time, backtracks, and backtrace steps. A decrease
is displayed as an improvement with an unambiguous sign and label. SmartATPG
graph parsing, graph encoding, and descriptor export are measured separately as
offline preprocessing and are not included in ATPG time.

The benchmark writes `final_comparison.json`, `final_comparison.csv`, and
`FINAL_RESULTS.md`, and prints a `COMPARISON` summary. Benchmark progress is
persisted independently so an interrupted benchmark can resume without
retraining the model.

## Failure Handling

Training stops with a clear error before mutation when the manifest hash,
relocated artifact hash, backend, circuit graph identity, reward configuration,
or checkpoint configuration is inconsistent. Non-finite rewards, targets,
losses, or evaluation summaries are rejected. A failed round evaluation does
not mark the round evaluated. A failed final benchmark leaves the completed
training checkpoint intact and can be retried separately.

## Verification

Automated tests cover:

- parsing and compatibility of `--curriculum-rounds` and legacy
  `--stage-sweeps`;
- exact Easy/Medium/Hard ordering for 20 rounds and deterministic fault
  selection;
- expected episode counts for a full round;
- deterministic evaluation without buffer, optimizer, policy, or RND mutation;
- baseline-relative per-fault reward and train/test micro-average calculations;
- checkpoint interruption and resume within a stage and between training and
  round evaluation;
- prevention of duplicate round evaluations;
- Windows-manifest relocation and SHA256 validation on Linux-style paths;
- final comparison aggregation and percentage-delta calculations.

A reduced two-round SmartATPG smoke run uses a small manifest to exercise native
training, round evaluation, checkpoint resume, actor export, and the heuristic
comparison without requiring the full 66,000 fault executions.

## Acceptance Criteria

The work is complete when a Linux user can install the project, launch one
command for a 20-round SmartATPG curriculum run, observe train/test mean reward
after every round, interrupt and resume safely, and receive the final heuristic
versus RL best/final comparison in terminal, JSON, CSV, and Markdown formats.
Existing stage-sweep experiments continue to run with their previous semantics.
