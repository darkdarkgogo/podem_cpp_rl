# Multi-Circuit Curriculum Training Design

## Goal

Train a new backtrace-only V2 actor that preserves fault detection while
reducing both gate-to-input backtrace traversals and PODEM backtracks relative
to the original heuristic across the available benchmark circuits.

The new workflow is versioned as V4. Existing V3 manifests, checkpoints,
actors, scripts, and native inference behavior remain usable and unchanged.
Propagation RL remains disabled.

## Approaches Considered

### Recommended: heuristic pretraining followed by curriculum PPO

First train the actor to reproduce the original C++ backtrace heuristic, then
fine-tune it with PPO on progressively harder faults. This gives PPO a safe
starting point and directly addresses the current policy's severe regressions
on `c499` and `c1355`.

### Curriculum PPO from random initialization

This requires less data plumbing, but early random actions still generate long,
low-quality episodes. The existing experiment already shows that sparse hard
fault training can regress after later sweeps, so this option is not selected.

### Runtime hybrid between RL and the heuristic

A confidence threshold could fall back to the heuristic during native
inference. This is safer but adds another inference decision and cannot prove
that a high-confidence RL choice is better. It remains a possible later safety
layer, not part of V4 training.

## Training Circuits and Artifacts

V4 uses the binary-netlist versions of:

- `c432`, `c499`, `c1355`, `c1908`, `c2670`, `c3540`, `c5315`, `c7552`;
- `c6288`;
- full-scan `s38417_scan`.

Each circuit must have a matching fixed DeepGate embedding artifact and fault
map. DeepGate is not retrained by this workflow. The preparation script records
SHA-256 hashes for every circuit, fault map, and embedding file.

The original heuristic profiles every collapsed stuck-at fault independently
with a backtrack limit of 500. Each profile record stores outcome, backtracks,
backtrace steps, and a stable fault identifier. Faults are split within each
circuit and difficulty stratum so training and validation IDs never overlap.

Difficulty is deterministic and local to each circuit. Non-redundant faults are
sorted by `(backtracks, backtrace_steps, fault_id)` and divided by rank into:

- easy: lowest 40 percent;
- medium: next 40 percent;
- hard: highest 20 percent, including limit-reaching faults.

This percentile definition avoids empty stages on circuits whose absolute
backtrack distributions differ substantially. The manifest stores the exact
stratum and baseline metrics for every selected fault.

## C++ Teacher Action

`DecisionRequest` gains a `heuristic_action` field for backtrace decisions. It
is the candidate index that the unchanged original PODEM rule would select:

- easiest controllability selects the first currently unknown input;
- hardest controllability selects the last currently unknown input;
- the gate type and objective value choose between those two rules exactly as
  the existing non-RL `find_pi_assignment` implementation does.

The pybind request dictionary exposes this integer. A teacher callback records
the objective gate name, objective value, and heuristic action, then returns the
same action so collection follows the teacher. The existing path lock and
action mask remain active during collection. Native actor inference ignores the
new field, so its hot path and actor file format do not change.

The C++ helper computes the label from the same candidate vector presented to
the policy. Invalid or missing labels fail immediately rather than silently
producing corrupted supervision data.

## Heuristic Pretraining

The preparation phase collects teacher decisions from training faults only.
Duplicate `(circuit, objective gate, objective value)` states are collapsed
with action counts. Conflicting labels are retained as a two-class empirical
target distribution instead of being overwritten.

The V2 actor is initialized from scratch and trained with cross-entropy against
the teacher distribution. Data are balanced by circuit and difficulty stratum,
and the validation fault split is never used for gradient updates. The best
behavior-cloning checkpoint is selected by validation action accuracy, with
mean per-circuit accuracy used instead of a global sample count.

PPO starts from this pretrained actor and critic/RND parameters initialized
normally. Pretraining is resumable and stores optimizer, RNG, epoch, and best
policy state.

## Curriculum PPO

Training proceeds through three cumulative stages:

1. easy faults only;
2. easy and medium faults;
3. easy, medium, and hard faults.

Within a stage, circuits and strata are sampled evenly before fault order is
shuffled with a stored seed. Later stages continue replaying earlier strata to
avoid catastrophic forgetting. Stage sweep counts are explicit checkpointed
configuration values and cannot change on resume.

RND is training-only and decays by stage:

- easy: `rnd_beta = 0.05`;
- medium: `rnd_beta = 0.02`;
- hard: `rnd_beta = 0.0`.

The final stage therefore optimizes detection and search cost without a novelty
bonus that could reward unnecessary detours. PPO entropy follows the same
principle: exploration may be configured per stage but reaches its minimum in
the final stage.

## Baseline-Relative Reward

Every episode looks up the original heuristic profile for its fault. Let
`B_h`, `T_h` be heuristic backtracks and backtrace steps and `B_r`, `T_r` be the
RL values. The terminal extrinsic reward is:

```text
detection = +100 if detected, otherwise -100
backtrack_gain = 20 * clip((B_h - B_r) / max(B_h, 1), -2, 1)
backtrace_gain = 10 * clip((T_h - T_r) / max(T_h, 1), -2, 1)
reward = detection + backtrack_gain + backtrace_gain
```

Backtracks receive twice the weight of backtrace steps because a backtrack
usually repeats a larger portion of the search. Clipping prevents a single
pathological episode from dominating PPO while preserving improvement signs.
Detection remains the primary objective because the combined search gain is
bounded to `[-60, +30]` and cannot compensate for changing a detected fault to
an undetected one.

The V3 exponential PI reward and per-edge `-0.1` reward are not mixed into V4.
Using one terminal objective avoids double-counting the same backtrace and
backtrack costs. Discount factor is `1.0` for V4 so every action in the episode
receives the same terminal comparison signal; behavior cloning provides the
initial local credit assignment.

## Validation and Model Selection

Validation is deterministic actor argmax with no PPO, RND, optimizer, or
running-stat updates. It covers every circuit and all three validation strata
after behavior cloning and after every complete PPO sweep.

Models are ordered lexicographically by:

1. more detected validation faults;
2. fewer aborted validation faults;
3. lower mean per-circuit normalized backtrack ratio;
4. lower mean per-circuit normalized backtrace ratio;
5. fewer actor decisions.

For a circuit, normalized ratios compare aggregate RL counts with aggregate
heuristic baselines for the same validation fault IDs. Averaging circuit ratios
prevents `s38417_scan` from dominating smaller circuits. Exact ties keep the
earlier model.

The best model may come from behavior cloning or any curriculum sweep. Latest
and best actors are exported separately using the unchanged
`SMARTATPG_ACTOR_V2` native format.

## Files and Compatibility

Implementation adds new V4 preparation, pretraining, curriculum training, and
verification scripts rather than changing V3 command semantics. Shared Python
classes may gain baseline-aware helpers, but old constructors retain their
defaults.

The V4 manifest and checkpoint have new format identifiers and strict config
validation. Writes use temporary-file replacement. Resuming rejects changed
artifact hashes, splits, stage schedules, reward weights, RND schedules, or
model dimensions.

Artifacts are written under `artifacts/paper_v4_curriculum/`. Existing V3
artifacts are neither read as resumable V4 state nor deleted.

## Verification

Automated verification outside a `tests/` directory covers:

- C++ heuristic labels for every supported two-input gate type and objective
  value;
- pybind exposure and teacher callback replay;
- deterministic stratified splits with no training/validation overlap;
- behavior-cloning loss reduction, resume, and best-checkpoint selection;
- exact baseline-relative reward values and clipping boundaries;
- stage composition, balanced sampling, RND decay, and resume rejection;
- deterministic validation with no model or RND mutation;
- per-circuit normalized model ranking;
- Python/C++ V2 actor logit parity after export;
- a small end-to-end smoke run before formal training.

Formal training starts from a fresh V4 checkpoint only after all verification
and smoke checks pass. Final reporting compares detection, backtracks,
backtrace steps, and test-generation CPU time against the original heuristic on
all ten circuits, excluding artifact loading time.
