# Live TensorBoard Round Metrics Design

## Objective

Extend the Linux SmartATPG curriculum workflow so each completed curriculum
round immediately publishes deterministic training and validation metrics to
TensorBoard while retaining concise live console output. The feature must not
change training, evaluation, reward, checkpoint, or model-selection semantics.

## Metrics and Step

The TensorBoard global step is the one-based curriculum round number. After the
Hard stage of each round, reuse the existing deterministic evaluation over all
1,000 training faults and all 500 validation faults. Publish these six scalar
tags:

- `return/train_mean`
- `return/validation_mean`
- `backtracks/train_total`
- `backtracks/validation_total`
- `backtraces/train_total`
- `backtraces/validation_total`

Mean return is the existing micro-average extrinsic reward. It excludes RND and
does not update the policy, optimizer, rollout buffer, or RND model. Backtrack
and backtrace values are the existing aggregate totals for each split.

The console `ROUND_EVAL` line reports the same six values with explicit train
and validation names. The structured JSON and checkpoint histories retain their
existing schemas so historical tooling remains compatible.

## Writer Lifecycle and Output

`scripts/train_curriculum.py` accepts an optional `--tensorboard-log-dir` path.
When supplied, it creates a `torch.utils.tensorboard.SummaryWriter`, writes and
flushes all six scalars immediately after each successful round evaluation, and
closes the writer during normal or exceptional shutdown. Without the option,
legacy callers behave exactly as before and do not require TensorBoard.

`scripts/run_smartatpg_linux.py` supplies
`<output-dir>/tensorboard` automatically. The training Python environment must
have the `tensorboard` package installed; the Linux requirements file declares
that dependency. A separate environment may run the TensorBoard web server
against the generated files.

TensorBoard can be started before training and opened at its default local URL:

```bash
python -m tensorboard.main \
  --logdir artifacts/smartatpg_linux_20rounds/tensorboard
```

The first six points appear after round 1 evaluation completes. The dashboard
discovers later points as subsequent rounds flush their event data.

## Resume and Failure Semantics

TensorBoard event publication occurs before the round-complete checkpoint is
committed. On resume, the writer uses the first uncommitted round as its purge
step, so an event written immediately before a crash is replaced when that round
is evaluated again instead of producing a misleading duplicate branch.

If a pre-feature checkpoint already contains completed round evaluations and
the TensorBoard directory has no event files, the training script backfills the
six historical scalars once before continuing. If event files already exist,
the existing completed rounds are not rewritten.

Failure to import or initialize TensorBoard when `--tensorboard-log-dir` is
requested stops before new training work with an actionable installation error.
Non-finite scalar values remain rejected by the existing evaluation checks.

## Verification

Automated tests cover argument parsing, exact scalar tags and values, immediate
flush, writer closure, launcher log-directory wiring, initial historical
backfill, and interruption/resume purging. Existing round-evaluation and legacy
stage-sweep tests must continue to pass.

## Acceptance Criteria

During a 20-round Linux SmartATPG run, a user can open TensorBoard and observe
the training and validation mean-return curves plus total backtrack and
backtrace curves update once per completed round. Console values match the
TensorBoard values, resume does not lose completed training or create misleading
duplicate curve branches, and callers that omit TensorBoard retain their prior
behavior.
