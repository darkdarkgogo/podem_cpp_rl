# GAT Backtrack Reward Design

## Scope

Change the `level_gat_gru` SmartATPG reward to the bounded nonlinear backtrack reward defined in `SmartATPG_Backtrack_Reward_Design_CN.docx`. Keep the `fanin_mean` reward behavior unchanged. Change the formal SmartATPG backtrack limit from 200 to 100 for both encoders.

Do not add reward variants, experiment switches, ablation runs, training runs, or benchmark runs. Verification is limited to focused unit tests, static checks, and the native extension build/tests needed to establish implementation correctness.

## Reward protocols

The implementation exposes two named reward schemes:

- `cubic_backtrack_v1` for `level_gat_gru`.
- `legacy_pi_exponential` for `fanin_mean`.

The encoder determines the scheme. Callers must not select an arbitrary scheme that conflicts with the encoder metadata.

Both schemes retain the existing RL-associated backtrace-step reward of `-0.1` and terminal rewards of `+100` for detected faults and `-100` otherwise.

### GAT reward

For GAT, each episode starts with backtrack count zero. The `B`th policy-associated backtrack, where `1 <= B <= 100`, produces:

```text
r_bt(B) = -(0.5 + 9.802960494 * (B / 100.0)^3)
```

The reward is assigned to the decision step identified by the backtrack event. `pi_not_done` remains a valid event but contributes no reward. A backtrack count above 100 is a protocol error.

The sum of the first 100 backtrack penalties must equal `-300` within `1e-6`.

### Mean reward

For mean, backtrack events continue to contribute no reward. `pi_not_done` continues to use the existing reward:

```text
10.0 - 7.5 * exp(0.07 * (backtracks + pi_visits))
```

No clipping, replacement, or other numerical change is applied to the mean reward. The only behavioral protocol change for mean is the backtrack limit changing from 200 to 100. Existing finite-number checks remain strict and may reject a non-finite mean validation return.

## Shared implementation boundaries

The Python trainer receives the reward scheme from the agent encoder variant. Its event handler applies either the GAT backtrack reward or the legacy mean PI reward, never both.

Native C++ validation receives the encoder-compatible reward scheme explicitly and mirrors the Python event accounting. The native journal and returned per-fault record must reject non-finite returns before persistence.

Python model validation and SCOAP validation use the same reward implementation and event semantics as the model being evaluated. Shared helpers centralize constants and reward calculations so Python training and Python validation cannot drift.

## Protocol identity and compatibility

The formal SmartATPG `backtrack_limit` is 100 in training-manifest preparation, Linux launchers, checkpoint configuration, validation identity, exported model metadata, portable/native model loading, benchmark preparation, and comparison validation.

Checkpoint, validation, and exported-model metadata record the reward scheme. Resume and artifact loading require the scheme to match the encoder. Existing artifacts with backtrack limit 200 are incompatible with the new protocol and must fail with a clear identity or metadata error rather than being silently reused.

## Metrics and comparison

Per-fault return, `return_total`, and `return_mean` must be finite before they are written. Existing comparison-side finite checks remain unchanged.

Returns remain part of each model's validation metrics and best-round score. Because GAT and mean use different reward definitions, cross-model and model-versus-SCOAP report rows must not subtract or otherwise compare return values. Coverage, backtracks, backtrace steps, and ATPG runtime remain directly comparable.

Reward-composition logging records backtrace-step penalty, GAT backtrack penalty, legacy PI reward, and terminal reward separately without changing the PPO combined-reward calculation or the existing RND coefficient.

## Verification

Focused tests cover:

- The exact GAT formula at representative counts and a cumulative penalty of `-300 +/- 1e-6` at 100 backtracks.
- Strict rejection of GAT backtrack counts outside 1 through 100.
- GAT event handling rewards backtracks and ignores `pi_not_done` for reward purposes.
- Mean event handling ignores backtracks and preserves the legacy `pi_not_done` reward.
- Python training, Python validation, and native C++ validation produce the same extrinsic return for equivalent event traces under each scheme.
- All formal protocol validators require backtrack limit 100 and correct encoder-to-reward-scheme metadata.
- Validation persistence rejects non-finite per-fault and aggregate returns.
- Comparison outputs do not calculate return deltas across different reward schemes.

No training, benchmark, reward comparison, or ablation experiment is part of verification.
