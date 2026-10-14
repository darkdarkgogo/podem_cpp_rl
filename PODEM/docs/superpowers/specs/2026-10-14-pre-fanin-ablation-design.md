# Pre-fanin GAT ablation design

Use the GAT-GRU policy at commit `7d92ab8`: its two-output Actor scores the objective gate embedding and objective value, without a candidate fanin embedding. Keep this checkout separate from the current fanin scorer.

Read the existing `full` baseline training and validation artifacts. Train only two model/PODEM variants from the same prepared GAT manifest, seed, reward, backtrack limit, and two-round training protocol. `no_lock` disables the backtrace path lock in training and validation. `no_object_value` removes the objective value from Actor and Critic inputs while PODEM still uses it for logic. `no_cache` is a validation-only run of the *same full checkpoint and actor*, with native Actor logits caching disabled; it does not retrain a second network.

Validation reads the existing full group's per-fault results and runs the three ablations on one ordered validation fault catalog. It records per-fault outcomes and compares per-circuit and total detected, redundant, aborted, coverage, backtracks, backtrace steps, and ATPG time. The runner rejects mismatched manifest, seed, checkpoint variant, path-lock state, or full/no-cache actor snapshot. Training and validation are separate explicit commands and use isolated artifact directories. No experiment is launched automatically by implementation or tests.
