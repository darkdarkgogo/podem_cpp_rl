# GAT Failure-Fault Reinforcement Design

## Scope

After the normal 20-round training finishes, reinforce only the `level_gat_gru`
model. The `fanin_mean` model and its artifacts remain unchanged.

The reinforcement candidates are restricted to the original 200 training faults.
Every candidate was detected by the heuristic baseline within the configured
backtrack limit, but is not detected by the best GAT checkpoint under the same
limit. Benchmark faults outside this training set must not enter reinforcement.

## Training Flow

1. Restore the full best GAT training state, including policy, critic, PPO
   optimizer, RND model, RND optimizer, and normalization statistics.
2. Evaluate the restored model on all 200 training faults and form the current
   unresolved set.
3. Run five reinforcement rounds. In each round, train each fault in the current
   unresolved set exactly once, in a deterministic shuffled order.
4. After each round, evaluate all 200 faults again. The next round uses only the
   faults that remain unresolved in that evaluation.
5. If no faults remain unresolved, stop early because subsequent rounds would
   contain no training episodes.

## Model Selection And Artifacts

Select the reinforced checkpoint by the existing validation order: maximize the
number of detected faults first, then minimize total backtracks, backtrace steps,
and finally maximize return. Compare every reinforced candidate with the original
best GAT model, so reinforcement cannot replace it with a worse all-200 result.

Keep the normal GAT artifacts intact. Write the best reinforced full checkpoint,
portable actor, per-round metrics, and unresolved-fault lists as separate files.
The benchmark bundle uses the reinforced actor when reinforcement completed
successfully; otherwise it falls back to the original best GAT actor.

## Resume And Verification

The reinforcement stage is resumable at fault granularity and validates the
manifest hash, encoder type, backtrack limit, and source best-checkpoint identity.
Automated tests cover residual-set extraction, one-visit-per-round scheduling,
dynamic shrinking, all-200 checkpoint selection, early completion, and the rule
that mean training is never reinforced.
