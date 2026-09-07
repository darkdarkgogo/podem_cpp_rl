# SmartATPG Actor Timing Design

## Goal

Measure the native Actor decision cost separately from ATPG search cost for
`smartatpg_mean` and `smartatpg_gat_gru`, and report offline embedding
preparation time beside those measurements.

## Native inference behavior

`smartatpg_mean` must evaluate its Actor on every eligible RL backtrace
decision. Its native logit cache is disabled. `smartatpg_gat_gru` keeps the
existing cache keyed by objective gate and objective value.

`NativeActorPolicy` records:

- total calls to `select()`;
- complete time spent in `select()`;
- calls that execute `ActorModel::backtrace_action_logits_into()`;
- time spent only in that Actor forward function.

The complete selection timer includes validation, cache lookup, input
preparation, Actor execution when needed, masking, and action selection. The
Actor timer excludes embedding generation and all other ATPG work. Timing uses
`std::chrono::steady_clock` and is printed in seconds for machine parsing and
microseconds per call for human inspection.

## Benchmark aggregation

The benchmark parser records the native counters and durations. As with ATPG
time, it takes the median duration for each circuit across measured repeats,
then sums circuit durations. Overall mean time is calculated as summed median
duration divided by the summed deterministic call count.

The report adds a native inference table containing selection calls, Actor
forward calls, cache hits, average selection time, and average Actor forward
time. Cache hits are derived as selection calls minus Actor forward calls.

Existing `graph_embedding` measurements in `preprocessing.json` are summed per
model and added to the final JSON and Markdown report as embedding preparation
seconds. The shared graph feature construction time remains a separate value.

## Validation

Tests cover native timing parsing and aggregation, embedding-time aggregation,
the report fields, and source-level cache policy. The native code is compiled
where the available environment supports it; Python unit tests and syntax
checks run in the existing project environment.
