# Hard-Detected Fault Selection

The approved training set contains 100 faults from c6288 and 100 from
full-scan s38417, shared by the baseline and level-GAT-GRU agents.

Keep the default heuristic profiling backtrack limit at 500. Select only
profiles with outcome 1 (detected), then rank by descending backtracks,
descending backtrace steps, and ascending fault ID. Fail explicitly if
fewer than 100 detected faults are available in either circuit. Do not
substitute aborted or untestable faults.

Use manifest V2 with a baseline_detected_only filter marker. Validate
the selected records and IDs against the original profiles at preparation
resume and training entry. Reject old manifests; a new output directory
preserves prior results and prevents incompatible checkpoint reuse.

This change does not alter PPO failure updates, evaluation fault catalogs,
or graph/Actor architecture. A baseline-detected fault can still fail under
the learned policy. Regression tests cover filtering, ranking, insufficient
eligible faults, valid resume, altered selections, and legacy rejection.
