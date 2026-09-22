# SCOAP Validation Batch Design

## Goal

Run the final comparison SCOAP baseline once per validation circuit instead of once per fault while preserving fault order, PODEM decisions, reward values, and the existing comparison record schema.

## Scope

- Add a native C++ SCOAP validation entry point that accepts all requested fault IDs for one circuit.
- Load, levelize, rearrange, create dummy gates, and generate the complete fault list once per circuit.
- Execute retained faults in catalog order with detected-fault dropping disabled.
- Keep `backtrack_limit=100`, `backtrace_rl`, SCOAP features, and the current reward schemes.
- Preserve the existing per-fault record fields and summary calculations.
- Add circuit-qualified progress output every 1000 faults and at circuit completion.
- Add an end-to-end wall-time log per circuit because existing `atpg_seconds` intentionally excludes circuit initialization.
- Do not add checkpoint or resume behavior for the SCOAP comparison baseline.

## Architecture

Introduce a `NativeHeuristicPolicy` implementing `smartatpg::DecisionPolicy`. Its `select()` method returns `request.heuristic_action`, and `needs_gate_names()` returns `false` because SCOAP selection does not consume names.

Reuse the existing `NativeValidationPolicy` as the recording wrapper. This preserves the current reward implementation, validation rules, per-fault timing, journaling behavior, and progress accounting without creating a second copy of that logic. Extend its progress prefix so the SCOAP entry point can emit `SCOAP_VALIDATE circuit=<name> completed=<n>/<total>` while learned validation retains its current output.

Add `run_native_scoap_validation(circuit_path, backtrack_limit, seed, fault_ids, reward_scheme, circuit_name)` to `python_bindings.cpp`. It constructs one `ATPG`, configures SCOAP and `backtrace_rl`, installs the heuristic policy through `NativeValidationPolicy`, initializes the circuit once, retains the complete requested fault list, disables detected-fault dropping, and calls `test()` once.

The recording wrapper must call `std::srand(seed_)` in every `on_episode_start()`. This matches the old one-`test()`-per-fault behavior and keeps random fill reproducible across batching.

## Python Integration

Replace the per-fault list comprehension in `_load_or_run_scoap()` with one native call per circuit. Normalize each native result into the existing comparison record schema and verify the returned count and order before extending the aggregate record list.

Keep `SMARTATPG_SCOAP_VALIDATION_V1` and the persisted JSON schema unchanged. Add a `force` argument to `_load_or_run_scoap()` and a `--force-scoap` command-line option so acceptance and benchmarking can bypass a valid existing cache without deleting it manually.

Print one circuit-level completion line containing the full wall time, for example:

```text
SCOAP_CIRCUIT_DONE circuit=b15_C faults=25910 wall_s=1234.567
```

The existing per-fault `atpg_seconds` remains defined by native validation intervals and therefore excludes circuit parsing and fault-list generation.

## Error Handling

- Reject an empty fault list.
- Reject a backtrack limit other than 100.
- Reject unknown reward schemes.
- Fail if the native extension lacks `run_native_scoap_validation`, with an explicit rebuild instruction.
- Fail if result count or fault order differs from the request.
- Reject non-finite reward or timing values.
- Preserve existing cache identity validation unless `--force-scoap` is supplied.

## Testing

- Compare native batched SCOAP results against the current Python-callback, one-fault-at-a-time reference for both reward schemes.
- Assert exact equality for fault ID, outcome, backtracks, and backtrace steps, and floating-point equality for return.
- Assert result order and non-negative per-fault time.
- Verify invalid reward schemes, invalid backtrack limits, and empty fault lists are rejected.
- Verify Python normalization preserves the current record schema.
- Verify `_load_or_run_scoap()` calls the native interface once per circuit and preserves global validation order.
- Verify a valid cache is reused normally and bypassed when force mode is enabled.
- Verify the CLI forwards `--force-scoap`.

## Non-goals

- Changing GAT or MEAN training behavior.
- Changing PODEM, SCOAP scoring, reward formulas, or fault catalogs.
- Adding SCOAP checkpoint or resume state.
- Changing the comparison JSON/CSV schema.
