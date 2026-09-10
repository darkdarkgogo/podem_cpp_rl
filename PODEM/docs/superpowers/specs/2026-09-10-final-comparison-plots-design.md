# Final Comparison Plot Design

## Goal

Add one Python command that reads the existing `final_comparison.csv` report
without conversion and writes four independent grouped bar charts. The charts
compare all reported models for every circuit using backtrace steps,
backtracks, native ATPG runtime, and fault coverage.

## Input Contract

The command accepts the path to `final_comparison.csv` as a positional
argument. The CSV is the long-form report produced by
`scripts/benchmark_smartatpg.py`; each row represents one `circuit` and
`model` pair. The plotter requires these columns:

- `circuit`
- `model`
- `backtrace_steps`
- `backtracks`
- `atpg_seconds`
- `fault_coverage`

The required model values are `heuristic`, `smartatpg_mean`, and
`smartatpg_gat_gru`. Every circuit must have one row for each required model.
Their order and colors remain consistent across all four figures. Unknown
models are plotted after the required models when they are present for every
circuit, keeping the plotter compatible with extended benchmark reports.

## Command and Outputs

The implementation adds `scripts/plot_final_comparison.py` with this interface:

```text
python scripts/plot_final_comparison.py FINAL_COMPARISON_CSV [--output-dir DIR]
```

When `--output-dir` is omitted, images are written beside the input CSV. One
run writes these high-resolution PNG files:

- `backtrace_steps_by_circuit.png`
- `backtracks_by_circuit.png`
- `runtime_by_circuit.png`
- `fault_coverage_by_circuit.png`

Existing files with these names are replaced because they are derived outputs
of the supplied CSV.

## Plot Behavior

Each figure uses circuit names on the x-axis and one adjacent bar per model.
The backtrace, backtrack, and runtime figures use the numeric CSV values
directly. Runtime comes from `atpg_seconds`, which measures the native C++ ATPG
interval and excludes graph embedding and orchestration. Fault coverage is
converted from its stored zero-to-one ratio to a percentage and plotted on a
zero-to-100-percent axis.

The plot width grows with the number of circuits, x-axis labels rotate when
needed, and layout padding prevents label clipping. Titles, units, legends,
model colors, and grid styling are shared across figures. The script uses the
standard-library `csv` module for input and Matplotlib for plotting, avoiding a
Pandas dependency.

## Validation and Errors

The command fails with a concise error when the input does not exist, is empty,
lacks a required column, contains a non-numeric metric, duplicates a
`circuit`/`model` pair, or omits a model for a circuit. It creates the output
directory when necessary.

Automated tests use a small temporary CSV matching the benchmark schema. They
verify that all four PNG files are created, coverage is converted to percent,
and malformed input reports a useful error. A command-line smoke test confirms
the script runs from the repository root with a non-interactive Matplotlib
backend.
