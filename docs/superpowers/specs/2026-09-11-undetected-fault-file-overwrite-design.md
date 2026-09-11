# Undetected Fault File Overwrite Design

## Goal

Ensure every ATPG run writes a fresh `<circuit>.uf` undetected-fault report. A
previous run's records must not remain in the file, because accumulated records
can be mistaken for the current run's aborted or redundant fault counts.

## Scope

Change only `PODEM/src/display.cpp`, in `ATPG::display_undetect()`. Open the
output file in truncate/output mode instead of append mode. The report name,
record format, fault classification, and all other outputs remain unchanged.

This behavior applies uniformly to every path that calls `display_undetect()`,
including stuck-at ATPG, transition-delay ATPG, and fault-simulation output.

## Behavior

When `display_undetect()` opens `<circuit>.uf`, an existing file is truncated
before current undetected faults are written. If the current run has no
undetected faults, the result is an empty `.uf` file rather than stale content
from an earlier run. Existing file-open failure handling is retained.

## Verification

Add a regression test that exercises the file-open behavior twice and verifies
that the second report replaces the first instead of appending to it. Also run
the relevant automated tests and inspect the source to confirm no other output
stream was changed.

After this change, coverage diagnosis must use either the current run's native
summary or its freshly overwritten `.uf` file. Formal SmartATPG comparison
continues to use `-bt 2000`, converted binary/full-scan netlists, and the
corresponding source-preserving fault maps.
