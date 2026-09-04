# Full-Scan BENCH Conversion Design

## Goal

Convert the original sequential `sample_circuits/s38417.bench` into a
combinational full-scan benchmark at `sample_circuits/s38417_scan.bench`
without modifying the source benchmark.

## Conversion

For every record `q = DFF(d)`, remove the DFF record, declare `q` as a
primary input, and declare `d` as a primary output. Existing primary input and
primary output declarations are retained. Duplicate declarations are emitted
only once, while their first-seen order is preserved. All combinational gate
records and comments remain otherwise unchanged.

The converter must reject malformed DFF records and any case where a DFF
output is also driven by a remaining combinational gate. It must not silently
change gate or net names.

## Validation

After conversion:

- The output contains no DFF gate records.
- Every pseudo PI and pseudo PO is declared exactly once.
- The graph parser can build a directed acyclic graph with no
  unreachable nodes.
- The C++ PODEM executable can parse and level the converted circuit.
- The original `s38417.bench` remains byte-for-byte unchanged.

## Integration

The converter is a reusable Python command under `scripts/`. Embedding export
and PPO/RND training will use `s38417_scan.bench`; the raw
sequential benchmark remains archival input only.
