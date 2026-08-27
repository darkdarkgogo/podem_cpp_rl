# Expanded XOR Fault Filter Design

## Goal

Keep the existing NAND/NOT implementation of commented two-input XOR cells for
PODEM search, but remove faults that exist only because each XOR was expanded
into implementation gates. The resulting fault set represents each expanded
XOR cell by faults on its original output wire, not by faults on its private
implementation nodes.

This is an isolation experiment. It changes fault selection and coverage
accounting without changing circuit logic, graph embeddings, PODEM backtrace
behavior, or the trained actor.

## XOR Cell Recognition

The BENCH source declares an expanded XOR with a comment and five active gates:

```text
# G223 = XOR(G203,G154)
W223 = NOT(G203)
Z223 = NOT(G154)
X223 = NAND(G203,Z223)
Y223 = NAND(G154,W223)
G223 = NAND(X223,Y223)
```

The converter recognizes a cell only when the comment and all five active gate
definitions match this structure, including signal names and connections. A
comment that resembles an XOR declaration but has a missing or mismatched
implementation is an error. This prevents a malformed source from silently
removing unrelated faults.

The four outputs `W223`, `Z223`, `X223`, and `Y223` are private implementation
wires. `G223` is the original logical XOR output wire.

## Fault Filtering Rules

For every recognized expanded XOR cell:

1. Remove every fault whose target gate is one of the private `W`, `Z`, `X`, or
   `Y` implementation gates. This removes their gate-output faults and any
   gate-input branch faults represented inside the expanded cell.
2. Keep the `GO-SA0` and `GO-SA1` records on the original XOR output, such as
   `G223`.
3. Normalize each retained XOR output record to `eqv_fault_num = 1`. The source
   collapsed catalog can otherwise carry internal equivalent faults in this
   weight even after their explicit records have been removed.
4. Keep faults on upstream original gate outputs. These are boundary wires and
   are not implementation-only nodes.
5. Keep downstream branch faults driven by the XOR output when the existing
   collapsed model requires them. They are outside the XOR implementation and
   can differ from an output-stem fault when the output has fan-out.

The converter recomputes the fault-map record count and uncollapsed total from
the filtered records. The C++ fault-map loader remains unchanged and continues
to validate both values.

## Other Multi-Input Gates

The existing binary conversion of multi-input AND, OR, NAND, and NOR gates uses
`__smartatpg_bin_*` nodes. Its fault map is generated from the original source
catalog, so those synthetic nodes already receive no independent fault records.
This behavior remains unchanged and is covered by regression checks.

Original gate-input branch faults are not globally removed. Only faults inside
a recognized XOR implementation are filtered. Changing the entire project to
an output-only fault model is outside this experiment.

## Artifacts And Compatibility

Regenerate fault maps for all binary circuits through the converter. Only
circuits containing recognized expanded XOR cells should change; currently
these are `c432` and `c499`. The binary BENCH files remain logically and
textually unchanged unless their generated headers require synchronization.

Existing DeepGate embeddings and actor weights remain usable because gate IDs
and topology do not change. Curriculum manifests and teacher data reference a
specific fault catalog and must be regenerated before any further training.
Old training and benchmark artifacts are retained for comparison.

Based on the current catalogs, the expected filtered collapsed counts are 452
records for `c432` and 534 records for `c499`. These values are regression-test
expectations rather than constants in production filtering logic.

## Verification

Verification must cover:

1. A valid expanded XOR is recognized and its circuit logic remains unchanged.
2. No generated fault record targets an XOR-private `W`, `Z`, `X`, or `Y` gate.
3. Every recognized XOR output has exactly one `GO-SA0` and one `GO-SA1`, each
   with `eqv_fault_num = 1`.
4. A malformed or partially matching expanded XOR causes conversion to fail.
5. No `__smartatpg_bin_*` node receives an independent fault record.
6. `c6288`, which contains no commented expanded XOR cells, retains the same
   fault catalog and acts as the control circuit.
7. The C++ loader accepts every regenerated map and reports totals matching its
   header.
8. Heuristic and current deterministic RL inference are benchmarked on `c432`,
   `c499`, and control circuit `c6288`, reporting detected, redundant, aborted,
   backtrack, and backtrace counts separately.

The benchmark comparison isolates the effect of removing implementation-only
faults. It does not claim equivalence to a native XOR PODEM implementation,
because search still traverses the NAND/NOT expansion.
