# Expanded XOR Fanin Fault Preservation Design

## Goal

Make the fault map for an expanded XOR follow the original collapsed line-fault
rule. An XOR input branch keeps its `GI-SA0` and `GI-SA1` faults when the
upstream wire has another logical fanout. When the XOR is the wire's only
logical load, those input faults remain collapsed into the upstream gate-output
fault.

The change does not add gates or wires to the BENCH circuit. The existing
five-gate NAND/NOT XOR implementation remains unchanged.

## Logical Fanout

Fanout is computed at the logical-cell boundary. The two physical connections
from one XOR input into its NAND/NOT implementation count together as one XOR
load. All consumers outside that expanded XOR count as separate loads.

For a logical XOR input `a`:

- XOR is the only logical consumer of `a`: do not add an XOR-input fault.
- `a` also feeds another gate or output branch: retain `a`'s XOR-branch
  `GI-SA0` and `GI-SA1` faults.

## Fault-map Representation

The converter derives the collapsed catalog from the source BENCH as before.
For each recognized expanded XOR, it identifies the source-catalog records for
each logical input branch and preserves the two stuck-at polarities only when
that input wire has logical fanout. Their external fault IDs use the logical XOR
output and input occurrence so they remain stable and distinguishable from the
upstream stem fault.

All faults belonging solely to the private `W`, `Z`, `X`, and `Y` expansion
nodes remain excluded. Each XOR output continues to retain exactly one
`GO-SA0` and one `GO-SA1` record with unit equivalent-fault weight. Fault-map
counts and uncollapsed totals are recomputed from the final records and checked
by the C++ loader.

## Verification

Regression checks use small expanded-XOR circuits to verify:

1. An input with no other logical consumer produces no XOR `GI` records.
2. An input with another logical consumer produces both XOR-branch `GI-SA0`
   and `GI-SA1` records, while the upstream `GO` records remain present.
3. The second XOR input is handled independently.
4. No unrelated private expansion fault survives.
5. XOR output `GO-SA0` and `GO-SA1` behavior remains unchanged.
6. Generated fault maps load successfully and report consistent collapsed and
   uncollapsed totals.

