# Normal BENCH Subcircuit Training Dataset Design

## Goal

Create exactly 1,024 deterministic, unique, combinational BENCH circuits for
future reinforcement learning in the C++ PODEM project. Source circuits come
from the ISCAS'89 and ITC'99 benchmark suites. The output must preserve a
normal gate-level representation and must never pass through ABC, AIGER, an
AIG file, or an AIG-only graph representation.

DeepTPI code and data are out of scope. Its existing test dataset remains
unchanged and is not an input to this generator.

## Output

The generated dataset lives at:

`PODEM/datasets/iscas89_itc99_normal_1024/`

It contains:

- `circuits/`: exactly 1,024 files named `<suite>_<source>_<index>.bench`.
- `manifest.json`: generator settings, source provenance, source and output
  SHA-256 hashes, extraction boundaries, and per-circuit statistics.
- `summary.json`: aggregate source, gate-type, node-count, depth, input, and
  output distributions.

Raw downloaded archives and expanded source files live under
`PODEM/datasets/_sources/` and are not mixed with generated training files.

## Source Acquisition

The generator downloads the public benchmark archives from the CVUT digital
design benchmark collection:

- `https://ddd.fit.cvut.cz/www/prj/Benchmarks/ISCAS.7z`
- `https://ddd.fit.cvut.cz/www/prj/Benchmarks/ITC99.7z`

Existing original BENCH files under `PODEM/sample_circuits/` may be reused when
their circuit name and content hash are recorded in the manifest. Downloads
are cached. Re-running the generator does not fetch an archive that is already
present. Extraction accepts `7z`, `7zz`, or a local archive path supplied on
the command line.

The source inventory records suite, archive URL or local path, archive hash,
relative file path, and file hash. Generated and previously converted files
whose names contain `_aig`, `_binary`, or `_scan` are excluded from the source
inventory.

## Normalization

Each source BENCH file is parsed into a small internal circuit model. Names,
ports, gate type, fanins, and source line numbers are retained for diagnostics.

Sequential circuits are converted to a full-scan combinational model using the
existing `scripts/convert_full_scan_bench.py` behavior:

- A DFF output Q becomes a primary input.
- The corresponding D input becomes a primary output.
- The DFF itself and its sequential edge are removed.
- All combinational gates remain ordinary gates.

The existing `scripts/convert_binary_bench.py` rules then normalize circuits
for the SmartATPG graph loader:

- `BUFF` is normalized to `BUF`, and `INV` to `NOT`.
- Multi-input AND, NAND, OR, and NOR gates become deterministic binary trees
  of the same ordinary gate family.
- XOR, XNOR, and EQV are expanded using the project's existing normal-gate
  expansion rules.
- Final gate types are exactly `AND`, `NAND`, `OR`, `NOR`, `NOT`, and `BUF`.

No AIG conversion executable or AIG-only rewrite is permitted.

## Subcircuit Extraction

The normalized circuit is topologically sorted and assigned logic levels.
Candidate subcircuits are backward logic cones rooted at deterministic sink
gates. A candidate may cover at most 25 logic levels and at most 820 nodes.
Traversal stops at the depth or node limit.

Boundary handling is structural:

- A fanin entering the selected cone becomes a primary input.
- A selected signal used outside the cone becomes a primary output.
- A source primary input inside the cone remains a primary input.
- A source primary output inside the cone remains a primary output.
- Internal gate types and connections are unchanged.

Candidates with fewer than 80 nodes, no primary input, no primary output, an
undefined fanin, a duplicate driver, or a combinational cycle are rejected.
Overlapping cones are allowed because they provide distinct training problems.
Exact structural duplicates are removed using a canonical gate-and-edge hash
that ignores source-specific signal names.

## Selection

All operations use seed `208`. Source files and sink gates are processed in a
stable sorted order. Valid candidates are selected round-robin across source
circuits so one large benchmark cannot fill the dataset by itself. Selection
continues until exactly 1,024 unique circuits are chosen.

If fewer than 1,024 valid unique candidates exist, generation fails with a
per-source rejection summary. It never duplicates circuits to reach the
target.

## Compatibility

Every output file uses the C++ PODEM BENCH syntax:

```text
INPUT(a)
INPUT(b)
OUTPUT(z)
n1 = NAND(a,b)
z = BUF(n1)
```

The generator produces BENCH files only. It does not produce DeepTPI NPZ
files, labels, feature arrays, policies, faults, or training manifests.

## Validation

Generation succeeds only when all checks pass:

1. Exactly 1,024 `.bench` files exist in the output set.
2. Every file parses with `rl_podem.smartatpg_features.load_circuit_graph`.
3. Every circuit is acyclic and has complete drivers, inputs, and outputs.
4. Every non-input gate is binary, except unary `NOT` and `BUF`.
5. Every file contains 80 through 820 nodes and spans at most 25 logic levels.
6. All canonical structural hashes and output file hashes are unique.
7. Only `AND`, `NAND`, `OR`, `NOR`, `NOT`, and `BUF` occur in gate records.
8. At least one output circuit contains OR-family logic, demonstrating that
   the dataset is not constrained to AIG structure.
9. A stratified sample from both suites loads successfully through the C++
   PODEM executable.
10. Re-running with the same inputs and seed reproduces identical manifest and
    circuit hashes.

## Failure Behavior

Malformed records and unsupported cells report the source file and line
number. Download, archive extraction, parsing, scan conversion, candidate
shortage, validation, and C++ smoke-test failures stop the run with a nonzero
exit code. Output is built in a staging directory and replaces the final
dataset only after every validation passes, so a failed run cannot leave a
partial 1,024-file dataset.

## Project Changes

Implementation adds one dataset generator under `PODEM/scripts/`, focused unit
tests under `PODEM/tests/`, and the generated dataset directory. Existing
DeepTPI files and existing C++ PODEM sample circuits are not modified.
