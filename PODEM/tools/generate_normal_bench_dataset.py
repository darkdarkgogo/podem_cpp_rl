"""Generate scan-cut, binary normal-BENCH circuits for SmartATPG training.

The generator intentionally does not read DeepTPI's AIG/NPZ data.  It accepts
ordinary ISCAS-style BENCH source circuits, cuts DFFs into pseudo ports, lowers
XOR/XNOR into normal logic gates, and extracts bounded backward cones.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PORT_RE = re.compile(r"^\s*(INPUT|OUTPUT)\s*\(\s*([^()\s]+)\s*\)\s*$", re.I)
GATE_RE = re.compile(
    r"^\s*([^\s=(),]+)\s*=\s*([A-Za-z][A-Za-z0-9_]*)\s*\(([^()]*)\)\s*$"
)
VALID_NAME_RE = re.compile(r"^[A-Za-z0-9_.\[\]]+$")
SUPPORTED_GATES = {"AND", "NAND", "OR", "NOR", "NOT"}
COMMUTATIVE_GATES = {"AND", "NAND", "OR", "NOR"}


@dataclass(frozen=True)
class Gate:
    output: str
    kind: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class BenchCircuit:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    gates: tuple[Gate, ...]

    @property
    def by_output(self) -> dict[str, Gate]:
        return {gate.output: gate for gate in self.gates}


@dataclass(frozen=True)
class SourceCircuit:
    family: str
    name: str
    path: Path
    circuit: BenchCircuit
    sha256: str


@dataclass(frozen=True)
class ExtractedCircuit:
    source: SourceCircuit
    root: str
    circuit: BenchCircuit
    structural_hash: str
    logic_depth: int


def _append_unique(values: list[str], seen: set[str], value: str) -> None:
    if value not in seen:
        seen.add(value)
        values.append(value)


def _safe_name(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.\[\]]", "_", name)
    if not result:
        result = "n"
    return result


def _new_name(used: set[str], prefix: str) -> str:
    base = _safe_name(prefix)
    candidate = base
    number = 0
    while candidate in used:
        number += 1
        candidate = f"{base}_{number}"
    used.add(candidate)
    return candidate


def _gate_alias(kind: str) -> str:
    kind = kind.upper()
    return {"BUFF": "BUF", "EQV": "XNOR"}.get(kind, kind)


def parse_bench(path: Path) -> BenchCircuit:
    """Parse an ordinary BENCH file without changing its sequential semantics."""
    inputs: list[str] = []
    outputs: list[str] = []
    input_seen: set[str] = set()
    output_seen: set[str] = set()
    gate_outputs: set[str] = set()
    gates: list[Gate] = []

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        port = PORT_RE.fullmatch(line)
        if port:
            kind, name = port.groups()
            if not VALID_NAME_RE.fullmatch(name):
                raise ValueError(f"Invalid name at {path}:{line_number}: {name}")
            if kind.upper() == "INPUT":
                _append_unique(inputs, input_seen, name)
            else:
                _append_unique(outputs, output_seen, name)
            continue
        gate = GATE_RE.fullmatch(line)
        if not gate:
            raise ValueError(f"Malformed BENCH line at {path}:{line_number}: {raw}")
        output, kind, raw_inputs = gate.groups()
        inputs_for_gate = tuple(item.strip() for item in raw_inputs.split(",") if item.strip())
        if not VALID_NAME_RE.fullmatch(output) or any(
            not VALID_NAME_RE.fullmatch(item) for item in inputs_for_gate
        ):
            raise ValueError(f"Invalid wire name at {path}:{line_number}")
        if not inputs_for_gate:
            raise ValueError(f"Gate without inputs at {path}:{line_number}")
        if output in gate_outputs:
            raise ValueError(f"Duplicate driver {output} at {path}:{line_number}")
        gate_outputs.add(output)
        gates.append(Gate(output, _gate_alias(kind), inputs_for_gate))

    if not inputs or not outputs or not gates:
        raise ValueError(f"{path} must contain INPUT, OUTPUT, and gate records")
    return BenchCircuit(tuple(inputs), tuple(outputs), tuple(gates))


def parse_blif(path: Path) -> BenchCircuit:
    """Parse the combinational .names subset of BLIF into normal gate logic.

    EPFL's original circuits use this subset.  Sequential latches and hierarchy
    are rejected deliberately: they need an explicit scan-cut source netlist,
    rather than an implicit and potentially incorrect interpretation here.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines: list[str] = []
    pending = ""
    for raw in raw_lines:
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        if text.endswith("\\"):
            pending += text[:-1].rstrip() + " "
            continue
        lines.append((pending + text).strip())
        pending = ""
    if pending:
        raise ValueError(f"Unterminated BLIF line continuation in {path}")

    inputs: list[str] = []
    outputs: list[str] = []
    input_seen: set[str] = set()
    output_seen: set[str] = set()
    used: set[str] = set()
    gates: list[Gate] = []
    negated: dict[str, str] = {}
    constants: dict[int, str] = {}

    def add_ports(tokens: list[str], target: list[str], seen: set[str]) -> None:
        for name in tokens:
            if not VALID_NAME_RE.fullmatch(name):
                raise ValueError(f"Unsupported BLIF wire name in {path}: {name}")
            _append_unique(target, seen, name)
            used.add(name)

    def negation(name: str) -> str:
        if name not in negated:
            target = _new_name(used, f"__blif_not_{name}")
            gates.append(Gate(target, "NOT", (name,)))
            negated[name] = target
        return negated[name]

    def constant(value: int) -> str:
        if value in constants:
            return constants[value]
        if not inputs:
            raise ValueError(f"Cannot lower a constant-only BLIF circuit: {path}")
        zero = constants.get(0)
        if zero is None:
            anchor = inputs[0]
            zero = _new_name(used, "__blif_zero")
            gates.append(Gate(zero, "AND", (anchor, negation(anchor))))
            constants[0] = zero
        if value == 0:
            return zero
        one = _new_name(used, "__blif_one")
        gates.append(Gate(one, "NOT", (zero,)))
        constants[1] = one
        return one

    def reduce_gate(output: str, kind: str, wires: list[str]) -> None:
        if not wires:
            gates.append(Gate(output, "BUF", (constant(1 if kind == "AND" else 0),)))
            return
        if len(wires) == 1:
            gates.append(Gate(output, "BUF", (wires[0],)))
            return
        current = wires[0]
        for index, wire in enumerate(wires[1:], 1):
            target = output if index == len(wires) - 1 else _new_name(used, f"__blif_{output}_{kind.lower()}{index}")
            gates.append(Gate(target, kind, (current, wire)))
            current = target

    index = 0
    while index < len(lines):
        tokens = lines[index].split()
        command = tokens[0].lower()
        if command == ".model" or command == ".end":
            index += 1
            continue
        if command == ".inputs":
            add_ports(tokens[1:], inputs, input_seen)
            index += 1
            continue
        if command == ".outputs":
            add_ports(tokens[1:], outputs, output_seen)
            index += 1
            continue
        if command in {".latch", ".subckt", ".gate", ".blackbox"}:
            raise ValueError(f"Sequential or hierarchical BLIF is unsupported: {path}:{index + 1}")
        if command != ".names":
            raise ValueError(f"Unsupported BLIF command {tokens[0]} at {path}:{index + 1}")
        if len(tokens) < 2:
            raise ValueError(f"Malformed .names record at {path}:{index + 1}")
        names = tokens[1:]
        output = names[-1]
        fanins = names[:-1]
        if any(not VALID_NAME_RE.fullmatch(name) for name in names):
            raise ValueError(f"Unsupported BLIF wire name at {path}:{index + 1}")
        used.update(names)
        index += 1
        onset: list[str] = []
        while index < len(lines) and not lines[index].startswith("."):
            row = lines[index].split()
            if not fanins:
                if row == ["1"]:
                    onset.append("")
                elif row != ["0"]:
                    raise ValueError(f"Malformed constant .names row at {path}:{index + 1}")
            else:
                if len(row) != 2 or row[1] not in {"0", "1"}:
                    raise ValueError(f"Malformed .names row at {path}:{index + 1}")
                cube, value = row
                if len(cube) != len(fanins) or any(bit not in "01-" for bit in cube):
                    raise ValueError(f"Malformed cube at {path}:{index + 1}")
                if value == "1":
                    onset.append(cube)
            index += 1
        # Preserve standard two-input truth tables as ordinary gates.  This
        # keeps the BENCH output from collapsing every BLIF node into an
        # AND/NOT-only representation.
        onset_set = set(onset)
        direct_gate: tuple[str, tuple[str, ...]] | None = None
        if len(fanins) == 1:
            if onset_set == {"0"}:
                direct_gate = ("NOT", (fanins[0],))
            elif onset_set == {"1"}:
                direct_gate = ("BUF", (fanins[0],))
        elif len(fanins) == 2:
            if onset_set == {"11"}:
                direct_gate = ("AND", tuple(fanins))
            elif onset_set == {"00"}:
                direct_gate = ("NOR", tuple(fanins))
            elif onset_set == {"00", "01", "10"}:
                direct_gate = ("NAND", tuple(fanins))
            elif onset_set == {"01", "10", "11"}:
                direct_gate = ("OR", tuple(fanins))
        if direct_gate is not None:
            gates.append(Gate(output, *direct_gate))
            continue
        terms: list[str] = []
        for cube_number, cube in enumerate(onset):
            literals = [
                wire if bit == "1" else negation(wire)
                for wire, bit in zip(fanins, cube)
                if bit != "-"
            ]
            term = _new_name(used, f"__blif_{output}_term{cube_number}")
            reduce_gate(term, "AND", literals)
            terms.append(term)
        reduce_gate(output, "OR", terms)

    if not inputs or not outputs or not gates:
        raise ValueError(f"{path} must contain .inputs, .outputs, and .names records")
    return BenchCircuit(tuple(inputs), tuple(outputs), tuple(gates))


def _topological_order(circuit: BenchCircuit) -> BenchCircuit:
    by_output = circuit.by_output
    if len(by_output) != len(circuit.gates):
        raise ValueError("Circuit has conflicting gate outputs")
    input_set = set(circuit.inputs)
    if input_set & set(by_output):
        names = ", ".join(sorted(input_set & set(by_output)))
        raise ValueError(f"INPUT(s) also driven by gates: {names}")
    fanouts: dict[str, list[str]] = defaultdict(list)
    pending: dict[str, int] = {}
    for gate in circuit.gates:
        unknown = [wire for wire in gate.inputs if wire not in by_output and wire not in input_set]
        if unknown:
            raise ValueError("Undefined drivers: " + ", ".join(sorted(set(unknown))))
        pending[gate.output] = sum(wire in by_output for wire in gate.inputs)
        for wire in gate.inputs:
            if wire in by_output:
                fanouts[wire].append(gate.output)
    ready = [name for name, count in pending.items() if count == 0]
    heapq.heapify(ready)
    ordered: list[Gate] = []
    while ready:
        name = heapq.heappop(ready)
        ordered.append(by_output[name])
        for child in sorted(fanouts[name]):
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(circuit.gates):
        raise ValueError("Cycle detected in BENCH circuit")
    missing_outputs = set(circuit.outputs) - input_set - set(by_output)
    if missing_outputs:
        raise ValueError("OUTPUT(s) without drivers: " + ", ".join(sorted(missing_outputs)))
    return BenchCircuit(circuit.inputs, circuit.outputs, tuple(ordered))


def _emit_binary(
    output: str, kind: str, inputs: tuple[str, ...], used: set[str]
) -> list[Gate]:
    """Lower one BENCH gate while preserving its Boolean function."""
    if kind == "NOT":
        if len(inputs) != 1:
            raise ValueError(f"{kind}({output}) must have one input")
        return [Gate(output, kind, inputs)]
    if kind == "BUF":
        if len(inputs) != 1:
            raise ValueError(f"BUF({output}) must have one input")
        inverted = _new_name(used, f"__nb_{output}_buf_not")
        return [
            Gate(inverted, "NOT", inputs),
            Gate(output, "NOT", (inverted,)),
        ]
    if kind in {"AND", "OR"}:
        if len(inputs) < 2:
            raise ValueError(f"{kind}({output}) must have at least two inputs")
        gates: list[Gate] = []
        current = inputs[0]
        for index, item in enumerate(inputs[1:], 1):
            target = output if index == len(inputs) - 1 else _new_name(used, f"__nb_{output}_a{index}")
            gates.append(Gate(target, kind, (current, item)))
            current = target
        return gates
    if kind in {"NAND", "NOR"}:
        if len(inputs) < 2:
            raise ValueError(f"{kind}({output}) must have at least two inputs")
        base = "AND" if kind == "NAND" else "OR"
        if len(inputs) == 2:
            return [Gate(output, kind, inputs)]
        intermediate = _new_name(used, f"__nb_{output}_reduce")
        return _emit_binary(intermediate, base, inputs, used) + [Gate(output, kind, (intermediate, intermediate))]
    if kind in {"XOR", "XNOR"}:
        if len(inputs) != 2:
            raise ValueError(f"{kind}({output}) must have exactly two inputs")
        first, second = inputs
        not_first = _new_name(used, f"__nb_{output}_na")
        not_second = _new_name(used, f"__nb_{output}_nb")
        left = _new_name(used, f"__nb_{output}_left")
        right = _new_name(used, f"__nb_{output}_right")
        xor_out = output if kind == "XOR" else _new_name(used, f"__nb_{output}_xor")
        gates = [
            Gate(not_first, "NOT", (first,)),
            Gate(not_second, "NOT", (second,)),
            Gate(left, "AND", (first, not_second)),
            Gate(right, "AND", (not_first, second)),
            Gate(xor_out, "OR", (left, right)),
        ]
        if kind == "XNOR":
            gates.append(Gate(output, "NOT", (xor_out,)))
        return gates
    if kind == "DFF":
        raise AssertionError("DFF must be handled before logic lowering")
    raise ValueError(f"Unsupported BENCH gate type {kind}")


def normalize_bench(circuit: BenchCircuit) -> BenchCircuit:
    """Scan-cut DFFs and emit only binary SmartATPG-supported gates."""
    primary_inputs: list[str] = list(circuit.inputs)
    primary_outputs: list[str] = list(circuit.outputs)
    input_seen = set(primary_inputs)
    output_seen = set(primary_outputs)
    combinational: list[Gate] = []
    used = set(circuit.inputs) | set(circuit.outputs)
    used.update(gate.output for gate in circuit.gates)
    used.update(wire for gate in circuit.gates for wire in gate.inputs)

    for gate in circuit.gates:
        if gate.kind == "DFF":
            if len(gate.inputs) != 1:
                raise ValueError(f"DFF({gate.output}) must have one input")
            _append_unique(primary_inputs, input_seen, gate.output)
            _append_unique(primary_outputs, output_seen, gate.inputs[0])
            continue
        combinational.extend(_emit_binary(gate.output, gate.kind, gate.inputs, used))
    normalized = BenchCircuit(tuple(primary_inputs), tuple(primary_outputs), tuple(combinational))
    ordered = _topological_order(normalized)
    for gate in ordered.gates:
        if gate.kind not in SUPPORTED_GATES:
            raise AssertionError(f"Lowering leaked unsupported gate {gate.kind}")
        expected = 1 if gate.kind == "NOT" else 2
        if len(gate.inputs) != expected:
            raise AssertionError(f"Lowering emitted non-binary {gate.kind}")
    return ordered


def _levels(circuit: BenchCircuit) -> dict[str, int]:
    values = {name: 0 for name in circuit.inputs}
    for gate in circuit.gates:
        values[gate.output] = 1 + max(values[name] for name in gate.inputs)
    return values


def _circuit_analysis(
    circuit: BenchCircuit,
) -> tuple[dict[str, Gate], dict[str, int], dict[str, list[str]]]:
    by_output = circuit.by_output
    levels = _levels(circuit)
    fanouts: dict[str, list[str]] = defaultdict(list)
    for gate in circuit.gates:
        for wire in gate.inputs:
            fanouts[wire].append(gate.output)
    return by_output, levels, fanouts


def structural_hash(circuit: BenchCircuit) -> str:
    """Return a name-independent hash, preserving gate topology and outputs."""
    by_output = circuit.by_output
    cache: dict[str, str] = {}
    fanout_count: dict[str, int] = defaultdict(int)
    for gate in circuit.gates:
        for wire in gate.inputs:
            fanout_count[wire] += 1
    for output in circuit.outputs:
        fanout_count[output] += 1
    output_set = set(circuit.outputs)

    def signature(name: str) -> str:
        if name in cache:
            return cache[name]
        if name not in by_output:
            # Fanout and output membership retain DAG sharing information.  In
            # particular AND(a,a) must not collide with AND(a,b).
            payload = f"PI[{fanout_count[name]},{int(name in output_set)}]"
        else:
            gate = by_output[name]
            parts = [signature(item) for item in gate.inputs]
            if gate.kind in COMMUTATIVE_GATES:
                parts.sort()
            payload = (
                f"{gate.kind}[{fanout_count[name]},{int(name in output_set)}]"
                f"({','.join(parts)})"
            )
        # Keep each recursive result fixed-size.  A textual expansion of a
        # shared DAG can otherwise grow exponentially for large cones.
        cache[name] = hashlib.sha256(payload.encode("ascii")).hexdigest()
        return cache[name]

    payload = "|".join(sorted(signature(name) for name in circuit.outputs))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def extract_cone(
    source: SourceCircuit, root: str, min_nodes: int = 80, max_nodes: int = 820,
    max_levels: int = 25,
    analysis: tuple[dict[str, Gate], dict[str, int], dict[str, list[str]]] | None = None,
) -> ExtractedCircuit | None:
    """Extract a backward cone, turning every cut edge into a pseudo input."""
    circuit = source.circuit
    by_output, levels, fanouts = analysis or _circuit_analysis(circuit)
    if root not in by_output:
        raise ValueError(f"Unknown extraction root {root}")
    selected = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for wire in by_output[current].inputs:
            if wire not in by_output or wire in selected:
                continue
            if levels[root] - levels[wire] >= max_levels:
                continue
            # Even without counting cut-boundary PIs, this cone can no longer
            # satisfy the inclusive node upper bound.
            if len(selected) >= max_nodes:
                return None
            selected.add(wire)
            queue.append(wire)

    input_set = set(circuit.inputs)
    boundary = {
        wire
        for output in selected
        for wire in by_output[output].inputs
        if wire not in selected
    }
    if not boundary <= input_set | set(by_output):
        raise AssertionError("Unexpected extraction boundary")
    total_nodes = len(selected) + len(boundary)
    if not min_nodes <= total_nodes <= max_nodes:
        return None

    outputs = [
        gate.output for gate in circuit.gates
        if gate.output in selected
        and (gate.output == root or any(child not in selected for child in fanouts[gate.output]))
    ]
    extracted = BenchCircuit(
        tuple(sorted(boundary)), tuple(sorted(set(outputs))),
        tuple(gate for gate in circuit.gates if gate.output in selected),
    )
    extracted = _topological_order(extracted)
    return ExtractedCircuit(
        source=source,
        root=root,
        circuit=extracted,
        structural_hash=structural_hash(extracted),
        logic_depth=max(_levels(extracted).values()),
    )


def write_bench(circuit: BenchCircuit, destination: Path, header: Iterable[str] = ()) -> None:
    for name in (*circuit.inputs, *circuit.outputs, *(gate.output for gate in circuit.gates)):
        if not VALID_NAME_RE.fullmatch(name):
            raise ValueError(f"Cannot write unsupported PODEM name: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {line}" for line in header]
    lines.extend(f"INPUT({name})" for name in circuit.inputs)
    lines.extend(f"OUTPUT({name})" for name in circuit.outputs)
    lines.extend(
        f"{gate.output} = {gate.kind}({','.join(gate.inputs)})" for gate in circuit.gates
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(family: str, path: Path) -> SourceCircuit:
    suffix = path.suffix.lower()
    if suffix == ".bench":
        parsed = parse_bench(path)
    elif suffix == ".blif":
        parsed = parse_blif(path)
    else:
        raise ValueError(f"Source must be .bench or combinational .blif, got {path}")
    return SourceCircuit(
        family=family.upper(), name=path.stem, path=path.resolve(),
        circuit=normalize_bench(parsed), sha256=_source_hash(path),
    )


def _round_robin_extract(
    sources: list[SourceCircuit], target: int, seed: int, min_nodes: int, max_nodes: int,
    max_levels: int,
) -> list[ExtractedCircuit]:
    if target < 1:
        raise ValueError("target must be positive")
    buckets: dict[str, deque[str]] = {}
    # SourceCircuit includes the full gate tuple, so using it as a dictionary
    # key would rehash an entire large source circuit for every candidate.
    analysis = {id(source): _circuit_analysis(source.circuit) for source in sources}
    for source in sources:
        roots = [gate.output for gate in source.circuit.gates]
        random.Random(f"{seed}:{source.name}").shuffle(roots)
        buckets[source.name] = deque(roots)
    family_sources: dict[str, list[SourceCircuit]] = defaultdict(list)
    for source in sorted(sources, key=lambda item: (item.family, item.name)):
        family_sources[source.family].append(source)
    families = sorted(family_sources)
    if {"ISCAS89", "ITC99"} - set(families):
        raise ValueError("Training sources must include ISCAS89 and ITC99")

    result: list[ExtractedCircuit] = []
    seen: set[str] = set()
    source_offsets = {family: 0 for family in families}
    while len(result) < target:
        progress = False
        for family in families:
            group = family_sources[family]
            source = group[source_offsets[family] % len(group)]
            source_offsets[family] += 1
            while buckets[source.name]:
                candidate = extract_cone(
                    source, buckets[source.name].popleft(), min_nodes, max_nodes, max_levels,
                    analysis[id(source)],
                )
                if candidate is None or candidate.structural_hash in seen:
                    continue
                seen.add(candidate.structural_hash)
                result.append(candidate)
                progress = True
                break
            if len(result) == target:
                break
        if not progress:
            raise RuntimeError(
                f"Only extracted {len(result)} unique circuits; need {target}. "
                "Add larger source circuits or loosen extraction bounds."
            )
    return result


def generate_dataset(
    train_sources: list[SourceCircuit], validation_sources: dict[str, SourceCircuit],
    output_root: Path, train_count: int = 1024, seed: int = 208,
    min_nodes: int = 80, max_nodes: int = 820, max_levels: int = 25,
) -> dict:
    output_root = output_root.resolve()
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    if train_dir.exists() and any(train_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {train_dir}")
    if validation_dir.exists() and any(validation_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty {validation_dir}")

    extracted = _round_robin_extract(
        train_sources, train_count, seed, min_nodes, max_nodes, max_levels
    )
    train_manifest = []
    for index, item in enumerate(extracted, 1):
        filename = f"train_{index:04d}_{item.source.family.lower()}_{item.source.name}_{item.root}.bench"
        write_bench(item.circuit, train_dir / filename, [
            "Normal BENCH subcircuit; sequential elements are cut to ports.",
            f"Source: {item.source.name} ({item.source.family})", f"Root: {item.root}",
        ])
        train_manifest.append({
            "file": filename, "family": item.source.family, "source": item.source.name,
            "root": item.root, "nodes": len(item.circuit.inputs) + len(item.circuit.gates),
            "logic_depth": item.logic_depth, "structural_hash": item.structural_hash,
        })
    validation_manifest = []
    for name, source in sorted(validation_sources.items()):
        filename = f"{name}.bench"
        write_bench(source.circuit, validation_dir / filename, [
            "Normal BENCH validation circuit; sequential elements are cut to ports.",
            f"Source: {source.name} ({source.family})",
        ])
        validation_manifest.append({
            "file": filename, "name": name, "family": source.family,
            "source": source.name, "nodes": len(source.circuit.inputs) + len(source.circuit.gates),
        })
    manifest = {
        "format": "normal-bench-v1", "seed": seed, "train_count": train_count,
        "bounds": {"min_nodes": min_nodes, "max_nodes": max_nodes, "max_levels": max_levels},
        "sources": [
            {"name": source.name, "family": source.family, "path": str(source.path), "sha256": source.sha256}
            for source in sorted({*train_sources, *validation_sources.values()}, key=lambda item: (item.family, item.name))
        ],
        "train": train_manifest, "validation": validation_manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _parse_assignment(value: str, option: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"{option} must be NAME=PATH")
    name, text_path = value.split("=", 1)
    if not name or not text_path:
        raise ValueError(f"{option} must be NAME=PATH")
    return name, Path(text_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--train-source", action="append", default=[], metavar="FAMILY=PATH",
        help="Repeat for normal BENCH sources; FAMILY must be ISCAS89 or ITC99.",
    )
    parser.add_argument(
        "--validation-source", action="append", default=[], metavar="NAME=PATH",
        help="Repeat for each validation source BENCH.",
    )
    parser.add_argument("--train-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=208)
    parser.add_argument("--min-nodes", type=int, default=80)
    parser.add_argument("--max-nodes", type=int, default=820)
    parser.add_argument("--max-levels", type=int, default=25)
    args = parser.parse_args()
    train_sources = []
    for value in args.train_source:
        family, path = _parse_assignment(value, "--train-source")
        if family.upper() not in {"ISCAS89", "ITC99"}:
            raise ValueError("--train-source family must be ISCAS89 or ITC99")
        train_sources.append(load_source(family, path))
    validation_sources = {
        name: load_source("VALIDATION", path)
        for name, path in (_parse_assignment(value, "--validation-source") for value in args.validation_source)
    }
    manifest = generate_dataset(
        train_sources, validation_sources, args.output_root, args.train_count, args.seed,
        args.min_nodes, args.max_nodes, args.max_levels,
    )
    print(f"Generated {len(manifest['train'])} training BENCH circuits")
    print(f"Generated {len(manifest['validation'])} validation BENCH circuits")
    print(f"Manifest: {(args.output_root / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()
