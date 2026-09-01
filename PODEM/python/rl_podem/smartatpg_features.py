"""Deterministic structural inputs for the trainable SmartATPG backend."""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

FEATURE_SCHEMA = "SMARTATPG_FEATURES_V1"
GATE_TYPES = ("PI", "AND", "NAND", "OR", "NOR", "NOT", "BUF", "XOR", "XNOR")
FEATURE_DIM = len(GATE_TYPES) + 5
COST_CAP = 10**9
GRAPH_CONFIG = {"layers": 2, "hidden_dim": 64, "aggregation": "fanin_mean"}
PORT_RE = re.compile(r"^(INPUT|OUTPUT)\s*\(\s*([^()\s]+)\s*\)$", re.I)
GATE_RE = re.compile(r"^([^\s=(),]+)\s*=\s*(\w+)\s*\(([^()]*)\)$")


def circuit_hash(path):
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


@dataclass(frozen=True)
class CircuitGraph:
    circuit_hash: str
    names: tuple[str, ...]
    gate_types: tuple[str, ...]
    fanins: tuple[tuple[int, ...], ...]
    x: torch.Tensor
    edge_index: torch.Tensor
    levels: tuple[int, ...]
    fanouts: tuple[int, ...]
    cc0: tuple[float, ...]
    cc1: tuple[float, ...]
    co: tuple[float, ...]

    @property
    def name_to_index(self):
        return {name: index for index, name in enumerate(self.names)}


def _bounded(value):
    return min(COST_CAP, value) if math.isfinite(value) else value


def load_circuit_graph(path):
    path = Path(path)
    drivers, outputs = {}, set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        port = PORT_RE.fullmatch(line)
        if port:
            kind, name = port.groups()
            if kind.upper() == "OUTPUT":
                if name in outputs:
                    raise ValueError(f"Duplicate OUTPUT {name} at line {line_number}")
                outputs.add(name)
                continue
            gate_type, inputs = "PI", ()
        else:
            gate = GATE_RE.fullmatch(line)
            if gate is None:
                raise ValueError(f"Malformed BENCH line {line_number}: {line}")
            name, gate_type, input_text = gate.groups()
            gate_type = {"BUFF": "BUF", "EQV": "XNOR"}.get(gate_type.upper(), gate_type.upper())
            inputs = tuple(value.strip() for value in input_text.split(","))
            if gate_type not in GATE_TYPES[1:]:
                raise ValueError(f"Unsupported gate type {gate_type} at line {line_number}")
            expected = 1 if gate_type in ("NOT", "BUF") else 2
            if len(inputs) != expected or any(not value or re.search(r"\s", value) for value in inputs):
                raise ValueError(f"Gate {name} requires {expected} input(s); use a binary BENCH")
        if name in drivers:
            raise ValueError(f"Conflicting driver or INPUT declaration for {name}")
        drivers[name] = (gate_type, inputs)
    if not outputs or not drivers:
        raise ValueError("Circuit must have named wires and at least one OUTPUT")
    missing = (outputs | {wire for _, inputs in drivers.values() for wire in inputs}) - drivers.keys()
    if missing:
        raise ValueError("Undefined driver(s): " + ", ".join(sorted(missing)))

    children = {name: [] for name in drivers}
    pending = {name: len(inputs) for name, (_, inputs) in drivers.items()}
    for name, (_, inputs) in drivers.items():
        for wire in inputs:
            children[wire].append(name)
    ready = [name for name, count in pending.items() if count == 0]
    heapq.heapify(ready)
    names = []
    while ready:
        name = heapq.heappop(ready)
        names.append(name)
        for child in children[name]:
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, child)
    if len(names) != len(drivers):
        raise ValueError("Cycle detected in BENCH circuit")
    index = {name: i for i, name in enumerate(names)}
    fanins = tuple(tuple(index[v] for v in drivers[name][1]) for name in names)
    types = tuple(drivers[name][0] for name in names)
    levels, cc0, cc1 = [], [], []
    for kind, inputs in zip(types, fanins):
        levels.append(0 if not inputs else 1 + max(levels[v] for v in inputs))
        a = [cc0[v] for v in inputs]
        b = [cc1[v] for v in inputs]
        if kind == "PI":
            zero, one = 1, 1
        elif kind in ("BUF", "NOT"):
            zero, one = a[0] + 1, b[0] + 1
        elif kind in ("AND", "NAND"):
            zero, one = min(a) + 1, sum(b) + 1
        elif kind in ("OR", "NOR"):
            zero, one = sum(a) + 1, min(b) + 1
        else:
            zero = 1 + min(a[0] + a[1], b[0] + b[1])
            one = 1 + min(a[0] + b[1], b[0] + a[1])
        if kind in ("NOT", "NAND", "NOR", "XNOR"):
            zero, one = one, zero
        cc0.append(_bounded(zero))
        cc1.append(_bounded(one))
    co = [0 if name in outputs else math.inf for name in names]
    for output in reversed(range(len(names))):
        for position, wire in enumerate(fanins[output]):
            others = [v for j, v in enumerate(fanins[output]) if j != position]
            kind = types[output]
            if kind in ("AND", "NAND"):
                cost = sum(cc1[v] for v in others)
            elif kind in ("OR", "NOR"):
                cost = sum(cc0[v] for v in others)
            elif kind in ("XOR", "XNOR"):
                cost = sum(min(cc0[v], cc1[v]) for v in others)
            else:
                cost = 0
            co[wire] = min(co[wire], _bounded(co[output] + cost + 1))
    fanouts = tuple(len(children[name]) for name in names)
    features = torch.zeros((len(names), FEATURE_DIM), dtype=torch.float32)
    features[torch.arange(len(names)), torch.tensor([GATE_TYPES.index(kind) for kind in types])] = 1
    features[:, 9] = torch.tensor(levels, dtype=torch.float32) / max(1, max(levels))
    for column, values in enumerate((fanouts, cc0, cc1, co), 10):
        maximum = max((v for v in values if math.isfinite(v)), default=0)
        scale = max(1.0, math.log1p(maximum))
        features[:, column] = torch.tensor([
            math.log1p(v) / scale if math.isfinite(v) else 1.0 for v in values
        ])
    edges = [(v, out) for out, inputs in enumerate(fanins) for v in inputs]
    edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return CircuitGraph(circuit_hash(path), tuple(names), types, fanins, features,
                        edge_index, tuple(levels), fanouts, tuple(cc0), tuple(cc1), tuple(co))
