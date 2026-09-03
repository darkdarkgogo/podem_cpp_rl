"""Torch-free SmartATPG V5 model and GraphSAGE inference utilities."""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from dataclasses import dataclass
from pathlib import Path


MODEL_FORMAT = "SMARTATPG_MODEL_V5"
EMBEDDING_FORMAT = "SMARTATPG_EMBEDDINGS_V3"
FEATURE_SCHEMA = "SMARTATPG_FEATURES_V2_11D"
GRAPH_CONFIG = "fanin_mean_1x22x11"
GATE_TYPES = ("PI", "AND", "NAND", "OR", "NOR", "NOT", "BUF")
GATE_EMBEDDING_DIM = 11
POLICY_STATE_DIM = 13
COST_CAP = 10**9
CIRCUITS = (
    "c432", "c499", "c1355", "c1908", "c2670", "c3540", "c5315",
    "c6288", "c7552", "s5378", "s9234", "s13207", "s15850",
    "s35932", "s38417", "s38584",
)
PORT_RE = re.compile(r"^(INPUT|OUTPUT)\s*\(\s*([^()\s]+)\s*\)$", re.I)
GATE_RE = re.compile(r"^([^\s=(),]+)\s*=\s*(\w+)\s*\(([^()]*)\)$")

MODEL_TENSORS = (
    "graph_encoder.layer.weight",
    "graph_encoder.layer.bias",
    "gate_encoder.0.weight",
    "gate_encoder.0.bias",
    "objective_value_embedding.weight",
    "backtrace_actor.0.weight",
    "backtrace_actor.0.bias",
    "backtrace_actor.2.weight",
    "backtrace_actor.2.bias",
)


@dataclass(frozen=True)
class Tensor:
    rows: int
    cols: int
    values: tuple[float, ...]


@dataclass(frozen=True)
class PortableModel:
    snapshot: str
    best_round: int
    best_score: tuple[float, ...] | None
    hidden_dim: int
    tensors: dict[str, Tensor]


@dataclass(frozen=True)
class PortableGraph:
    circuit_hash: str
    names: tuple[str, ...]
    fanins: tuple[tuple[int, ...], ...]
    features: tuple[tuple[float, ...], ...]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circuit_hash(path):
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _next(tokens, context):
    try:
        return next(tokens)
    except StopIteration as error:
        raise ValueError(f"Truncated SmartATPG model while reading {context}") from error


def _field(tokens, expected):
    key = _next(tokens, expected)
    if key != expected:
        raise ValueError(f"Expected model field {expected}, found {key}")
    return _next(tokens, expected)


def load_model(path):
    path = Path(path)
    tokens = iter(path.read_text(encoding="utf-8").split())
    if _next(tokens, "header") != MODEL_FORMAT:
        raise ValueError("SmartATPG benchmark requires a V5 model with GraphSAGE weights")
    expected_metadata = {
        "backend": "smartatpg",
        "feature_schema": FEATURE_SCHEMA,
        "graph_config": GRAPH_CONFIG,
        "gate_embedding_dim": str(GATE_EMBEDDING_DIM),
        "policy_state_dim": str(POLICY_STATE_DIM),
    }
    for key, expected in expected_metadata.items():
        if _field(tokens, key) != expected:
            raise ValueError(f"Invalid SmartATPG model {key}")
    snapshot = _field(tokens, "snapshot")
    if len(snapshot) != 64 or any(value not in "0123456789abcdef" for value in snapshot):
        raise ValueError("Invalid SmartATPG model snapshot")
    best_round = int(_field(tokens, "best_round"))
    best_score_text = _field(tokens, "best_score")
    if best_score_text == "none":
        best_score = None
    else:
        try:
            best_score = tuple(float(value) for value in best_score_text.split(","))
        except ValueError as error:
            raise ValueError("Invalid SmartATPG best score") from error
        if len(best_score) != 5 or any(
            not math.isfinite(value) for value in best_score
        ):
            raise ValueError("SmartATPG best score must contain five finite values")
    hidden_dim = int(_field(tokens, "hidden_dim"))
    if hidden_dim <= 0:
        raise ValueError("SmartATPG model hidden_dim must be positive")

    tensors = {}
    while True:
        marker = _next(tokens, "tensor or end")
        if marker == "end":
            break
        if marker != "tensor":
            raise ValueError(f"Expected tensor entry, found {marker}")
        name = _next(tokens, "tensor name")
        rows = int(_next(tokens, f"{name} rows"))
        cols = int(_next(tokens, f"{name} columns"))
        if rows <= 0 or cols <= 0 or name in tensors:
            raise ValueError(f"Invalid or duplicate SmartATPG tensor {name}")
        values = tuple(
            float(_next(tokens, f"{name} values")) for _ in range(rows * cols)
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite SmartATPG tensor {name}")
        tensors[name] = Tensor(rows, cols, values)
    try:
        trailing = next(tokens)
    except StopIteration:
        trailing = None
    if trailing is not None:
        raise ValueError(f"Unexpected trailing model data: {trailing}")
    if set(tensors) != set(MODEL_TENSORS):
        missing = sorted(set(MODEL_TENSORS) - set(tensors))
        extra = sorted(set(tensors) - set(MODEL_TENSORS))
        raise ValueError(f"Invalid V5 tensor set; missing={missing}, extra={extra}")
    graph_weight = tensors["graph_encoder.layer.weight"]
    graph_bias = tensors["graph_encoder.layer.bias"]
    if (graph_weight.rows, graph_weight.cols) != (11, 22):
        raise ValueError("GraphSAGE weight must have shape [11,22]")
    if graph_bias.rows * graph_bias.cols != 11:
        raise ValueError("GraphSAGE bias must contain 11 values")
    if (tensors["gate_encoder.0.weight"].rows,
            tensors["gate_encoder.0.weight"].cols) != (hidden_dim, 13):
        raise ValueError("Actor gate encoder must have shape [hidden_dim,13]")
    return PortableModel(snapshot, best_round, best_score, hidden_dim, tensors)


def load_graph(path):
    path = Path(path)
    drivers = {}
    outputs = set()
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
            gate_type = {"BUFF": "BUF", "EQV": "XNOR"}.get(
                gate_type.upper(), gate_type.upper()
            )
            inputs = tuple(value.strip() for value in input_text.split(","))
            if gate_type not in GATE_TYPES[1:]:
                raise ValueError(
                    f"Unsupported SmartATPG gate type {gate_type} at line "
                    f"{line_number}; convert XOR/XNOR before embedding"
                )
            expected = 1 if gate_type in ("NOT", "BUF") else 2
            if len(inputs) != expected or any(
                not value or re.search(r"\s", value) for value in inputs
            ):
                raise ValueError(f"Gate {name} requires {expected} input(s)")
        if name in drivers:
            raise ValueError(f"Conflicting driver or INPUT declaration for {name}")
        drivers[name] = (gate_type, inputs)
    if not outputs or not drivers:
        raise ValueError("Circuit must have named wires and at least one OUTPUT")
    missing = (
        outputs | {wire for _, inputs in drivers.values() for wire in inputs}
    ) - drivers.keys()
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

    index = {name: position for position, name in enumerate(names)}
    fanins = tuple(tuple(index[value] for value in drivers[name][1]) for name in names)
    gate_types = tuple(drivers[name][0] for name in names)
    levels = []
    cc0 = []
    cc1 = []
    for gate_type, inputs in zip(gate_types, fanins):
        levels.append(0 if not inputs else 1 + max(levels[value] for value in inputs))
        zeros = [cc0[value] for value in inputs]
        ones = [cc1[value] for value in inputs]
        if gate_type == "PI":
            zero, one = 1, 1
        elif gate_type in ("BUF", "NOT"):
            zero, one = zeros[0] + 1, ones[0] + 1
        elif gate_type in ("AND", "NAND"):
            zero, one = min(zeros) + 1, sum(ones) + 1
        else:
            zero, one = sum(zeros) + 1, min(ones) + 1
        if gate_type in ("NOT", "NAND", "NOR"):
            zero, one = one, zero
        cc0.append(min(COST_CAP, zero))
        cc1.append(min(COST_CAP, one))

    fanouts = tuple(len(children[name]) for name in names)
    max_level = max(1, max(levels))
    normalized = []
    for values in (fanouts, cc0, cc1):
        scale = max(1.0, math.log1p(max(values, default=0)))
        normalized.append([math.log1p(value) / scale for value in values])
    features = []
    for position, gate_type in enumerate(gate_types):
        row = [0.0] * GATE_EMBEDDING_DIM
        row[GATE_TYPES.index(gate_type)] = 1.0
        row[7] = levels[position] / max_level
        row[8] = normalized[0][position]
        row[9] = normalized[1][position]
        row[10] = normalized[2][position]
        features.append(tuple(row))
    return PortableGraph(
        circuit_hash(path), tuple(names), fanins, tuple(features)
    )


def compute_embeddings(model, graph):
    weight = model.tensors["graph_encoder.layer.weight"]
    bias = model.tensors["graph_encoder.layer.bias"].values
    embeddings = []
    for position, feature in enumerate(graph.features):
        inputs = graph.fanins[position]
        if inputs:
            neighbor = tuple(
                sum(graph.features[source][column] for source in inputs) / len(inputs)
                for column in range(GATE_EMBEDDING_DIM)
            )
        else:
            neighbor = (0.0,) * GATE_EMBEDDING_DIM
        combined = feature + neighbor
        row = []
        for output in range(GATE_EMBEDDING_DIM):
            offset = output * weight.cols
            value = bias[output] + sum(
                weight.values[offset + column] * combined[column]
                for column in range(weight.cols)
            )
            row.append(max(0.0, value))
        embeddings.append(tuple(row))
    return tuple(embeddings)


def export_embeddings(model, graph, path):
    values = compute_embeddings(model, graph)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write(f"{EMBEDDING_FORMAT}\n")
        output.write(
            f"backend smartatpg\nfeature_schema {FEATURE_SCHEMA}\n"
            f"graph_config {GRAPH_CONFIG}\n"
            f"gate_embedding_dim {GATE_EMBEDDING_DIM}\n"
            f"policy_state_dim {POLICY_STATE_DIM}\n"
            f"snapshot {model.snapshot}\n"
            f"circuit_hash {graph.circuit_hash}\n"
            f"dimension {GATE_EMBEDDING_DIM}\ncount {len(graph.names)}\n"
        )
        for name, row in zip(graph.names, values):
            output.write(name + " " + " ".join(format(value, ".9g") for value in row) + "\n")
    temporary.replace(path)
    return len(graph.names), GATE_EMBEDDING_DIM
