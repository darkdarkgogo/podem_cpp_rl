"""Static features for the NN-All PODEM backtrace baseline."""

from dataclasses import dataclass
from pathlib import Path
import math

from .smartatpg_features import circuit_hash, load_circuit_graph


FEATURE_SCHEMA = "NN_ALL_PODEM_11D_V1"
NATIVE_FEATURE_FORMAT = "NN_ALL_PODEM_FEATURES_V1"
PARENT_TYPES = ("AND", "NAND", "OR", "NOR", "PI", "BUF", "NOT")


@dataclass(frozen=True)
class NnAllFeatures:
    circuit_hash: str
    names: tuple[str, ...]
    gate_types: tuple[str, ...]
    levels: tuple[int, ...]
    cc0: tuple[float, ...]
    cc1: tuple[float, ...]
    cop: tuple[float, ...]

    @property
    def indices(self):
        return {name: index for index, name in enumerate(self.names)}


def load_nn_all_features(path: Path) -> NnAllFeatures:
    graph = load_circuit_graph(path)
    cop = []
    for kind, fanins in zip(graph.gate_types, graph.fanins):
        inputs = [cop[index] for index in fanins]
        if kind == "PI":
            value = 0.5
        elif kind == "AND":
            value = math.prod(inputs)
        elif kind == "NAND":
            value = 1.0 - math.prod(inputs)
        elif kind == "OR":
            value = 1.0 - math.prod(1.0 - item for item in inputs)
        elif kind == "NOR":
            value = math.prod(1.0 - item for item in inputs)
        elif kind == "NOT":
            value = 1.0 - inputs[0]
        else:
            raise ValueError(f"Unsupported NN-All gate type: {kind}")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid COP probability for {graph.names[len(cop)]}")
        cop.append(value)
    return NnAllFeatures(
        circuit_hash(path), graph.names, graph.gate_types, graph.levels,
        graph.cc0, graph.cc1, tuple(cop),
    )


def candidate_vector(features: NnAllFeatures, objective: str, candidate: str) -> tuple[float, ...]:
    indices = features.indices
    parent = features.gate_types[indices[objective]]
    index = indices[candidate]
    if parent not in PARENT_TYPES:
        raise ValueError(f"Unsupported NN-All objective type: {parent}")
    vector = (
        float(features.levels[index]), float(features.cc0[index]),
        float(features.cc1[index]), float(features.cop[index]),
        *(float(parent == kind) for kind in PARENT_TYPES),
    )
    if len(vector) != 11 or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid NN-All candidate vector")
    return vector


def write_native_features(path: Path, features: NnAllFeatures) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kind, level, cc0, cc1, cop in zip(
        features.names, features.gate_types, features.levels,
        features.cc0, features.cc1, features.cop,
    ):
        if any(character.isspace() for character in name):
            raise ValueError(f"Invalid NN-All gate name: {name!r}")
        rows.append(
            f"gate {name} {kind} {level} {cc0:.17g} {cc1:.17g} {cop:.17g}"
        )
    checksum = 14695981039346656037
    for byte in ("\n".join(rows) + "\n").encode("utf-8"):
        checksum = ((checksum ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    lines = [
        NATIVE_FEATURE_FORMAT, f"circuit_hash {features.circuit_hash}",
        f"feature_hash {checksum:016x}", f"gate_count {len(rows)}", *rows,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
