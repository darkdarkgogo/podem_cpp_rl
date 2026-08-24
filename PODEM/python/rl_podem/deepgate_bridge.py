from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Union

import torch


def _fnv1a_file_hash(path: Union[str, Path]) -> str:
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_project_roots() -> list[Path]:
    base_root = _project_root()
    return [
        base_root,
        base_root.parent,
        Path.cwd(),
    ]


def _ensure_deepgate_importable() -> None:
    candidate_roots = []
    for project_root in _candidate_project_roots():
        candidate_roots.extend(
            [
                project_root / "vendor" / "deepgate_recgnn_extractor",
                project_root / "deepgate_recgnn_extractor",
            ]
        )

    seen = set()
    for deepgate_root in candidate_roots:
        resolved = deepgate_root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deepgate_root_str = str(resolved)
        if resolved.exists() and deepgate_root_str not in sys.path:
            sys.path.insert(0, deepgate_root_str)
            print(f"[DeepGateBridge] Using deepgate_recgnn_extractor from: {deepgate_root_str}")
            return
        if resolved.exists():
            print(f"[DeepGateBridge] deepgate_recgnn_extractor already importable from: {deepgate_root_str}")
            return

    raise ModuleNotFoundError(
        "Unable to locate deepgate_recgnn_extractor. Expected it under "
        "'<project-root>/vendor/deepgate_recgnn_extractor', "
        "'<project-root>/deepgate_recgnn_extractor', "
        "or a sibling project root on the current search path."
    )


def load_aligned_gate_embeddings(circuit, checkpoint_path: str) -> int:
    _ensure_deepgate_importable()

    from deepgate_recgnn_extractor import encode_bench

    bench_path = circuit.bench_path
    if bench_path is None:
        raise ValueError("Circuit is missing the source .bench path required for DeepGate encoding.")

    print(f"[DeepGateBridge] Loading embeddings for bench: {bench_path}")
    print(f"[DeepGateBridge] DeepGate checkpoint: {checkpoint_path}")
    result = encode_bench(bench_path, checkpoint_path=checkpoint_path, verbose=True)
    node_embeddings = result["node_embeddings"]
    gate_meta = result["gate_meta"]
    embedding_by_name = {
        meta["name"]: node_embeddings[meta["index"]].detach().clone().float()
        for meta in gate_meta
    }

    if not embedding_by_name:
        raise ValueError("DeepGate extractor returned no node embeddings.")

    cache = {}
    visiting = set()

    def resolve_gate_embedding(gate):
        if gate.id in cache:
            return cache[gate.id]
        if gate.id in visiting:
            raise ValueError(f"Cycle detected while resolving embedding for gate '{gate.outputpin}'.")

        visiting.add(gate.id)
        gate_name = gate.outputpin
        if gate_name in embedding_by_name:
            embedding = embedding_by_name[gate_name]
        elif gate.type == "output_pin":
            if len(gate.input_gates) != 1:
                raise ValueError(
                    f"Expected output pin '{gate.outputpin}' to have exactly one driver, "
                    f"found {len(gate.input_gates)}."
                )
            embedding = resolve_gate_embedding(gate.input_gates[0])
        else:
            available_examples = ", ".join(sorted(embedding_by_name.keys())[:10])
            raise KeyError(
                f"DeepGate embedding not found for gate '{gate_name}' (type={gate.type}). "
                f"Sample available node names: {available_examples}"
            )

        visiting.remove(gate.id)
        cache[gate.id] = embedding
        return embedding

    for gate in circuit.gates.values():
        gate.deepgate_embedding = resolve_gate_embedding(gate)

    print(
        "[DeepGateBridge] Gate embedding alignment complete: "
        f"mapped_gates={len(circuit.gates)} embedding_dim={int(node_embeddings.shape[1])}"
    )
    return int(node_embeddings.shape[1])


def export_cpp_embeddings(
    bench_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Tuple[int, int]:
    """Encode one circuit and write the versioned C++ embedding table."""
    _ensure_deepgate_importable()
    from deepgate_recgnn_extractor import encode_bench

    bench_path = Path(bench_path).resolve()
    output_path = Path(output_path).resolve()
    result = encode_bench(
        str(bench_path),
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        verbose=True,
    )
    embeddings = result["node_embeddings"].detach().cpu().float()
    gate_meta = result["gate_meta"]
    if embeddings.ndim != 2 or embeddings.shape[0] != len(gate_meta):
        raise ValueError("DeepGate node embeddings and gate metadata are not aligned.")

    names = [meta["name"] for meta in gate_meta]
    if len(names) != len(set(names)):
        raise ValueError("DeepGate returned duplicate node names.")
    invalid_names = [name for name in names if any(char.isspace() for char in name)]
    if invalid_names:
        raise ValueError(
            "C++ embedding format does not support whitespace in node names: "
            + ", ".join(invalid_names[:5])
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("SMARTATPG_EMBEDDINGS_V1\n")
        output.write(f"circuit_hash {_fnv1a_file_hash(bench_path)}\n")
        output.write(f"dimension {embeddings.shape[1]}\n")
        output.write(f"count {embeddings.shape[0]}\n")
        for meta in gate_meta:
            values = embeddings[meta["index"]].tolist()
            output.write(meta["name"])
            output.write(" ")
            output.write(" ".join(format(value, ".9g") for value in values))
            output.write("\n")

    return int(embeddings.shape[0]), int(embeddings.shape[1])
