"""Snapshot-paired SmartATPG export, independent of DeepGate."""

import hashlib
import json
from pathlib import Path

import torch

from .backends import resolve_backend, smartatpg_metadata
from .cpp_bridge import export_actor_v2_state_dict
from .smartatpg import (
    GATE_EMBEDDING_DIM, POLICY_STATE_DIM, SmartATPGPolicy,
)
from .smartatpg_features import (
    FEATURE_SCHEMA, GRAPH_CONFIG_ID, load_circuit_graph,
)


def snapshot_id(state):
    digest = hashlib.sha256(json.dumps(smartatpg_metadata(), sort_keys=True).encode("ascii"))
    for name in sorted(state):
        if name.startswith("critic."):
            continue
        tensor = state[name].detach().cpu().float().contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite inference tensor: {name}")
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def export_actor(state, path, best_round=0, best_score=None):
    identity = snapshot_id(state)
    score_text = (
        "none" if best_score is None
        else ",".join(format(float(value), ".17g") for value in best_score)
    )
    export_actor_v2_state_dict(state, path, metadata={
        "backend": "smartatpg", "feature_schema": FEATURE_SCHEMA,
        "graph_config": GRAPH_CONFIG_ID,
        "gate_embedding_dim": GATE_EMBEDDING_DIM,
        "policy_state_dim": POLICY_STATE_DIM,
        "snapshot": identity,
        "best_round": int(best_round),
        "best_score": score_text,
    })
    return identity


def policy_from_state(state):
    hidden_dim = state["gate_encoder.0.weight"].shape[0]
    policy = SmartATPGPolicy(hidden_dim)
    policy.load_state_dict(state)
    return policy.eval()


def export_descriptors(state, graph, path, policy=None):
    identity = snapshot_id(state)
    policy = policy_from_state(state) if policy is None else policy
    if snapshot_id(policy.state_dict()) != identity:
        raise ValueError("Descriptor encoder does not match the selected inference snapshot")
    with torch.no_grad():
        values = policy.graph_embeddings(graph).cpu()
    if values.shape != (len(graph.names), GATE_EMBEDDING_DIM) or not bool(torch.isfinite(values).all()):
        raise ValueError("Invalid SmartATPG gate embeddings")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as out:
        out.write("SMARTATPG_EMBEDDINGS_V3\n")
        out.write(
            f"backend smartatpg\nfeature_schema {FEATURE_SCHEMA}\n"
            f"graph_config {GRAPH_CONFIG_ID}\n"
            f"gate_embedding_dim {GATE_EMBEDDING_DIM}\n"
            f"policy_state_dim {POLICY_STATE_DIM}\nsnapshot {identity}\n"
        )
        out.write(f"circuit_hash {graph.circuit_hash}\ndimension {GATE_EMBEDDING_DIM}\ncount {len(graph.names)}\n")
        for name, row in zip(graph.names, values.tolist()):
            out.write(name + " " + " ".join(format(v, ".9g") for v in row) + "\n")
    temporary.replace(path)
    return len(graph.names), GATE_EMBEDDING_DIM


def export_snapshot(state, graphs, actor_path):
    actor_path = Path(actor_path).resolve()
    identity = snapshot_id(state)
    snapshot_dir = actor_path.parent / (actor_path.stem + "_snapshots") / identity
    native_actor = snapshot_dir / "actor.txt"
    export_actor(state, native_actor)
    policy = policy_from_state(state)
    circuits = {}
    for name, graph in graphs.items():
        path = snapshot_dir / (graph.circuit_hash + ".emb")
        export_descriptors(state, graph, path, policy)
        circuits[name] = {"embeddings": str(path), "circuit_hash": graph.circuit_hash}
    export_actor(state, actor_path)
    manifest = {"format": "SMARTATPG_INFERENCE_SNAPSHOT_V1", **smartatpg_metadata(),
                "snapshot": identity, "actor": str(native_actor), "circuits": circuits}
    manifest_path = actor_path.with_suffix(actor_path.suffix + ".json")
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def checkpoint_policy(checkpoint, selection="best", requested=None):
    backend = resolve_backend(checkpoint.get("agent", checkpoint), requested)
    if backend != "smartatpg":
        raise ValueError("This export requires a SmartATPG training checkpoint")
    if selection == "best":
        if "best_policy_state" not in checkpoint:
            raise ValueError("Checkpoint has no best snapshot; select latest explicitly")
        return checkpoint["best_policy_state"]
    agent = checkpoint.get("agent", checkpoint)
    return agent["policy_old"]
