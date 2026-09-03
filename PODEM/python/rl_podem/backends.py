from .smartatpg import GATE_EMBEDDING_DIM, POLICY_STATE_DIM
from .smartatpg_features import FEATURE_SCHEMA, GRAPH_CONFIG, GRAPH_CONFIG_ID

MANIFEST_V5 = "RL_PODEM_CURRICULUM_V5"
CHECKPOINT_V5 = "RL_PODEM_CURRICULUM_TRAINING_V5"


def resolve_backend(metadata, requested=None):
    backend = metadata.get("embedding_backend", "deepgate")
    if backend not in ("smartatpg", "deepgate"):
        raise ValueError(f"Unsupported embedding backend: {backend}")
    if requested is not None and requested != backend:
        raise ValueError(f"Requested backend {requested} conflicts with artifact backend {backend}")
    if backend == "smartatpg":
        if metadata.get("feature_schema") != FEATURE_SCHEMA or metadata.get("graph_config") != GRAPH_CONFIG:
            raise ValueError("SmartATPG feature schema or graph configuration changed")
        if (
            int(metadata.get("gate_embedding_dim", -1)) != GATE_EMBEDDING_DIM
            or int(metadata.get("policy_state_dim", -1)) != POLICY_STATE_DIM
        ):
            raise ValueError("SmartATPG gate embedding or policy state dimension changed")
    return backend


def smartatpg_metadata():
    return {"embedding_backend": "smartatpg", "feature_schema": FEATURE_SCHEMA,
            "graph_config": dict(GRAPH_CONFIG), "graph_config_id": GRAPH_CONFIG_ID,
            "gate_embedding_dim": GATE_EMBEDDING_DIM,
            "policy_state_dim": POLICY_STATE_DIM}
