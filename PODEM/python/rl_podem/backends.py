from .smartatpg_features import FEATURE_SCHEMA, GRAPH_CONFIG

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
    return backend


def smartatpg_metadata():
    return {"embedding_backend": "smartatpg", "feature_schema": FEATURE_SCHEMA,
            "graph_config": dict(GRAPH_CONFIG)}
