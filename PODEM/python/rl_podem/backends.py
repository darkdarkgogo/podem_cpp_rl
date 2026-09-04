from .smartatpg import (
    ACTION_MASK_DIM, ACTOR_INPUT_DIM, DECISION_STATE_DIM,
    ENCODER_VARIANT, GATE_EMBEDDING_DIM, POLICY_STATE_DIM,
)
from .smartatpg_features import FEATURE_SCHEMA, GRAPH_CONFIG, GRAPH_CONFIG_ID

MANIFEST_V5 = "RL_PODEM_CURRICULUM_V5"
CHECKPOINT_V5 = "RL_PODEM_CURRICULUM_TRAINING_V5"


def resolve_backend(metadata, requested=None):
    backend = metadata.get("embedding_backend")
    if backend != "smartatpg":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    if requested is not None and requested != backend:
        raise ValueError(f"Requested backend {requested} conflicts with artifact backend {backend}")
    variant = metadata.get("encoder_variant", ENCODER_VARIANT)
    actor_dim = ACTOR_INPUT_DIM + int(variant == "level_gat_gru")
    state_dim = actor_dim + ACTION_MASK_DIM
    if variant == ENCODER_VARIANT:
        expected_graph_config = GRAPH_CONFIG
    elif variant == "level_gat_gru":
        from .gat_gru import GRAPH_CONFIG as expected_graph_config
    else:
        raise ValueError(f"Unsupported SmartATPG encoder variant: {variant}")
    if metadata.get("feature_schema") != FEATURE_SCHEMA or metadata.get("graph_config") != expected_graph_config:
        raise ValueError("SmartATPG feature schema or graph configuration changed")
    if (
        int(metadata.get("gate_embedding_dim", -1)) != GATE_EMBEDDING_DIM
        or int(metadata.get("policy_state_dim", -1)) != state_dim
    ):
        raise ValueError("SmartATPG gate embedding or policy state dimension changed")
    optional_dimensions = {
        "actor_input_dim": actor_dim,
        "action_mask_dim": ACTION_MASK_DIM,
        "decision_state_dim": state_dim,
    }
    if any(
        key in metadata and int(metadata[key]) != expected
        for key, expected in optional_dimensions.items()
    ):
        raise ValueError("SmartATPG Actor input, mask, or decision state dimension changed")
    return backend


def smartatpg_metadata(encoder_variant=ENCODER_VARIANT):
    actor_dim = ACTOR_INPUT_DIM + int(encoder_variant == "level_gat_gru")
    if encoder_variant == ENCODER_VARIANT:
        graph_config, graph_config_id = GRAPH_CONFIG, GRAPH_CONFIG_ID
    elif encoder_variant == "level_gat_gru":
        from .gat_gru import GRAPH_CONFIG as graph_config, GRAPH_CONFIG_ID as graph_config_id
    else:
        raise ValueError(f"Unsupported SmartATPG encoder variant: {encoder_variant}")
    return {"embedding_backend": "smartatpg", "encoder_variant": encoder_variant,
            "feature_schema": FEATURE_SCHEMA,
            "graph_config": dict(graph_config), "graph_config_id": graph_config_id,
            "gate_embedding_dim": GATE_EMBEDDING_DIM,
            "actor_input_dim": actor_dim,
            "action_mask_dim": ACTION_MASK_DIM,
            "decision_state_dim": actor_dim + ACTION_MASK_DIM,
            "policy_state_dim": actor_dim + ACTION_MASK_DIM}
