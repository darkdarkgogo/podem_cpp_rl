from .smartatpg import (
    ACTION_MASK_DIM, ACTOR_INPUT_DIM, ENCODER_VARIANT, GATE_EMBEDDING_DIM,
)
from .smartatpg_features import FEATURE_SCHEMA, GRAPH_CONFIG, GRAPH_CONFIG_ID

MANIFEST_V5 = "SMARTATPG_DATA_SPLIT_TRAINING_V5_11D_CO_NO_BUF"
MANIFEST_V6 = "SMARTATPG_DATA_SPLIT_MANIFEST_V6_TOP30_11D_CO_NO_BUF"
CHECKPOINT_V5 = "SMARTATPG_DATA_SPLIT_TRAINING_V6_11D_CO_NO_BUF"
GAT_VARIANTS = (
    "level_gat_gru", "level_gat_no_gru", "level_gat_gru_no_object_value",
)


def resolve_backend(metadata, requested=None):
    backend = metadata.get("embedding_backend")
    if backend != "smartatpg":
        raise ValueError(f"Unsupported embedding backend: {backend}")
    if requested is not None and requested != backend:
        raise ValueError(f"Requested backend {requested} conflicts with artifact backend {backend}")
    variant = metadata.get("encoder_variant", ENCODER_VARIANT)
    is_gat = variant in GAT_VARIANTS
    has_value = variant != "level_gat_gru_no_object_value"
    actor_dim = (
        2 * GATE_EMBEDDING_DIM + int(has_value)
        if is_gat else ACTOR_INPUT_DIM
    )
    critic_dim = GATE_EMBEDDING_DIM + int(is_gat and has_value)
    state_dim = actor_dim + ACTION_MASK_DIM
    if variant == ENCODER_VARIANT:
        expected_graph_config = GRAPH_CONFIG
    elif is_gat:
        from .gat_gru import GRAPH_CONFIG
        expected_graph_config = (
            {**GRAPH_CONFIG, "update": "residual_elu"}
            if variant == "level_gat_no_gru" else GRAPH_CONFIG
        )
    else:
        raise ValueError(f"Unsupported SmartATPG encoder variant: {variant}")
    if metadata.get("feature_schema") != FEATURE_SCHEMA or metadata.get("graph_config") != expected_graph_config:
        raise ValueError("SmartATPG feature schema or graph configuration changed")
    if (
        int(metadata.get("gate_embedding_dim", -1)) != GATE_EMBEDDING_DIM
        or int(metadata.get("policy_state_dim", -1)) != state_dim
        or (
            is_gat
            and int(metadata.get("critic_input_dim", -1)) != critic_dim
        )
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
    is_gat = encoder_variant in GAT_VARIANTS
    has_value = encoder_variant != "level_gat_gru_no_object_value"
    actor_dim = (
        2 * GATE_EMBEDDING_DIM + int(has_value)
        if is_gat else ACTOR_INPUT_DIM
    )
    if encoder_variant == ENCODER_VARIANT:
        graph_config, graph_config_id = GRAPH_CONFIG, GRAPH_CONFIG_ID
    elif is_gat:
        from .gat_gru import GRAPH_CONFIG, GRAPH_CONFIG_ID, NO_GRU_GRAPH_CONFIG_ID
        graph_config = (
            {**GRAPH_CONFIG, "update": "residual_elu"}
            if encoder_variant == "level_gat_no_gru" else GRAPH_CONFIG
        )
        graph_config_id = (
            NO_GRU_GRAPH_CONFIG_ID
            if encoder_variant == "level_gat_no_gru" else GRAPH_CONFIG_ID
        )
    else:
        raise ValueError(f"Unsupported SmartATPG encoder variant: {encoder_variant}")
    metadata = {"embedding_backend": "smartatpg", "encoder_variant": encoder_variant,
            "feature_schema": FEATURE_SCHEMA,
            "graph_config": dict(graph_config), "graph_config_id": graph_config_id,
            "gate_embedding_dim": GATE_EMBEDDING_DIM,
            "actor_input_dim": actor_dim,
            "action_mask_dim": ACTION_MASK_DIM,
            "decision_state_dim": actor_dim + ACTION_MASK_DIM,
            "policy_state_dim": actor_dim + ACTION_MASK_DIM}
    if is_gat:
        metadata["critic_input_dim"] = GATE_EMBEDDING_DIM + int(has_value)
    return metadata
