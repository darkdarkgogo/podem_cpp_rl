import torch


def one_hot(indices, length):
    if isinstance(indices, int):
        indices = torch.LongTensor([indices]).unsqueeze(0)
    else:
        indices = torch.LongTensor(indices).unsqueeze(0).t()
    return torch.zeros((len(indices), length)).scatter_(1, indices, 1)


def build_model_features(circuit, config):
    gate_type_ids = [meta["gate_type_id"] for meta in circuit.gate_meta]
    features = one_hot(gate_type_ids, config.num_gate_types)

    max_level = max((meta["level"] for meta in circuit.gate_meta), default=0)
    max_fanout = max((meta["fanout_count"] for meta in circuit.gate_meta), default=0)
    level_scale = float(max(1, max_level))
    fanout_scale = float(max(1, max_fanout))

    levels = torch.tensor(
        [meta["level"] / level_scale for meta in circuit.gate_meta],
        dtype=torch.float32,
    ).unsqueeze(1)
    fanouts = torch.tensor(
        [meta["fanout_count"] / fanout_scale for meta in circuit.gate_meta],
        dtype=torch.float32,
    ).unsqueeze(1)
    return torch.cat([features, levels, fanouts], dim=1)


def build_raw_feature_tensor(circuit):
    rows = []
    for meta in circuit.gate_meta:
        rows.append([
            float(meta["gate_type_id"]),
            float(meta["level"]),
            float(meta["c1"]),
            float(meta["c0"]),
            float(meta["co"]),
            float(meta["fanout_count"]),
            float(meta["fanout_flag"]),
            float(meta["is_reconvergent"]),
            float(meta["reconv_source_index"]),
        ])
    return torch.tensor(rows, dtype=torch.float32)

