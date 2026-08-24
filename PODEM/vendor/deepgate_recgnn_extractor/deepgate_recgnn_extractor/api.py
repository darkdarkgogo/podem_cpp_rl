from types import SimpleNamespace
from typing import Optional

import torch

from .bench_parser import parse_bench
from .config import EncoderConfig
from .dag_utils import return_order_info
from .data import OrderedData
from .featurizer import build_model_features, build_raw_feature_tensor
from .graph_builder import build_circuit
from .models.recgnn import RecGNN
from .pooling import pool_graph


def _build_args(config: EncoderConfig, device):
    model = config.model
    return SimpleNamespace(
        aggr_function=model.aggr_function,
        update_function=model.update_function,
        dim_hidden=model.dim_hidden,
        dim_mlp=model.dim_mlp,
        dim_pred=model.dim_pred,
        num_fc=model.num_fc,
        wx_update=model.wx_update,
        wx_mlp=model.wx_mlp,
        intermediate_supervision=model.intermediate_supervision,
        reverse=model.reverse,
        custom_backward=model.custom_backward,
        use_edge_attr=model.use_edge_attr,
        mask=model.mask,
        num_rounds=model.num_rounds,
        num_aggr=model.num_aggr,
        dim_edge_feature=model.dim_edge_feature,
        norm_layer=model.norm_layer,
        activation_layer=model.activation_layer,
        dim_node_feature=config.dim_node_feature,
        num_gate_types=config.num_gate_types,
        verbose=False,
        log_rounds=False,
        device=device,
    )


def build_graph_from_bench(bench_path, config: EncoderConfig, verbose: bool = False):
    if verbose:
        print(f"[DeepGate] Parsing bench netlist: {bench_path}")
    parsed = parse_bench(bench_path)
    circuit = build_circuit(parsed, config.gate_to_index)
    model_x = build_model_features(circuit, config)
    raw_x = build_raw_feature_tensor(circuit)
    edge_index = torch.tensor(circuit.edge_index, dtype=torch.long).t().contiguous()
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, model_x.size(0))
    graph = OrderedData(
        x=model_x,
        edge_index=edge_index,
        forward_level=forward_level,
        forward_index=forward_index,
        backward_level=backward_level,
        backward_index=backward_index,
    )
    graph.gate_type = torch.tensor(
        [meta["gate_type_id"] for meta in circuit.gate_meta],
        dtype=torch.long,
    )
    graph.prob = torch.tensor(
        [[meta["c1"]] for meta in circuit.gate_meta],
        dtype=torch.float32,
    )
    graph.raw_x = raw_x
    graph.po_indices = torch.tensor(circuit.po_indices, dtype=torch.long)
    if verbose:
        print(
            "[DeepGate] Built graph: "
            f"nodes={graph.num_nodes} edges={graph.edge_index.size(1)} "
            f"po_nodes={len(circuit.po_indices)} feature_dim={graph.x.size(1)}"
        )
    return graph, circuit


def load_model(config: Optional[EncoderConfig] = None, checkpoint_path=None, device=None):
    if config is None:
        config = EncoderConfig()
    if device is None:
        device = torch.device("cpu")
    args = _build_args(config, device)
    model = RecGNN(args).to(device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def encode_bench(
    bench_path,
    model=None,
    checkpoint_path=None,
    config: Optional[EncoderConfig] = None,
    device=None,
    verbose: bool = False,
):
    if config is None:
        config = EncoderConfig()
    if device is None:
        device = torch.device("cpu")
    if model is None:
        if verbose:
            print(f"[DeepGate] Loading checkpoint: {checkpoint_path or '<random-init>'}")
        model = load_model(config=config, checkpoint_path=checkpoint_path, device=device)
    if hasattr(model, "args"):
        model.args.verbose = verbose
        model.args.log_rounds = verbose

    graph, circuit = build_graph_from_bench(bench_path, config, verbose=verbose)
    graph = graph.to(device)
    if verbose:
        print(
            f"[DeepGate] Start message passing: rounds={model.args.num_rounds} "
            f"reverse={model.args.reverse} device={device}"
        )
    predictions = model(graph)
    node_embeddings = model.last_node_embedding.detach().cpu()
    graph_embedding = pool_graph(node_embeddings, method=config.graph_pool)
    if verbose:
        print(
            "[DeepGate] Encoding complete: "
            f"node_embeddings={tuple(node_embeddings.shape)} "
            f"graph_embedding={tuple(graph_embedding.shape)}"
        )

    return {
        "node_embeddings": node_embeddings,
        "graph_embedding": graph_embedding,
        "model_node_features": graph.x.detach().cpu(),
        "raw_node_features": graph.raw_x.detach().cpu(),
        "edge_index": graph.edge_index.detach().cpu(),
        "po_indices": graph.po_indices.detach().cpu(),
        "gate_meta": circuit.gate_meta,
        "predictions": [pred.detach().cpu() for pred in predictions],
        "config": config,
    }

