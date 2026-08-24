import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .api import build_graph_from_bench
from .config import EncoderConfig
from .models.recgnn import RecGNN


def _build_model_args(config, device):
    model = config.model
    return argparse.Namespace(
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
        device=device,
    )


def _simulate_signatures(circuit, num_patterns, exact_pi_limit, seed):
    pi_indices = [meta["index"] for meta in circuit.gate_meta if meta["is_pi"]]
    num_pis = len(pi_indices)
    if num_pis == 0:
        raise ValueError("Circuit has no primary inputs.")

    if num_pis <= exact_pi_limit:
        pattern_count = 1 << num_pis
        assignments = torch.zeros((pattern_count, num_pis), dtype=torch.bool)
        for bit in range(num_pis):
            period = 1 << (num_pis - bit - 1)
            block = torch.arange(pattern_count) // period
            assignments[:, bit] = (block % 2 == 1)
    else:
        generator = torch.Generator().manual_seed(seed)
        assignments = torch.randint(
            0,
            2,
            (num_patterns, num_pis),
            generator=generator,
            dtype=torch.int64,
        ).bool()

    values = [None] * len(circuit.gate_meta)
    for local_idx, node_idx in enumerate(pi_indices):
        values[node_idx] = assignments[:, local_idx]

    for meta in sorted(circuit.gate_meta, key=lambda item: (item["level"], item["index"])):
        gate_type = meta["gate_type_id"]
        node_idx = meta["index"]
        if gate_type == 0:
            continue

        fanins = circuit.fanin_list[node_idx]
        fanin_values = [values[src_idx] for src_idx in fanins]
        if gate_type == 1:
            values[node_idx] = fanin_values[0]
        elif gate_type == 2:
            values[node_idx] = torch.stack(fanin_values, dim=0).all(dim=0)
        elif gate_type == 3:
            values[node_idx] = ~torch.stack(fanin_values, dim=0).all(dim=0)
        elif gate_type == 4:
            values[node_idx] = torch.stack(fanin_values, dim=0).any(dim=0)
        elif gate_type == 5:
            values[node_idx] = ~torch.stack(fanin_values, dim=0).any(dim=0)
        elif gate_type == 6:
            values[node_idx] = ~fanin_values[0]
        elif gate_type == 7:
            xor_value = fanin_values[0].clone()
            for other in fanin_values[1:]:
                xor_value = torch.logical_xor(xor_value, other)
            values[node_idx] = xor_value
        elif gate_type in {8, 9}:
            values[node_idx] = fanin_values[0]
        elif gate_type == 10:
            xnor_value = fanin_values[0].clone()
            for other in fanin_values[1:]:
                xnor_value = torch.logical_xor(xnor_value, other)
            values[node_idx] = ~xnor_value
        else:
            raise ValueError(f"Unsupported gate type id during simulation: {gate_type}")

    signatures = torch.stack(values, dim=0).to(torch.float32)
    probabilities = signatures.mean(dim=1, keepdim=True)
    return signatures, probabilities


def _sample_pairs(num_nodes, max_pairs, generator):
    if num_nodes < 2:
        return []
    all_pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    if max_pairs <= 0 or len(all_pairs) <= max_pairs:
        return all_pairs
    return random.Random(generator).sample(all_pairs, max_pairs)


def build_training_graph(bench_path, config, num_patterns, exact_pi_limit, max_pairs, seed):
    print(f"[DeepGate][Dataset] Parsing circuit: {bench_path}")
    graph, circuit = build_graph_from_bench(str(bench_path), config, verbose=False)
    signatures, probabilities = _simulate_signatures(
        circuit=circuit,
        num_patterns=num_patterns,
        exact_pi_limit=exact_pi_limit,
        seed=seed,
    )
    pairs = _sample_pairs(graph.num_nodes, max_pairs=max_pairs, generator=seed)
    if not pairs:
        raise ValueError(f"Bench '{bench_path}' does not have enough nodes to form training pairs.")

    pair_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    tt_distance = []
    for src_idx, dst_idx in pairs:
        distance = torch.ne(signatures[src_idx], signatures[dst_idx]).float().mean()
        tt_distance.append(distance)

    graph.prob = probabilities
    graph.tt_pair_index = pair_index
    graph.tt_dis = torch.stack(tt_distance)
    print(
        "[DeepGate][Dataset] Prepared graph: "
        f"name={Path(bench_path).name} nodes={graph.num_nodes} "
        f"edges={graph.edge_index.size(1)} pairs={graph.tt_dis.numel()} "
        f"pi={sum(1 for meta in circuit.gate_meta if meta['is_pi'])}"
    )
    return graph


def load_training_graphs(bench_dir, config, num_patterns, exact_pi_limit, max_pairs, seed):
    bench_paths = sorted(Path(bench_dir).glob("*.bench"))
    if not bench_paths:
        raise ValueError(f"No .bench files found under '{bench_dir}'.")

    print(f"[DeepGate] Loading training benches from: {bench_dir}")
    print(f"[DeepGate] Found {len(bench_paths)} bench circuits")
    graphs = []
    for index, bench_path in enumerate(bench_paths):
        print(f"[DeepGate] ({index + 1}/{len(bench_paths)}) build graph for {bench_path.name}")
        graphs.append(
            build_training_graph(
                bench_path=bench_path,
                config=config,
                num_patterns=num_patterns,
                exact_pi_limit=exact_pi_limit,
                max_pairs=max_pairs,
                seed=seed + index,
            )
        )
    return graphs


def split_graphs(graphs, train_ratio):
    cutoff = max(1, min(len(graphs) - 1, math.floor(len(graphs) * train_ratio)))
    return graphs[:cutoff], graphs[cutoff:]


def run_epoch(model, loader, optimizer, device, prob_weight, func_weight):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_prob = 0.0
    total_func = 0.0
    total_graphs = 0

    phase = "train" if is_train else "val"
    print(f"[DeepGate] Start {phase} epoch pass: batches={len(loader)}")
    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device)
        predictions = model(batch)
        prob_pred = predictions[-1]
        node_emb = model.last_node_embedding

        prob_loss = F.smooth_l1_loss(prob_pred, batch.prob)
        node_a = node_emb[batch.tt_pair_index[0]]
        node_b = node_emb[batch.tt_pair_index[1]]
        emb_distance = 1.0 - F.cosine_similarity(node_a, node_b, dim=1, eps=1e-8)
        func_loss = F.smooth_l1_loss(emb_distance, batch.tt_dis)

        loss = prob_weight * prob_loss + func_weight * func_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_graphs = batch.num_graphs
        total_loss += loss.item() * batch_graphs
        total_prob += prob_loss.item() * batch_graphs
        total_func += func_loss.item() * batch_graphs
        total_graphs += batch_graphs
        print(
            f"[DeepGate][{phase}] batch={batch_idx}/{len(loader)} "
            f"graphs={batch_graphs} loss={loss.item():.4f} "
            f"prob={prob_loss.item():.4f} func={func_loss.item():.4f}"
        )

    return {
        "loss": total_loss / max(1, total_graphs),
        "prob_loss": total_prob / max(1, total_graphs),
        "func_loss": total_func / max(1, total_graphs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train DeepGate-style embeddings on bench graphs for C++ PODEM."
    )
    parser.add_argument("--bench_dir", required=True, help="Directory containing training .bench files.")
    parser.add_argument("--save_path", required=True, help="Path to save the trained checkpoint.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/validation split ratio.")
    parser.add_argument("--num_patterns", type=int, default=256, help="Random simulation patterns for large-PI circuits.")
    parser.add_argument("--exact_pi_limit", type=int, default=10, help="Use exact truth tables up to this PI count.")
    parser.add_argument("--max_pairs", type=int, default=512, help="Maximum gate pairs sampled per circuit.")
    parser.add_argument("--prob_weight", type=float, default=0.2, help="Weight of probability supervision.")
    parser.add_argument("--func_weight", type=float, default=1.0, help="Weight of truth-table distance supervision.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--num_rounds", type=int, default=10, help="Forward/backward message passing rounds.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    args = parser.parse_args()

    print(f"[DeepGate] Training settings: {args}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    config = EncoderConfig()
    config.model.num_rounds = args.num_rounds
    device = torch.device(args.device)
    print(
        "[DeepGate] Model config: "
        f"hidden={config.model.dim_hidden} rounds={config.model.num_rounds} "
        f"reverse={config.model.reverse} feature_dim={config.dim_node_feature}"
    )
    print(f"[DeepGate] Using device: {device}")
    model = RecGNN(_build_model_args(config, device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    graphs = load_training_graphs(
        bench_dir=args.bench_dir,
        config=config,
        num_patterns=args.num_patterns,
        exact_pi_limit=args.exact_pi_limit,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )
    train_graphs, val_graphs = split_graphs(graphs, args.train_ratio)
    print(
        f"[DeepGate] Dataset split: train_graphs={len(train_graphs)} "
        f"val_graphs={len(val_graphs)}"
    )
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"[DeepGate] ===== Epoch {epoch}/{args.epochs} =====")
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            prob_weight=args.prob_weight,
            func_weight=args.func_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                prob_weight=args.prob_weight,
                func_weight=args.func_weight,
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), save_path)
            print(f"[DeepGate] New best checkpoint saved to: {save_path}")

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_prob={train_metrics['prob_loss']:.4f} "
            f"train_func={train_metrics['func_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_prob={val_metrics['prob_loss']:.4f} "
            f"val_func={val_metrics['func_loss']:.4f}"
        )

    print(f"saved_checkpoint={save_path}")


if __name__ == "__main__":
    main()
