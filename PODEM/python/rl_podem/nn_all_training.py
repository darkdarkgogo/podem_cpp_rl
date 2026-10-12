"""Train an independent all-level backtrace scorer using PODEM paths."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch

from .nn_all_features import FEATURE_SCHEMA, candidate_vector, load_nn_all_features
from .native_io import _native_circuit_path


MODEL_FORMAT = "NN_ALL_PODEM_MODEL_V1"
DEFAULT_SEED = 2026


class PathSampleCollector:
    """Attribute a PODEM decision to backtrack or eventual detection."""

    def __init__(self, features, seed: int):
        self.features = features
        self.random = random.Random(seed)
        self.pending = {}
        self.samples = []
        self.excluded = 0
        self.current_fault = ""
        self.outcomes = Counter()

    def decision_callback(self, request):
        mask = tuple(bool(value) for value in request["action_mask"])
        legal = [index for index, allowed in enumerate(mask) if allowed]
        if not legal:
            raise ValueError("NN-All collector received no legal fanin")
        heuristic = int(request["heuristic_action"])
        action = heuristic if heuristic in legal else legal[0]
        if len(legal) > 1 and self.random.randrange(2):
            action = legal[1 - legal.index(action)]
        objective = str(request["objective_name"])
        candidate = str(request["candidate_names"][action])
        vector = candidate_vector(self.features, objective, candidate)
        self.pending[int(request["sequence"])] = (
            vector, int(self.features.levels[self.features.indices[objective]]),
            self.current_fault,
        )
        return action

    def event_callback(self, event):
        kind = event["event"]
        if kind == "episode_start":
            if self.pending:
                raise RuntimeError("NN-All collector episode started with pending paths")
            self.current_fault = str(event["fault_id"])
        elif kind == "backtrack":
            item = self.pending.pop(int(event["decision_sequence"]), None)
            if item is not None:
                self.samples.append((*item, 0))
        elif kind == "episode_end":
            outcome = int(event["outcome"])
            self.outcomes[outcome] += 1
            if outcome == 1:
                self.samples.extend((*item, 1) for item in self.pending.values())
            else:
                self.excluded += len(self.pending)
            self.pending.clear()
            self.current_fault = ""


class NnAllClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(11, 60), torch.nn.Sigmoid(),
            torch.nn.Linear(60, 1), torch.nn.Sigmoid(),
        )
        self.register_buffer("feature_mean", torch.zeros(11))
        self.register_buffer("feature_scale", torch.ones(11))

    def forward(self, vectors):
        transformed = torch.cat((torch.log1p(vectors[..., :3]), vectors[..., 3:]), dim=-1)
        return self.network(
            (transformed - self.feature_mean) / self.feature_scale
        ).squeeze(-1)


def _circuit_split(paths, seed):
    ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
    if len(ordered) < 2:
        raise ValueError("NN-All training requires at least two circuits")
    random.Random(seed).shuffle(ordered)
    holdout_count = max(1, round(len(ordered) * 0.2))
    return ordered[holdout_count:], ordered[:holdout_count]


def _sample_fault_ids(circuit_path, limit, seed):
    import cpp_podem

    faults = [entry["fault_id"] for entry in cpp_podem.catalog_stuck_at(_native_circuit_path(circuit_path))["faults"]]
    if not faults:
        raise ValueError(f"No stuck-at faults in {circuit_path}")
    indices = list(range(len(faults)))
    random.Random(seed).shuffle(indices)
    return [faults[index] for index in indices[:limit]]


def _collect_circuit(circuit_path, fault_ids, seed):
    import cpp_podem

    features = load_nn_all_features(circuit_path)
    collector = PathSampleCollector(features, seed)
    cpp_podem.run_stuck_at(
        _native_circuit_path(circuit_path), collector.decision_callback, collector.event_callback,
        100, seed, fault_ids, True, "backtrace_rl", "", True, False,
    )
    return collector


def _check_labels(samples, description):
    counts = Counter(sample[3] for sample in samples)
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            f"NN-All {description} must contain both path classes; "
            f"negative={counts[0]} positive={counts[1]}"
        )
    return counts


def _fit(train_samples, holdout_samples, seed, epochs):
    if epochs < 1:
        raise ValueError("NN-All epochs must be positive")
    _check_labels(train_samples, "train split")
    _check_labels(holdout_samples, "holdout split")
    torch.manual_seed(seed)
    model = NnAllClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vectors = torch.tensor([sample[0] for sample in train_samples], dtype=torch.float32)
    labels = torch.tensor([sample[3] for sample in train_samples], dtype=torch.float32)
    holdout_vectors = torch.tensor([sample[0] for sample in holdout_samples], dtype=torch.float32)
    holdout_labels = torch.tensor([sample[3] for sample in holdout_samples], dtype=torch.float32)
    transformed = torch.cat((torch.log1p(vectors[:, :3]), vectors[:, 3:]), dim=-1)
    with torch.no_grad():
        model.feature_mean.copy_(transformed.mean(dim=0))
        model.feature_scale.copy_(transformed.std(dim=0, unbiased=False).clamp_min(1.0))
    counts = Counter(labels.tolist())
    weights = torch.tensor(
        [len(labels) / (2 * counts[float(value)]) for value in labels.tolist()],
        dtype=torch.float32,
    )
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(labels), generator=generator).split(512):
            optimizer.zero_grad(set_to_none=True)
            predictions = model(vectors[batch])
            loss = torch.nn.functional.binary_cross_entropy(
                predictions, labels[batch], weight=weights[batch],
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            holdout_counts = Counter(holdout_labels.tolist())
            holdout_weights = torch.tensor(
                [len(holdout_labels) / (2 * holdout_counts[float(label)])
                 for label in holdout_labels.tolist()], dtype=torch.float32,
            )
            value = float(torch.nn.functional.binary_cross_entropy(
                model(holdout_vectors), holdout_labels, weight=holdout_weights,
            ))
        if value < best_loss:
            best_loss = value
            best_epoch = epoch
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def export_native_model(path, model, manifest_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = model.network[0]
    second = model.network[2]
    arrays = {
        "mean": model.feature_mean.detach().tolist(),
        "scale": model.feature_scale.detach().tolist(),
        "w1": first.weight.detach().flatten().tolist(),
        "b1": first.bias.detach().flatten().tolist(),
        "w2": second.weight.detach().flatten().tolist(),
        "b2": second.bias.detach().flatten().tolist(),
    }
    lines = [MODEL_FORMAT, f"feature_schema {FEATURE_SCHEMA}",
             "hidden_dim 60", f"manifest_sha256 {manifest_hash}"]
    for name, values in arrays.items():
        lines.append(f"{name} {len(values)} " + " ".join(f"{value:.9g}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    dataset_root, output_dir, *, seed=DEFAULT_SEED, faults_per_circuit=30,
    epochs=20, max_circuits=None,
):
    from .data_split import discover_training_dataset

    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = list(discover_training_dataset(dataset_root))
    if max_circuits is not None:
        if max_circuits < 2:
            raise ValueError("NN-All max-circuits must be at least two")
        paths = paths[:max_circuits]
    if faults_per_circuit < 1:
        raise ValueError("NN-All faults-per-circuit must be positive")
    train_paths, holdout_paths = _circuit_split(paths, seed)
    manifest = []
    samples = {"train": [], "holdout": []}
    excluded = Counter()
    outcomes = {"train": Counter(), "holdout": Counter()}
    for split, circuit_paths in (("train", train_paths), ("holdout", holdout_paths)):
        for circuit_path in circuit_paths:
            file_seed = (
                int.from_bytes(hashlib.sha256(circuit_path.name.encode()).digest()[:4], "big") ^ seed
            ) % 2147483647
            fault_ids = _sample_fault_ids(circuit_path, faults_per_circuit, file_seed)
            collector = _collect_circuit(circuit_path, fault_ids, file_seed)
            samples[split].extend(collector.samples)
            excluded[split] += collector.excluded
            outcomes[split].update(collector.outcomes)
            manifest.append({
                "split": split, "circuit": circuit_path.name,
                "sha256": hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
                "fault_ids": fault_ids,
            })
    manifest.sort(key=lambda item: (item["split"], item["circuit"]))
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    model, best_epoch, best_loss = _fit(samples["train"], samples["holdout"], seed, epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    export_native_model(output_dir / "model_best.txt", model, manifest_hash)
    summary = {
        "format": "NN_ALL_PODEM_TRAINING_V1",
        "feature_schema": FEATURE_SCHEMA,
        "dataset_root": str(dataset_root), "seed": seed,
        "circuits": len(paths), "faults_per_circuit": faults_per_circuit,
        "manifest_sha256": manifest_hash,
        "best_epoch": best_epoch, "holdout_loss": best_loss,
        "sample_counts": {
            split: {
                "positive": sum(item[3] == 1 for item in values),
                "negative": sum(item[3] == 0 for item in values),
                "excluded": excluded[split],
                "levels": dict(sorted(Counter(item[1] for item in values).items())),
                "outcomes": dict(outcomes[split]),
            }
            for split, values in samples.items()
        },
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary
