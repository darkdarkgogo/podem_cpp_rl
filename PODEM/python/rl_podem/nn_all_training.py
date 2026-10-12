"""Train an independent all-level backtrace scorer using PODEM paths."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .nn_all_features import FEATURE_SCHEMA, candidate_vector, load_nn_all_features
from .native_io import _native_circuit_path


MODEL_FORMAT = "NN_ALL_PODEM_MODEL_V1"
TRAINING_FORMAT = "NN_ALL_PODEM_TRAINING_V3_PAPER_CIRCUITS_SINGLE_EPOCH"
DEFAULT_SEED = 2026
PAPER_TRAIN_CIRCUITS = ("c1908.bench", "c2670.bench")
SAMPLE_WIDTH = 12
BATCH_SIZE = 512


class PathSampleCollector:
    """Attribute a PODEM decision to backtrack or eventual detection."""

    def __init__(self, features, seed: int):
        self.features = features
        self.random = random.Random(seed)
        self.pending = {}
        self.pi_paths = {}
        self.samples = []
        self.excluded = 0
        self.current_fault = ""
        self.outcomes = Counter()

    def decision_callback(self, request):
        mask = tuple(bool(value) for value in request["action_mask"])
        legal = [index for index, allowed in enumerate(mask) if allowed]
        if not legal:
            raise ValueError("NN-All collector received no legal fanin")
        if len(legal) == 1:
            return legal[0]
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
            self.pi_paths.clear()
            self.current_fault = str(event["fault_id"])
        elif kind == "pi_assignment":
            first = int(event["first_sequence"])
            last = int(event["last_sequence"])
            self.pi_paths[str(event["pi_name"])] = tuple(
                sequence for sequence in range(first, last + 1)
                if sequence in self.pending
            )
        elif kind == "backtrack_pi":
            for sequence in self.pi_paths.pop(str(event["pi_name"]), ()):
                item = self.pending.pop(sequence, None)
                if item is not None:
                    self.samples.append((*item, 0))
        elif kind == "backtrace_dead_end":
            for sequence in range(
                int(event["first_sequence"]), int(event["last_sequence"]) + 1,
            ):
                item = self.pending.pop(sequence, None)
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
            self.pi_paths.clear()
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


def _sample_fault_ids(circuit_path):
    import cpp_podem

    faults = [entry["fault_id"] for entry in cpp_podem.catalog_stuck_at(_native_circuit_path(circuit_path))["faults"]]
    if not faults:
        raise ValueError(f"No stuck-at faults in {circuit_path}")
    return faults


def _collect_circuit(circuit_path, fault_ids, seed):
    import cpp_podem

    features = load_nn_all_features(circuit_path)
    collector = PathSampleCollector(features, seed)
    cpp_podem.run_stuck_at(
        _native_circuit_path(circuit_path), collector.decision_callback, collector.event_callback,
        100, seed, fault_ids, True, "backtrace_rl", "", True, False,
    )
    return collector


def _check_labels(counts):
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(
            "NN-All training data must contain both path classes; "
            f"negative={counts[0]} positive={counts[1]}"
        )


def _fit(samples, seed):
    """Make one training pass over every collected path, then keep final weights."""
    counts = Counter(int(value) for value in samples[:, 11])
    _check_labels(counts)
    torch.manual_seed(seed)
    model = NnAllClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total = np.zeros(11, dtype=np.float64)
    total_squared = np.zeros(11, dtype=np.float64)
    for start in range(0, len(samples), 8192):
        vectors = np.array(samples[start:start + 8192, :11], dtype=np.float64)
        vectors[:, :3] = np.log1p(vectors[:, :3])
        total += vectors.sum(axis=0)
        total_squared += np.square(vectors).sum(axis=0)
    mean = total / len(samples)
    scale = np.sqrt(np.maximum(total_squared / len(samples) - np.square(mean), 0.0))
    with torch.no_grad():
        model.feature_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        model.feature_scale.copy_(torch.from_numpy(np.maximum(scale, 1.0).astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    loss_sum = 0.0
    model.train()
    for batch in torch.randperm(len(samples), generator=generator).split(BATCH_SIZE):
        rows = np.asarray(samples[batch.numpy()], dtype=np.float32)
        vectors = torch.from_numpy(rows[:, :11].copy())
        labels = torch.from_numpy(rows[:, 11].copy())
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy(model(vectors), labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(batch)
    return model, loss_sum / len(samples)


def export_native_model(path, model, manifest_hash, seed=DEFAULT_SEED):
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
             "hidden_dim 60", f"manifest_sha256 {manifest_hash}",
             f"seed {int(seed)}"]
    for name, values in arrays.items():
        lines.append(f"{name} {len(values)} " + " ".join(f"{value:.9g}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(dataset_root, output_dir, *, seed=DEFAULT_SEED):
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    paper_train_dir = dataset_root / "nn_all_paper_train"
    paths = [paper_train_dir / name for name in PAPER_TRAIN_CIRCUITS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing NN-All paper training circuit(s): " + ", ".join(missing))
    manifest = []
    counts = Counter()
    levels = Counter()
    outcomes = Counter()
    excluded = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "training_samples.f32"
    try:
        with sample_path.open("wb") as stream:
            for circuit_path in paths:
                file_seed = (
                    int.from_bytes(hashlib.sha256(circuit_path.name.encode()).digest()[:4], "big") ^ seed
                ) % 2147483647
                fault_ids = _sample_fault_ids(circuit_path)
                collector = _collect_circuit(circuit_path, fault_ids, file_seed)
                rows = np.empty((len(collector.samples), SAMPLE_WIDTH), dtype="<f4")
                for index, (vector, level, _fault_id, label) in enumerate(collector.samples):
                    rows[index, :11] = vector
                    rows[index, 11] = label
                    counts[label] += 1
                    levels[level] += 1
                stream.write(rows.tobytes())
                excluded += collector.excluded
                outcomes.update(collector.outcomes)
                manifest.append({
                    "circuit": circuit_path.name,
                    "sha256": hashlib.sha256(circuit_path.read_bytes()).hexdigest(),
                    "fault_ids": fault_ids,
                })
        _check_labels(counts)
        samples = np.memmap(sample_path, dtype="<f4", mode="r",
                            shape=(sum(counts.values()), SAMPLE_WIDTH))
        model, epoch_loss = _fit(samples, seed)
        del samples
    finally:
        sample_path.unlink(missing_ok=True)
    manifest.sort(key=lambda item: item["circuit"])
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    model_path = output_dir / "model_final.txt"
    export_native_model(model_path, model, manifest_hash, seed)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = {
        "format": TRAINING_FORMAT,
        "feature_schema": FEATURE_SCHEMA,
        "dataset_root": str(dataset_root), "seed": seed,
        "circuits": len(paths), "training_circuits": list(PAPER_TRAIN_CIRCUITS),
        "fault_selection": "all_catalog_stuck_at",
        "manifest_sha256": manifest_hash,
        "model_sha256": model_hash,
        "epochs": 1, "checkpoint_policy": "final_epoch",
        "loss": "binary_cross_entropy", "epoch_loss": epoch_loss,
        "sample_counts": {
            "positive": counts[1], "negative": counts[0],
            "excluded": excluded,
            "levels": dict(sorted(levels.items())),
            "outcomes": dict(outcomes),
        },
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return summary
