from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import torch
import torch.nn.functional as F

from .cpp_bridge import (
    CppPodemBacktraceV2Trainer,
    _fnv1a_file_hash,
    _load_cpp_embedding_artifact,
    _native_circuit_path,
)
from .ppo import BacktracePPOAgentV2, device


DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_TRAIN_COUNTS = {"easy": 40, "medium": 40, "hard": 20}
DEFAULT_VALIDATION_COUNTS = {"easy": 20, "medium": 20, "hard": 10}
REWARD_CONFIG = {
    "detected": 100.0,
    "undetected": -100.0,
    "backtrack_weight": 20.0,
    "backtrace_weight": 10.0,
    "gain_min": -2.0,
    "gain_max": 1.0,
}
REWARD_DISTRIBUTION = "incremental_potential_v1"


def rollout_length_stats(lengths: Iterable[int]) -> dict[str, Any]:
    """Include zero-decision faults; p90 uses the nearest-rank convention."""
    ordered = sorted(int(length) for length in lengths)
    if any(length < 0 for length in ordered):
        raise ValueError("Rollout lengths cannot be negative.")
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else 0,
        "mean": statistics.mean(ordered) if ordered else 0.0,
        "median": statistics.median(ordered) if ordered else 0.0,
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1] if ordered else 0,
        "max": ordered[-1] if ordered else 0,
        "zero_decision_faults": ordered.count(0),
    }


def _difficulty(index: int, count: int) -> str:
    fraction = (index + 1) / count
    if fraction <= 0.4:
        return "easy"
    if fraction <= 0.8:
        return "medium"
    return "hard"


def stratify_profiles(profiles: Iterable[Mapping[str, Any]]) -> dict[str, list[dict]]:
    eligible = [dict(item) for item in profiles if int(item["outcome"]) != 0]
    eligible.sort(
        key=lambda item: (
            int(item["backtracks"]),
            int(item.get("backtrace_steps", 0)),
            str(item["fault_id"]),
        )
    )
    if not eligible:
        raise ValueError("Cannot stratify an empty non-redundant fault set.")
    strata = {name: [] for name in DIFFICULTIES}
    for index, item in enumerate(eligible):
        difficulty = _difficulty(index, len(eligible))
        item["difficulty"] = difficulty
        strata[difficulty].append(item)
    return strata


def stratified_split(
    profiles: Iterable[Mapping[str, Any]],
    seed: Union[int, str],
    train_counts: Mapping[str, int] = DEFAULT_TRAIN_COUNTS,
    validation_counts: Mapping[str, int] = DEFAULT_VALIDATION_COUNTS,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    strata = stratify_profiles(profiles)
    training = []
    validation = []
    available = {}
    for difficulty in DIFFICULTIES:
        requested = int(train_counts[difficulty]) + int(validation_counts[difficulty])
        candidates = list(strata[difficulty])
        available[difficulty] = len(candidates)
        if len(candidates) < requested:
            raise ValueError(
                f"Difficulty '{difficulty}' has {len(candidates)} faults, "
                f"but {requested} are required."
            )
        random.Random(f"{seed}:{difficulty}").shuffle(candidates)
        train_count = int(train_counts[difficulty])
        validation_count = int(validation_counts[difficulty])
        training.extend(candidates[:train_count])
        validation.extend(candidates[train_count : train_count + validation_count])
    return training, validation, available


def baseline_relative_potential(
    backtracks: int,
    backtrace_steps: int,
    baseline: Mapping[str, Any],
    config: Mapping[str, float] = REWARD_CONFIG,
) -> tuple[float, dict[str, float]]:
    baseline_backtracks = int(baseline["backtracks"])
    baseline_backtrace = int(baseline["backtrace_steps"])
    gain_min = float(config["gain_min"])
    gain_max = float(config["gain_max"])
    backtrack_gain = (baseline_backtracks - int(backtracks)) / max(
        baseline_backtracks, 1
    )
    backtrace_gain = (baseline_backtrace - int(backtrace_steps)) / max(
        baseline_backtrace, 1
    )
    backtrack_gain = min(max(backtrack_gain, gain_min), gain_max)
    backtrace_gain = min(max(backtrace_gain, gain_min), gain_max)
    weighted_backtrack = float(config["backtrack_weight"]) * backtrack_gain
    weighted_backtrace = float(config["backtrace_weight"]) * backtrace_gain
    return weighted_backtrack + weighted_backtrace, {
        "backtrack_gain": backtrack_gain,
        "backtrace_gain": backtrace_gain,
        "weighted_backtrack_gain": weighted_backtrack,
        "weighted_backtrace_gain": weighted_backtrace,
    }


def baseline_relative_reward(
    outcome: int,
    backtracks: int,
    backtrace_steps: int,
    baseline: Mapping[str, Any],
    config: Mapping[str, float] = REWARD_CONFIG,
) -> tuple[float, dict[str, float]]:
    potential, parts = baseline_relative_potential(
        backtracks, backtrace_steps, baseline, config
    )
    detection_reward = (
        float(config["detected"])
        if int(outcome) == 1
        else float(config["undetected"])
    )
    parts["detection_reward"] = detection_reward
    return detection_reward + potential, parts


def collect_teacher_samples(
    circuit_name: str,
    circuit_path: Union[str, Path],
    fault_map_path: Union[str, Path],
    faults: Iterable[Mapping[str, Any]],
    backtrack_limit: int = 500,
    seed: int = 14,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Cannot import cpp_podem. Build the editable pybind extension first."
        ) from error

    grouped_faults: dict[str, list[str]] = {name: [] for name in DIFFICULTIES}
    for fault in faults:
        difficulty = str(fault["difficulty"])
        if difficulty not in grouped_faults:
            raise ValueError(f"Unknown curriculum difficulty: {difficulty}")
        grouped_faults[difficulty].append(str(fault["fault_id"]))

    samples: dict[tuple[str, int], dict[str, Any]] = {}
    summaries = {}
    active_difficulty = ""

    def decision_callback(request: dict[str, Any]) -> int:
        if request["mode"] != "backtrace":
            raise ValueError("Teacher collection supports backtrace decisions only.")
        candidates = list(request["candidate_names"])
        action = int(request["heuristic_action"])
        if len(candidates) != 2 or action not in (0, 1):
            raise ValueError(
                "V2 teacher collection requires exactly two candidates and a valid label."
            )
        key = (str(request["objective_name"]), int(request["objective_value"]))
        if key not in samples:
            samples[key] = {
                "circuit": circuit_name,
                "objective_name": key[0],
                "objective_value": key[1],
                "action_counts": [0, 0],
                "difficulty_counts": {name: 0 for name in DIFFICULTIES},
            }
        samples[key]["action_counts"][action] += 1
        samples[key]["difficulty_counts"][active_difficulty] += 1
        return action

    for difficulty in DIFFICULTIES:
        fault_ids = grouped_faults[difficulty]
        if not fault_ids:
            continue
        active_difficulty = difficulty
        summaries[difficulty] = dict(
            cpp_podem.run_stuck_at(
                _native_circuit_path(circuit_path),
                decision_callback,
                None,
                backtrack_limit,
                seed,
                fault_ids,
                True,
                "backtrace_rl",
                _native_circuit_path(fault_map_path),
            )
        )
    return sorted(
        samples.values(),
        key=lambda item: (item["objective_name"], item["objective_value"]),
    ), summaries


def _teacher_tensors(
    samples: Iterable[Mapping[str, Any]],
    embedding_tables: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Teacher dataset is empty.")
    group_sizes: dict[tuple[str, str], int] = defaultdict(int)
    for sample in sample_list:
        for difficulty, count in sample["difficulty_counts"].items():
            if int(count) > 0:
                group_sizes[(str(sample["circuit"]), str(difficulty))] += 1

    embeddings = []
    values = []
    targets = []
    weights = []
    circuits = []
    graph_indices = {name: table.name_to_index for name, table in embedding_tables.items()
                     if hasattr(table, "name_to_index")}
    for sample in sample_list:
        circuit = str(sample["circuit"])
        gate_name = str(sample["objective_name"])
        try:
            table = embedding_tables[circuit]
            if circuit in graph_indices:
                embeddings.append(torch.tensor([graph_indices[circuit][gate_name]], dtype=torch.long))
            else:
                embeddings.append(table[gate_name].float())
        except KeyError as error:
            raise KeyError(f"Missing teacher embedding for {circuit}:{gate_name}") from error
        values.append(int(sample["objective_value"]))
        counts = torch.tensor(sample["action_counts"], dtype=torch.float32)
        targets.append(counts / counts.sum())
        sample_weight = 0.0
        for difficulty, count in sample["difficulty_counts"].items():
            if int(count) > 0:
                sample_weight += 1.0 / group_sizes[(circuit, str(difficulty))]
        weights.append(sample_weight / len(DIFFICULTIES))
        circuits.append(circuit)
    return (
        torch.stack(embeddings),
        torch.tensor(values, dtype=torch.long),
        torch.stack(targets),
        torch.tensor(weights, dtype=torch.float32),
        circuits,
    )


def filter_unseen_teacher_samples(
    training_samples: Iterable[Mapping[str, Any]],
    validation_samples: Iterable[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    training_keys = {
        (
            str(sample["circuit"]),
            str(sample["objective_name"]),
            int(sample["objective_value"]),
        )
        for sample in training_samples
    }
    filtered = [
        sample
        for sample in validation_samples
        if (
            str(sample["circuit"]),
            str(sample["objective_name"]),
            int(sample["objective_value"]),
        )
        not in training_keys
    ]
    present_circuits = {str(sample["circuit"]) for sample in filtered}
    expected_circuits = {str(sample["circuit"]) for sample in validation_samples}
    missing = expected_circuits - present_circuits
    if missing:
        raise ValueError(
            "No unseen teacher validation states remain for: " + ", ".join(sorted(missing))
        )
    return filtered


def _teacher_logits(policy, embeddings, values, circuits, tables):
    if not hasattr(policy, "graph_encoder"):
        states = policy.gate_encoder(embeddings)
        return policy.backtrace_actor(states + policy.objective_value_embedding(values))
    grouped = defaultdict(list)
    for index, circuit in enumerate(circuits):
        grouped[circuit].append(index)
    logits, positions = [], []
    for circuit, indices in grouped.items():
        selected = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
        graph = tables[circuit]
        context = policy.context(graph, cached=not torch.is_grad_enabled())
        descriptors = policy.descriptors(graph, embeddings[selected, 0].long(), context=context)
        logits.append(policy.batch_logits(descriptors, values[selected])[0])
        positions.extend(indices)
    order = torch.tensor(positions, dtype=torch.long, device=embeddings.device).argsort()
    return torch.cat(logits)[order]


def teacher_accuracy(
    policy,
    samples: Iterable[Mapping[str, Any]],
    embedding_tables: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    embeddings, values, targets, _, circuits = _teacher_tensors(
        samples, embedding_tables
    )
    policy.eval()
    with torch.no_grad():
        logits = _teacher_logits(policy, embeddings.to(device), values.to(device),
                                 circuits, embedding_tables)
        predictions = torch.argmax(logits, dim=-1).cpu()
        labels = torch.argmax(targets, dim=-1)
    correct: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for prediction, label, circuit in zip(predictions, labels, circuits):
        correct[circuit] += int(int(prediction.item()) == int(label.item()))
        totals[circuit] += 1
    result = {
        circuit: correct[circuit] / totals[circuit] for circuit in sorted(totals)
    }
    result["mean_per_circuit"] = sum(result.values()) / len(result)
    return result


def pretrain_actor(
    agent: BacktracePPOAgentV2,
    training_samples: Iterable[Mapping[str, Any]],
    validation_samples: Iterable[Mapping[str, Any]],
    embedding_tables: Mapping[str, Mapping[str, torch.Tensor]],
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    seed: int = 2026,
    resume_state: Optional[Mapping[str, Any]] = None,
    epoch_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Pretraining epochs and batch size must be positive.")
    embeddings, values, targets, weights, circuits = _teacher_tensors(
        training_samples, embedding_tables
    )
    actor_parameters = list(agent.policy.gate_encoder.parameters())
    actor_parameters += list(agent.policy.objective_value_embedding.parameters())
    actor_parameters += list(agent.policy.backtrace_actor.parameters())
    if hasattr(agent.policy, "graph_encoder"):
        actor_parameters += list(agent.policy.graph_encoder.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_accuracy = -math.inf
    history = []
    start_epoch = 0
    if resume_state is not None:
        start_epoch = int(resume_state["epoch"])
        if start_epoch < 0 or start_epoch > epochs:
            raise ValueError("Behavior-cloning resume epoch is out of range.")
        agent.policy.load_state_dict(resume_state["policy_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        generator.set_state(resume_state["generator_state"])
        best_state = resume_state["best_policy_state"]
        best_accuracy = float(resume_state["best_validation_accuracy"])
        history = list(resume_state["history"])
        if len(history) != start_epoch:
            raise ValueError("Behavior-cloning history does not match resume epoch.")

    embeddings_device = embeddings.to(device)
    values_device = values.to(device)
    targets_device = targets.to(device)
    weights_device = weights.to(device)
    for epoch in range(start_epoch + 1, epochs + 1):
        permutation = torch.randperm(len(embeddings), generator=generator)
        loss_sum = 0.0
        weight_sum = 0.0
        agent.policy.train()
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size].to(device)
            batch_circuits = [circuits[index] for index in indices.cpu().tolist()]
            logits = _teacher_logits(agent.policy, embeddings_device[indices],
                                     values_device[indices], batch_circuits, embedding_tables)
            loss_tensor = -(
                targets_device[indices] * F.log_softmax(logits, dim=-1)
            ).sum(dim=-1)
            batch_weights_tensor = weights_device[indices]
            loss = (loss_tensor * batch_weights_tensor).sum() / batch_weights_tensor.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float((loss_tensor.detach() * batch_weights_tensor).sum().item())
            weight_sum += float(batch_weights_tensor.sum().item())

        validation = teacher_accuracy(
            agent.policy, validation_samples, embedding_tables
        )
        record = {
            "epoch": epoch,
            "loss": loss_sum / weight_sum,
            "validation": validation,
        }
        history.append(record)
        score = float(validation["mean_per_circuit"])
        if score > best_accuracy:
            best_accuracy = score
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in agent.policy.state_dict().items()
            }
        if epoch_callback is not None:
            epoch_callback(
                {
                    "epoch": epoch,
                    "policy_state": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in agent.policy.state_dict().items()
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                    "best_policy_state": best_state,
                    "best_validation_accuracy": best_accuracy,
                    "history": list(history),
                }
            )

    if best_state is None:
        raise RuntimeError("Behavior cloning did not produce a checkpoint.")
    agent.policy.load_state_dict(best_state)
    with torch.no_grad():
        # Zeroing every Tanh layer traps the Critic at a trainable constant bias.
        # A zero output head keeps V(s)=1 initially while preserving hidden features.
        agent.policy.critic[-1].weight.zero_()
        agent.policy.critic[-1].bias.fill_(1.0)
    agent.policy_old.load_state_dict(agent.policy.state_dict())
    initialized_best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in agent.policy.state_dict().items()
    }
    return {
        "history": history,
        "best_validation_accuracy": best_accuracy,
        "best_policy_state": initialized_best_state,
        "optimizer_state": optimizer.state_dict(),
    }


class CppPodemCurriculumTrainer(CppPodemBacktraceV2Trainer):
    def __init__(
        self,
        embedding_path: Union[str, Path],
        baselines: Mapping[str, Mapping[str, Any]],
        agent: Optional[BacktracePPOAgentV2] = None,
        reward_config: Mapping[str, float] = REWARD_CONFIG,
    ):
        super().__init__(embedding_path, agent=agent)
        self.baselines = {str(key): dict(value) for key, value in baselines.items()}
        self.reward_config = dict(reward_config)
        self.current_fault_id: Optional[str] = None
        self._event_backtracks = 0
        self._event_backtrace_steps = 0
        self._event_potential = 0.0
        self._distributed_potential_reward = 0.0
        self._attributed_backtracks = 0
        self._attributed_backtrace_steps = 0

    def _attribute_potential_delta(self, decision_sequence: int) -> bool:
        baseline = self.baselines[self.current_fault_id]
        potential, _ = baseline_relative_potential(
            self._event_backtracks,
            self._event_backtrace_steps,
            baseline,
            self.reward_config,
        )
        delta = potential - self._event_potential
        self._event_potential = potential
        step_idx = self.sequence_to_step.get(decision_sequence)
        if step_idx is None or not 0 <= step_idx < len(self.agent.buffer.steps):
            return False
        self.agent.add_reward_to_step(step_idx, delta)
        self._distributed_potential_reward += delta
        return True

    def event_callback(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "episode_start":
            self.sequence_to_step.clear()
            self.current_fault_id = str(event["fault_id"])
            if self.current_fault_id not in self.baselines:
                raise KeyError(f"Missing heuristic baseline for {self.current_fault_id}")
            self._event_backtracks = 0
            self._event_backtrace_steps = 0
            self._event_potential, _ = baseline_relative_potential(
                0,
                0,
                self.baselines[self.current_fault_id],
                self.reward_config,
            )
            self._distributed_potential_reward = 0.0
            self._attributed_backtracks = 0
            self._attributed_backtrace_steps = 0
            return
        if event_type == "backtrack":
            self._event_backtracks += 1
            sequence = int(event["decision_sequence"])
            if self._attribute_potential_delta(sequence):
                self._attributed_backtracks += 1
            return
        if event_type == "backtrace_step":
            self._event_backtrace_steps += 1
            sequence = int(event["decision_sequence"])
            if self._attribute_potential_delta(sequence):
                self._attributed_backtrace_steps += 1
            return
        if event_type == "pi_not_done":
            return
        if event_type != "episode_end":
            raise ValueError(f"Unknown C++ PODEM event: {event_type}")
        fault_id = str(event["fault_id"])
        if fault_id != self.current_fault_id:
            raise RuntimeError("PODEM episode-end fault does not match episode-start fault.")
        reward, components = baseline_relative_reward(
            int(event["outcome"]),
            int(event["backtracks"]),
            int(event["backtrace_steps"]),
            self.baselines[fault_id],
            self.reward_config,
        )
        terminal_residual = reward - self._distributed_potential_reward
        self.agent.finish_episode(terminal_residual)
        self.last_metrics = self.agent.update()
        if self.last_metrics is None:
            metrics = {
                "steps": 0,
                "reward_sum": reward,
                "intrinsic_reward_sum": 0.0,
                "rnd_loss": 0.0,
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "ratio_mean": 0.0,
                "updated": False,
            }
        else:
            metrics = dict(self.last_metrics)
            metrics["updated"] = True
        backtrack_mismatch = int(event["backtracks"]) - self._event_backtracks
        backtrace_mismatch = (
            int(event["backtrace_steps"]) - self._event_backtrace_steps
        )
        metrics.update(components)
        metrics["extrinsic_reward_sum"] = reward
        metrics["distributed_potential_reward"] = self._distributed_potential_reward
        metrics["terminal_reward_residual"] = terminal_residual
        metrics["attributed_backtracks"] = self._attributed_backtracks
        metrics["attributed_backtrace_steps"] = self._attributed_backtrace_steps
        metrics["backtrack_event_mismatch"] = backtrack_mismatch
        metrics["backtrace_event_mismatch"] = backtrace_mismatch
        metrics["backtrack_event_mismatch_abs"] = abs(backtrack_mismatch)
        metrics["backtrace_event_mismatch_abs"] = abs(backtrace_mismatch)
        metrics["counter_mismatch_episode"] = int(
            backtrack_mismatch != 0 or backtrace_mismatch != 0
        )
        metrics["scaled_intrinsic_reward_sum"] = (
            self.agent.rnd_beta * metrics["intrinsic_reward_sum"]
        )
        metrics["outcome"] = int(event["outcome"])
        metrics["fault_id"] = fault_id
        self.episode_metrics.append(metrics)

    def run(self, *args, **kwargs) -> dict[str, Any]:
        summary = super().run(*args, **kwargs)
        updated_metrics = [
            item for item in self.episode_metrics if item.get("updated", True)
        ]
        self.run_metrics["episodes_with_updates"] = len(updated_metrics)
        self.run_metrics["advantage_method"] = self.agent.advantage_method
        self.run_metrics["rollout_steps"] = rollout_length_stats(
            item["steps"] for item in self.episode_metrics
        )
        mean_keys = {
            "total_loss_mean": "total_loss",
            "policy_loss_mean": "policy_loss",
            "value_loss_mean": "value_loss",
            "entropy_mean": "entropy",
            "ratio_mean": "ratio_mean",
            "rnd_loss_mean": "rnd_loss",
            "scaled_reward_mean": "scaled_reward_mean",
            "scaled_reward_std_mean": "scaled_reward_std",
            "raw_adv_mean": "raw_adv_mean",
            "raw_adv_std_mean": "raw_adv_std",
            "actor_adv_mean": "actor_adv_mean",
            "actor_adv_std_mean": "actor_adv_std",
            "value_target_mean": "value_target_mean",
            "value_target_std_mean": "value_target_std",
        }
        for output_key, source_key in mean_keys.items():
            self.run_metrics[output_key] = (
                sum(item[source_key] for item in updated_metrics)
                / len(updated_metrics)
                if updated_metrics
                else 0.0
            )
        sum_keys = (
            "distributed_potential_reward",
            "terminal_reward_residual",
            "attributed_backtracks",
            "attributed_backtrace_steps",
            "backtrack_event_mismatch",
            "backtrace_event_mismatch",
            "backtrack_event_mismatch_abs",
            "backtrace_event_mismatch_abs",
            "counter_mismatch_episode",
        )
        for key in sum_keys:
            self.run_metrics[key] = sum(
                item.get(key, 0.0) for item in self.episode_metrics
            )
        return summary

    def set_exploration(self, rnd_beta: float, entropy_coef: float) -> None:
        if rnd_beta < 0 or entropy_coef < 0:
            raise ValueError("Exploration coefficients must be non-negative.")
        self.agent.rnd_beta = float(rnd_beta)
        self.agent.entropy_coef = float(entropy_coef)


def load_embedding_tables(circuits: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    tables = {}
    dimension = None
    for circuit in circuits:
        expected_hash = _fnv1a_file_hash(circuit["circuit"])
        artifact_hash, table = _load_cpp_embedding_artifact(circuit["embeddings"])
        if artifact_hash != expected_hash:
            raise ValueError(f"Embedding circuit hash mismatch for {circuit['name']}.")
        current_dimension = next(iter(table.values())).numel()
        if dimension is None:
            dimension = current_dimension
        elif current_dimension != dimension:
            raise ValueError("All curriculum embeddings must have the same dimension.")
        tables[str(circuit["name"])] = table
    return tables
