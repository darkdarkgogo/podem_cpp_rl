from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch

from .ppo import BacktracePPOAgentV2


def _load_cpp_embedding_artifact(
    path: Union[str, Path], *, expected_backend="smartatpg", include_metadata=False,
) -> Tuple[str, dict[str, torch.Tensor]]:
    def read_pair(handle, expected_key):
        line = handle.readline()
        if not line:
            raise ValueError(f"Truncated embedding file: {path}")
        parts = line.split()
        if len(parts) != 2 or parts[0] != expected_key:
            raise ValueError(f"Expected '{expected_key} <value>' in {path}")
        return parts[1]

    table: dict[str, torch.Tensor] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        if header not in (
            "SMARTATPG_EMBEDDINGS_V2", "SMARTATPG_EMBEDDINGS_V3",
            "SMARTATPG_EMBEDDINGS_V4",
        ):
            raise ValueError(f"Unsupported embedding format: {path}")
        if header == "SMARTATPG_EMBEDDINGS_V2":
            raise ValueError(
                "Legacy 80-dimensional SmartATPG descriptors are incompatible "
                "with the 11-dimensional GraphSAGE format"
            )
        keys = (
            ("backend", "feature_schema", "graph_config", "gate_embedding_dim",
             "policy_state_dim", "snapshot")
            if header == "SMARTATPG_EMBEDDINGS_V3" else
            ("backend", "feature_schema", "encoder_variant", "graph_config",
             "gate_embedding_dim", "actor_input_dim", "action_mask_dim",
             "decision_state_dim", "snapshot")
        )
        metadata = {key: read_pair(handle, key) for key in keys}
        valid_graph = (
            metadata["graph_config"] == "fanin_mean_1x22x11"
            if header == "SMARTATPG_EMBEDDINGS_V3" else
            metadata["graph_config"] in (
                "fanin_mean_1x22x11", "level_gat_gru_fwd_rev_11d_v1"
            )
        )
        dimensions_valid = (
            metadata.get("policy_state_dim") == "13"
            if header == "SMARTATPG_EMBEDDINGS_V3" else
            metadata.get("actor_input_dim") == "11"
            and metadata.get("action_mask_dim") == "2"
            and metadata.get("decision_state_dim") == "13"
        )
        if (
            metadata["backend"] != "smartatpg"
            or metadata["feature_schema"] != "SMARTATPG_FEATURES_V2_11D"
            or not valid_graph
            or metadata["gate_embedding_dim"] != "11"
            or not dimensions_valid
            or len(metadata["snapshot"]) != 64
            or any(char not in "0123456789abcdef" for char in metadata["snapshot"])
        ):
            raise ValueError("Invalid SmartATPG 11D embedding metadata")
        if expected_backend is not None and expected_backend != metadata["backend"]:
            raise ValueError(f"Descriptor backend {metadata['backend']} conflicts with {expected_backend}")
        circuit_hash = read_pair(handle, "circuit_hash")
        dimension = int(read_pair(handle, "dimension"))
        count = int(read_pair(handle, "count"))
        if dimension <= 0 or count < 0:
            raise ValueError(f"Invalid embedding dimensions in: {path}")
        if dimension != 11:
            raise ValueError("SmartATPG gate embedding dimension must be 11")

        for row in range(count):
            line = handle.readline()
            if not line:
                raise ValueError(f"Truncated embedding file at row {row}: {path}")
            parts = line.split()
            if len(parts) != dimension + 1:
                raise ValueError(
                    f"Expected {dimension} values at embedding row {row}, "
                    f"found {max(len(parts) - 1, 0)} in {path}"
                )
            name = parts[0]
            if name in table:
                raise ValueError(f"Duplicate embedding name: {name}")
            table[name] = torch.tensor(
                [float(value) for value in parts[1:]], dtype=torch.float32
            )
            if not bool(torch.isfinite(table[name]).all()):
                raise ValueError(f"Non-finite embedding for {name}")

        if any(line.strip() for line in handle):
            raise ValueError(f"Unexpected trailing data in embedding file: {path}")
    return (circuit_hash, table, metadata) if include_metadata else (circuit_hash, table)


def load_cpp_embedding_table(path: Union[str, Path]) -> dict[str, torch.Tensor]:
    return _load_cpp_embedding_artifact(path)[1]


def _fnv1a_file_hash(path: Union[str, Path]) -> str:
    value = 14695981039346656037
    for byte in Path(path).read_bytes():
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _native_circuit_path(path: Union[str, Path]) -> str:
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)
    try:
        # The legacy C++ reader uses narrow paths. A relative ASCII path avoids
        # losing Unicode characters from the absolute Windows workspace path.
        relative = os.path.relpath(resolved, Path.cwd())
    except ValueError:
        return str(resolved)
    return relative if relative.isascii() else str(resolved)


def profile_cpp_podem(
    circuit_path: Union[str, Path],
    backtrack_limit: int = 97,
    seed: int = 14,
    fault_map_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Cannot import cpp_podem. Install this project in the active environment with "
            "'python -m pip install -e .'."
        ) from error
    return list(
        cpp_podem.profile_stuck_at(
            _native_circuit_path(circuit_path),
            backtrack_limit,
            seed,
            _native_circuit_path(fault_map_path) if fault_map_path else "",
        )
    )


def catalog_cpp_podem(
    circuit_path: Union[str, Path],
    fault_map_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    try:
        import cpp_podem
    except ImportError as error:
        raise ImportError(
            "Cannot import cpp_podem. Install this project in the active environment with "
            "'python -m pip install -e .'."
        ) from error
    return dict(
        cpp_podem.catalog_stuck_at(
            _native_circuit_path(circuit_path),
            _native_circuit_path(fault_map_path) if fault_map_path else "",
        )
    )


def export_actor_v2_state_dict(
    state_dict: dict[str, torch.Tensor], path: Union[str, Path], *, metadata=None
) -> None:
    if metadata is None:
        raise ValueError("SmartATPG weights require a snapshot-paired model export")
    actor_tensor_names = [
        "gate_encoder.0.weight",
        "gate_encoder.0.bias",
        "objective_value_embedding.weight",
        "backtrace_actor.0.weight",
        "backtrace_actor.0.bias",
        "backtrace_actor.2.weight",
        "backtrace_actor.2.bias",
    ]
    variant = metadata.get("encoder_variant")
    if variant == "fanin_mean":
        encoder_tensor_names = (
            "graph_encoder.layer.weight", "graph_encoder.layer.bias",
        )
    elif variant == "level_gat_gru":
        encoder_tensor_names = tuple(
            f"graph_encoder.{direction}.{name}"
            for direction in ("forward_pass", "reverse_pass")
            for name in (
                "attention", "projection.weight", "gru.weight_ih",
                "gru.weight_hh", "gru.bias_ih", "gru.bias_hh",
            )
        )
    else:
        raise ValueError(f"Unsupported SmartATPG encoder variant: {variant}")
    tensor_names = [*encoder_tensor_names, *actor_tensor_names]
    missing = [name for name in tensor_names if name not in state_dict]
    if missing:
        raise KeyError("Checkpoint is missing V2 actor tensors: " + ", ".join(missing))

    gate_weight = state_dict["gate_encoder.0.weight"]
    hidden_dim, embedding_dim = gate_weight.shape
    if (
        metadata.get("backend") != "smartatpg"
        or int(metadata.get("gate_embedding_dim", -1)) != 11
        or int(metadata.get("actor_input_dim", -1)) != 11
        or int(metadata.get("action_mask_dim", -1)) != 2
        or int(metadata.get("decision_state_dim", -1)) != 13
        or embedding_dim != 11
    ):
        raise ValueError("SmartATPG Actor requires an 11D input and a separate 2D mask")
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        output.write("SMARTATPG_MODEL_V6\n")
        for key in (
            "backend", "feature_schema", "encoder_variant", "graph_config",
            "gate_embedding_dim", "actor_input_dim", "action_mask_dim",
            "decision_state_dim", "snapshot",
        ):
            output.write(f"{key} {metadata[key]}\n")
        output.write(f"best_round {int(metadata.get('best_round', 0))}\n")
        output.write(f"best_score {metadata.get('best_score', 'none')}\n")
        output.write(f"hidden_dim {hidden_dim}\n")
        for name in tensor_names:
            tensor = state_dict[name].detach().cpu().float().contiguous()
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Non-finite actor tensor: {name}")
            if tensor.ndim == 1:
                rows, cols = 1, tensor.shape[0]
            elif tensor.ndim == 2:
                rows, cols = tensor.shape
            else:
                raise ValueError(
                    f"Unsupported V2 actor tensor rank for {name}: {tensor.ndim}"
                )
            output.write(f"tensor {name} {rows} {cols}\n")
            output.write(
                " ".join(format(value, ".9g") for value in tensor.reshape(-1).tolist())
            )
            output.write("\n")
        output.write("end\n")
    temporary.replace(output_path)


def smartatpg_pi_reward(
    backtracks: int,
    pi_visits: int,
    alpha: float = 7.5,
    beta: float = 0.07,
) -> float:
    if backtracks < 0 or pi_visits <= 0:
        raise ValueError("SmartATPG reward counters are out of range.")
    return 10.0 - alpha * math.exp(beta * (backtracks + pi_visits))


class _CppPodemTrainerBase:
    def __init__(
        self,
        graph,
        agent: BacktracePPOAgentV2,
    ):
        from .smartatpg_features import CircuitGraph
        from .smartatpg import ACTOR_INPUT_DIM, GraphGate
        if not isinstance(graph, CircuitGraph):
            raise TypeError("SmartATPG training requires a CircuitGraph")
        if getattr(agent, "embedding_backend", None) != "smartatpg":
            raise ValueError("Circuit graph inputs require a SmartATPG agent")
        self.circuit_hash = graph.circuit_hash
        if self.circuit_hash not in agent.graphs:
            raise ValueError("Circuit graph is not registered with this agent")
        self.gates = {name: GraphGate(name, self.circuit_hash, index)
                      for index, name in enumerate(graph.names)}
        embedding_dim = ACTOR_INPUT_DIM
        self.agent = agent
        agent_input_dim = getattr(
            self.agent, "actor_input_dim", self.agent.gate_embedding_dim
        )
        if agent_input_dim != embedding_dim:
            raise ValueError(
                "Shared PPO agent input dimension does not match the circuit inputs."
            )
        self.sequence_to_step: dict[int, int] = {}
        self.last_metrics: Optional[dict[str, Any]] = None

    def _gate(self, name: str):
        try:
            return self.gates[name]
        except KeyError as error:
            raise KeyError(f"No policy input for C++ wire '{name}'.") from error

    def run(
        self,
        circuit_path: Union[str, Path],
        backtrack_limit: int = 97,
        seed: int = 14,
        fault_ids: Optional[list[str]] = None,
        quiet: bool = True,
        rl_mode: str = "backtrace_rl",
        fault_map_path: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]:
        resolved_circuit_path = Path(circuit_path).resolve()
        actual_hash = _fnv1a_file_hash(resolved_circuit_path)
        if actual_hash != self.circuit_hash:
            raise ValueError(
                "Embedding circuit hash mismatch: "
                f"expected {self.circuit_hash}, found {actual_hash}"
            )
        try:
            import cpp_podem
        except ImportError as error:
            raise ImportError(
                "Cannot import cpp_podem. Install this project in the active environment with "
                "'python -m pip install -e .'."
            ) from error
        return cpp_podem.run_stuck_at(
            _native_circuit_path(resolved_circuit_path),
            self.decision_callback,
            self.event_callback,
            backtrack_limit,
            seed,
            fault_ids,
            quiet,
            rl_mode,
            _native_circuit_path(fault_map_path) if fault_map_path else "",
        )

class CppPodemBacktraceV2Trainer(_CppPodemTrainerBase):
    def __init__(
        self,
        graph,
        agent: BacktracePPOAgentV2,
    ):
        super().__init__(graph, agent=agent)
        self.reward_alpha = 7.5
        self.reward_beta = 0.07
        self.non_pi_reward = -0.1
        self.detected_reward = 100.0
        self.undetected_reward = -100.0
        self.episode_metrics: list[dict[str, Any]] = []
        self.run_metrics: dict[str, Any] = {}
        self._episode_extrinsic_reward = 0.0

    def decision_callback(self, request: dict[str, Any]) -> int:
        if request["mode"] != "backtrace":
            raise ValueError("V2 actor supports backtrace decisions only.")
        candidates = [self._gate(name) for name in request["candidate_names"]]
        selected = self.agent.select_backtrace_action(
            self._gate(request["objective_name"]),
            int(request["objective_value"]),
            candidates,
            request.get("action_mask", [True, True]),
        )
        action = next(
            index for index, candidate in enumerate(candidates) if candidate is selected
        )
        if self.agent.last_selected_step_idx is not None:
            self.sequence_to_step[int(request["sequence"])] = (
                self.agent.last_selected_step_idx
            )
        return action

    def event_callback(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "episode_start":
            self.sequence_to_step.clear()
            self._episode_extrinsic_reward = 0.0
            return
        if event_type == "backtrack":
            return
        if event_type == "backtrace_step":
            step_idx = self.sequence_to_step.get(int(event["decision_sequence"]))
            if step_idx is not None and 0 <= step_idx < len(self.agent.buffer.steps):
                self.agent.add_reward_to_step(step_idx, self.non_pi_reward)
                self._episode_extrinsic_reward += self.non_pi_reward
            return
        if event_type == "pi_not_done":
            step_idx = self.sequence_to_step.get(int(event["decision_sequence"]))
            reward = smartatpg_pi_reward(
                int(event["backtracks"]),
                int(event["pi_visits"]),
                self.reward_alpha,
                self.reward_beta,
            )
            if step_idx is not None and 0 <= step_idx < len(self.agent.buffer.steps):
                self.agent.add_reward_to_step(step_idx, reward)
                self._episode_extrinsic_reward += reward
            return
        if event_type != "episode_end":
            raise ValueError(f"Unknown C++ PODEM event: {event_type}")

        terminal_reward = (
            self.detected_reward
            if int(event["outcome"]) == 1
            else self.undetected_reward
        )
        self.agent.finish_episode(terminal_reward)
        self._episode_extrinsic_reward += terminal_reward
        self.last_metrics = self.agent.update()
        metrics = dict(self.last_metrics or {
            "steps": 0,
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ratio_mean": 0.0,
            "rnd_loss": 0.0,
            "intrinsic_reward_sum": 0.0,
            "reward_sum": self._episode_extrinsic_reward,
        })
        metrics["extrinsic_reward_sum"] = self._episode_extrinsic_reward
        metrics["scaled_intrinsic_reward_sum"] = (
            self.agent.rnd_beta * metrics["intrinsic_reward_sum"]
        )
        metrics["combined_reward_sum"] = (
            metrics["extrinsic_reward_sum"]
            + metrics["scaled_intrinsic_reward_sum"]
        )
        metrics["fault_id"] = event.get("fault_id", "")
        metrics["outcome"] = int(event["outcome"])
        metrics["detected"] = int(event["outcome"] == 1)
        metrics["backtracks"] = int(event["backtracks"])
        metrics["backtrace_steps"] = int(event["backtrace_steps"])
        metrics["pi_visits"] = int(event["pi_visits"])
        self.episode_metrics.append(metrics)

    def run(self, *args, rl_mode: str = "backtrace_rl", **kwargs) -> dict[str, Any]:
        if rl_mode != "backtrace_rl":
            raise ValueError("V2 actor requires rl_mode='backtrace_rl'.")
        kwargs.setdefault("backtrack_limit", 500)
        self.episode_metrics = []
        summary = super().run(*args, rl_mode=rl_mode, **kwargs)
        metric_keys = {
            "total_loss_mean": "total_loss",
            "policy_loss_mean": "policy_loss",
            "value_loss_mean": "value_loss",
            "entropy_mean": "entropy",
            "ratio_mean": "ratio_mean",
            "rnd_loss_mean": "rnd_loss",
        }
        update_count = len(self.episode_metrics)
        self.run_metrics = {
            "episodes": int(summary["episodes"]),
            "episodes_with_updates": update_count,
            "steps": sum(item["steps"] for item in self.episode_metrics),
            "extrinsic_reward_sum": sum(
                item["extrinsic_reward_sum"] for item in self.episode_metrics
            ),
            "intrinsic_reward_sum": sum(
                item["intrinsic_reward_sum"] for item in self.episode_metrics
            ),
            "scaled_intrinsic_reward_sum": sum(
                item["scaled_intrinsic_reward_sum"] for item in self.episode_metrics
            ),
            "combined_reward_sum": sum(
                item["combined_reward_sum"] for item in self.episode_metrics
            ),
        }
        for output_key, source_key in metric_keys.items():
            self.run_metrics[output_key] = (
                sum(item[source_key] for item in self.episode_metrics) / update_count
                if update_count
                else 0.0
            )
        return summary

class CppPodemBacktraceV2Evaluator(CppPodemBacktraceV2Trainer):
    def decision_callback(self, request: dict[str, Any]) -> int:
        if request["mode"] != "backtrace":
            raise ValueError("V2 actor supports backtrace decisions only.")
        candidates = [self._gate(name) for name in request["candidate_names"]]
        selected = self.agent.select_backtrace_action_deterministic(
            self._gate(request["objective_name"]),
            int(request["objective_value"]),
            candidates,
            request.get("action_mask", [True, True]),
        )
        return next(
            index for index, candidate in enumerate(candidates) if candidate is selected
        )

    def run(
        self,
        circuit_path: Union[str, Path],
        backtrack_limit: int = 500,
        seed: int = 14,
        fault_ids: Optional[list[str]] = None,
        quiet: bool = True,
        rl_mode: str = "backtrace_rl",
        fault_map_path: Optional[Union[str, Path]] = None,
        event_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        if rl_mode != "backtrace_rl":
            raise ValueError("V2 actor requires rl_mode='backtrace_rl'.")
        resolved_circuit_path = Path(circuit_path).resolve()
        actual_hash = _fnv1a_file_hash(resolved_circuit_path)
        if actual_hash != self.circuit_hash:
            raise ValueError(
                "Embedding circuit hash mismatch: "
                f"expected {self.circuit_hash}, found {actual_hash}"
            )
        try:
            import cpp_podem
        except ImportError as error:
            raise ImportError(
                "Cannot import cpp_podem. Install this project in the active environment "
                "with 'python -m pip install -e .'."
            ) from error
        return dict(
            cpp_podem.run_stuck_at(
                _native_circuit_path(resolved_circuit_path),
                self.decision_callback,
                event_callback,
                backtrack_limit,
                seed,
                fault_ids,
                quiet,
                rl_mode,
                _native_circuit_path(fault_map_path) if fault_map_path else "",
            )
        )
