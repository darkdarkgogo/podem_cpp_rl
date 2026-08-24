from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import torch

from .ppo import RLGuidedPPOAgent


@dataclass(frozen=True)
class EmbeddingGate:
    outputpin: str
    deepgate_embedding: torch.Tensor


def _load_cpp_embedding_artifact(
    path: Union[str, Path],
) -> Tuple[str, dict[str, torch.Tensor]]:
    tokens = Path(path).read_text(encoding="utf-8").split()
    position = 0

    def take(expected: Optional[str] = None) -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError(f"Truncated embedding file: {path}")
        value = tokens[position]
        position += 1
        if expected is not None and value != expected:
            raise ValueError(f"Expected '{expected}', found '{value}' in {path}")
        return value

    take("SMARTATPG_EMBEDDINGS_V1")
    take("circuit_hash")
    circuit_hash = take()
    take("dimension")
    dimension = int(take())
    take("count")
    count = int(take())
    table: dict[str, torch.Tensor] = {}
    for _ in range(count):
        name = take()
        values = [float(take()) for _ in range(dimension)]
        if name in table:
            raise ValueError(f"Duplicate embedding name: {name}")
        table[name] = torch.tensor(values, dtype=torch.float32)
    if position != len(tokens):
        raise ValueError(f"Unexpected trailing data in embedding file: {path}")
    return circuit_hash, table


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


def export_actor_state_dict(
    state_dict: dict[str, torch.Tensor], path: Union[str, Path]
) -> None:
    tensor_names = [
        "gate_encoder.0.weight",
        "gate_encoder.0.bias",
        "mode_embedding.weight",
        "backtrace_actor.0.weight",
        "backtrace_actor.0.bias",
        "backtrace_actor.2.weight",
        "backtrace_actor.2.bias",
        "propagation_actor.0.weight",
        "propagation_actor.0.bias",
        "propagation_actor.2.weight",
        "propagation_actor.2.bias",
    ]
    missing = [name for name in tensor_names if name not in state_dict]
    if missing:
        raise KeyError("Checkpoint is missing actor tensors: " + ", ".join(missing))

    gate_weight = state_dict["gate_encoder.0.weight"]
    hidden_dim, embedding_dim = gate_weight.shape
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("SMARTATPG_ACTOR_V1\n")
        output.write(f"embedding_dim {embedding_dim}\n")
        output.write(f"hidden_dim {hidden_dim}\n")
        for name in tensor_names:
            tensor = state_dict[name].detach().cpu().float().contiguous()
            if tensor.ndim == 1:
                rows, cols = 1, tensor.shape[0]
            elif tensor.ndim == 2:
                rows, cols = tensor.shape
            else:
                raise ValueError(f"Unsupported actor tensor rank for {name}: {tensor.ndim}")
            values = tensor.reshape(-1).tolist()
            output.write(f"tensor {name} {rows} {cols}\n")
            output.write(" ".join(format(value, ".9g") for value in values))
            output.write("\n")
        output.write("end\n")


def export_actor_checkpoint(
    checkpoint_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    export_actor_state_dict(state_dict, output_path)


class CppPodemPPOTrainer:
    def __init__(
        self,
        embedding_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ):
        self.embedding_path = Path(embedding_path).resolve()
        self.circuit_hash, embeddings = _load_cpp_embedding_artifact(self.embedding_path)
        if not embeddings:
            raise ValueError("Embedding table is empty.")
        self.gates = {
            name: EmbeddingGate(name, embedding) for name, embedding in embeddings.items()
        }
        embedding_dim = next(iter(embeddings.values())).numel()
        self.agent = RLGuidedPPOAgent(gate_embedding_dim=embedding_dim)
        self.checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.agent.load(str(self.checkpoint_path))

        self.sequence_to_step: dict[int, int] = {}
        self.last_metrics: Optional[dict[str, Any]] = None
        self.step_penalty = -0.01
        self.backtrack_penalty = -0.1
        self.success_reward = 1.0
        self.failure_reward = -1.0

    def _gate(self, name: str) -> EmbeddingGate:
        try:
            return self.gates[name]
        except KeyError as error:
            raise KeyError(f"No DeepGate embedding for C++ wire '{name}'.") from error

    def decision_callback(self, request: dict[str, Any]) -> int:
        candidates = [self._gate(name) for name in request["candidate_names"]]
        if request["mode"] == "backtrace":
            selected = self.agent.select_backtrace_action(
                self._gate(request["objective_name"]), candidates
            )
        elif request["mode"] == "propagation":
            selected = self.agent.select_propagation_action(candidates)
        else:
            raise ValueError(f"Unknown C++ PODEM decision mode: {request['mode']}")

        action = next(
            index for index, candidate in enumerate(candidates) if candidate is selected
        )
        self.agent.add_reward(self.step_penalty)
        if self.agent.last_selected_step_idx is not None:
            self.sequence_to_step[int(request["sequence"])] = self.agent.last_selected_step_idx
        return action

    def event_callback(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "episode_start":
            self.sequence_to_step.clear()
            return
        if event_type == "backtrack":
            step_idx = self.sequence_to_step.get(int(event["decision_sequence"]))
            self.agent.add_reward_to_step(step_idx, self.backtrack_penalty)
            return
        if event_type != "episode_end":
            raise ValueError(f"Unknown C++ PODEM event: {event_type}")

        terminal_reward = (
            self.success_reward if int(event["outcome"]) == 1 else self.failure_reward
        )
        terminal_reward -= 0.05 * int(event["backtracks"])
        self.agent.finish_episode(terminal_reward)
        self.last_metrics = self.agent.update()

    def run(
        self,
        circuit_path: Union[str, Path],
        backtrack_limit: int = 97,
        seed: int = 14,
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
        )

    def save(self, actor_output_path: Optional[Union[str, Path]] = None) -> None:
        if self.checkpoint_path:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent.save(str(self.checkpoint_path))
        if actor_output_path:
            export_actor_state_dict(self.agent.policy_old.state_dict(), actor_output_path)
