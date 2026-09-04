"""Trainable graph policy for SmartATPG-guided PODEM."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributions import Categorical

from .ppo import (BacktraceActorCriticV2, BacktraceDecisionStepV2,
                  BacktracePPOAgentV2, RandomNetworkDistillation, device)
from .smartatpg_features import FEATURE_DIM, FEATURE_SCHEMA, GRAPH_CONFIG

GATE_EMBEDDING_DIM = FEATURE_DIM
ACTOR_INPUT_DIM = GATE_EMBEDDING_DIM
ACTION_MASK_DIM = 2
DECISION_STATE_DIM = GATE_EMBEDDING_DIM + ACTION_MASK_DIM
# Kept as the logical rollout-state dimension for metadata compatibility.
POLICY_STATE_DIM = DECISION_STATE_DIM
ENCODER_VARIANT = "fanin_mean"
RND_SCHEMA = "SMARTATPG_RAW_OBJECTIVE_V2_CO"
TRAINING_FORMAT = "RL_PODEM_SMARTATPG_PPO_V5_CO"


@dataclass(frozen=True)
class GraphGate:
    outputpin: str
    circuit_hash: str
    node_index: int


class MeanGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(FEATURE_DIM * 2, GATE_EMBEDDING_DIM)

    def forward(self, graph):
        model_device = self.layer.weight.device
        features = graph.x.to(model_device)
        source, target = graph.edge_index.to(model_device)
        neighbors = torch.zeros_like(features)
        counts = features.new_zeros((features.shape[0], 1))
        neighbors.index_add_(0, target, features[source])
        counts.index_add_(0, target, features.new_ones((target.numel(), 1)))
        neighbors = neighbors / counts.clamp_min(1)
        return torch.relu(self.layer(torch.cat((features, neighbors), dim=-1)))


class SmartATPGPolicy(BacktraceActorCriticV2):
    def __init__(self, hidden_dim=32, graph_encoder=None, actor_input_dim=ACTOR_INPUT_DIM):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.backtrace_actor = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(actor_input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.graph_encoder = MeanGraphEncoder() if graph_encoder is None else graph_encoder
        self._embedding_cache = {}
        self._embedding_version = None

    def clear_embeddings(self):
        self._embedding_cache.clear()
        self._embedding_version = None

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.clear_embeddings()
        return result

    def graph_embeddings(self, graph, cached=False):
        if not cached:
            return self.graph_encoder(graph)
        if torch.is_grad_enabled():
            raise RuntimeError("Cached gate embeddings are for no-grad inference only")
        version = tuple((id(p), p._version, p.device) for p in self.graph_encoder.parameters())
        if version != self._embedding_version:
            self.clear_embeddings()
            self._embedding_version = version
        if graph.circuit_hash not in self._embedding_cache:
            self._embedding_cache[graph.circuit_hash] = self.graph_encoder(graph)
        return self._embedding_cache[graph.circuit_hash]

    def descriptors(self, graph, node_indices, embeddings=None):
        embeddings = (
            self.graph_embeddings(graph) if embeddings is None else embeddings
        )
        indices = torch.as_tensor(
            node_indices, dtype=torch.long, device=embeddings.device
        )
        return embeddings[indices]

    def batch_logits(self, descriptors, values):
        if descriptors.ndim != 2 or descriptors.shape[1] != ACTOR_INPUT_DIM:
            raise ValueError("SmartATPG Actor accepts only 12D gate embeddings")
        state = descriptors.to(device=self.backtrace_actor[0].weight.device, dtype=torch.float32)
        return self.backtrace_actor(state), self.critic(state).squeeze(-1)

    def backtrace_logits(self, objective_embedding, objective_value):
        logits, values = self.batch_logits(objective_embedding.unsqueeze(0), [objective_value])
        return logits[0], values[0]


class SmartATPGPPOAgent(BacktracePPOAgentV2):
    actor_input_dim = ACTOR_INPUT_DIM
    decision_state_dim = DECISION_STATE_DIM
    embedding_backend = "smartatpg"
    encoder_variant = ENCODER_VARIANT
    graph_config = GRAPH_CONFIG
    training_format = TRAINING_FORMAT
    policy_class = SmartATPGPolicy

    def __init__(self, graphs, hidden_dim=32, **kwargs):
        kwargs.setdefault("lr_actor", 0.001)
        kwargs.setdefault("lr_critic", 0.01)
        super().__init__(ACTOR_INPUT_DIM, hidden_dim=hidden_dim, **kwargs)
        self.gate_embedding_dim = GATE_EMBEDDING_DIM
        self.action_mask_dim = ACTION_MASK_DIM
        self.policy_state_dim = self.decision_state_dim
        self.graphs = {graph.circuit_hash: graph for graph in graphs.values()}
        self.gate_indices = {key: graph.name_to_index for key, graph in self.graphs.items()}
        self.policy = self.policy_class(hidden_dim).to(device)
        self.policy_old = self.policy_class(hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        shared = [p for name, p in self.policy.named_parameters() if not name.startswith("critic.")]
        self.optimizer = torch.optim.Adam([
            {"params": shared, "lr": self.lr_actor},
            {"params": self.policy.critic.parameters(), "lr": self.lr_critic},
        ])
        self.rnd = RandomNetworkDistillation(FEATURE_DIM + 2, hidden_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.rnd_lr)

    def _rnd_observation(self, objective_embedding, objective_value):
        if objective_embedding.numel() != FEATURE_DIM:
            raise ValueError("SmartATPG RND expects the raw 12-value gate features")
        return super()._rnd_observation(objective_embedding.cpu(), objective_value)

    def _select(self, gate, value, candidates, mask, deterministic):
        if value not in (0, 1) or len(candidates) != 2:
            raise ValueError("SmartATPG requires a binary objective and two input positions")
        mask = torch.as_tensor([True, True] if mask is None else mask, dtype=torch.bool)
        if mask.shape != (2,) or not bool(mask.any()):
            raise ValueError("SmartATPG action mask must enable at least one of two inputs")
        graph = self.graphs[gate.circuit_hash]
        with torch.no_grad():
            embeddings = self.policy_old.graph_embeddings(graph, cached=True)
            descriptor = self.policy_old.descriptors(
                graph, [gate.node_index], embeddings
            )[0]
            logits, state_value = self.policy_old.backtrace_logits(descriptor, value)
            dist = Categorical(logits=logits.masked_fill(~mask.to(device), -1e9))
            action = torch.argmax(dist.logits) if deterministic else dist.sample()
            if not deterministic:
                observation = super()._rnd_observation(
                    graph.x[gate.node_index], value
                )
                intrinsic = self._intrinsic_reward(observation) if self.rnd_beta > 0 else 0.0
                self.buffer.steps.append(BacktraceDecisionStepV2(
                    objective_embedding=torch.empty(0), objective_value=value,
                    action_mask=mask.detach().cpu().clone(), action=int(action.item()), logprob=dist.log_prob(action),
                    state_value=state_value, rnd_observation=observation,
                    intrinsic_reward=intrinsic, reward=self.rnd_beta * intrinsic,
                    circuit_hash=graph.circuit_hash, objective_name=gate.outputpin,
                ))
                self.last_selected_step_idx = len(self.buffer.steps) - 1
                self.last_selected_mode = "backtrace"
        return candidates[int(action.item())]

    def select_backtrace_action(self, objective_gate, objective_value, candidate_gates, action_mask=None):
        return self._select(objective_gate, objective_value, candidate_gates, action_mask, False)

    def select_backtrace_action_deterministic(self, objective_gate, objective_value, candidate_gates, action_mask=None):
        return self._select(objective_gate, objective_value, candidate_gates, action_mask, True)

    def _evaluate_rollout(self):
        embeddings_by_graph = {}
        evaluated = []
        for step in self.buffer.steps:
            graph = self.graphs[step.circuit_hash]
            if graph.circuit_hash not in embeddings_by_graph:
                embeddings_by_graph[graph.circuit_hash] = (
                    self.policy.graph_embeddings(graph)
                )
            descriptor = self.policy.descriptors(
                graph, [self.gate_indices[graph.circuit_hash][step.objective_name]],
                embeddings_by_graph[graph.circuit_hash],
            )[0]
            evaluated.append(self.policy.evaluate_step(replace(step, objective_embedding=descriptor)))
        return evaluated

    def training_state_dict(self):
        state = super().training_state_dict()
        state.update(
            format=self.training_format,
            embedding_backend="smartatpg",
            encoder_variant=self.encoder_variant,
            feature_schema=FEATURE_SCHEMA,
            graph_config=dict(self.graph_config),
            rnd_schema=RND_SCHEMA,
            actor_input_dim=self.actor_input_dim,
            action_mask_dim=ACTION_MASK_DIM,
            decision_state_dim=self.decision_state_dim,
        )
        state["policy_state_dim"] = self.policy_state_dim
        return state

    def load_training_state_dict(self, state):
        expected = {"format": self.training_format, "embedding_backend": "smartatpg",
                    "encoder_variant": self.encoder_variant,
                    "feature_schema": FEATURE_SCHEMA, "graph_config": self.graph_config,
                    "rnd_schema": RND_SCHEMA, "gate_embedding_dim": GATE_EMBEDDING_DIM,
                    "actor_input_dim": self.actor_input_dim,
                    "action_mask_dim": ACTION_MASK_DIM,
                    "decision_state_dim": self.decision_state_dim,
                    "policy_state_dim": self.policy_state_dim}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible SmartATPG checkpoint backend or graph schema")
        super().load_training_state_dict({**state, "format": "RL_PODEM_PPO_RND_V2"})
