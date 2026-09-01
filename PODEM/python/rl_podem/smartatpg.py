"""Trainable graph policy; the fixed DeepGate policy remains independent."""

from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributions import Categorical

from .ppo import (BacktraceActorCriticV2, BacktraceDecisionStepV2,
                  BacktracePPOAgentV2, RandomNetworkDistillation, device)
from .smartatpg_features import FEATURE_DIM, FEATURE_SCHEMA, GRAPH_CONFIG

DESCRIPTOR_DIM = 64 + FEATURE_DIM + 2
RND_SCHEMA = "SMARTATPG_RAW_OBJECTIVE_V1"
TRAINING_FORMAT = "RL_PODEM_SMARTATPG_PPO_V1"


@dataclass(frozen=True)
class GraphGate:
    outputpin: str
    circuit_hash: str
    node_index: int


class MeanGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(FEATURE_DIM * 2, 64), nn.Linear(128, 64)])

    def forward(self, graph):
        model_device = self.layers[0].weight.device
        hidden = graph.x.to(model_device)
        source, target = graph.edge_index.to(model_device)
        counts = hidden.new_zeros((hidden.shape[0], 1))
        counts.index_add_(0, target, hidden.new_ones((target.numel(), 1)))
        for layer in self.layers:
            neighbors = torch.zeros_like(hidden)
            neighbors.index_add_(0, target, hidden[source])
            neighbors = neighbors / counts.clamp_min(1)
            hidden = torch.relu(layer(torch.cat((hidden, neighbors), dim=-1)))
        return hidden


class SmartATPGPolicy(BacktraceActorCriticV2):
    def __init__(self, hidden_dim=32):
        super().__init__(DESCRIPTOR_DIM, hidden_dim)
        self.graph_encoder = MeanGraphEncoder()
        self._contexts = {}
        self._context_version = None

    def clear_contexts(self):
        self._contexts.clear()
        self._context_version = None

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.clear_contexts()
        return result

    def context(self, graph, cached=False):
        if not cached:
            return self.graph_encoder(graph).mean(dim=0)
        if torch.is_grad_enabled():
            raise RuntimeError("Cached graph context is for no-grad inference only")
        version = tuple((id(p), p._version, p.device) for p in self.graph_encoder.parameters())
        if version != self._context_version:
            self.clear_contexts()
            self._context_version = version
        if graph.circuit_hash not in self._contexts:
            self._contexts[graph.circuit_hash] = self.graph_encoder(graph).mean(dim=0)
        return self._contexts[graph.circuit_hash]

    def descriptors(self, graph, node_indices, masks=None, context=None):
        context = self.context(graph) if context is None else context
        indices = torch.as_tensor(node_indices, dtype=torch.long, device="cpu")
        local = graph.x[indices].to(context.device)
        masks = torch.ones((len(indices), 2), device=context.device) if masks is None else torch.as_tensor(
            masks, dtype=torch.float32, device=context.device)
        if masks.shape != (len(indices), 2):
            raise ValueError("SmartATPG requires a two-position action mask")
        return torch.cat((context.expand(len(indices), -1), local, masks), dim=-1)

    def batch_logits(self, descriptors, values):
        values = torch.as_tensor(values, dtype=torch.long, device=descriptors.device)
        state = self.gate_encoder(descriptors) + self.objective_value_embedding(values)
        return self.backtrace_actor(state), self.critic(state).squeeze(-1)


class SmartATPGPPOAgent(BacktracePPOAgentV2):
    embedding_backend = "smartatpg"

    def __init__(self, graphs, hidden_dim=32, **kwargs):
        super().__init__(DESCRIPTOR_DIM, hidden_dim=hidden_dim, **kwargs)
        self.graphs = {graph.circuit_hash: graph for graph in graphs.values()}
        self.gate_indices = {key: graph.name_to_index for key, graph in self.graphs.items()}
        self.policy = SmartATPGPolicy(hidden_dim).to(device)
        self.policy_old = SmartATPGPolicy(hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        shared = [p for name, p in self.policy.named_parameters() if not name.startswith("critic.")]
        self.optimizer = torch.optim.Adam([
            {"params": shared, "lr": self.lr_actor},
            {"params": self.policy.critic.parameters(), "lr": self.lr_critic},
        ])
        self.rnd = RandomNetworkDistillation(FEATURE_DIM + 2, hidden_dim).to(device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=self.rnd_lr)

    def _rnd_observation(self, objective_embedding, objective_value):
        return super()._rnd_observation(objective_embedding[64:64 + FEATURE_DIM].cpu(), objective_value)

    def _select(self, gate, value, candidates, mask, deterministic):
        if value not in (0, 1) or len(candidates) != 2:
            raise ValueError("SmartATPG requires a binary objective and two input positions")
        mask = torch.as_tensor([True, True] if mask is None else mask, dtype=torch.bool)
        if mask.shape != (2,) or not bool(mask.any()):
            raise ValueError("SmartATPG action mask must enable at least one of two inputs")
        graph = self.graphs[gate.circuit_hash]
        with torch.no_grad():
            descriptor = self.policy_old.descriptors(
                graph, [gate.node_index], mask.unsqueeze(0), self.policy_old.context(graph, cached=True)
            )[0]
            logits, state_value = self.policy_old.backtrace_logits(descriptor, value)
            dist = Categorical(logits=logits.masked_fill(~mask.to(device), -1e9))
            action = torch.argmax(dist.logits) if deterministic else dist.sample()
            if not deterministic:
                observation = self._rnd_observation(descriptor, value)
                intrinsic = self._intrinsic_reward(observation) if self.rnd_beta > 0 else 0.0
                self.buffer.steps.append(BacktraceDecisionStepV2(
                    objective_embedding=descriptor.cpu(), objective_value=value,
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
        contexts = {}
        evaluated = []
        for step in self.buffer.steps:
            graph = self.graphs[step.circuit_hash]
            if graph.circuit_hash not in contexts:
                contexts[graph.circuit_hash] = self.policy.context(graph)
            descriptor = self.policy.descriptors(
                graph, [self.gate_indices[graph.circuit_hash][step.objective_name]],
                step.action_mask.unsqueeze(0), contexts[graph.circuit_hash],
            )[0]
            evaluated.append(self.policy.evaluate_step(replace(step, objective_embedding=descriptor)))
        return evaluated

    def training_state_dict(self):
        state = super().training_state_dict()
        state.update(format=TRAINING_FORMAT, embedding_backend="smartatpg",
                     feature_schema=FEATURE_SCHEMA, graph_config=dict(GRAPH_CONFIG), rnd_schema=RND_SCHEMA)
        return state

    def load_training_state_dict(self, state):
        expected = {"format": TRAINING_FORMAT, "embedding_backend": "smartatpg",
                    "feature_schema": FEATURE_SCHEMA, "graph_config": GRAPH_CONFIG, "rnd_schema": RND_SCHEMA}
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible SmartATPG checkpoint backend or graph schema")
        super().load_training_state_dict({**state, "format": "RL_PODEM_PPO_RND_V2"})
