"""Level-wise bidirectional GAT-GRU encoder for SmartATPG."""

import torch
from torch import nn
from torch.nn import functional as F

from .smartatpg import GATE_EMBEDDING_DIM, SmartATPGPPOAgent, SmartATPGPolicy
from .smartatpg_features import FEATURE_DIM


ENCODER_VARIANT = "level_gat_gru"
GRAPH_CONFIG = {
    "input_dim": FEATURE_DIM,
    "hidden_dim": GATE_EMBEDDING_DIM,
    "attention_heads": 1,
    "schedule": "forward_levels_then_reverse_levels",
    "directions": "independent",
}
GRAPH_CONFIG_ID = "level_gat_gru_fwd_rev_11d_v1"
TRAINING_FORMAT = "RL_PODEM_SMARTATPG_GAT_GRU_PPO_V1_11D"


class DirectionalGATGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=False)
        self.attention = nn.Parameter(torch.empty(FEATURE_DIM * 2))
        self.gru = nn.GRUCell(FEATURE_DIM, FEATURE_DIM)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attention.view(1, -1))

    def update_level(self, hidden, edge_index):
        if edge_index.numel() == 0:
            return hidden
        source, target = edge_index.to(hidden.device)
        source_states = self.projection(hidden.index_select(0, source))
        target_states = self.projection(hidden.index_select(0, target))
        scores = F.leaky_relu(
            torch.cat((target_states, source_states), dim=-1).matmul(self.attention),
            negative_slope=0.2,
        )
        max_scores = torch.scatter_reduce(
            scores, 0, target, reduce="amax", output_size=hidden.shape[0]
        )
        exponentials = torch.exp(scores - max_scores.index_select(0, target))
        totals = torch.scatter_reduce(
            exponentials, 0, target, reduce="sum", output_size=hidden.shape[0]
        )
        weights = exponentials / totals.index_select(0, target).clamp_min(1e-16)
        messages = torch.zeros_like(hidden)
        messages.index_add_(0, target, weights.unsqueeze(-1) * source_states)
        target_indices = torch.unique_consecutive(target)
        updates = self.gru(
            messages.index_select(0, target_indices),
            hidden.index_select(0, target_indices),
        )
        return hidden.index_copy(0, target_indices, updates)


class LevelWiseGATGRUEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_pass = DirectionalGATGRU()
        self.reverse_pass = DirectionalGATGRU()

    def forward(self, graph):
        device = self.forward_pass.projection.weight.device
        hidden = graph.x.to(device)
        for edges in graph.forward_level_edges[1:]:
            hidden = self.forward_pass.update_level(hidden, edges)
        for edges in reversed(graph.reverse_level_edges):
            hidden = self.reverse_pass.update_level(hidden, edges)
        if hidden.shape[1] != GATE_EMBEDDING_DIM:
            raise ValueError("GAT-GRU must preserve the 11D gate embedding")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("Non-finite GAT-GRU hidden state")
        return hidden


class GATGRUSmartATPGPolicy(SmartATPGPolicy):
    def __init__(self, hidden_dim=32):
        super().__init__(hidden_dim, graph_encoder=LevelWiseGATGRUEncoder())


class GATGRUSmartATPGPPOAgent(SmartATPGPPOAgent):
    encoder_variant = ENCODER_VARIANT
    graph_config = GRAPH_CONFIG
    training_format = TRAINING_FORMAT
    policy_class = GATGRUSmartATPGPolicy
