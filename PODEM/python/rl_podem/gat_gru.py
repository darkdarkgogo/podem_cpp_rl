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
GRAPH_CONFIG_ID = "level_gat_gru_fwd_rev_11d_v3_nobuf"
ACTOR_INPUT_DIM = 2 * FEATURE_DIM + 1
CRITIC_INPUT_DIM = FEATURE_DIM + 1
TRAINING_FORMAT = (
    "RL_PODEM_SMARTATPG_GAT_GRU_FANIN_SCORER_PPO_V9_BATCH4_MINIBATCH512_EPOCH4_11D_CO_NO_BUF"
)


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
        # PyTorch 1.11 used the older scatter_reduce signature.
        if hasattr(torch.Tensor, "scatter_reduce_"):
            max_scores = scores.new_full((hidden.shape[0],), -float("inf"))
            max_scores.scatter_reduce_(0, target, scores.detach(), reduce="amax")
        else:
            # The 1.11 reduction is CPU-only; the softmax shift is detached.
            max_scores = torch.scatter_reduce(
                scores.detach().cpu(), 0, target.cpu(), reduce="amax", output_size=hidden.shape[0]
            ).to(hidden.device)
        exponentials = torch.exp(scores - max_scores.index_select(0, target))
        totals = scores.new_zeros(hidden.shape[0]).index_add(0, target, exponentials)
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
            raise ValueError(
                f"GAT-GRU must preserve the {GATE_EMBEDDING_DIM}D gate embedding"
            )
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("Non-finite GAT-GRU hidden state")
        return hidden


class GATGRUSmartATPGPolicy(SmartATPGPolicy):
    def __init__(self, hidden_dim=32):
        super().__init__(
            hidden_dim,
            graph_encoder=LevelWiseGATGRUEncoder(),
            actor_input_dim=ACTOR_INPUT_DIM,
            critic_input_dim=CRITIC_INPUT_DIM,
            actor_output_dim=1,
        )

    def batch_candidate_logits(self, objective_embeddings, candidate_embeddings, values):
        model_device = self.backtrace_actor[0].weight.device
        objective_embeddings = objective_embeddings.to(
            device=model_device, dtype=torch.float32
        )
        candidate_embeddings = candidate_embeddings.to(
            device=model_device, dtype=torch.float32
        )
        values = torch.as_tensor(values, device=model_device).reshape(-1, 1)
        if objective_embeddings.ndim != 2 or objective_embeddings.shape[1] != FEATURE_DIM:
            raise ValueError(f"GAT objective embeddings must have shape [N, {FEATURE_DIM}]")
        if (candidate_embeddings.ndim != 3 or
                candidate_embeddings.shape[0] != objective_embeddings.shape[0] or
                candidate_embeddings.shape[1] != 2 or
                candidate_embeddings.shape[2] != FEATURE_DIM):
            raise ValueError(f"GAT candidate embeddings must have shape [N, 2, {FEATURE_DIM}]")
        if values.shape[0] != objective_embeddings.shape[0] or not bool(
            ((values == 0) | (values == 1)).all()
        ):
            raise ValueError("GAT scorer requires one binary objective value per state")
        objective_pairs = objective_embeddings.unsqueeze(1).expand(-1, 2, -1)
        value_pairs = values.to(objective_embeddings.dtype).unsqueeze(1).expand(-1, 2, -1)
        scorer_inputs = torch.cat(
            (objective_pairs, candidate_embeddings, value_pairs), dim=-1
        )
        return self.backtrace_actor(scorer_inputs.reshape(-1, ACTOR_INPUT_DIM)).reshape(-1, 2)

    def backtrace_logits(self, objective_embedding, candidate_embeddings, objective_value):
        model_device = self.backtrace_actor[0].weight.device
        objective_embedding = objective_embedding.to(device=model_device, dtype=torch.float32)
        candidate_embeddings = candidate_embeddings.to(device=model_device, dtype=torch.float32)
        if objective_embedding.shape != (FEATURE_DIM,):
            raise ValueError(
                f"GAT objective embedding must have shape [{FEATURE_DIM}]"
            )
        if candidate_embeddings.ndim != 2 or candidate_embeddings.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"GAT candidate embeddings must have shape [N, {FEATURE_DIM}]"
            )
        if candidate_embeddings.shape[0] < 2:
            raise ValueError("GAT scorer requires at least two fanin candidates")
        if objective_value not in (0, 1):
            raise ValueError("GAT objective value must be binary")
        logits = self.batch_candidate_logits(
            objective_embedding.unsqueeze(0), candidate_embeddings.unsqueeze(0),
            [objective_value],
        )[0]
        critic_state = torch.cat((
            objective_embedding,
            objective_embedding.new_tensor([float(objective_value)]),
        )).unsqueeze(0)
        state_value = self.critic(critic_state).squeeze(-1)[0]
        return logits, state_value

    def evaluate_step(self, step):
        if step.candidate_embeddings is None:
            raise ValueError("GAT rollout is missing fanin embeddings")
        logits, state_value = self.backtrace_logits(
            step.objective_embedding, step.candidate_embeddings,
            step.objective_value,
        )
        mask = step.action_mask.to(device=logits.device, dtype=torch.bool)
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~mask, -1e9)
        )
        action = torch.tensor(step.action, dtype=torch.long, device=logits.device)
        return distribution.log_prob(action), state_value, distribution.entropy()


class GATGRUSmartATPGPPOAgent(SmartATPGPPOAgent):
    actor_input_dim = ACTOR_INPUT_DIM
    critic_input_dim = CRITIC_INPUT_DIM
    decision_state_dim = ACTOR_INPUT_DIM + 2
    encoder_variant = ENCODER_VARIANT
    graph_config = GRAPH_CONFIG
    training_format = TRAINING_FORMAT
    policy_class = GATGRUSmartATPGPolicy

    def __init__(self, graphs, hidden_dim=32, **kwargs):
        kwargs.setdefault("lr_actor", 0.0003)
        kwargs.setdefault("lr_critic", 0.001)
        kwargs.setdefault("k_epochs", 4)
        super().__init__(graphs, hidden_dim=hidden_dim, **kwargs)
