from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


@dataclass
class DecisionStep:
    mode: str
    state_embedding: Optional[torch.Tensor]
    candidate_embeddings: list[torch.Tensor]
    action: int
    logprob: torch.Tensor
    state_value: torch.Tensor
    reward: float = 0.0
    is_terminal: bool = False


class RolloutBuffer:
    def __init__(self):
        self.steps = []

    def clear(self):
        self.steps.clear()


class RLActorCritic(nn.Module):
    def __init__(self, gate_embedding_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_encoder = nn.Sequential(
            nn.Linear(gate_embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mode_embedding = nn.Embedding(2, hidden_dim)
        self.backtrace_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.propagation_actor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_gate(self, gate_embedding, mode):
        gate_tensor = gate_embedding.to(device=device, dtype=torch.float32).unsqueeze(0)
        mode_tensor = torch.tensor([mode], dtype=torch.long, device=device)
        return (
            self.gate_encoder(gate_tensor)
            + self.mode_embedding(mode_tensor)
        ).squeeze(0)

    def build_backtrace_pair_repr(self, objective_embedding, candidate_embedding):
        objective_repr = self.encode_gate(objective_embedding, mode=0)
        candidate_repr = self.encode_gate(candidate_embedding, mode=0)
        pair_repr = torch.cat([objective_repr, candidate_repr], dim=-1)
        return pair_repr, objective_repr

    def backtrace_logits(self, objective_embedding, candidate_gate_embeddings):
        pair_reprs = []
        objective_repr = None
        for candidate_embedding in candidate_gate_embeddings:
            pair_repr, objective_repr = self.build_backtrace_pair_repr(
                objective_embedding,
                candidate_embedding,
            )
            pair_reprs.append(pair_repr)
        pair_reprs = torch.stack(pair_reprs, dim=0)
        logits = self.backtrace_actor(pair_reprs).squeeze(-1)
        state_repr = objective_repr
        state_value = self.critic(state_repr).squeeze(-1)
        return logits, state_value

    def propagation_logits(self, candidate_gate_embeddings):
        candidate_reprs = []
        for gate_embedding in candidate_gate_embeddings:
            candidate_reprs.append(self.encode_gate(gate_embedding, mode=1))
        candidate_reprs = torch.stack(candidate_reprs, dim=0)
        state_repr = candidate_reprs.mean(dim=0)
        pair_reprs = []
        for candidate_repr in candidate_reprs:
            pair_reprs.append(torch.cat([state_repr, candidate_repr], dim=-1))
        pair_reprs = torch.stack(pair_reprs, dim=0)
        logits = self.propagation_actor(pair_reprs).squeeze(-1)
        state_value = self.critic(state_repr).squeeze(-1)
        return logits, state_value

    def evaluate_step(self, step):
        if step.mode == "backtrace":
            logits, state_value = self.backtrace_logits(
                step.state_embedding,
                step.candidate_embeddings,
            )
        else:
            logits, state_value = self.propagation_logits(step.candidate_embeddings)

        dist = Categorical(logits=logits)
        action_tensor = torch.tensor(step.action, dtype=torch.long, device=device)
        logprob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
        return logprob, state_value, entropy


class RLGuidedPPOAgent:
    def __init__(
        self,
        gate_embedding_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        k_epochs=8,
        eps_clip=0.2,
    ):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.buffer = RolloutBuffer()
        self.last_selected_step_idx = None
        self.last_selected_mode = None

        self.policy = RLActorCritic(
            gate_embedding_dim=gate_embedding_dim,
        ).to(device)
        self.policy_old = RLActorCritic(
            gate_embedding_dim=gate_embedding_dim,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.gate_encoder.parameters(), "lr": lr_actor},
                {"params": self.policy.mode_embedding.parameters(), "lr": lr_actor},
                {"params": self.policy.backtrace_actor.parameters(), "lr": lr_actor},
                {"params": self.policy.propagation_actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.mse_loss = nn.MSELoss()
        self.update_count = 0

    def _gate_embedding(self, gate):
        if gate.deepgate_embedding is None:
            raise ValueError(f"Gate '{gate.outputpin}' is missing a fixed DeepGate embedding.")
        return gate.deepgate_embedding

    def select_backtrace_action(self, objective_gate, candidate_gates):
        objective_embedding = self._gate_embedding(objective_gate)
        candidate_embeddings = [self._gate_embedding(gate) for gate in candidate_gates]
        logits, state_value = self.policy_old.backtrace_logits(
            objective_embedding,
            candidate_embeddings,
        )
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        self.buffer.steps.append(
            DecisionStep(
                mode="backtrace",
                state_embedding=objective_embedding.detach().cpu(),
                candidate_embeddings=[embedding.detach().cpu() for embedding in candidate_embeddings],
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
            )
        )
        self.last_selected_step_idx = len(self.buffer.steps) - 1
        self.last_selected_mode = "backtrace"
        return candidate_gates[int(action.item())]

    def select_propagation_action(self, frontier_gates):
        candidate_embeddings = [self._gate_embedding(gate) for gate in frontier_gates]
        logits, state_value = self.policy_old.propagation_logits(candidate_embeddings)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        self.buffer.steps.append(
            DecisionStep(
                mode="propagation",
                state_embedding=None,
                candidate_embeddings=[embedding.detach().cpu() for embedding in candidate_embeddings],
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
            )
        )
        self.last_selected_step_idx = len(self.buffer.steps) - 1
        self.last_selected_mode = "propagation"
        return frontier_gates[int(action.item())]

    def add_reward(self, reward):
        if self.buffer.steps:
            self.buffer.steps[-1].reward += reward

    def add_reward_to_step(self, step_idx, reward):
        if step_idx is None:
            return
        if 0 <= step_idx < len(self.buffer.steps):
            self.buffer.steps[step_idx].reward += reward

    def finish_episode(self, final_reward):
        if not self.buffer.steps:
            return
        self.buffer.steps[-1].reward += final_reward
        self.buffer.steps[-1].is_terminal = True

    def update(self):
        if not self.buffer.steps:
            return None

        step_rewards = [step.reward for step in self.buffer.steps]
        returns = []
        discounted_reward = 0.0
        for step in reversed(self.buffer.steps):
            if step.is_terminal:
                discounted_reward = 0.0
            discounted_reward = step.reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_logprobs = torch.stack([step.logprob for step in self.buffer.steps]).to(device)
        old_state_values = (
            torch.stack([step.state_value for step in self.buffer.steps]).to(device).squeeze(-1)
        )
        advantages = returns.detach() - old_state_values.detach()

        final_metrics = None
        for _ in range(self.k_epochs):
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            ratios = []
            for idx, step in enumerate(self.buffer.steps):
                logprob, state_value, entropy = self.policy.evaluate_step(step)
                ratio = torch.exp(logprob - old_logprobs[idx].detach())
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2)
                value_loss = self.mse_loss(state_value.squeeze(), returns[idx])
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                losses.append(loss)
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                ratios.append(ratio)

            total_loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            final_metrics = {
                "total_loss": float(total_loss.detach().item()),
                "policy_loss": float(torch.stack(policy_losses).mean().detach().item()),
                "value_loss": float(torch.stack(value_losses).mean().detach().item()),
                "entropy": float(torch.stack(entropies).mean().detach().item()),
                "ratio_mean": float(torch.stack(ratios).mean().detach().item()),
            }

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.update_count += 1
        metrics = {
            "update": self.update_count,
            "steps": len(self.buffer.steps),
            "reward_sum": float(sum(step_rewards)),
            "reward_last": float(step_rewards[-1]),
            "return_mean": float(returns.mean().detach().item()),
            "return_std": float(returns.std().detach().item()) if returns.numel() > 1 else 0.0,
            "adv_mean": float(advantages.mean().detach().item()),
            "adv_std": float(advantages.std().detach().item()) if advantages.numel() > 1 else 0.0,
            "epochs": self.k_epochs,
        }
        if final_metrics is not None:
            metrics.update(final_metrics)
        self.buffer.clear()
        return metrics

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)

