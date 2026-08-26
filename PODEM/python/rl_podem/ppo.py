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
    rnd_observation: torch.Tensor
    intrinsic_reward: float = 0.0
    reward: float = 0.0
    is_terminal: bool = False


class RolloutBuffer:
    def __init__(self):
        self.steps = []

    def clear(self):
        self.steps.clear()


class RunningMeanVariance:
    def __init__(self):
        self.mean = 0.0
        self.variance = 1.0
        self.count = 1.0

    def update(self, value):
        value = float(value)
        delta = value - self.mean
        total = self.count + 1.0
        new_mean = self.mean + delta / total
        self.variance = (
            self.variance * self.count + delta * (value - new_mean)
        ) / total
        self.mean = new_mean
        self.count = total

    def state_dict(self):
        return {
            "mean": self.mean,
            "variance": self.variance,
            "count": self.count,
        }

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.variance = float(state["variance"])
        self.count = float(state["count"])


class RandomNetworkDistillation(nn.Module):
    def __init__(self, observation_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def prediction_error(self, observations):
        with torch.no_grad():
            target = self.target(observations)
        prediction = self.predictor(observations)
        return (prediction - target).pow(2).mean(dim=-1)


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
        rnd_beta=0.05,
        rnd_lr=1e-4,
        rnd_bonus_clip=5.0,
    ):
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gate_embedding_dim = gate_embedding_dim
        self.rnd_beta = rnd_beta
        self.rnd_bonus_clip = rnd_bonus_clip
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
        self.rnd = RandomNetworkDistillation(
            observation_dim=gate_embedding_dim * 2 + 2,
        ).to(device)
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(), lr=rnd_lr
        )
        self.rnd_error_stats = RunningMeanVariance()
        self.mse_loss = nn.MSELoss()
        self.update_count = 0

    def _gate_embedding(self, gate):
        if gate.deepgate_embedding is None:
            raise ValueError(f"Gate '{gate.outputpin}' is missing a fixed DeepGate embedding.")
        return gate.deepgate_embedding

    def _rnd_observation(self, mode, objective_embedding, candidate_embeddings):
        candidate_mean = torch.stack(candidate_embeddings).mean(dim=0)
        if objective_embedding is None:
            objective_embedding = torch.zeros_like(candidate_mean)
        mode_vector = torch.tensor(
            [1.0, 0.0] if mode == "backtrace" else [0.0, 1.0],
            dtype=torch.float32,
        )
        return torch.cat(
            [objective_embedding.float(), candidate_mean.float(), mode_vector], dim=0
        ).detach().cpu()

    def _intrinsic_reward(self, observation):
        with torch.no_grad():
            error = float(
                self.rnd.prediction_error(observation.to(device).unsqueeze(0)).item()
            )
        scale = max(self.rnd_error_stats.variance, 1e-8) ** 0.5
        normalized = min(error / scale, self.rnd_bonus_clip)
        self.rnd_error_stats.update(error)
        return normalized

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
        rnd_observation = self._rnd_observation(
            "backtrace", objective_embedding, candidate_embeddings
        )
        intrinsic_reward = self._intrinsic_reward(rnd_observation)
        self.buffer.steps.append(
            DecisionStep(
                mode="backtrace",
                state_embedding=objective_embedding.detach().cpu(),
                candidate_embeddings=[embedding.detach().cpu() for embedding in candidate_embeddings],
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
                rnd_observation=rnd_observation,
                intrinsic_reward=intrinsic_reward,
                reward=self.rnd_beta * intrinsic_reward,
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
        rnd_observation = self._rnd_observation(
            "propagation", None, candidate_embeddings
        )
        intrinsic_reward = self._intrinsic_reward(rnd_observation)
        self.buffer.steps.append(
            DecisionStep(
                mode="propagation",
                state_embedding=None,
                candidate_embeddings=[embedding.detach().cpu() for embedding in candidate_embeddings],
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
                rnd_observation=rnd_observation,
                intrinsic_reward=intrinsic_reward,
                reward=self.rnd_beta * intrinsic_reward,
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

        rnd_observations = torch.stack(
            [step.rnd_observation for step in self.buffer.steps]
        ).to(device)

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

        rnd_loss = self.rnd.prediction_error(rnd_observations).mean()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.update_count += 1
        metrics = {
            "update": self.update_count,
            "steps": len(self.buffer.steps),
            "reward_sum": float(sum(step_rewards)),
            "intrinsic_reward_sum": float(
                sum(step.intrinsic_reward for step in self.buffer.steps)
            ),
            "rnd_loss": float(rnd_loss.detach().item()),
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

    def training_state_dict(self):
        return {
            "format": "RL_PODEM_PPO_RND_V1",
            "policy": self.policy.state_dict(),
            "policy_old": self.policy_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rnd": self.rnd.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict(),
            "rnd_error_stats": self.rnd_error_stats.state_dict(),
            "update_count": self.update_count,
            "rnd_beta": self.rnd_beta,
        }

    def load_training_state_dict(self, state):
        if state.get("format") != "RL_PODEM_PPO_RND_V1":
            raise ValueError("Unsupported PPO/RND training checkpoint format.")
        self.policy.load_state_dict(state["policy"])
        self.policy_old.load_state_dict(state["policy_old"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rnd.load_state_dict(state["rnd"])
        self.rnd_optimizer.load_state_dict(state["rnd_optimizer"])
        self.rnd_error_stats.load_state_dict(state["rnd_error_stats"])
        self.update_count = int(state["update_count"])
        self.rnd_beta = float(state["rnd_beta"])

    def load(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.policy_old.load_state_dict(state_dict)
        self.policy.load_state_dict(state_dict)


@dataclass
class BacktraceDecisionStepV2:
    objective_embedding: torch.Tensor
    objective_value: int
    action_mask: torch.Tensor
    action: int
    logprob: torch.Tensor
    state_value: torch.Tensor
    rnd_observation: torch.Tensor
    intrinsic_reward: float = 0.0
    reward: float = 0.0
    is_terminal: bool = False


class BacktraceActorCriticV2(nn.Module):
    def __init__(self, gate_embedding_dim, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_encoder = nn.Sequential(
            nn.Linear(gate_embedding_dim, hidden_dim),
            nn.Tanh(),
        )
        self.objective_value_embedding = nn.Embedding(2, hidden_dim)
        self.backtrace_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def backtrace_logits(self, objective_embedding, objective_value):
        gate_tensor = objective_embedding.to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)
        value_tensor = torch.tensor(
            [objective_value], dtype=torch.long, device=device
        )
        state = (
            self.gate_encoder(gate_tensor)
            + self.objective_value_embedding(value_tensor)
        ).squeeze(0)
        return self.backtrace_actor(state), self.critic(state).squeeze(-1)

    def evaluate_step(self, step):
        logits, state_value = self.backtrace_logits(
            step.objective_embedding, step.objective_value
        )
        mask = step.action_mask.to(device=device, dtype=torch.bool)
        dist = Categorical(logits=logits.masked_fill(~mask, -1e9))
        action = torch.tensor(step.action, dtype=torch.long, device=device)
        return dist.log_prob(action), state_value, dist.entropy()


class BacktracePPOAgentV2:
    def __init__(
        self,
        gate_embedding_dim,
        hidden_dim=32,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        k_epochs=8,
        eps_clip=0.2,
        rnd_beta=0.05,
        rnd_lr=1e-4,
        rnd_bonus_clip=5.0,
    ):
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.gamma = float(gamma)
        self.k_epochs = int(k_epochs)
        self.eps_clip = float(eps_clip)
        self.gate_embedding_dim = int(gate_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.rnd_beta = float(rnd_beta)
        self.rnd_lr = float(rnd_lr)
        self.rnd_bonus_clip = float(rnd_bonus_clip)
        self.buffer = RolloutBuffer()
        self.last_selected_step_idx = None
        self.last_selected_mode = None

        self.policy = BacktraceActorCriticV2(gate_embedding_dim, hidden_dim).to(device)
        self.policy_old = BacktraceActorCriticV2(gate_embedding_dim, hidden_dim).to(
            device
        )
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.gate_encoder.parameters(), "lr": lr_actor},
                {
                    "params": self.policy.objective_value_embedding.parameters(),
                    "lr": lr_actor,
                },
                {"params": self.policy.backtrace_actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )
        self.rnd = RandomNetworkDistillation(
            observation_dim=gate_embedding_dim + 2,
            hidden_dim=hidden_dim,
        ).to(device)
        self.rnd_optimizer = torch.optim.Adam(
            self.rnd.predictor.parameters(), lr=rnd_lr
        )
        self.rnd_error_stats = RunningMeanVariance()
        self.mse_loss = nn.MSELoss()
        self.update_count = 0

    def _gate_embedding(self, gate):
        if gate.deepgate_embedding is None:
            raise ValueError(
                f"Gate '{gate.outputpin}' is missing a fixed DeepGate embedding."
            )
        return gate.deepgate_embedding

    def _rnd_observation(self, objective_embedding, objective_value):
        value_vector = torch.tensor(
            [1.0, 0.0] if objective_value == 0 else [0.0, 1.0],
            dtype=torch.float32,
        )
        return torch.cat(
            [objective_embedding.float(), value_vector], dim=0
        ).detach().cpu()

    def _intrinsic_reward(self, observation):
        with torch.no_grad():
            error = float(
                self.rnd.prediction_error(observation.to(device).unsqueeze(0)).item()
            )
        scale = max(self.rnd_error_stats.variance, 1e-8) ** 0.5
        normalized = min(error / scale, self.rnd_bonus_clip)
        self.rnd_error_stats.update(error)
        return normalized

    def select_backtrace_action(
        self, objective_gate, objective_value, candidate_gates, action_mask=None
    ):
        if objective_value not in (0, 1):
            raise ValueError("Backtrace objective value must be 0 or 1.")
        if len(candidate_gates) != 2:
            raise ValueError("V2 backtrace actor requires exactly two input positions.")
        if action_mask is None:
            action_mask = [True, True]
        mask = torch.tensor(action_mask, dtype=torch.bool)
        if mask.numel() != 2 or not bool(mask.any()):
            raise ValueError("V2 backtrace action mask must enable at least one input.")

        objective_embedding = self._gate_embedding(objective_gate)
        logits, state_value = self.policy_old.backtrace_logits(
            objective_embedding, objective_value
        )
        dist = Categorical(logits=logits.masked_fill(~mask.to(device), -1e9))
        action = dist.sample()
        logprob = dist.log_prob(action)
        rnd_observation = self._rnd_observation(objective_embedding, objective_value)
        intrinsic_reward = self._intrinsic_reward(rnd_observation)
        self.buffer.steps.append(
            BacktraceDecisionStepV2(
                objective_embedding=objective_embedding.detach().cpu(),
                objective_value=objective_value,
                action_mask=mask,
                action=int(action.item()),
                logprob=logprob.detach(),
                state_value=state_value.detach(),
                rnd_observation=rnd_observation,
                intrinsic_reward=intrinsic_reward,
                reward=self.rnd_beta * intrinsic_reward,
            )
        )
        self.last_selected_step_idx = len(self.buffer.steps) - 1
        self.last_selected_mode = "backtrace"
        return candidate_gates[int(action.item())]

    def select_backtrace_action_deterministic(
        self, objective_gate, objective_value, candidate_gates, action_mask=None
    ):
        if objective_value not in (0, 1):
            raise ValueError("Backtrace objective value must be 0 or 1.")
        if len(candidate_gates) != 2:
            raise ValueError("V2 backtrace actor requires exactly two input positions.")
        if action_mask is None:
            action_mask = [True, True]
        mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
        if mask.numel() != 2 or not bool(mask.any()):
            raise ValueError("V2 backtrace action mask must enable at least one input.")

        with torch.no_grad():
            logits, _ = self.policy_old.backtrace_logits(
                self._gate_embedding(objective_gate), objective_value
            )
            action = torch.argmax(logits.masked_fill(~mask, -1e9))
        return candidate_gates[int(action.item())]

    def add_reward(self, reward):
        if self.buffer.steps:
            self.buffer.steps[-1].reward += reward

    def add_reward_to_step(self, step_idx, reward):
        if step_idx is not None and 0 <= step_idx < len(self.buffer.steps):
            self.buffer.steps[step_idx].reward += reward

    def finish_episode(self, final_reward):
        if self.buffer.steps:
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
        returns = torch.tensor(returns, dtype=torch.float64, device=device)
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-12)
        elif returns.abs().item() > 1.0:
            returns = returns / returns.abs()
        returns = returns.to(dtype=torch.float32)

        old_logprobs = torch.stack(
            [step.logprob for step in self.buffer.steps]
        ).to(device)
        old_values = torch.stack(
            [step.state_value for step in self.buffer.steps]
        ).to(device).squeeze(-1)
        advantages = returns.detach() - old_values.detach()
        rnd_observations = torch.stack(
            [step.rnd_observation for step in self.buffer.steps]
        ).to(device)

        final_metrics = None
        for _ in range(self.k_epochs):
            losses = []
            policy_losses = []
            value_losses = []
            entropies = []
            ratios = []
            for index, step in enumerate(self.buffer.steps):
                logprob, state_value, entropy = self.policy.evaluate_step(step)
                ratio = torch.exp(logprob - old_logprobs[index].detach())
                surrogate1 = ratio * advantages[index]
                surrogate2 = torch.clamp(
                    ratio, 1 - self.eps_clip, 1 + self.eps_clip
                ) * advantages[index]
                policy_loss = -torch.min(surrogate1, surrogate2)
                value_loss = self.mse_loss(state_value.squeeze(), returns[index])
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
                "policy_loss": float(
                    torch.stack(policy_losses).mean().detach().item()
                ),
                "value_loss": float(
                    torch.stack(value_losses).mean().detach().item()
                ),
                "entropy": float(torch.stack(entropies).mean().detach().item()),
                "ratio_mean": float(torch.stack(ratios).mean().detach().item()),
            }

        rnd_loss = self.rnd.prediction_error(rnd_observations).mean()
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.update_count += 1
        metrics = {
            "update": self.update_count,
            "steps": len(self.buffer.steps),
            "reward_sum": float(sum(step_rewards)),
            "intrinsic_reward_sum": float(
                sum(step.intrinsic_reward for step in self.buffer.steps)
            ),
            "rnd_loss": float(rnd_loss.detach().item()),
            "reward_last": float(step_rewards[-1]),
            "return_mean": float(returns.mean().detach().item()),
            "return_std": float(returns.std().detach().item())
            if returns.numel() > 1
            else 0.0,
            "adv_mean": float(advantages.mean().detach().item()),
            "adv_std": float(advantages.std().detach().item())
            if advantages.numel() > 1
            else 0.0,
            "epochs": self.k_epochs,
        }
        if final_metrics is not None:
            metrics.update(final_metrics)
        self.buffer.clear()
        return metrics

    def training_state_dict(self):
        return {
            "format": "RL_PODEM_PPO_RND_V2",
            "gate_embedding_dim": self.gate_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "policy": self.policy.state_dict(),
            "policy_old": self.policy_old.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rnd": self.rnd.state_dict(),
            "rnd_optimizer": self.rnd_optimizer.state_dict(),
            "rnd_error_stats": self.rnd_error_stats.state_dict(),
            "update_count": self.update_count,
            "hyperparameters": self.hyperparameters(),
        }

    def hyperparameters(self):
        return {
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "gamma": self.gamma,
            "k_epochs": self.k_epochs,
            "eps_clip": self.eps_clip,
            "rnd_beta": self.rnd_beta,
            "rnd_lr": self.rnd_lr,
            "rnd_bonus_clip": self.rnd_bonus_clip,
        }

    def load_training_state_dict(self, state):
        if state.get("format") != "RL_PODEM_PPO_RND_V2":
            raise ValueError("Unsupported V2 PPO/RND training checkpoint format.")
        if int(state["gate_embedding_dim"]) != self.gate_embedding_dim:
            raise ValueError("V2 checkpoint embedding dimension changed.")
        if int(state["hidden_dim"]) != self.hidden_dim:
            raise ValueError("V2 checkpoint hidden dimension changed.")
        if state.get("hyperparameters") != self.hyperparameters():
            raise ValueError("V2 checkpoint PPO/RND hyperparameters changed.")
        self.policy.load_state_dict(state["policy"])
        self.policy_old.load_state_dict(state["policy_old"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rnd.load_state_dict(state["rnd"])
        self.rnd_optimizer.load_state_dict(state["rnd_optimizer"])
        self.rnd_error_stats.load_state_dict(state["rnd_error_stats"])
        self.update_count = int(state["update_count"])
