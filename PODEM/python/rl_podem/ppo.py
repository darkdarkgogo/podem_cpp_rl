import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .advantages import full_fault_targets, stable_population_std


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")


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
    circuit_hash: Optional[str] = None
    objective_name: Optional[str] = None


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
        model_device = self.gate_encoder[0].weight.device
        gate_tensor = objective_embedding.to(
            device=model_device, dtype=torch.float32
        ).unsqueeze(0)
        value_tensor = torch.tensor(
            [objective_value], dtype=torch.long, device=model_device
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
        mask = step.action_mask.to(device=logits.device, dtype=torch.bool)
        dist = Categorical(logits=logits.masked_fill(~mask, -1e9))
        action = torch.tensor(step.action, dtype=torch.long, device=logits.device)
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
        minibatch_size=128,
        eps_clip=0.2,
        rnd_beta=0.05,
        rnd_lr=1e-4,
        rnd_bonus_clip=5.0,
        normalize_returns=True,
        entropy_coef=0.01,
        return_scale=1.0,
        max_grad_norm=0.0,
        advantage_method="mc",
        gae_lambda=0.97,
        normalize_advantages=False,
    ):
        self.lr_actor = float(lr_actor)
        self.lr_critic = float(lr_critic)
        self.gamma = float(gamma)
        self.k_epochs = int(k_epochs)
        self.minibatch_size = int(minibatch_size)
        self.eps_clip = float(eps_clip)
        self.gate_embedding_dim = int(gate_embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.rnd_beta = float(rnd_beta)
        self.rnd_lr = float(rnd_lr)
        self.rnd_bonus_clip = float(rnd_bonus_clip)
        self.normalize_returns = bool(normalize_returns)
        self.entropy_coef = float(entropy_coef)
        self.return_scale = float(return_scale)
        self.max_grad_norm = float(max_grad_norm)
        self.advantage_method = str(advantage_method)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = bool(normalize_advantages)
        if (not math.isfinite(self.return_scale) or self.return_scale <= 0.0
                or not math.isfinite(self.max_grad_norm) or self.max_grad_norm < 0.0):
            raise ValueError("Return scale must be positive and gradient norm non-negative.")
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if not math.isfinite(self.gae_lambda) or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be finite and in [0, 1].")
        if self.advantage_method not in ("mc", "gae"):
            raise ValueError("advantage_method must be 'mc' or 'gae'.")
        if self.k_epochs <= 0:
            raise ValueError("k_epochs must be positive.")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive.")
        if self.advantage_method == "gae" and self.normalize_returns:
            raise ValueError("GAE requires normalize_returns=False.")
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

    def _validate_rollout_indices(self, indices):
        indices = tuple(indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(self.buffer.steps)
            for index in indices
        ):
            raise IndexError("Rollout indices must reference buffered transitions")
        return indices

    def _evaluate_rollout(self, indices):
        indices = self._validate_rollout_indices(indices)
        return [
            self.policy.evaluate_step(self.buffer.steps[index])
            for index in indices
        ]

    def update(self, rollout_faults=None):
        if not self.buffer.steps:
            return None

        step_count = len(self.buffer.steps)
        if rollout_faults is None:
            rollout_faults = sum(step.is_terminal for step in self.buffer.steps)
        if isinstance(rollout_faults, bool) or not isinstance(rollout_faults, int):
            raise ValueError("rollout_faults must be a positive integer.")
        if rollout_faults <= 0:
            raise ValueError("rollout_faults must be a positive integer.")
        step_rewards = [step.reward for step in self.buffer.steps]
        old_logprobs = torch.stack(
            [step.logprob for step in self.buffer.steps]
        ).to(device)
        old_values = torch.stack(
            [step.state_value for step in self.buffer.steps]
        ).to(device).reshape(-1)
        targets = full_fault_targets(
            step_rewards,
            old_values,
            [step.is_terminal for step in self.buffer.steps],
            gamma=self.gamma,
            advantage_method=self.advantage_method,
            gae_lambda=self.gae_lambda,
            return_scale=self.return_scale,
            normalize_advantages=self.normalize_advantages,
            normalize_returns=self.normalize_returns,
        )
        returns = targets.value_targets
        advantages = targets.actor_advantages
        rnd_observations = torch.stack(
            [step.rnd_observation for step in self.buffer.steps]
        ).to(device)

        metric_names = (
            "total_loss", "policy_loss", "value_loss", "entropy",
            "ratio_mean", "approx_kl", "clip_fraction",
        )
        weighted_metrics = {name: 0.0 for name in metric_names}
        last_epoch_metrics = {name: 0.0 for name in metric_names}
        optimizer_steps = 0
        for epoch in range(self.k_epochs):
            permutation = torch.randperm(step_count).tolist()
            for start in range(0, step_count, self.minibatch_size):
                indices = permutation[start:start + self.minibatch_size]
                batch_indices = torch.as_tensor(
                    indices, dtype=torch.long, device=device,
                )
                evaluated = self._evaluate_rollout(indices)
                logprobs = torch.stack([item[0] for item in evaluated])
                state_values = torch.stack([item[1] for item in evaluated]).reshape(-1)
                entropies = torch.stack([item[2] for item in evaluated])
                selected_old_logprobs = old_logprobs.index_select(
                    0, batch_indices,
                ).detach()
                selected_advantages = advantages.index_select(0, batch_indices)
                selected_returns = returns.index_select(0, batch_indices)
                log_ratios = logprobs - selected_old_logprobs
                ratios = torch.exp(log_ratios)
                surrogate1 = ratios * selected_advantages
                surrogate2 = torch.clamp(
                    ratios, 1 - self.eps_clip, 1 + self.eps_clip
                ) * selected_advantages
                policy_losses = -torch.min(surrogate1, surrogate2)
                value_losses = (state_values - selected_returns).pow(2)
                losses = (
                    policy_losses + 0.5 * value_losses
                    - self.entropy_coef * entropies
                )
                total_loss = losses.mean()
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()
                optimizer_steps += 1

                batch_metrics = {
                    "total_loss": total_loss.detach(),
                    "policy_loss": policy_losses.mean().detach(),
                    "value_loss": value_losses.mean().detach(),
                    "entropy": entropies.mean().detach(),
                    "ratio_mean": ratios.mean().detach(),
                    "approx_kl": ((ratios - 1.0) - log_ratios).mean().detach(),
                    "clip_fraction": (
                        (ratios - 1.0).abs() > self.eps_clip
                    ).float().mean().detach(),
                }
                batch_size = len(indices)
                for name, value in batch_metrics.items():
                    weighted_metrics[name] += float(value.item()) * batch_size
                    if epoch == self.k_epochs - 1:
                        last_epoch_metrics[name] += float(value.item()) * batch_size

        if self.rnd_beta > 0.0:
            rnd_loss = self.rnd.prediction_error(rnd_observations).mean()
            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()
        else:
            rnd_loss = torch.zeros((), dtype=torch.float32, device=device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.update_count += 1
        metrics = {
            "update": self.update_count,
            "steps": step_count,
            "rollout_faults": rollout_faults,
            "rollout_transitions": step_count,
            "minibatches_per_epoch": math.ceil(
                step_count / self.minibatch_size
            ),
            "optimizer_steps": optimizer_steps,
            "reward_sum": float(sum(step_rewards)),
            "intrinsic_reward_sum": float(
                sum(step.intrinsic_reward for step in self.buffer.steps)
            ),
            "rnd_loss": float(rnd_loss.detach().item()),
            "reward_last": float(step_rewards[-1]),
            "advantage_method": self.advantage_method,
            "scaled_reward_mean": float(targets.scaled_rewards.mean().item()),
            "scaled_reward_std": float(stable_population_std(targets.scaled_rewards).item()),
            "scaled_reward_sum": float(targets.scaled_rewards.sum().item()),
            "raw_adv_mean": float(targets.raw_advantages.mean().item()),
            "raw_adv_std": float(targets.raw_advantages.std(unbiased=False).item()),
            "value_target_mean": float(returns.mean().item()),
            "value_target_std": float(returns.std(unbiased=False).item()),
            "actor_adv_mean": float(advantages.mean().item()),
            "actor_adv_std": float(advantages.std(unbiased=False).item()),
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
        metric_denominator = self.k_epochs * step_count
        for name in metric_names:
            metrics[name] = weighted_metrics[name] / metric_denominator
            metrics[f"{name}_last_epoch"] = (
                last_epoch_metrics[name] / step_count
            )
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
            "minibatch_size": self.minibatch_size,
            "eps_clip": self.eps_clip,
            "rnd_beta": self.rnd_beta,
            "rnd_lr": self.rnd_lr,
            "rnd_bonus_clip": self.rnd_bonus_clip,
            "normalize_returns": self.normalize_returns,
            "entropy_coef": self.entropy_coef,
            "return_scale": self.return_scale,
            "max_grad_norm": self.max_grad_norm,
            "advantage_method": self.advantage_method,
            "gae_lambda": self.gae_lambda,
            "normalize_advantages": self.normalize_advantages,
        }

    def load_actor_state_dict(self, state_dict):
        """Warm-start a fresh agent without importing an old Critic or optimizer."""
        if self.update_count or self.buffer.steps or self.optimizer.state:
            raise ValueError("Actor weights-only warm start requires a fresh agent.")
        current = self.policy.state_dict()
        actor_keys = {key for key in current if not key.startswith("critic.")}
        if not actor_keys.issubset(state_dict) or set(state_dict) - set(current):
            raise ValueError("Invalid V2 Actor state dictionary.")
        current.update({key: state_dict[key] for key in actor_keys})
        self.policy.load_state_dict(current)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def load_training_state_dict(self, state):
        if state.get("format") != "RL_PODEM_PPO_RND_V2":
            raise ValueError("Unsupported V2 PPO/RND training checkpoint format.")
        if int(state["gate_embedding_dim"]) != self.gate_embedding_dim:
            raise ValueError("V2 checkpoint embedding dimension changed.")
        if int(state["hidden_dim"]) != self.hidden_dim:
            raise ValueError("V2 checkpoint hidden dimension changed.")
        saved_hyperparameters = dict(state.get("hyperparameters", {}))
        saved_hyperparameters.setdefault("normalize_returns", True)
        saved_hyperparameters.setdefault("entropy_coef", 0.01)
        saved_hyperparameters.setdefault("return_scale", 1.0)
        saved_hyperparameters.setdefault("max_grad_norm", 0.0)
        saved_hyperparameters.setdefault("advantage_method", "mc")
        saved_hyperparameters.setdefault("gae_lambda", 0.97)
        saved_hyperparameters.setdefault("normalize_advantages", False)
        if saved_hyperparameters["normalize_returns"] and not self.normalize_returns:
            raise ValueError(
                "Legacy checkpoint uses per-fault return normalization. "
                "Use a fresh training checkpoint or an explicit weights-only warm start."
            )
        if saved_hyperparameters != self.hyperparameters():
            raise ValueError("V2 checkpoint PPO/RND hyperparameters changed.")
        self.policy.load_state_dict(state["policy"])
        self.policy_old.load_state_dict(state["policy_old"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rnd.load_state_dict(state["rnd"])
        self.rnd_optimizer.load_state_dict(state["rnd_optimizer"])
        self.rnd_error_stats.load_state_dict(state["rnd_error_stats"])
        self.update_count = int(state["update_count"])
