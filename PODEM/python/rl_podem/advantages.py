"""Detached value targets for complete fault-solving rollouts."""

import math
from dataclasses import dataclass

import torch


@dataclass
class RolloutTargets:
    scaled_rewards: torch.Tensor
    value_targets: torch.Tensor
    raw_advantages: torch.Tensor
    actor_advantages: torch.Tensor


def stable_population_std(values):
    """Avoid variance overflow for finite, extreme legacy paper rewards."""
    values = values.detach().double()
    magnitude = values.abs().max()
    if magnitude.item() == 0.0:
        return magnitude
    return (values / magnitude).std(unbiased=False) * magnitude


@torch.no_grad()
def full_fault_targets(
    rewards,
    old_values,
    terminals,
    *,
    gamma=0.99,
    advantage_method="gae",
    gae_lambda=0.97,
    return_scale=100.0,
    normalize_advantages=True,
    normalize_returns=False,
):
    """Use zero terminal continuation; reject an unfinished rollout.

    Return normalization is supported only for legacy paper-reward MC runs.
    New curriculum MC/GAE runs normalize only the Actor's Advantage copy.
    """
    if advantage_method not in ("mc", "gae"):
        raise ValueError("advantage_method must be 'mc' or 'gae'.")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1].")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be finite and in [0, 1].")
    if not math.isfinite(return_scale) or return_scale <= 0.0:
        raise ValueError("return_scale must be finite and positive.")
    if normalize_returns and advantage_method == "gae":
        raise ValueError("GAE requires normalize_returns=False; normalize Actor Advantages instead.")

    values = torch.as_tensor(old_values).detach().to(dtype=torch.float64).reshape(-1)
    rewards = torch.as_tensor(rewards, device=values.device, dtype=torch.float64)
    terminals = torch.as_tensor(terminals, device=values.device, dtype=torch.bool)
    if rewards.ndim != 1 or terminals.ndim != 1:
        raise ValueError("Rewards and terminal flags must be one-dimensional.")
    if not values.numel() or not (values.numel() == rewards.numel() == terminals.numel()):
        raise ValueError("Rewards, values and terminal flags must have equal nonzero lengths.")
    if not bool(terminals[-1]):
        raise ValueError("Full-fault update requires a terminal final step, not a truncated rollout.")
    if not bool(torch.isfinite(values).all() and torch.isfinite(rewards).all()):
        raise ValueError("Rollout rewards and old values must be finite.")

    scaled_rewards = rewards / return_scale
    targets = torch.empty_like(values)
    if advantage_method == "mc":
        running_return = values.new_zeros(())
        for index in range(values.numel() - 1, -1, -1):
            if bool(terminals[index]):
                running_return = values.new_zeros(())
            running_return = scaled_rewards[index] + gamma * running_return
            targets[index] = running_return
        if normalize_returns:
            if targets.numel() > 1:
                targets = (targets - targets.mean()) / (targets.std() + 1e-12)
            elif targets.abs().item() > 1.0:
                targets = targets / targets.abs()
        raw_advantages = targets - values
    else:
        raw_advantages = torch.empty_like(values)
        running_advantage = values.new_zeros(())
        for index in range(values.numel() - 1, -1, -1):
            if bool(terminals[index]):
                next_value = values.new_zeros(())
                running_advantage = values.new_zeros(())
            else:
                next_value = values[index + 1]
            delta = scaled_rewards[index] + gamma * next_value - values[index]
            running_advantage = delta + gamma * gae_lambda * running_advantage
            raw_advantages[index] = running_advantage
        targets = raw_advantages + values

    actor_advantages = raw_advantages.clone()
    if normalize_advantages and actor_advantages.numel() > 1:
        actor_advantages = (actor_advantages - actor_advantages.mean()) / (
            actor_advantages.std(unbiased=False) + 1e-8
        )
    # Float64 protects legacy large rewards during accumulation/normalization.
    targets = targets.float()
    raw_advantages = raw_advantages.float()
    actor_advantages = actor_advantages.float()
    if not all(bool(torch.isfinite(tensor).all()) for tensor in (
        scaled_rewards, targets, raw_advantages, actor_advantages
    )):
        raise ValueError("Non-finite PPO targets; check rewards or increase return_scale.")
    return RolloutTargets(scaled_rewards, targets, raw_advantages, actor_advantages)
