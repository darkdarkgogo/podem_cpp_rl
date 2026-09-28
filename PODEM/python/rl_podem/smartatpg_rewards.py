import math


BACKTRACK_MAX = 100
BACKTRACK_BASE = 0.5
BACKTRACK_POWER = 3
BACKTRACK_SCALE = 9.802960494
GAT_STEP_PENALTY_BUDGET = 10.0

GAT_REWARD_SCHEME = "cubic_backtrack_depthnorm_v2"
MEAN_REWARD_SCHEME = "legacy_pi_exponential"
PAPER_REWARD = {
    "non_pi": -0.1,
    "alpha": 7.5,
    "beta": 0.07,
    "detected": 100.0,
    "undetected": -100.0,
}


def reward_scheme_for_encoder(encoder_variant: str) -> str:
    schemes = {
        "level_gat_gru": GAT_REWARD_SCHEME,
        "fanin_mean": MEAN_REWARD_SCHEME,
    }
    try:
        return schemes[encoder_variant]
    except KeyError as error:
        raise ValueError(
            f"Unsupported SmartATPG encoder variant: {encoder_variant}"
        ) from error


def smartatpg_circuit_depth(levels) -> int:
    values = tuple(levels)
    if not values:
        raise ValueError("SmartATPG circuit levels must not be empty.")
    if any(
        isinstance(level, bool) or not isinstance(level, int) or level < 0
        for level in values
    ):
        raise ValueError("SmartATPG circuit levels must be non-negative integers.")
    return max(1, max(values))


def smartatpg_backtrace_step_reward(
    reward_scheme: str,
    circuit_depth: int,
) -> float:
    if (
        isinstance(circuit_depth, bool)
        or not isinstance(circuit_depth, int)
        or circuit_depth <= 0
    ):
        raise ValueError("SmartATPG circuit depth must be a positive integer.")
    if reward_scheme == GAT_REWARD_SCHEME:
        return -GAT_STEP_PENALTY_BUDGET / circuit_depth
    if reward_scheme == MEAN_REWARD_SCHEME:
        return PAPER_REWARD["non_pi"]
    raise ValueError(f"Unknown SmartATPG reward scheme: {reward_scheme}")


def smartatpg_backtrack_reward(backtrack_count: int) -> float:
    if isinstance(backtrack_count, bool) or not isinstance(backtrack_count, int):
        raise TypeError("backtrack_count must be an integer")
    if not 1 <= backtrack_count <= BACKTRACK_MAX:
        raise ValueError("backtrack_count out of range")
    x = backtrack_count / BACKTRACK_MAX
    return -(BACKTRACK_BASE + BACKTRACK_SCALE * (x ** BACKTRACK_POWER))


def smartatpg_pi_reward(
    backtracks: int,
    pi_visits: int,
    alpha: float = 7.5,
    beta: float = 0.07,
) -> float:
    if backtracks < 0 or pi_visits <= 0:
        raise ValueError("SmartATPG reward counters are out of range.")
    return 10.0 - alpha * math.exp(beta * (backtracks + pi_visits))
