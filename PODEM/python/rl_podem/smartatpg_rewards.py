import math


BACKTRACK_MAX = 100
BACKTRACK_BASE = 0.5
BACKTRACK_POWER = 3
BACKTRACK_SCALE = 9.802960494

GAT_REWARD_SCHEME = "cubic_backtrack_v1"
MEAN_REWARD_SCHEME = "legacy_pi_exponential"


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
