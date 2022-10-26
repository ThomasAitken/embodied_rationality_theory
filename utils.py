from typing import Callable


def compute_max_gain_kelly_choice_from_reward_function(
    reward_function: Callable[[int], int],
    resource_cost: int,
    win_probability: float = 1.0,
) -> tuple[int, float, int]:
    """
    Returns the Kelly fraction from the given parameters.
    """
    profit_function = lambda resource_cost: reward_function(resource_cost) - resource_cost
    max_profit_cost = max(range(resource_cost), key=profit_function)
    max_profit = profit_function(max_profit_cost)
    b = (max_profit - max_profit_cost) / max_profit_cost  # proportion gained
    fraction = win_probability - (1 - win_probability) / b
    max_kelly_profit = fraction * max_profit
    return max_kelly_profit
