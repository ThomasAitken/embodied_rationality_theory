from typing import TypedDict


class Payout(TypedDict):
    discharge_reached: bool
    reward: int
    resource_profit: int
    resources_spent: int
