from dataclasses import dataclass

from models.deterministic.minimal.classes import InvestmentMinimal


class InvestmentMinimalUnselfish(InvestmentMinimal):
    def __init__(self, beneficiary_id: str, weighting: float, **kwargs):
        super().__init__(**kwargs)
        self.beneficiary_id = beneficiary_id
        self.weighting = weighting
        self.reward_discharge_amount *= weighting
        self.resource_discharge_amount *= weighting


@dataclass
class ResourcePath:
    resources_spent: int
    resources_to_spend: int
    reward_to_date: int
    reward_for_others_to_date: dict[str, int]  # maps beneficiary id to reward level
    world_copy: list[InvestmentMinimalUnselfish]
    # fields for plotting
    resource_level_at_each_step: list[int]
    reward_level_at_each_step: list[int]
    reward_level_for_others_at_each_step: list[dict[str, int]]
