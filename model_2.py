from dataclasses import dataclass
from types import MethodType
from typing import Callable

from model_1 import InvestmentV1


class InvestmentV2(InvestmentV1):
    """
    Same as InvestmentV1 but now with a baseline reward depletion rate reflecting entropy.
    """

    baseline_reward_depletion_rate: float
    time_since_last_injection: int = 0

    _total_resources_discharged: int  # internal variable tracking the resources discharged to the agent

    def __init__(
        self, resources_to_resources: Callable[[int], int], baseline_reward_depletion_rate: float = 1.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.resources_to_resources = MethodType(resources_to_resources, self)
        self.baseline_reward_depletion_rate = baseline_reward_depletion_rate

    """"
    @staticmethod
    def resources_to_resources(resources: int) -> int:
        \"\"\"
        Maps the total resources put into an investment to the reward that the agent should receive.
        Overridden via __init__.
        \"\"\"
        pass
    """

    def register_injection(self):
        self.time_since_last_injection = 0

    def compute_payouts(self, added_resources: int = 0) -> tuple[int, int, int]:
        """
        Returns three values:
            1.) the reward payout to be discharged at a given point in time (the total accrued, undischarged reward);
            2.) the resources payout to be discharged at a given point in time (the total accrued, undischarged resources);
            3.) the resources, of the "added_resources", actually spent by the agent.

        Optional arg "added_resources" allows one to compute the effect of a new resource injection.
        """
        resources_expended = 0
        if added_resources != 0:
            resources_expended = max(added_resources, self.resource_capacity)
        return (
            self.resources_to_reward(self.total_resources_invested + resources_expended)
            - self._total_reward_discharged,
            self.resources_to_resources(self.total_resources_invested + resources_expended)
            - self._total_resources_discharged,
            resources_expended,
        )

    def update_values_post_discharge(self, resources_invested: int, reward_payout: int, resources_payout: int):
        """
        Executed after a reward/resource discharge.
        """
        self.resource_capacity -= resources_invested
        self._total_reward_discharged -= reward_payout
        self._total_resources_discharged -= resources_payout

    def update_values_post_discharge(self, resources_invested: int, reward_payout: int, resources_payout: int):
        """
        Executed after a reward/resource discharge.
        """
        self.resource_capacity -= resources_invested
        self._total_reward_discharged -= reward_payout
        self._total_resources_discharged -= resources_payout


@dataclass
class OtherEnvironmentalFactors:
    """
    Contains the parameters determine other aspects of the agent's environment, apart from the investments within it.
    Not used for the V1 model, just here as an illustration.
    """

    energetic_depletion_rate: float = 1.0  # baseline energetic resources of the agent depleted per time step


@dataclass
class AgentV1:
    energetic_resources: int = 100  # ignore this in V1
    expected_lifespan: int = (
        700800  # lifespan in model timesteps ~ 80 years if each timestep is an hour (24 * 365 * 80)
    )
    instrumental_resources: int = 500
