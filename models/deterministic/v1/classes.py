from dataclasses import dataclass

from models.deterministic.types import Payout


class InvestmentV1:
    """
    A thing in the environment into which an agent invests resources (energetic and instrumental), from which the agent
    expects to receive reward from their total investment according to the given "resources_to_reward" method and/or
    additional resources according to the given "resources_to_resources" method. These payouts are given according to a
    schedule operating on a fixed period (specified by the "reward_period" parameter). The payouts discharge all
    reward/resources accrued in the investment. The relationship between the level of investment and the payout is
    specified by the methods "resources_to_reward" and "resources_to_resources". For now, these generate payout
    deterministically; we will introduce stochasticity later.

    Each investment can only accept so many resources at a given time, as determined by the "resource_capacity".
    This capacity then recovers according to the parameter "capacity_recovery_rate"
    """

    ### Identifiers ###
    id: str
    name: str

    ### Constants ###
    discharge_threshold: int  # threshold of resources invested before reward/resource discharge
    reward_discharge_amount: int  # amount of reward discharged
    resource_discharge_amount: int  # amount of resources discharged
    capacity_recovery_rate: int  # resource investment capacity recovered per time step after a discharge

    ### Variables ###
    resource_capacity: int  # max capacity for investment at one time, initially set to discharge_threshold
    current_resources_invested: int  # resources put into this investment by the agent so far (reset after discharge)

    _total_reward_discharged: int  # internal variable tracking the reward discharged to the agent
    _total_resources_discharged: int  # internal variable tracking the resources discharged to the agent

    def __init__(
        self,
        id: str,
        name: str,
        discharge_threshold: int,
        reward_discharge_amount: int,
        resource_discharge_amount: int,
        capacity_recovery_rate: int = 1,
    ):
        self.id = id
        self.name = name
        self.discharge_threshold = discharge_threshold
        self.reward_discharge_amount = reward_discharge_amount
        self.resource_discharge_amount = resource_discharge_amount
        self.resource_capacity = discharge_threshold
        self.capacity_recovery_rate = capacity_recovery_rate
        self.current_resources_invested = 0
        self._total_resources_discharged = 0
        self._total_resources_discharged = 0

    def compute_payout(self, added_resources: int = 0) -> Payout:
        """
        Returns three values:
            1.) the reward payout to be discharged at a given point in time;
            2.) the resources profit to be discharged at a given point in time;
            3.) the resources, of the "added_resources", actually spent by the agent.
        """

        resources_expended = 0
        if added_resources != 0:
            resources_expended = min(added_resources, self.resource_capacity)

        discharge_reached = self.current_resources_invested + resources_expended >= self.discharge_threshold
        reward_payout = self.reward_discharge_amount if discharge_reached else 0
        resource_payout = self.resource_discharge_amount if discharge_reached else 0
        resource_profit = resource_payout - resources_expended
        return (
            reward_payout,
            resource_profit,
            resources_expended,
        )

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
    baseline_death_probability: float = 0  # baseline chance of the agent dying


@dataclass
class AgentV1:
    energetic_resources: int = 100
    expected_lifespan: int = 1000


@dataclass
class ResourcePath:
    resources_spent: int
    resources_to_spend: int
    reward_to_date: int
    investments_chosen: list[InvestmentV1]
    world_copy: list[InvestmentV1]
    best_next_investments_by_resource_profit: dict[int, InvestmentV1]
