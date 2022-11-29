from dataclasses import dataclass
from typing import TypedDict


class Payout(TypedDict):
    discharge_reached: bool
    reward: int
    resource_profit: int
    resources_spent: int


class InvestmentV1:
    """
    A thing in the environment into which an agent invests resources, from which the agent expects to receive reward
    from their total investment according to the given "resources_to_reward" method and/or additional resources according
    to the given "resources_to_resources" method. These payouts are given according to a schedule operating on a fixed
    period (specified by the "reward_period" parameter). The payouts discharge all reward/resources accrued in the
    investment. The relationship between the level of investment and the payout is specified by the methods
    "resources_to_reward" and "resources_to_resources". For now, these generate payout deterministically; we will
    introduce stochasticity later.

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
        self._total_reward_discharged = 0
        self._total_resources_discharged = 0

    def compute_payout(self, added_resources: int = 0) -> Payout:
        resources_expended = 0
        if added_resources != 0:
            resources_expended = min(added_resources, self.resource_capacity)

        discharge_reached = self.current_resources_invested + resources_expended >= self.discharge_threshold
        reward_payout = self.reward_discharge_amount if discharge_reached else 0
        resource_payout = self.resource_discharge_amount if discharge_reached else 0
        resource_profit = resource_payout - resources_expended
        return Payout(
            discharge_reached=discharge_reached,
            reward=reward_payout,
            resource_profit=resource_profit,
            resources_spent=resources_expended,
        )

    def update_values_post_investment(
        self, discharge_reached: bool, resources_invested: int, reward_payout: int, resources_profit: int
    ):
        """
        Executed after a reward/resource discharge.
        """
        if discharge_reached:
            self.current_resources_invested = 0
        else:
            self.current_resources_invested += resources_invested
        self.resource_capacity -= resources_invested
        self._total_reward_discharged -= reward_payout
        self._total_resources_discharged -= resources_profit + resources_invested

    def get_payout_given_resource_parameters(self, agent_resources_available: int, resources_profit: int):
        resources_investible = min(agent_resources_available, self.resource_capacity)
        for possible_expenditure in range(resources_investible + 1):
            possible_payout = self.compute_payout(possible_expenditure)
            if possible_payout[1] == resources_profit:
                return possible_payout
        return None

    def get_min_resource_profit(self, agent_resources_available: int) -> int:
        """
        Calculates the minimum possible resources profit for a given investment with the given amount of resources.
        """
        resources_investible = min(agent_resources_available, self.resource_capacity)
        return min(self.compute_payout(expenditure)[1] for expenditure in range(resources_investible + 1))

    # def get_max_resource_profit

    def get_resources_until_payout_post_injection(self, resources_invested: int) -> int:
        return self.discharge_threshold - (self.current_resources_invested + resources_invested)

    @property
    def resources_until_payout(self) -> int:
        return self.discharge_threshold - self.current_resources_invested

    @property
    def is_net_resource_positive(self):
        return self.resource_discharge_amount > self.discharge_threshold

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.id == other.id

    def __str__(self):
        return f"Id: {self.id}, Name: {self.name}, Discharge threshold: {self.discharge_threshold}, Reward discharge amount: {self.reward_discharge_amount}, Resource discharge amount: {self.resource_discharge_amount}, Capacity Recovery Rate: {self.capacity_recovery_rate}, Resource capacity: {self.resource_capacity}, Current Resources Invested: {self.current_resources_invested}"


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
    investments_chosen: list[str]
    world_copy: list[InvestmentV1]
    # fields for plotting
    resource_level_at_each_step: list[int]
    reward_level_at_each_step: list[int]

    def __repr__(self):
        return f"Resources spent: {self.resources_spent}, Resources to spend: {self.resources_to_spend}, Reward to date: {self.reward_to_date}"
