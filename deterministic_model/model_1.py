from dataclasses import dataclass
from types import MethodType
from typing import Callable

from .algorithms import select_max_investment_by_reward_maximisation


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

    id: str
    name: str

    resource_capacity: int  # max capacity for investment at one time
    capacity_recovery_rate: float  # capacity recovered per time step
    total_resources_invested: int  # total resources put into this investment by the agent
    reward_period: int  # number of time steps defining the payout schedule (payout of reward/resources)

    _total_reward_discharged: int  # internal variable tracking the reward discharged to the agent
    _total_resources_discharged: int  # internal variable tracking the resources discharged to the agent

    def __init__(
        self,
        id: str,
        name: str,
        resources_to_reward: Callable[[int], float],
        resources_to_resources: Callable[[int], float],
        resource_capacity: int = 100,
        capacity_recovery_rate: float = 1.0,
        total_resources_invested: int = 0,
        reward_period: int = 10,
    ):
        self.id = id
        self.name = name
        self.resources_to_reward = MethodType(resources_to_reward, self)
        self.resources_to_resources = MethodType(resources_to_resources, self)
        self.resource_capacity = resource_capacity
        self.capacity_recovery_rate = capacity_recovery_rate
        self.total_resources_invested = total_resources_invested
        self.reward_period = reward_period

    """
    These methods are overriden in the __init__:

    @staticmethod
    def resources_to_reward(resources: int) -> float:
        \"\"\"
        Maps the total resources put into an investment to the reward that the agent should receive.
        Overridden via __init__.
        \"\"\"
        pass

    @staticmethod
    def resources_to_resources(resources: int) -> float:
        \"\"\"
        Maps the total resources put into an investment to the reward that the agent should receive.
        Overridden via __init__.
        \"\"\"
        pass
    """

    def compute_payouts_simple(self, added_resources: int = 0) -> tuple[int, int, int]:
        """
        Returns three values:
            1.) the reward payout to be discharged at a given point in time (the total accrued, undischarged reward);
            2.) the resources profit to be discharged at a given point in time (the total accrued, undischarged resource profit);
            3.) the resources, of the "added_resources", actually spent by the agent.

        Optional arg "added_resources" allows one to compute the effect of a new resource injection.
        """
        resources_expended = 0
        if added_resources != 0:
            resources_expended = max(added_resources, self.resource_capacity)

        reward_payout = (
            self.resources_to_reward(self.total_resources_invested + resources_expended) - self._total_reward_discharged
        )
        resource_payout = (
            self.resources_to_resources(self.total_resources_invested + resources_expended)
            - self._total_resources_discharged
        )
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


@dataclass
class AgentV1:
    energetic_resources: int = 100
    expected_lifespan: int = 1000


# def get_max_of_function(function: Callable[[int], int], max_bound=1000) -> float:
#     solution = scipy.optimize.minimize_scalar(lambda x: -function(x), bounds=[0,max_bound], method='bounded')
#     return solution.x


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


if __name__ == "main":

    perceived_investments: list[InvestmentV1] = generate_investments()

    agent = AgentV1()

    print("SIMPLE DETERMINISTIC OPTIMISATION\n")
    for timestep in range(agent.expected_lifespan):
        (
            max_investment,
            reward_payout,
            resources_payout,
            max_resources_expended,
        ) = select_max_investment_by_reward_maximisation(perceived_investments, agent.energetic_resources)
        max_investment.update_values_post_discharge(resources_expended, reward_payout, resources_payout)
        agent.instrumental_resources -= resources_expended

        print(f"Timestep {timestep+1}: agent spends {resources_expended} on ")
        update_investments(perceived_investments)
