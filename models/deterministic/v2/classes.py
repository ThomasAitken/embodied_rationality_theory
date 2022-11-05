from types import MethodType
from typing import Callable

from models.deterministic.types import Payout


class InvestmentV2:
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
        resources_to_reward: Callable[[int], int],
        resources_to_resources: Callable[[int], int],
        inverse_resources_to_resources: Callable[[int], int],
        resource_capacity: int = 100,
        capacity_recovery_rate: float = 1.0,
        total_resources_invested: int = 0,
        reward_period: int = 10,
    ):
        self.id = id
        self.name = name
        self.resources_to_reward = MethodType(resources_to_reward, self)
        self.resources_to_resources = MethodType(resources_to_resources, self)
        self.inverse_resources_to_resources = MethodType(inverse_resources_to_resources, self)
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

    def compute_payout(self, added_resources: int = 0) -> Payout:
        """
        Returns three values:
            1.) the reward payout to be discharged at a given point in time (the total accrued, undischarged reward);
            2.) the resources profit to be discharged at a given point in time (the total accrued, undischarged resource profit);
            3.) the resources, of the "added_resources", actually spent by the agent.

        Optional arg "added_resources" allows one to compute the effect of a new resource injection.
        """
        resources_expended = 0
        if added_resources != 0:
            resources_expended = min(added_resources, self.resource_capacity)

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

    # def can_achieve_given_resources_profit(self, resources_profit: int) -> bool:
    #     for possible_expenditure in range(self.resource_capacity + 1):
    #         possible_profit = self.compute_payout(possible_expenditure)[1]
    #         if possible_profit == resources_profit:
    #             return True
    #     return False

    def get_payout_given_resource_parameters(
        self, agent_resources_available: int, resources_profit: int
    ) -> Payout | None:
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

    def get_max_resource_profit(self, agent_resources_available: int) -> int:
        resources_investible = min(agent_resources_available, self.resource_capacity)
        return max(self.compute_payout(expenditure)[1] for expenditure in range(resources_investible + 1))
