import dataclasses
from copy import copy, deepcopy
from typing import TYPE_CHECKING

from .utils import get_resource_gain_bounds, update_investments

if TYPE_CHECKING:
    from .classes import InvestmentV1, OtherEnvironmentalFactors, Payout, ResourcePath

# max_investment, reward_payout, resources_payout, resources_expended
InvestmentSelection = tuple[InvestmentV1, dict[str, int]]


def adjust_payout_for_environment(environment: OtherEnvironmentalFactors) -> Payout:
    pass


def select_max_investment_by_reward_maximisation(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    """
    Finds the selection that maximises reward. If there is more than one, finds the one with the highest resource profit.
    """
    investments_with_payouts = [(investment, investment.compute_payout(resources)) for investment in investments]
    max_reward = max(map(lambda x: x[1]["reward"], investments_with_payouts))
    # select investment with max reward and highest net resources
    max_investment = max(
        filter(lambda x: x[1][0] == max_reward, investments_with_payouts),
        key=lambda x: x[1]["resource_profit"],
    )
    investment, payout = max_investment
    return investment, payout


def select_max_investment_by_reward_over_resources_profit(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    max_investment = max(
        investments, key=lambda x: x.compute_payout(resources)["reward"] - x.compute_payout(resources)["resource_profit"]
    )  # reward - resources_expended
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_fixed_tradeoff_heuristic(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)["reward"] + (1 - reward_bias) * x.compute_payout(resources)["resource_profit"],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended

def select_max_investment_by_reward_minus_resources(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)["reward"] + (1 - reward_bias) * x.compute_payout(resources)["resource_profit"],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended

def compute_min_reward_bound_by_resource_maxing(
    investments: list[InvestmentV1], resources: int
) -> int:
    """
    Returns the amount of reward obtained by choosing the investment that maximises resource profit.
    This serves as a lower bound requirement for a given investment selection decision. If the resource discharge amount
    for a given investment is below this, it is provably suboptimal by Theorem X.
    """
    max_investment = max(
        investments,
        key=lambda x: x.compute_payout(resources)["resource_profit"],
    )
    reward_payout = max_investment.compute_payout(resources)["reward"]
    return reward_payout

def compute_min_resource_bound_by_reward_maxing(
    investments: list[InvestmentV1], resources: int
) -> int:
    """
    Returns the amount of resources obtained by choosing the investment that maximises reward.
    This serves as a lower bound requirement for a given investment selection decision. If the reward discharge amount
    for a given investment is below this, it is provably suboptimal by Theorem X.
    """
    max_investment = max(
        investments,
        key=lambda x: x.reward_discharge_amount
    )
    resources_profit = max_investment.compute_payout(resources)["resource_profit"]
    return resources_profit

def get_best_investments_by_resource_profit(
    investments: list[InvestmentV1], agent_resources: int, min_profit, max_profit
) -> dict[int, tuple[InvestmentV1, int]]:
    """
    Returns a map of investments that get the highest reward by each possible level of resource profit.
    """
    resource_to_selection_map = {}
    for r in range(min_profit, max_profit + 1):
        investment_and_payouts_given_resource_constraints = [
            (investment, investment.get_payout_given_resource_parameters(agent_resources, r)[0])
            for investment in investments
        ]
        investment_and_payouts_given_resource_constraints = filter(
            lambda i: i[1] is not None,
            investment_and_payouts_given_resource_constraints,
        )
        resource_to_selection_map[r] = max(investment_and_payouts_given_resource_constraints, lambda p: p[1])
    return resource_to_selection_map

def compute_path_of_immediate_reward_maximisation(resource_path: ResourcePath, timesteps_remaining: int) -> ResourcePath:
    for _ in range(timesteps_remaining):
        chosen_investment, payout = select_max_investment_by_reward_maximisation(resource_path.world_copy)
        chosen_investment_copy = copy(chosen_investment)
        chosen_investment_copy.update_values_post_discharge(payout["resources_spent"], payout["reward"], payout["resource_profit"])



    
def get_nondominated_consumption_choices(investment: InvestmentV1, resources: int):
    """
    Assume: we can't pay over the discharge threshold for a given investment.
    If the investment cannot deliver a resource profit on the current turn, include every choice up to the discharge
    threshold.
    If the investment can deliver a resource profit on the current turn, include only the choice that matches the
    discharge threshold. Any other choice is provably inferior by Theorem X.
    """
    max_resources_to_spend = min(investment.resources_until_payout, investment.resource_capacity, resources)

    can_reach_discharge = max_resources_to_spend >= investment.resources_until_payout
    if investment.is_net_resource_positive and can_reach_discharge: 
        return [investment.compute_payout(investment.resources_until_payout)]
    return [investment.compute_payout(r) for r in range(max_resources_to_spend)]

def boundedly_optimise_max_investment(
    investments: set[InvestmentV1], resources: int, lookahead_steps: int
) -> ResourcePath:
    """
    A branch-and-bound programming approach considering all optimal paths through time for a given level of total resource
    consumption. The search needs to be bounded by lookahead_steps, since it is a potentially unbounded problem.

    Guaranteed to find the path through time that is optimal for maximising reward after n lookahead_steps.
    If more than one, select the path that has the highest available resources.
    """

    resource_paths: list[ResourcePath] = []
    for investment in investments:
        investment_copy = copy(investment)
        for choice in get_nondominated_consumption_choices(investment_copy, resources):
            investment_copy.update_values_post_discharge(choice["resources_spent"], choice["reward"], choice["resource_profit"])
            resource_paths.append(ResourcePath(
                resources_spent=choice["resource_spent"],
                resources_to_spend=resources + choice["resource_profit"],
                reward_to_date=choice["reward"],
                investments_chosen=[investment_copy.id],
                world_copy=[investment_copy]
            ))
    for _ in range(lookahead_steps):
        new_resource_paths = []
        for r in resource_paths:
            if r.resources_to_spend == 0: # agent is dead
                continue
            
            min_resource_take = compute_min_resource_bound_by_reward_maxing(r.world_copy, r.resources_to_spend)
            min_reward_take = compute_min_reward_bound_by_resource_maxing(r.world_copy, r.resources_to_spend)
            for investment in r.world_copy:
                # fails reward lower bound
                if investment.reward_discharge_amount < min_reward_take:
                    continue
                # fails resource lower bound
                if investment.resource_discharge_amount < min_resource_take:
                    continue

                investment_copy = copy(investment)
                for choice in get_nondominated_consumption_choices(investment_copy, r.resources_to_spend):
 
                    investment_copy.update_values_post_discharge(choice["resources_spent"], choice["reward"], choice["resource_profit"])

                    world_copy_copy = [investment_copy if invest.id == investment.id else invest for invest in r.world_copy]

                    new_resource_paths.append(ResourcePath(
                        resources_spent=r.resources_spent + choice["resources_spent"],
                        resources_to_spend=r.resources_to_spend + choice["resource_profit"],
                        reward_to_date=r.reward_to_date + choice["reward"],
                        investments_chosen=r.investments_chosen+[investment_copy.id],
                        world_copy=world_copy_copy
                    ))
        resource_paths = new_resource_paths
    
    return max(resource_paths, key=lambda r_p: (r_p.reward_to_date, r_p.resources_to_spend))

