from copy import copy
from typing import TYPE_CHECKING

from models.deterministic.utils import update_investments

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
        investments,
        key=lambda x: x.compute_payout(resources)["reward"] - x.compute_payout(resources)["resource_profit"],
    )  # reward - resources_expended
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_fixed_tradeoff_heuristic(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)["reward"]
        + (1 - reward_bias) * x.compute_payout(resources)["resource_profit"],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_reward_minus_resources(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)["reward"]
        + (1 - reward_bias) * x.compute_payout(resources)["resource_profit"],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


# def compute_path_of_immediate_reward_maximisation(
#     resource_path: ResourcePath, timesteps_remaining: int
# ) -> ResourcePath:
#     for _ in range(timesteps_remaining):
#         chosen_investment, payout = select_max_investment_by_reward_maximisation(resource_path.world_copy)
#         chosen_investment_copy = copy(chosen_investment)
#         chosen_investment_copy.update_values_post_discharge(
#             payout["resources_spent"], payout["reward"], payout["resource_profit"]
#         )


def compute_min_reward_bound_by_resource_maxing(investments: list[InvestmentV1], resources: int) -> tuple[int, int]:
    """
    Returns the amount of reward obtained by choosing the investment that maximises resource profit.
    This serves as a lower bound requirement for a given investment selection decision. If the resource discharge amount
    for a given investment is below this, it is provably suboptimal by Theorem 1.2.
    """
    max_investment = max(
        investments,
        key=lambda x: x.compute_payout(resources)["resource_profit"],
    )
    payout = max_investment.compute_payout(resources)
    return payout["reward"], payout["resource_profit"]


def compute_min_resource_bound_by_reward_maxing(investments: list[InvestmentV1], resources: int) -> tuple[int, int]:
    """
    Returns the amount of resources obtained by choosing the investment that maximises reward.
    This serves as a lower bound requirement for a given investment selection decision. If the reward discharge amount
    for a given investment is below this, it is provably suboptimal by Theorem 1.3.
    """
    max_investment = max(investments, key=lambda x: x.reward_discharge_amount)
    payout = max_investment.compute_payout(resources)
    return payout["resource_profit"], payout["reward"]


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


def get_max_possible_resource_sum(investments: list[InvestmentV1], resources_now: int) -> int:
    return resources_now + sum(i.resource_discharge_amount - i.resources_until_payout for i in investments)


def is_resource_level_unreachable(
    investments: list[InvestmentV1], resources_now: int, resource_level_to_reach: int, timesteps: int
) -> list[InvestmentV1]:
    """
    This is useful for determining whether a given investment with discharge_threshold = resource_level_to_reach can
    possibly be exploited before the end of the game.
    A value of False doesn't always mean provably reachable since this is too expensive to compute (but True is
    always correct).
    """
    if resource_level_to_reach <= resources_now:
        return False

    # only care about those investments that will recharge in time
    temporally_available_investments = list(
        filter(
            lambda i: i.resource_capacity + i.capacity_recovery_rate * timesteps >= i.discharge_threshold, investments
        )
    )
    if len(temporally_available_investments) < timesteps:
        return True
    investments_sorted_by_cheapness = sorted(investments, key=lambda i: i.discharge_threshold)
    investments_sorted_by_resource_profit = sorted(
        investments, key=lambda i: i.resource_discharge_amount - i.resources_until_payout, reverse=True
    )
    max_possible_sum = get_max_possible_resource_sum(investments_sorted_by_resource_profit[:timesteps], resources_now)
    if max_possible_sum < resource_level_to_reach:
        return True

    start_idx = 0
    cheapest_viable_investment_set = investments_sorted_by_cheapness[:timesteps]
    while get_max_possible_resource_sum(cheapest_viable_investment_set, resources_now) < resource_level_to_reach:
        cheapest_viable_investment_set = investments_sorted_by_cheapness[start_idx : timesteps + start_idx]
        start_idx += 1
    if (
        cheapest_viable_investment_set[0].resources_until_payout > resources_now
    ):  # can't even get the cheapest investment in one step
        return True
    return False


def is_investment_discharge_unreachable(
    resource_path: ResourcePath, investment: InvestmentV1, timesteps_remaining: int
) -> bool:
    return is_resource_level_unreachable(
        resource_path.world_copy, resource_path.resources_to_spend, investment.discharge_threshold, timesteps_remaining
    )


def get_nondominated_consumption_choices(investment: InvestmentV1, resources: int):
    """
    Assume: we can't pay over the discharge threshold for a given investment.
    If the investment cannot deliver a resource profit on the current turn, include every choice up to the discharge
    threshold.
    If the investment can deliver a resource profit on the current turn, include only the choice that matches the
    discharge threshold. Any other choice is provably inferior by Theorem 1.1.
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
            investment_copy.update_values_post_discharge(
                choice["resources_spent"], choice["reward"], choice["resource_profit"]
            )
            resource_paths.append(
                ResourcePath(
                    resources_spent=choice["resource_spent"],
                    resources_to_spend=resources + choice["resource_profit"],
                    reward_to_date=choice["reward"],
                    investments_chosen=[investment_copy.id],
                    world_copy=[investment_copy],
                )
            )

    for t in range(1, lookahead_steps):
        timesteps_remaining = lookahead_steps - t
        new_resource_paths = []
        for r in resource_paths:
            if r.resources_to_spend == 0:  # agent is dead
                continue

            reward_max_resource_take, reward_max_reward_take = compute_min_resource_bound_by_reward_maxing(
                r.world_copy, r.resources_to_spend
            )
            resource_max_reward_take, resource_max_resource_take = compute_min_reward_bound_by_resource_maxing(
                r.world_copy, r.resources_to_spend
            )
            for investment in r.world_copy:
                # determine if not enough time/resources to achieve discharge for given investment
                if is_investment_discharge_unreachable(r, investment, timesteps_remaining - 1):
                    continue

                # fails reward lower bound
                if (
                    investment.reward_discharge_amount < resource_max_reward_take
                    and investment.resource_discharge_amount <= resource_max_resource_take
                ):
                    continue
                # fails resource lower bound
                if (
                    investment.resource_discharge_amount < reward_max_resource_take
                    and investment.reward_discharge_amount <= reward_max_reward_take
                ):
                    continue

                investment_copy = copy(investment)
                for choice in get_nondominated_consumption_choices(investment_copy, r.resources_to_spend):

                    investment_copy.update_values_post_discharge(
                        choice["resources_spent"], choice["reward"], choice["resource_profit"]
                    )

                    world_copy_copy = [
                        investment_copy if invest.id == investment.id else copy(invest) for invest in r.world_copy
                    ]
                    update_investments(world_copy_copy)

                    new_resource_paths.append(
                        ResourcePath(
                            resources_spent=r.resources_spent + choice["resources_spent"],
                            resources_to_spend=r.resources_to_spend + choice["resource_profit"],
                            reward_to_date=r.reward_to_date + choice["reward"],
                            investments_chosen=r.investments_chosen + [investment_copy.id],
                            world_copy=world_copy_copy,
                        )
                    )
        resource_paths = new_resource_paths

    return max(resource_paths, key=lambda r_p: (r_p.reward_to_date, r_p.resources_to_spend))
