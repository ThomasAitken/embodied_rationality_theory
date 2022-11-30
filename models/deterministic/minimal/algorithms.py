import logging
from copy import copy
from typing import Optional

from models.deterministic.utils import update_investments

from .classes import InvestmentMinimal, Payout, ResourcePath

logger = logging.getLogger(__name__)

# max_investment, reward_payout, resources_payout, resources_expended
InvestmentSelection = tuple[InvestmentMinimal, int, int, int]


def select_max_investment_by_reward_maximisation(
    investments: list[InvestmentMinimal], resources: int
) -> tuple[InvestmentMinimal, Payout]:
    """
    Finds the selection that maximises reward. If there is more than one, finds the one with the highest resource profit.
    """
    investments_with_payouts = [(investment, investment.compute_payout(resources)) for investment in investments]
    max_reward = max(map(lambda x: x[1]["reward"], investments_with_payouts))
    # select investment with max reward and highest net resources
    max_investment = max(
        filter(lambda x: x[1]["reward"] == max_reward, investments_with_payouts),
        key=lambda x: x[1]["resource_profit"],
    )
    investment, payout = max_investment
    return investment, payout


def select_max_investment_by_reward_over_resources_profit(
    investments: list[InvestmentMinimal], resources: int
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: x.compute_payout(resources)["reward"] - x.compute_payout(resources)["resource_profit"],
    )  # reward - resources_expended
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_fixed_tradeoff_heuristic(
    investments: list[InvestmentMinimal], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)["reward"]
        + (1 - reward_bias) * x.compute_payout(resources)["resource_profit"],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def compute_min_reward_bound_by_resource_maxing(
    investments: list[InvestmentMinimal], resources: int
) -> tuple[int, int]:
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


def compute_min_resource_bound_by_reward_maxing(
    investments: list[InvestmentMinimal], resources: int
) -> tuple[int, int]:
    """
    Returns the amount of resources obtained by choosing the investment that maximises reward.
    This serves as a lower bound requirement for a given investment selection decision. If the reward discharge amount
    for a given investment is below this, it is provably suboptimal by Theorem 1.3.
    """
    max_investment = max(investments, key=lambda x: x.reward_discharge_amount)
    payout = max_investment.compute_payout(resources)
    return payout["resource_profit"], payout["reward"]


# def get_best_investments_by_resource_profit(
#     investments: list[InvestmentMinimal], agent_resources: int, min_profit, max_profit
# ) -> dict[int, tuple[InvestmentMinimal, int]]:
#     """
#     Returns a map of investments that get the highest reward by each possible level of resource profit.
#     """
#     resource_to_selection_map = {}
#     for r in range(min_profit, max_profit + 1):
#         investment_and_payouts_given_resource_constraints = [
#             (investment, investment.get_payout_given_resource_parameters(agent_resources, r)[0])
#             for investment in investments
#         ]
#         investment_and_payouts_given_resource_constraints = filter(
#             lambda i: i["resource_profit"] is not None,
#             investment_and_payouts_given_resource_constraints,
#         )
#         resource_to_selection_map[r] = max(investment_and_payouts_given_resource_constraints, lambda p: p[1])
#     return resource_to_selection_map


def get_max_possible_resource_sum(investments: list[InvestmentMinimal], resources_now: int) -> int:
    return resources_now + sum(i.resource_discharge_amount - i.resources_until_payout for i in investments)


def is_resource_level_unreachable(
    investments: list[InvestmentMinimal], resources_now: int, resource_level_to_reach: int, timesteps: int
) -> bool:
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
    resource_path: ResourcePath, investment: InvestmentMinimal, timesteps_remaining: int
) -> bool:
    return is_resource_level_unreachable(
        resource_path.world_copy, resource_path.resources_to_spend, investment.discharge_threshold, timesteps_remaining
    )


# using Optional as "| None" syntax bugs out here https://github.com/python/mypy/issues/11098
DischargeResult = Optional[tuple[int, int]]
LatentResult = Optional[tuple[int, int, int]]


def update_best_result_so_far(
    max_choice: Payout,
    investment: InvestmentMinimal,
    best_discharge_result: DischargeResult,
    best_latent_result: LatentResult,
    is_last_timestep: bool,
) -> tuple[DischargeResult, LatentResult] | None:
    """
    Takes the highest-return choice of resource-expenditure for a given investment and determines if there is another choice for
    some other investment that dominates it in the current resource path.
    If max_choice is dominated, return None.
    Otherwise return updated values for best_discharge_result and best_latent_result.

    If we are on the last timestep, we ignore resources
    """
    if max_choice["discharge_reached"]:
        if best_discharge_result is None:
            best_discharge_result = (max_choice["reward"], max_choice["resource_profit"])
            return best_discharge_result, best_latent_result

        # existing best choice dominates
        if max_choice["reward"] < best_discharge_result[0] and (
            max_choice["resource_profit"] < best_discharge_result[1] or is_last_timestep
        ):
            return None

        # new choice is new best
        if max_choice["reward"] >= best_discharge_result[0] and (
            max_choice["resource_profit"] >= best_discharge_result[1] or is_last_timestep
        ):
            best_discharge_result = (max_choice["reward"], max_choice["resource_profit"])
    else:
        if best_discharge_result is not None:
            # best discharge result is better than this latent result
            if investment.reward_discharge_amount < best_discharge_result[0] and (
                investment.resource_discharge_amount < best_discharge_result[1] or is_last_timestep
            ):
                return None

        resources_until_payout_post_injection = investment.get_resources_until_payout_post_injection(
            max_choice["resources_spent"]
        )
        if best_latent_result is None:
            best_latent_result = (
                investment.reward_discharge_amount,
                investment.resource_discharge_amount,
                resources_until_payout_post_injection,
            )
            return best_discharge_result, best_latent_result

        # existing best choice dominates
        if investment.reward_discharge_amount < best_latent_result[0] and (
            (
                investment.resource_discharge_amount < best_latent_result[1]
                and resources_until_payout_post_injection > best_latent_result[2]
            )
            or is_last_timestep
        ):
            return None

        # new choice is new best
        if investment.reward_discharge_amount >= best_latent_result[0] and (
            (
                investment.resource_discharge_amount >= best_latent_result[1]
                and resources_until_payout_post_injection <= best_latent_result[2]
            )
            or is_last_timestep
        ):
            best_latent_result = (
                investment.reward_discharge_amount,
                investment.resource_discharge_amount,
                resources_until_payout_post_injection,
            )

    return best_discharge_result, best_latent_result


def get_nondominated_consumption_choices(investment: InvestmentMinimal, resources: int):
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
        return [investment.compute_payout(max_resources_to_spend)]
    return [investment.compute_payout(r) for r in range(max_resources_to_spend)]


def boundedly_optimise_max_investment(
    investments: set[InvestmentMinimal], resources: int, lookahead_steps: int
) -> ResourcePath:
    """
    A branch-and-bound programming approach considering all optimal paths through time for a given level of total resource
    consumption. The search needs to be bounded by lookahead_steps, since it is a potentially unbounded problem.

    Guaranteed to find the path through time that is optimal for maximising reward after n lookahead_steps.
    If more than one, select the path that has the highest available resources.
    """

    resource_paths: list[ResourcePath] = []
    best_discharge_result = None
    best_latent_result = None
    for investment in investments:
        nondominated_consumption_choices = get_nondominated_consumption_choices(investment, resources)
        max_choice = max(nondominated_consumption_choices, key=lambda c: (c["reward"], c["resource_profit"]))
        best_results = update_best_result_so_far(
            max_choice, investment, best_discharge_result, best_latent_result, lookahead_steps == 1
        )
        if best_results is None:
            # logging.debug(f"Investment {investment.id} dominated on timestep 0")
            continue
        best_discharge_result, best_latent_result = best_results

        for choice in nondominated_consumption_choices:
            investment_copy = copy(investment)
            investment_copy.update_values_post_investment(
                choice["discharge_reached"], choice["resources_spent"], choice["reward"], choice["resource_profit"]
            )
            resources_to_spend = resources + choice["resource_profit"]
            reward_to_date = choice["reward"]
            update_investments([investment_copy])
            resource_paths.append(
                ResourcePath(
                    resources_spent=choice["resources_spent"],
                    resources_to_spend=resources_to_spend,
                    reward_to_date=reward_to_date,
                    investments_chosen=[investment_copy.id],
                    world_copy=[investment_copy]
                    + [copy(invest) for invest in investments if invest.id != investment_copy.id],
                    resource_level_at_each_step=[resources_to_spend],
                    reward_level_at_each_step=[reward_to_date],
                )
            )
    print(len(resource_paths))
    # logging.debug(resource_paths[-1])

    for t in range(1, lookahead_steps):
        # timesteps to go including the current timestep
        timesteps_remaining = lookahead_steps - t
        new_resource_paths = []
        print(f"ITERATION {t}")
        for r in resource_paths:
            best_discharge_result = None
            best_latent_result = None
            if r.resources_to_spend == 0:  # agent is dead
                logging.debug(f"AGENT IS DEAD")
                continue

            reward_max_resource_take, reward_max_reward_take = compute_min_resource_bound_by_reward_maxing(
                r.world_copy, r.resources_to_spend
            )
            resource_max_reward_take, resource_max_resource_take = compute_min_reward_bound_by_resource_maxing(
                r.world_copy, r.resources_to_spend
            )
            for investment in sorted(
                r.world_copy, key=lambda i: (i.reward_discharge_amount, i.resource_discharge_amount)
            ):
                # determine if not enough time/resources to achieve discharge for given investment
                if is_investment_discharge_unreachable(r, investment, timesteps_remaining - 1):
                    logging.debug(f"Investment {investment.id} pruned because discharge is unreachable on timestep {t}")
                    continue

                # fails reward lower bound
                if (
                    investment.reward_discharge_amount < resource_max_reward_take
                    and investment.resource_discharge_amount <= resource_max_resource_take
                ):
                    logging.debug(f"Investment {investment.id} pruned for failing reward bound on timestep {t}")
                    continue
                # fails resource lower bound
                if (
                    investment.resource_discharge_amount < reward_max_resource_take
                    and investment.reward_discharge_amount <= reward_max_reward_take
                ):
                    logging.debug(f"Investment {investment.id} pruned for failing resource bound on timestep {t}")
                    continue

                nondominated_consumption_choices = get_nondominated_consumption_choices(investment, resources)
                max_choice = max(nondominated_consumption_choices, key=lambda c: (c["reward"], c["resource_profit"]))
                best_results = update_best_result_so_far(
                    max_choice, investment, best_discharge_result, best_latent_result, t == lookahead_steps - 1
                )
                if best_results is None:
                    # logging.debug(f"Investment {investment.id} dominated on timestep {t}")
                    continue
                best_discharge_result, best_latent_result = best_results

                for choice in nondominated_consumption_choices:
                    investment_copy = copy(investment)
                    investment_copy.update_values_post_investment(
                        choice["discharge_reached"],
                        choice["resources_spent"],
                        choice["reward"],
                        choice["resource_profit"],
                    )

                    world_copy_copy = [
                        investment_copy if invest.id == investment.id else copy(invest) for invest in r.world_copy
                    ]
                    update_investments(world_copy_copy)

                    resources_to_spend = r.resources_to_spend + choice["resource_profit"]
                    reward_to_date = r.reward_to_date + choice["reward"]
                    new_resource_paths.append(
                        ResourcePath(
                            resources_spent=r.resources_spent + choice["resources_spent"],
                            resources_to_spend=resources_to_spend,
                            reward_to_date=reward_to_date,
                            investments_chosen=r.investments_chosen + [investment_copy.id],
                            world_copy=world_copy_copy,
                            resource_level_at_each_step=r.resource_level_at_each_step + [resources_to_spend],
                            reward_level_at_each_step=r.reward_level_at_each_step + [reward_to_date],
                        )
                    )
        resource_paths = new_resource_paths
        print(len(resource_paths))

    return max(resource_paths, key=lambda r_p: (r_p.reward_to_date, r_p.resources_to_spend))
