from copy import deepcopy
from typing import TYPE_CHECKING

from .utils import get_resource_gain_bounds, update_investments

if TYPE_CHECKING:
    from .classes import InvestmentV1, OtherEnvironmentalFactors, Payout, ResourcePath

# max_investment, reward_payout, resources_payout, resources_expended
InvestmentSelection = tuple[InvestmentV1, int, int, int]


def adjust_payout_for_environment(environment: OtherEnvironmentalFactors) -> Payout:
    pass


def select_max_investment_by_reward_maximisation(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    """
    Finds the selection that maximises reward. If there is more than one, finds the one that costs the least
    resources.
    """
    investments_with_payouts = [(investment, investment.compute_payout(resources)) for investment in investments]
    max_reward = max(map(lambda x: x[1][0], investments_with_payouts))
    # select investment with max reward and highest net resources
    max_investment = max(
        filter(lambda x: x[1][0] == max_reward, investments_with_payouts),
        key=lambda x: x[1][1] - x[1][2],
    )
    reward_payout, resources_profit, resources_expended = max_investment[1]
    return max_investment[0], reward_payout, resources_profit, resources_expended


def select_max_investment_by_reward_over_resources_profit(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    max_investment = max(
        investments, key=lambda x: x.compute_payout(resources)[0] - x.compute_payout(resources)[2]
    )  # reward - resources_expended
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_fixed_tradeoff_heuristic(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payout(resources)[0] + (1 - reward_bias) * x.compute_payout(resources)[1],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payout(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


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


def boundedly_optimise_max_investment(
    investments: list[InvestmentV1], resources: int, lookahead_steps: int
) -> ResourcePath:
    """
    A dynamic programming approach considering all optimal paths through time for a given level of total resource
    consumption. The search needs to be bounded by lookahead_steps, since it is a potentially unbounded problem.

    Guaranteed to find the path through time that is optimal for maximising reward after n lookahead_steps.
    If more than one, select the path that has the highest available resources.
    """

    max_resources_consumable = resources

    # Stores optimal resource path after timesteps t for total resource consumption r.
    # Each timeslice of the dictionary grows in size as the total possible resource consumption grows.
    optimal_lookup = {
        0: {
            r: ResourcePath(
                resources_spent=0,
                resources_to_spend=resources,
                reward_to_date=0,
                investments_chosen=[],
                world_copy=deepcopy(investments),
            )
        }
        for r in range(resources + 1)
    }
    for t in range(1, lookahead_steps + 1):
        max_resources_consumable_for_next_round = max_resources_consumable
        for r_max_spend in range(1, max_resources_consumable + 1):
            best_selection: InvestmentSelection | None = None
            best_prev_path: ResourcePath | None = None
            # We have the optimal paths for each level of total resource consumption for the previous timestep.
            # We try all possibilities with the best new selection that can be achieved constrained by r_max_spend.
            for r, prev_path in enumerate(list(optimal_lookup[t - 1].values())[:r_max_spend]):
                new_selection = select_max_investment_by_reward_maximisation(investments, r_max_spend - r)
                best_reward = 0 if best_selection is None else best_selection[1]
                new_reward = new_selection[1]
                if (
                    best_prev_path is None
                    or new_reward + prev_path.reward_to_date > best_reward + best_prev_path.reward_to_date
                ):
                    best_selection = new_selection
                    best_prev_path = prev_path

            assert best_selection is not None  # type assertion
            assert best_prev_path is not None  # type assertion
            (
                max_investment,
                reward_payout,
                resources_payout,
                resources_expended,
            ) = best_selection
            new_path = deepcopy(best_prev_path)  # fork history
            new_path.resources_spent += resources_expended
            new_path.resources_to_spend += resources_payout - resources_expended
            new_path.reward_to_date += reward_payout
            max_investment.update_values_post_discharge(resources_expended, reward_payout, resources_payout)
            new_path.investments_chosen.append(max_investment)
            optimal_lookup[t][r_max_spend] = new_path
            update_investments(new_path.world_copy)

            if new_path.resources_spent + new_path.resources_to_spend > max_resources_consumable_for_next_round:
                max_resources_consumable_for_next_round = new_path.resources_spent + new_path.resources_to_spend

        max_resources_consumable = max_resources_consumable_for_next_round

    final_step_values = list(optimal_lookup[lookahead_steps].values())
    top_value = final_step_values[-1]
    return max(
        filter(lambda r: r.reward_to_date == top_value, final_step_values), key=lambda r: r.resources_to_spend
    )  # best value with most available resources


def boundedly_optimise_max_investment(
    investments: list[InvestmentV1], resources: int, lookahead_steps: int
) -> ResourcePath:
    """
    A dynamic programming approach considering all optimal paths through time for a given level of total resource
    consumption. The search needs to be bounded by lookahead_steps, since it is a potentially unbounded problem.

    Guaranteed to find the path through time that is optimal for maximising reward after n lookahead_steps.
    If more than one, select the path that has the highest available resources.
    """

    max_resources_consumable = resources

    min_possible_resource_gain = -resources
    max_possible_resource_gain = get_max_possible_resources(investments, resources)

    # Stores optimal resource path after timesteps t for total resource consumption r.
    # Each timeslice of the dictionary grows in size as the total possible resource consumption grows.
    optimal_lookup = {
        0: [
            ResourcePath(
                resources_spent=0,
                resources_to_spend=resources,
                reward_to_date=0,
                investments_chosen=[],
                world_copy=deepcopy(investments),
                best_next_investments_by_resource_profit=get_best_investments_by_resource_profit(
                    resources, investments, min_possible_resource_gain, max_possible_resource_gain
                ),
            )
            for r in range(min_possible_resource_gain, max_possible_resource_gain + 1)
        ]
    }

    for t in range(1, lookahead_steps + 1):
        next_round_min_resource_gain = min_possible_resource_gain
        next_round_max_resource_gain = max_possible_resource_gain
        for resource_profit in range(min_possible_resource_gain, max_possible_resource_gain + 1):
            best_selection: InvestmentSelection | None = None
            best_prev_path: ResourcePath | None = None

            relevant_paths = optimal_lookup[t - 1]
            best_path = max(relevant_paths, key=lambda p: p.best_next_investments_by_resource_profit[resource_profit])

            # for r, prev_path in enumerate(relevant_paths)[:r_max_profit]:
            #     new_selection = select_max_investment_by_reward_maximisation(prev_path.world_copy, r_max_profit - r)
            #     if new_selection is None:  # no investment delivers the given resource profit on this path of history
            #         continue
            #     best_reward = 0 if best_selection is None else best_selection[1]
            #     new_reward = new_selection[1]
            #     if (
            #         best_prev_path is None
            #         or new_reward + prev_path.reward_to_date > best_reward + best_prev_path.reward_to_date
            #     ):
            #         best_selection = new_selection
            #         best_prev_path = prev_path

            assert best_selection is not None  # type assertion
            assert best_prev_path is not None  # type assertion
            (
                max_investment,
                reward_payout,
                resources_profit,
                resources_expended,
            ) = best_selection
            resources_to_spend = best_prev_path.resources_to_spend + resources_profit
            if resources_to_spend <= 0:
                optimal_lookup[t][r_max_profit] = None  # agent is dead

            new_path = deepcopy(best_prev_path)  # fork history
            new_path.resources_spent += resources_expended
            new_path.resources_to_spend += resources_profit
            new_path.reward_to_date += reward_payout
            max_investment.update_values_post_discharge(resources_expended, reward_payout, resources_profit)
            new_path.investments_chosen.append(max_investment)
            update_investments(new_path.world_copy)
            min_possible_resource_gain = min(
                investment.get_min_resource_profit(new_path.resources_to_spend) for investment in new_path.world_copy
            )
            max_possible_resource_gain = max(
                investment.get_max_resource_profit(new_path.resources_to_spend) for investment in new_path.world_copy
            )
            new_path.best_next_investments_by_resource_profit = get_best_investments_by_resource_profit(
                new_path.world_copy, new_path.resources_to_spend, min_possible_resource_gain, max_possible_resource_gain
            )
            optimal_lookup[t][r_max_profit] = new_path

            if max_possible_resource_gain > next_round_max_resource_gain:
                next_round_max_resource_gain = max_possible_resource_gain
            if min_possible_resource_gain < next_round_min_resource_gain:
                next_round_min_resource_gain = min_possible_resource_gain

        min_possible_resource_gain = next_round_min_resource_gain
        max_possible_resource_gain = next_round_max_resource_gain

    final_step_values = list(optimal_lookup[lookahead_steps].values())
    top_value = final_step_values[-1]
    return max(
        filter(lambda r: r.reward_to_date == top_value, final_step_values), key=lambda r: r.resources_to_spend
    )  # best value with most available resources
