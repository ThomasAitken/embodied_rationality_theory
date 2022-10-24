from typing import TYPE_CHECKING, TypedDict
from copy import deepcopy

if TYPE_CHECKING:
    from .model_1 import InvestmentV1

# max_investment, reward_payout, resources_payout, resources_expended
InvestmentSelection = tuple[InvestmentV1, int, int, int]


def select_max_investment_by_reward_maximisation(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    """
    Finds the selection that maximises reward. If there is more than one, finds the one that costs the least
    resources.
    """
    investments_with_payouts = [
        (investment, investment.compute_payouts_simple(resources)) for investment in investments
    ]
    max_reward = max(map(lambda x: x[1][0], investments_with_payouts))
    # select investment with max reward and highest net resources
    max_investment = max(
        filter(lambda x: x[1][0] == max_reward, investments_with_payouts),
        key=lambda x: x[1][1] - x[1][2],
    )
    reward_payout, resources_payout, resources_expended = max_investment[1]
    return max_investment[0], reward_payout, resources_payout, resources_expended


def select_max_investment_by_reward_over_resources_profit(
    investments: list[InvestmentV1], resources: int
) -> InvestmentSelection:
    max_investment = max(
        investments, key=lambda x: x.compute_payouts_simple(resources)[0] - x.compute_payouts_simple(resources)[2]
    )  # reward - resources_expended
    reward_payout, resources_payout, resources_expended = max_investment.compute_payouts_simple(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


def select_max_investment_by_fixed_tradeoff_heuristic(
    investments: list[InvestmentV1], resources: int, reward_bias: float
) -> InvestmentSelection:
    max_investment = max(
        investments,
        key=lambda x: reward_bias * x.compute_payouts_simple(resources)[0]
        + (1 - reward_bias) * x.compute_payouts_simple(resources)[1],
    )  # maximise weighted average of reward/resources with weights given by "reward_bias"
    reward_payout, resources_payout, resources_expended = max_investment.compute_payouts_simple(resources)
    return max_investment, reward_payout, resources_payout, resources_expended


class ResourcePath(TypedDict):
    resources_spent: int
    resources_to_spend: int
    reward_to_date: int
    investments_chosen: list[InvestmentV1]
    world_copy: list[InvestmentV1]


def update_investments(investments: list[InvestmentV1]):
    """
    Updates resource capacity of investments at end of time step.
    """
    for investment in investments:
        investment.resource_capacity += int(investment.capacity_recovery_rate)


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
                    or new_reward + prev_path["reward_to_date"] > best_reward + best_prev_path["reward_to_date"]
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
            new_path["resources_spent"] += resources_expended
            new_path["resources_to_spend"] += resources_payout - resources_expended
            new_path["reward_to_date"] += reward_payout
            max_investment.update_values_post_discharge(resources_expended, reward_payout, resources_payout)
            new_path["investments_chosen"].append(max_investment)
            optimal_lookup[t][r_max_spend] = new_path
            update_investments(new_path["world_copy"])

            if new_path["resources_spent"] + new_path["resources_to_spend"] > max_resources_consumable_for_next_round:
                max_resources_consumable_for_next_round = new_path["resources_spent"] + new_path["resources_to_spend"]

        max_resources_consumable = max_resources_consumable_for_next_round

    return list(optimal_lookup[lookahead_steps].values())[-1]
