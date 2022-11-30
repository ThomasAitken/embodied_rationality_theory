from typing import TYPE_CHECKING

from models.deterministic.minimal.classes import InvestmentMinimal

ResourceGainBoundsList = list[dict[str, list[tuple[int, int]]]]


def get_max_possible_resources(investments: list[InvestmentMinimal], resources: int) -> int:
    """
    Finds the highest resource profit that can be made from the given options.
    """
    investments_with_payouts = [(investment, investment.compute_payout(resources)) for investment in investments]
    max_resources = max(map(lambda x: x[1][1], investments_with_payouts))
    return max_resources


# def get_min_resource_gains(investments: list[InvestmentMinimal], resources: int) -> ResourceGainList:
#     min_resource_gains: ResourceGainList = []
#     for investment in investments:
#         invest_id = investment.id
#         possible_payouts = [investment.compute_payout(r) for r in range(0, resources + 1)]
#         min_resource_profit, resources_invested = min(possible_payouts, key=lambda x: x[1])[1:]
#         min_resource_gains.append((invest_id, min_resource_profit, resources_invested))
#     return min_resource_gains


# def get_max_resource_gains(investments: list[InvestmentMinimal], resources: int) -> ResourceGainList:
#     max_resource_gains: ResourceGainList = []
#     for investment in investments:
#         invest_id = investment.id
#         possible_payouts = [investment.compute_payout(r) for r in range(0, resources + 1)]
#         min_resource_profit, resources_invested = max(possible_payouts, key=lambda x: x[1])[1:]
#         max_resource_gains.append((invest_id, min_resource_profit, resources_invested))
#     return max_resource_gains


def get_resource_gain_bounds(investments: list[InvestmentMinimal], resources: int) -> ResourceGainBoundsList:
    resource_gain_bounds: ResourceGainBoundsList = []
    for investment in investments:
        invest_id = investment.id
        possible_payouts = [investment.compute_payout(r) for r in range(0, resources + 1)]
        min_resource_profit, resources_invested_a = min(possible_payouts, key=lambda x: x[1])[1:]
        max_resource_profit, resources_invested_b = max(possible_payouts, key=lambda x: x[1])[1:]
        resource_gain_bounds.append(
            {invest_id: [(min_resource_profit, resources_invested_a), (max_resource_profit, resources_invested_b)]}
        )
    return resource_gain_bounds


# def get_all_possible_worlds_where_given_resource_profit_is_possible(resource_profit: int, list[ResourcePath]) -> list[ResourcePath]:
#     pass


def update_investments(investments: list[InvestmentMinimal]):
    """
    Updates resource capacity of investments at end of time step.
    """
    for investment in investments:
        if investment.resource_capacity + investment.capacity_recovery_rate > investment.discharge_threshold:
            investment.resource_capacity = investment.discharge_threshold
        else:
            investment.resource_capacity += investment.capacity_recovery_rate
