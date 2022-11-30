from models.deterministic.minimal_unselfish.classes import (
    InvestmentMinimalUnselfish,
    ResourcePath,
)


def boundedly_optimise_max_investment(
    investments: set[InvestmentMinimalUnselfish], resources: int, lookahead_steps: int
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
