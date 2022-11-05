from cli import InvestmentSeed

from .algorithms import select_max_investment_by_reward_maximisation
from .classes import AgentV1, InvestmentV1, OtherEnvironmentalFactors


def generate_investments(seed: list[InvestmentSeed]) -> list[InvestmentV1]:
    """
    Generates the agent's environment.
    """
    pass


def main():
    perceived_investments: list[InvestmentV1] = generate_investments()

    agent = AgentV1()

    print("SIMPLE DETERMINISTIC OPTIMISATION\n")
    for timestep in range(agent.expected_lifespan):
        (
            max_investment,
            reward_payout,
            resources_payout,
            resources_expended,
        ) = select_max_investment_by_reward_maximisation(perceived_investments, agent.resources)
        max_investment.update_values_post_discharge(resources_expended, reward_payout, resources_payout)
        agent.energetic_resources -= resources_expended

        print(f"Timestep {timestep+1}: agent spends {resources_expended} on ")
        update_investments(perceived_investments)

        agent.energetic_resources -= OtherEnvironmentalFactors.energetic_depletion_rate
