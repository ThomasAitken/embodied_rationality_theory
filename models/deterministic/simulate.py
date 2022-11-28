import logging
from dataclasses import dataclass
from random import normalvariate, randint
from typing import Literal

from dataclasses_json import dataclass_json
from matplotlib import pyplot as plt

from models.deterministic.v1.algorithms import boundedly_optimise_max_investment
from models.deterministic.v1.classes import InvestmentV1

MEAN_PAYOFF_MAP = {
    "low": 100,
    "medium": 200,
    "high": 400,
}
SD_MAP = {
    "low": 1,
    "medium": 20,
    "high": 50,
}


@dataclass_json
@dataclass
class DeterministicEnvironmentSeed:
    resource_abundance: Literal["low", "medium", "high"]
    resource_variance: Literal["low", "medium", "high"]
    reward_abundance: Literal["low", "medium", "high"]
    reward_variance: Literal["low", "medium", "high"]


def generate_investments(num_investments: int, seed: DeterministicEnvironmentSeed) -> list[InvestmentV1]:

    investments: list[InvestmentV1] = []
    for i in range(num_investments):
        investment = InvestmentV1(
            id=str(i + 1),
            name="",
            discharge_threshold=randint(50, MEAN_PAYOFF_MAP[seed.resource_abundance]),
            reward_discharge_amount=int(
                normalvariate(MEAN_PAYOFF_MAP[seed.reward_abundance], SD_MAP[seed.reward_variance])
            ),
            resource_discharge_amount=int(
                normalvariate(MEAN_PAYOFF_MAP[seed.resource_abundance], SD_MAP[seed.resource_variance])
            ),
            capacity_recovery_rate=randint(10, 100),
        )
        logging.debug(f"Generated investment: {investment}")
        investments.append(investment)
    return investments


def simulate(
    agent_starting_resources: int,
    num_investments: int,
    num_timesteps: int,
    seed: str | None,
    plot: bool,
    algorithms: list[Literal["optimal"]],
):
    if seed:
        seed_parsed = DeterministicEnvironmentSeed.from_json(seed)
    else:
        seed_parsed = DeterministicEnvironmentSeed(
            resource_abundance="medium",
            resource_variance="medium",
            reward_abundance="medium",
            reward_variance="medium",
        )
    investments = generate_investments(num_investments, seed_parsed)
    max_path = boundedly_optimise_max_investment(investments, agent_starting_resources, num_timesteps)
    print(
        max_path, max_path.resource_level_at_each_step, max_path.reward_level_at_each_step, max_path.investments_chosen
    )
    # plt.scatter(list(range(num_timesteps + 1)), max_path.resource_level_at_each_step, color="green")
    # plt.scatter(list(range(num_timesteps + 1)), max_path.reward_level_at_each_step, color="blue")
    # plt.show()

    # agent = AgentV1(energetic_resources=agent_starting_resources)
