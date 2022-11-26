import logging
from dataclasses import dataclass
from random import normalvariate, randint
from typing import Literal

from dataclasses_json import dataclass_json
from matplotlib import pyplot as plt
from v1.algorithms import boundedly_optimise_max_investment
from v1.classes import InvestmentV1

logger = logging.getLogger(__name__)

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


def generate_investments(num_investments: int, seed: str | None) -> list[InvestmentV1]:
    if seed is not None:
        seed = DeterministicEnvironmentSeed.from_json(seed)
    else:
        seed = DeterministicEnvironmentSeed(
            resource_abundance="medium",
            resource_variance="medium",
            reward_abundance="medium",
            reward_variance="medium",
        )
    investments: list[InvestmentV1] = []
    for i in range(num_investments):
        investment = InvestmentV1(
            id=str(i + 1),
            name="",
            discharge_threshold=randint(50, 1000),
            reward_discharge_amount=normalvariate(MEAN_PAYOFF_MAP[seed.reward_abundance], SD_MAP[seed.reward_variance]),
            resource_discharge_amount=normalvariate(
                MEAN_PAYOFF_MAP[seed.resource_abundance], SD_MAP[seed.resource_variance]
            ),
            capacity_recovery_rate=randint(10, 200),
        )
        logger.debug(f"Generated investment: {investment}")
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
    investments = generate_investments(num_investments, seed)
    max_path = boundedly_optimise_max_investment(investments, agent_starting_resources)
    plt.scatter(list(range(num_timesteps + 1)), max_path.resource_level_at_each_step, color="green")
    plt.scatter(list(range(num_timesteps + 1)), max_path.reward_level_at_each_step, color="blue")
    plt.show()

    # agent = AgentV1(energetic_resources=agent_starting_resources)
