import random
from dataclasses import dataclass
from typing import Literal

from dataclasses_json import dataclass_json
from typed_argparse import Parser, TypedArgs, arg


@dataclass
@dataclass_json
class StochasticInvestmentSeed:
    reward_function_class: Literal["constant", "linear", "exponential", "logarithmic", "logistic"]
    resources_function_class: Literal["constant", "linear", "exponential", "logarithmic", "logistic"]
    reward_period: Literal["hourly", "daily", "yearly", "decanually"]
    reward_to_resources_payout_ratio: Literal["high", "low", "inverse_high", "inverse_low"]
    resource_capacity: Literal["low", "medium", "high"]
    capacity_recovery_rate: Literal["slow", "medium", "fast"]
    reward_sampling_variance: Literal["low", "medium", "high"]
    reward_sampling_skew: Literal["low", "medium", "high"]
    resources_sampling_skew: Literal["low", "medium", "high"]
    reward_sampling_kurtosis: Literal["low", "medium", "high"]
    resources_sampling_variance: Literal["low", "medium", "high"]
    resources_sampling_kurtosis: Literal["low", "medium", "high"]


LOW_ABUNDANCE_MEAN = 100
MEDIUM_ABUNDANCE_MEAN = 200
HIGH_ABUNDANCE_MEAN = 400

LOW_SD = 1
MEDIUM_SD = 10
HIGH_SD = 20

LOW_STARTING_RESOURCES = 333
MEDIUM_STARTING_RESOURCES = 666
HIGH_STARTING_RESOURCES = 999


@dataclass_json
@dataclass
class DeterministicEnvironmentSeed:
    resource_abundance: Literal["low", "medium", "high"]
    resource_variance: Literal["low", "medium", "high"]
    reward_abundance: Literal["low", "medium", "high"]
    reward_variance: Literal["low", "medium", "high"]
    num_investments: Literal["low", "medium", "high"]
    agent_starting_resources: Literal["low", "medium", "high"]


class Args(TypedArgs):
    verbose: bool = arg("-v", "--verbose", help="Print DEBUG level logs.")
    class_: Literal["deterministic", "mean_variance", "total"] = arg(
        "-c",
        "--class",
        help="The class of model to simulate",
    )
    version: Literal["v1", "v2"] = arg("-w", "--version", help="The version of the model to simulate")
    seed: str = arg("-s", "--seed", help="JSON format seed for investment. See cli.py file to understand.")
    # TODO: add args plot_data, export_data, compare_algorithms


def simulate(args: Args):
    # Currently assuming _class="deterministic" and version="version"
    #  TODO: extend
    if args.seed is not None:
        seed = DeterministicEnvironmentSeed.from_json(args.seed)
    else:
        seed = DeterministicEnvironmentSeed(
            resource_abundance="medium",
            resource_variance="medium",
            reward_abundance="medium",
            reward_variance="medium",
            num_investments="medium",
            agent_starting_resources="medium",
        )
    # random.normalvariate

    # print(args)


if __name__ == "__main__":
    args = Parser(Args).bind(simulate).run()
    # if args.command == "simulate":
    #     simulate_args = simulate_cmd.parse_args()
    #     simulate(simulate_args)
