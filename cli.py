import logging
from dataclasses import dataclass
from typing import Literal

from dataclasses_json import dataclass_json
from typed_argparse import Parser, TypedArgs, arg

from models.deterministic.simulate import simulate as simulate_deterministic


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


class Args(TypedArgs):
    class_: Literal["deterministic", "mean_variance", "total"] = arg(
        "class", positional=True, help="The class of model to simulate"
    )
    version: Literal["minimal", "minimal_unselfish"] = arg(
        "version", positional=True, help="The version of the model to simulate"
    )
    agent_starting_resources: int = arg("resources", positional=True, help="The agent's starting resources")
    num_investments: int = arg("investments", positional=True, help="The number of investments in the world")
    num_timesteps: int = arg("timesteps", positional=True, help="The number of timesteps in the simulation")
    verbose: bool = arg("-v", "--verbose", help="Print DEBUG level logs.")
    seed: str = arg("-s", "--seed", default="", help="JSON format seed for investment. See cli.py file to understand.")
    plot: bool = arg("-p", "--plot", default=False, help="Plot the data")
    algorithms: list[Literal["optimal"]] = arg(
        "--algorithms", default=[], help="Plot the result of each of these algorithms"
    )
    # TODO: add args export_data


def simulate(args: Args):
    # Currently assuming _class="deterministic" and version="version"
    #  TODO: extend
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.class_ == "deterministic" and args.version == "minimal":
        simulate_deterministic(
            args.agent_starting_resources,
            args.num_investments,
            args.num_timesteps,
            args.seed,
            args.plot,
            args.algorithms,
        )

    # print(args)


if __name__ == "__main__":
    args = Parser(Args).bind(simulate).run()
