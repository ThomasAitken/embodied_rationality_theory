import argparse
from dataclasses import dataclass
from typing import Literal

from dataclasses_json import dataclass_json

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Action commands", dest="command")
subparsers.required = True

simulate_cmd = subparsers.add_parser(
    "simulate", description="Runs a simulation", formatter_class=argparse.RawTextHelpFormatter
)
simulate_cmd.add_argument(
    "-c",
    "--class",
    dest="class",
    type=str,
    choices={"deterministic", "mean_variance", "total"},
    help="The class of model to simulate",
)
simulate_cmd.add_argument(
    "-s", "--seed", dest="seed", type=str, help="JSON format seed for investment. See cli.py file to understand."
)
simulate_cmd.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Print DEBUG level logs.")


@dataclass
@dataclass_json
class InvestmentSeed:
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


if __name__ == "main":
    pass
