from typing import TYPE_CHECKING
from typed_argparse import TypedArgs

if TYPE_CHECKING:
    from argparse import ArgumentParser

# Step 1: Add an argument type.
class SimulateArgs(TypedArgs):
    class_: str
    version: str
    files: List[str]



def simulate(parser: ArgumentParser):