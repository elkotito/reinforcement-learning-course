import importlib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from base import PolicyIterationBase, ValueIterationBase
from frozen_lake import FrozenLakeModel
from visualization import plot_policy


class Algorithm(str, Enum):
    policy_iteration = "policy_iteration"
    value_iteration = "value_iteration"


class Implementation(str, Enum):
    solution = "solution"
    student = "student"


class MapName(str, Enum):
    small = "4x4"
    large = "8x8"


def main(
    algorithm: Annotated[Algorithm, typer.Option()] = Algorithm.policy_iteration,
    implementation: Annotated[
        Implementation,
        typer.Option(help="Use solution.py or the student's algorithms.py."),
    ] = Implementation.solution,
    map_name: Annotated[MapName, typer.Option()] = MapName.large,
    slippery: Annotated[bool, typer.Option()] = False,
    gamma: Annotated[float, typer.Option()] = 0.9,
    seed: Annotated[int, typer.Option()] = 0,
    output: Annotated[Path | None, typer.Option()] = None,
    show: Annotated[bool, typer.Option()] = False,
) -> None:
    env = FrozenLakeModel(
        map_name=map_name.value,
        is_slippery=slippery,
        seed=seed,
    )
    solver = make_solver(
        implementation=implementation.value,
        algorithm=algorithm.value,
        gamma=gamma,
    )
    policy = solver.solve(env)

    output_path = output or Path(
        f"lesson1/outputs/{implementation.value}_{algorithm.value}_{map_name.value}.png"
    )
    plot_policy(env, policy, output_path=output_path, show=show)
    print(f"Saved visualization to {output_path.resolve()}")


def make_solver(
    *,
    implementation: str,
    algorithm: str,
    gamma: float,
) -> PolicyIterationBase | ValueIterationBase:
    module_name = "solution" if implementation == "solution" else "algorithms"
    module = importlib.import_module(module_name)

    if algorithm == "value_iteration":
        return module.ValueIteration(gamma=gamma)  # pyright: ignore[reportAny]
    return module.PolicyIteration(gamma=gamma)  # pyright: ignore[reportAny]


if __name__ == "__main__":
    typer.run(main)
