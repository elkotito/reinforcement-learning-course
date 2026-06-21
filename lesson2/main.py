import importlib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from base import (
    ExpectedSarsaBase,
    ExplorationConfig,
    MonteCarloControlBase,
    QLearningBase,
    SarsaBase,
)
from cliff_walking import CliffWalkingModel
from visualization import plot_greedy_path, plot_learning_curve


class Algorithm(str, Enum):
    monte_carlo_control = "monte_carlo_control"
    sarsa = "sarsa"
    q_learning = "q_learning"
    expected_sarsa = "expected_sarsa"


class Implementation(str, Enum):
    solution = "solution"
    student = "student"


ALGORITHM_DISPLAY_NAMES: dict[str, str] = {
    Algorithm.monte_carlo_control.value: "Monte Carlo Control",
    Algorithm.sarsa.value: "SARSA",
    Algorithm.q_learning.value: "Q-learning",
    Algorithm.expected_sarsa.value: "Expected SARSA",
}


def main(
    algorithm: Annotated[Algorithm, typer.Option()] = Algorithm.monte_carlo_control,
    implementation: Annotated[
        Implementation,
        typer.Option(help="Use solution.py or the student's algorithms.py."),
    ] = Implementation.solution,
    gamma: Annotated[float, typer.Option()] = 1.0,
    epsilon: Annotated[
        float,
        typer.Option(
            help=(
                "Initial epsilon-greedy exploration probability for sampled control."
            ),
        ),
    ] = 0.1,
    epsilon_end: Annotated[
        float,
        typer.Option(help="Final epsilon-greedy exploration probability."),
    ] = 0.1,
    epsilon_decay_episodes: Annotated[
        int,
        typer.Option(help="Number of episodes used to decay epsilon."),
    ] = 10_000,
    alpha: Annotated[
        float,
        typer.Option(help="Step size for SARSA, Q-learning, and Expected SARSA."),
    ] = 0.1,
    episodes: Annotated[int, typer.Option()] = 10_000,
    max_steps_per_episode: Annotated[int, typer.Option()] = 1_000,
    evaluation_interval: Annotated[
        int,
        typer.Option(help="Evaluate the greedy policy every N training episodes."),
    ] = 100,
    smoothing_window: Annotated[
        int,
        typer.Option(help="Moving-average window for the learning curve."),
    ] = 100,
    seed: Annotated[int, typer.Option()] = 0,
    output: Annotated[Path | None, typer.Option()] = None,
    learning_output: Annotated[Path | None, typer.Option()] = None,
    show: Annotated[bool, typer.Option()] = False,
    progress: Annotated[
        bool,
        typer.Option(help="Show a tqdm progress bar while training."),
    ] = True,
) -> None:
    env = CliffWalkingModel(seed=seed)
    solver = make_solver(
        implementation=implementation.value,
        algorithm=algorithm.value,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        alpha=alpha,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        evaluation_interval=evaluation_interval,
        seed=seed,
        progress=progress,
    )
    solver.solve(env)
    greedy_policy = solver.greedy_policy(env)
    final_greedy_return = solver.greedy_policy_return(env, seed=seed)

    output_path = output or Path(
        f"lesson2/outputs/{implementation.value}_{algorithm.value}_deterministic_path.png"
    )
    plot_greedy_path(
        env,
        greedy_policy,
        output_path=output_path,
        show=show,
        seed=seed,
    )
    print(f"Saved greedy path to {output_path.resolve()}")

    learning_output_path = learning_output or Path(
        f"lesson2/outputs/{implementation.value}_{algorithm.value}_deterministic_learning.png"
    )
    algorithm_title = ALGORITHM_DISPLAY_NAMES[algorithm.value]
    optimal_return = -13.0
    title = (
        f"{algorithm_title} Learning Curve; "
        f"final greedy return={final_greedy_return:.0f}; "
        f"optimal={optimal_return:.0f}"
    )
    plot_learning_curve(
        solver.episode_returns,
        output_path=learning_output_path,
        show=show,
        title=title,
        smoothing_window=smoothing_window,
        optimal_return=optimal_return,
    )
    print(f"Saved learning curve to {learning_output_path.resolve()}")

    if solver.episode_returns:
        recent_returns = solver.episode_returns[-100:]
        average_return = sum(recent_returns) / len(recent_returns)
        print(
            f"Average return over last {len(recent_returns)} episodes: {average_return:.2f}"
        )
    print(f"Final greedy return: {final_greedy_return:.2f}")


def make_solver(
    *,
    implementation: str,
    algorithm: str,
    gamma: float,
    epsilon: float,
    epsilon_end: float,
    epsilon_decay_episodes: int,
    alpha: float,
    episodes: int,
    max_steps_per_episode: int,
    evaluation_interval: int,
    seed: int,
    progress: bool,
) -> MonteCarloControlBase | SarsaBase | QLearningBase | ExpectedSarsaBase:
    module_name = "solution" if implementation == "solution" else "algorithms"
    module = importlib.import_module(module_name)
    exploration_strategy = module.Exploration(  # pyright: ignore[reportAny]
        ExplorationConfig(
            epsilon_start=epsilon,
            epsilon_end=epsilon_end,
            decay_iterations=epsilon_decay_episodes,
        )
    )

    if algorithm == "sarsa":
        return module.Sarsa(  # pyright: ignore[reportAny]
            gamma=gamma,
            alpha=alpha,
            exploration=exploration_strategy,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=progress,
        )

    if algorithm == "q_learning":
        return module.QLearning(  # pyright: ignore[reportAny]
            gamma=gamma,
            alpha=alpha,
            exploration=exploration_strategy,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=progress,
        )

    if algorithm == "expected_sarsa":
        return module.ExpectedSarsa(  # pyright: ignore[reportAny]
            gamma=gamma,
            alpha=alpha,
            exploration=exploration_strategy,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=progress,
        )

    return module.MonteCarloControl(  # pyright: ignore[reportAny]
        gamma=gamma,
        epsilon=epsilon,
        exploration=exploration_strategy,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        evaluation_interval=evaluation_interval,
        seed=seed,
        show_progress=progress,
    )


if __name__ == "__main__":
    typer.run(main)
