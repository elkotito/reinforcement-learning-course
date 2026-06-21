from collections.abc import Sequence
import math
from pathlib import Path
from typing import cast

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from base import CliffWalkingAction as Action
from base import CliffWalkingPolicy as PolicyDistribution
from cliff_walking import CliffWalkingModel
from matplotlib.patches import FancyArrowPatch

DIRECTIONS: dict[Action, tuple[float, float]] = {
    0: (0.0, -0.30),
    1: (0.30, 0.0),
    2: (0.0, 0.30),
    3: (-0.30, 0.0),
}


def plot_greedy_path(
    env: CliffWalkingModel,
    policy: PolicyDistribution,
    *,
    output_path: str | Path,
    show: bool = False,
    title: str | None = None,
    seed: int | None = None,
) -> None:
    env.reset(seed=seed)
    image = env.render()
    if image is None:
        raise RuntimeError("Create CliffWalkingModel with render_mode='rgb_array'.")

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]

    if title is not None:
        ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks([])  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks([])  # pyright: ignore[reportUnknownMemberType]

    image_height, image_width, _ = cast(tuple[int, int, int], image.shape)
    cell_width = image_width / env.ncol
    cell_height = image_height / env.nrow

    for state, action in _greedy_path(env, policy):
        row = state // env.ncol
        col = state % env.ncol
        direction_x, direction_y = DIRECTIONS[action]
        center_x = col * cell_width + cell_width / 2
        center_y = row * cell_height + cell_height / 2

        arrow = FancyArrowPatch(
            (
                center_x - direction_x * cell_width,
                center_y - direction_y * cell_height,
            ),
            (
                center_x + direction_x * cell_width,
                center_y + direction_y * cell_height,
            ),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=3.8,
            color="#e11d48",
            shrinkA=0,
            shrinkB=0,
            zorder=5,
        )
        arrow.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=6.2,
                    foreground="white",
                    alpha=0.82,
                ),
                path_effects.Normal(),
            ]
        )
        ax.add_patch(arrow)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", transparent=True)  # pyright: ignore[reportUnknownMemberType]

    if show:
        plt.show()  # pyright: ignore[reportUnknownMemberType]
    else:
        plt.close(fig)


def _greedy_path(
    env: CliffWalkingModel,
    policy: PolicyDistribution,
) -> list[tuple[int, Action]]:
    state = env.start_state
    path: list[tuple[int, Action]] = []
    visited_states: set[int] = set()

    while state in policy and state not in visited_states:
        visited_states.add(state)
        action_distribution = policy[state]
        action = max(
            action_distribution,
            key=lambda candidate: action_distribution[candidate],
        )
        path.append((state, action))

        next_state, done = _deterministic_next_state(env, state, action)
        if done:
            break

        state = next_state

    return path


def _deterministic_next_state(
    env: CliffWalkingModel,
    state: int,
    action: Action,
) -> tuple[int, bool]:
    row = state // env.ncol
    col = state % env.ncol

    if action == 0:
        row = max(row - 1, 0)
    elif action == 1:
        col = min(col + 1, env.ncol - 1)
    elif action == 2:
        row = min(row + 1, env.nrow - 1)
    elif action == 3:
        col = max(col - 1, 0)

    next_state = row * env.ncol + col
    if next_state == env.goal_state:
        return next_state, True

    if next_state in env.cliff_states():
        return env.start_state, False

    return next_state, False


def plot_learning_curve(
    episode_returns: Sequence[float],
    *,
    output_path: str | Path,
    show: bool = False,
    title: str | None = None,
    smoothing_window: int = 100,
    optimal_return: float | None = -13.0,
    display_floor: float | None = None,
) -> None:
    if not episode_returns:
        raise ValueError("Cannot plot a learning curve without episode returns.")

    x_values = list(range(1, len(episode_returns) + 1))
    smoothed_returns = _moving_average(episode_returns, smoothing_window)
    smoothed_x_values = list(range(smoothing_window, len(episode_returns) + 1))
    resolved_display_floor = _resolve_display_floor(
        episode_returns,
        smoothed_returns,
        display_floor,
    )
    displayed_episode_returns = _clip_lower(
        episode_returns,
        resolved_display_floor,
    )
    displayed_smoothed_returns = _clip_lower(
        smoothed_returns,
        resolved_display_floor,
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x_values,
        displayed_episode_returns,
        color="#94a3b8",
        alpha=0.22,
        linewidth=0.8,
        label=_clipped_label(
            "episode return",
            episode_returns,
            resolved_display_floor,
        ),
    )

    if displayed_smoothed_returns:
        ax.plot(  # pyright: ignore[reportUnknownMemberType]
            smoothed_x_values,
            displayed_smoothed_returns,
            color="#e11d48",
            linewidth=2.2,
            label=_clipped_label(
                f"{smoothing_window}-episode moving average",
                smoothed_returns,
                resolved_display_floor,
            ),
        )

    if optimal_return is not None:
        ax.axhline(  # pyright: ignore[reportUnknownMemberType]
            optimal_return,
            color="#16a34a",
            linestyle="--",
            linewidth=1.4,
            label=f"shortest safe path ({optimal_return:.0f})",
        )

    if title is not None:
        ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Episode")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Return")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylim(bottom=resolved_display_floor, top=10)
    ax.grid(True, alpha=0.28)  # pyright: ignore[reportUnknownMemberType]
    ax.legend(loc="lower right")  # pyright: ignore[reportUnknownMemberType]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]

    if show:
        plt.show()  # pyright: ignore[reportUnknownMemberType]
    else:
        plt.close(fig)


def _moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 0 or len(values) < window:
        return []

    averages: list[float] = []
    running_sum = sum(values[:window])
    averages.append(running_sum / window)

    for index in range(window, len(values)):
        running_sum += values[index]
        running_sum -= values[index - window]
        averages.append(running_sum / window)

    return averages


def _clip_lower(values: Sequence[float], floor: float | None) -> list[float]:
    if floor is None:
        return list(values)

    return [max(value, floor) for value in values]


def _clipped_label(
    label: str,
    values: Sequence[float],
    floor: float | None,
) -> str:
    if floor is None or all(value >= floor for value in values):
        return label

    return f"{label} (clipped below {floor:.0f})"


def _resolve_display_floor(
    episode_returns: Sequence[float],
    smoothed_returns: Sequence[float],
    requested_floor: float | None,
) -> float:
    if requested_floor is not None:
        return requested_floor

    scale_values = list(episode_returns)
    scale_values.extend(smoothed_returns)
    tenth_percentile = _percentile(scale_values, 0.10)
    floor = min(-50.0, tenth_percentile - 20.0)
    floor = max(floor, -500.0)
    return 50.0 * math.floor(floor / 50.0)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return -50.0

    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * quantile)
    return sorted_values[index]
