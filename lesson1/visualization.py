from pathlib import Path
from typing import cast

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from base import FrozenLakeAction as Action
from base import FrozenLakePolicy as PolicyDistribution
from frozen_lake import FrozenLakeModel
from matplotlib.patches import FancyArrowPatch

DIRECTIONS: dict[Action, tuple[float, float]] = {
    0: (-0.30, 0.0),
    1: (0.0, 0.30),
    2: (0.30, 0.0),
    3: (0.0, -0.30),
}


def plot_policy(
    env: FrozenLakeModel,
    policy: PolicyDistribution,
    *,
    output_path: str | Path,
    show: bool = False,
    title: str | None = None,
) -> None:
    image = env.render()
    if image is None:
        raise RuntimeError("Create FrozenLakeModel with render_mode='rgb_array'.")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(image)  # pyright: ignore[reportUnknownMemberType]
    if title is not None:
        ax.set_title(title)  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks([])  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks([])  # pyright: ignore[reportUnknownMemberType]

    image_height, image_width, _ = cast(tuple[int, int, int], image.shape)
    cell_width = image_width / env.ncol
    cell_height = image_height / env.nrow

    for row in range(env.nrow):
        for col in range(env.ncol):
            state = row * env.ncol + col
            if state not in policy:
                continue

            action_distribution = policy[state]
            action = max(
                action_distribution,
                key=lambda candidate: action_distribution[candidate],
            )
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
    fig.savefig(path, dpi=170, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]

    if show:
        plt.show()  # pyright: ignore[reportUnknownMemberType]
    else:
        plt.close(fig)
