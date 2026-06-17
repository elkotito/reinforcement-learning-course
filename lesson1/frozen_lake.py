from typing import TypeAlias, cast

import numpy as np
from gymnasium.envs.toy_text import FrozenLakeEnv
from numpy.typing import NDArray

State: TypeAlias = int
Action: TypeAlias = int
Transition: TypeAlias = tuple[float, State, float, bool]
FrozenLakeDescription: TypeAlias = NDArray[np.bytes_]
FrozenLakeImage: TypeAlias = NDArray[np.uint8]


class FrozenLakeModel:
    def __init__(
        self,
        *,
        map_name: str = "8x8",
        is_slippery: bool = False,
        render_mode: str = "rgb_array",
        seed: int = 2137,
    ) -> None:
        self.gym_env: FrozenLakeEnv = FrozenLakeEnv(
            render_mode=render_mode,
            map_name=map_name,
            is_slippery=is_slippery,
        )
        self.gym_env.reset(seed=seed)  # pyright: ignore[reportUnknownMemberType]

    @property
    def nrow(self) -> int:
        return self.gym_env.nrow  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def ncol(self) -> int:
        return self.gym_env.ncol  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def desc(self) -> FrozenLakeDescription:
        return self.gym_env.desc

    def states(self) -> range:
        return range(int(self.gym_env.observation_space.n))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

    def actions(self) -> range:
        return range(int(self.gym_env.action_space.n))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

    def terminal_states(self) -> set[State]:
        terminals: set[State] = set()
        for row in range(self.nrow):
            for col in range(self.ncol):
                tile = cast(bytes, self.desc[row, col]).decode("utf-8")
                if tile in {"H", "G"}:
                    terminals.add(row * self.ncol + col)

        return terminals

    def non_terminal_states(self) -> list[State]:
        terminals = self.terminal_states()
        return [state for state in self.states() if state not in terminals]

    def transitions(self, state: State, action: Action) -> tuple[Transition, ...]:
        return tuple(self.gym_env.P[state][action])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]

    def render(self) -> FrozenLakeImage | None:
        frame = self.gym_env.render()
        if frame is None:
            return None

        if isinstance(frame, np.ndarray):
            return cast(FrozenLakeImage, frame)

        raise TypeError("Create FrozenLakeModel with render_mode='rgb_array'.")
