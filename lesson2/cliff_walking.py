from typing import TypeAlias, cast

import numpy as np
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from numpy.typing import NDArray

State: TypeAlias = int
Action: TypeAlias = int
StepResult: TypeAlias = tuple[State, float, bool, bool, dict[str, object]]
CliffWalkingImage: TypeAlias = NDArray[np.uint8]


class CliffWalkingModel:
    def __init__(
        self,
        *,
        render_mode: str = "rgb_array",
        seed: int = 2137,
    ) -> None:
        self.gym_env: CliffWalkingEnv = CliffWalkingEnv(
            render_mode=render_mode,
        )
        self.gym_env.reset(seed=seed)  # pyright: ignore[reportUnknownMemberType]

    @property
    def nrow(self) -> int:
        return int(self.gym_env.shape[0])

    @property
    def ncol(self) -> int:
        return int(self.gym_env.shape[1])

    @property
    def start_state(self) -> State:
        return int(self.gym_env.start_state_index)

    @property
    def goal_state(self) -> State:
        return self.nrow * self.ncol - 1

    def states(self) -> range:
        return range(int(self.gym_env.observation_space.n))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

    def actions(self) -> range:
        return range(int(self.gym_env.action_space.n))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]

    def cliff_states(self) -> set[State]:
        return {(self.nrow - 1) * self.ncol + col for col in range(1, self.ncol - 1)}

    def terminal_states(self) -> set[State]:
        return {self.goal_state}

    def non_terminal_states(self) -> list[State]:
        skipped_states = self.terminal_states() | self.cliff_states()
        return [state for state in self.states() if state not in skipped_states]

    def reset(self, *, seed: int | None = None) -> State:
        state, _ = self.gym_env.reset(seed=seed)  # pyright: ignore[reportUnknownMemberType]
        return int(state)

    def step(self, action: Action) -> tuple[State, float, bool]:
        next_state, reward, terminated, truncated, _ = cast(
            StepResult,
            self.gym_env.step(action),  # pyright: ignore[reportUnknownMemberType]
        )
        done = bool(terminated or truncated)
        return int(next_state), float(reward), done

    def render(self) -> CliffWalkingImage | None:
        frame = self.gym_env.render()
        if frame is None:
            return None

        if isinstance(frame, np.ndarray):
            return cast(CliffWalkingImage, frame)

        raise TypeError("Create CliffWalkingModel with render_mode='rgb_array'.")
