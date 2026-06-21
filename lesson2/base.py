from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias, TypeVar, override

import numpy as np
from cliff_walking import CliffWalkingModel
from tqdm.auto import tqdm

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)
ActionDistribution: TypeAlias = dict[Action, float]
ActionValueFunction: TypeAlias = dict[State, dict[Action, float]]
VisitCounts: TypeAlias = dict[State, dict[Action, int]]
Policy: TypeAlias = dict[State, ActionDistribution[Action]]

CliffWalkingState: TypeAlias = int
CliffWalkingAction: TypeAlias = int
CliffWalkingActionDistribution: TypeAlias = ActionDistribution[CliffWalkingAction]
CliffWalkingActionValueFunction: TypeAlias = ActionValueFunction[
    CliffWalkingState,
    CliffWalkingAction,
]
CliffWalkingVisitCounts: TypeAlias = VisitCounts[
    CliffWalkingState,
    CliffWalkingAction,
]
CliffWalkingPolicy: TypeAlias = Policy[CliffWalkingState, CliffWalkingAction]


@dataclass(frozen=True)
class EpisodeStep:
    state: CliffWalkingState
    action: CliffWalkingAction
    reward: float


@dataclass(frozen=True)
class Transition:
    state: CliffWalkingState
    action: CliffWalkingAction
    reward: float
    next_state: CliffWalkingState
    done: bool
    # For SARSA is not None. For Q-Learning and Expected SARSA is None.
    next_action: CliffWalkingAction | None = None


@dataclass(frozen=True)
class ExplorationConfig:
    epsilon_start: float = 0.1
    epsilon_end: float = 0.1
    decay_iterations: int = 10_000


class ExplorationBase(ABC):
    def __init__(self, config: ExplorationConfig | None = None) -> None:
        self.config: ExplorationConfig = config or ExplorationConfig()
        self.iteration: int = 0

    @property
    def epsilon(self) -> float:
        if self.config.decay_iterations <= 1:
            return self.config.epsilon_end

        progress = min(self.iteration / (self.config.decay_iterations - 1), 1.0)
        return (
            self.config.epsilon_start
            + (self.config.epsilon_end - self.config.epsilon_start) * progress
        )

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration

    def action_probabilities(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        state: CliffWalkingState,
    ) -> CliffWalkingActionDistribution:
        return self.epsilon_greedy_action_probabilities(
            env,
            action_values,
            state,
            epsilon=self.epsilon,
        )

    def policy_distribution(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingPolicy:
        return {
            state: self.action_probabilities(env, action_values, state)
            for state in env.non_terminal_states()
        }

    def select_action(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        state: CliffWalkingState,
        rng: np.random.Generator,
    ) -> CliffWalkingAction:
        probabilities_by_action = self.action_probabilities(
            env,
            action_values,
            state,
        )
        draw = rng.random()
        cumulative_probability = 0.0
        fallback_action = next(iter(env.actions()))

        for action in env.actions():
            fallback_action = action
            cumulative_probability += probabilities_by_action[action]
            if draw <= cumulative_probability:
                return action

        return fallback_action

    @abstractmethod
    def epsilon_greedy_action_probabilities(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        state: CliffWalkingState,
        *,
        epsilon: float,
    ) -> CliffWalkingActionDistribution:
        """Return epsilon-greedy action probabilities for one state."""


class GreedyExploration(ExplorationBase):
    @override
    def epsilon_greedy_action_probabilities(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        state: CliffWalkingState,
        *,
        epsilon: float,
    ) -> CliffWalkingActionDistribution:
        best_action = max(
            env.actions(),
            key=lambda action: (action_values[state][action], -action),
        )
        return {
            action: 1.0 if action == best_action else 0.0 for action in env.actions()
        }


class SampledControlBase(ABC):
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        episodes: int = 10_000,
        max_steps_per_episode: int = 1_000,
        evaluation_interval: int = 100,
        seed: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.gamma: float = gamma
        self.episodes: int = episodes
        self.max_steps_per_episode: int = max_steps_per_episode
        self.evaluation_interval: int = evaluation_interval
        self.seed: int = seed
        self.show_progress: bool = show_progress
        self.episode_returns: list[float] = []
        self.evaluation_returns: list[tuple[int, float]] = []
        self.action_values: CliffWalkingActionValueFunction = {}

    def _episode_range(self, *, description: str) -> Iterable[int]:
        return tqdm(
            range(self.episodes),
            desc=description,
            unit="episode",
            disable=not self.show_progress,
        )

    def _zero_action_values(
        self,
        env: CliffWalkingModel,
        *,
        initial_value: float = 0.0,
    ) -> CliffWalkingActionValueFunction:
        return {
            state: {action: initial_value for action in env.actions()}
            for state in env.non_terminal_states()
        }

    def _zero_visit_counts(self, env: CliffWalkingModel) -> CliffWalkingVisitCounts:
        return {
            state: {action: 0 for action in env.actions()}
            for state in env.non_terminal_states()
        }

    def _uniform_policy(self, env: CliffWalkingModel) -> CliffWalkingPolicy:
        probability = 1.0 / len(env.actions())
        return {
            state: {action: probability for action in env.actions()}
            for state in env.non_terminal_states()
        }

    def _deterministic_policy(
        self,
        env: CliffWalkingModel,
        best_actions: Mapping[CliffWalkingState, CliffWalkingAction],
    ) -> CliffWalkingPolicy:
        return {
            state: {
                action: 1.0 if action == best_actions[state] else 0.0
                for action in env.actions()
            }
            for state in env.non_terminal_states()
        }

    def _greedy_action(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        state: CliffWalkingState,
    ) -> CliffWalkingAction:
        return max(
            env.actions(),
            key=lambda action: (action_values[state][action], -action),
        )

    def _greedy_policy(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingPolicy:
        best_actions = {
            state: self._greedy_action(env, action_values, state)
            for state in env.non_terminal_states()
        }
        return self._deterministic_policy(env, best_actions)

    def _epsilon_greedy_policy(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        epsilon: float,
    ) -> CliffWalkingPolicy:
        exploration_probability = epsilon / len(env.actions())
        policy: CliffWalkingPolicy = {}

        for state in env.non_terminal_states():
            best_action = self._greedy_action(env, action_values, state)
            policy[state] = {
                action: exploration_probability for action in env.actions()
            }
            policy[state][best_action] += 1.0 - epsilon

        return policy

    def _sample_action(
        self,
        env: CliffWalkingModel,
        policy: CliffWalkingPolicy,
        state: CliffWalkingState,
        rng: np.random.Generator,
    ) -> CliffWalkingAction:
        draw = rng.random()
        cumulative_probability = 0.0
        fallback_action = next(iter(env.actions()))

        for action in env.actions():
            fallback_action = action
            cumulative_probability += policy[state][action]
            if draw <= cumulative_probability:
                return action

        return fallback_action

    def _validate_policy(
        self,
        env: CliffWalkingModel,
        policy: CliffWalkingPolicy,
    ) -> None:
        for state in env.non_terminal_states():
            if state not in policy:
                raise ValueError(f"Policy is missing state {state}.")

            distribution = policy[state]
            if set(distribution) != set(env.actions()):
                raise ValueError(f"Policy for state {state} must include every action.")

            probability_sum = sum(distribution.values())
            if abs(probability_sum - 1.0) > 1e-9:
                raise ValueError(
                    f"Policy probabilities for state {state} sum to {probability_sum}."
                )

    def _greedy_policy_return(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        *,
        seed: int | None = None,
    ) -> float:
        policy = self._greedy_policy(env, action_values)
        state = env.reset(seed=seed)
        episode_return = 0.0

        for _ in range(self.max_steps_per_episode):
            action = max(
                policy[state],
                key=lambda candidate: policy[state][candidate],
            )
            state, reward, done = env.step(action)
            episode_return += reward

            if done:
                break

        return episode_return

    def greedy_policy(self, env: CliffWalkingModel) -> CliffWalkingPolicy:
        return self._greedy_policy(env, self.action_values)

    def greedy_policy_return(
        self,
        env: CliffWalkingModel,
        *,
        seed: int | None = None,
    ) -> float:
        return self._greedy_policy_return(env, self.action_values, seed=seed)

    def _record_greedy_policy_return(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
        *,
        episode_index: int,
    ) -> None:
        if self.evaluation_interval <= 0:
            return

        episode_number = episode_index + 1
        if episode_number % self.evaluation_interval != 0:
            return

        episode_return = self._greedy_policy_return(
            env,
            action_values,
        )
        self.evaluation_returns.append((episode_number, episode_return))


class MonteCarloControlBase(SampledControlBase, ABC):
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        epsilon: float = 0.1,
        exploration: ExplorationBase | None = None,
        episodes: int = 10_000,
        max_steps_per_episode: int = 1_000,
        evaluation_interval: int = 100,
        seed: int = 0,
        show_progress: bool = True,
    ) -> None:
        super().__init__(
            gamma=gamma,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=show_progress,
        )
        self.epsilon: float = epsilon
        self.exploration: ExplorationBase | None = exploration

    def _set_exploration_iteration(self, iteration: int) -> None:
        if self.exploration is None:
            return

        self.exploration.set_iteration(iteration)
        self.epsilon = self.exploration.epsilon

    def solve(self, env: CliffWalkingModel) -> CliffWalkingPolicy:
        rng = np.random.default_rng(self.seed)
        action_values = self._zero_action_values(env)
        visit_counts = self._zero_visit_counts(env)
        self.episode_returns: list[float] = []
        self.evaluation_returns: list[tuple[int, float]] = []

        for episode_index in self._episode_range(description=type(self).__name__):
            self._set_exploration_iteration(episode_index)
            policy = self.policy_improvement(env, action_values)
            episode = self.generate_episode(
                env,
                policy,
                rng,
                seed=self.seed + episode_index,
            )
            self.episode_returns.append(sum(step.reward for step in episode))
            action_values = self.policy_evaluation(
                env,
                episode,
                action_values,
                visit_counts,
            )
            policy = self.policy_improvement(env, action_values)
            self._record_greedy_policy_return(
                env,
                action_values,
                episode_index=episode_index,
            )

        self._set_exploration_iteration(max(self.episodes - 1, 0))
        policy = self.policy_improvement(env, action_values)
        self.action_values: CliffWalkingActionValueFunction = action_values
        self._validate_policy(env, policy)
        return policy

    def generate_episode(
        self,
        env: CliffWalkingModel,
        policy: CliffWalkingPolicy,
        rng: np.random.Generator,
        *,
        seed: int | None = None,
    ) -> list[EpisodeStep]:
        state = env.reset(seed=seed)
        episode: list[EpisodeStep] = []

        for _ in range(self.max_steps_per_episode):
            action = self._sample_action(env, policy, state, rng)
            next_state, reward, done = env.step(action)
            episode.append(EpisodeStep(state, action, reward))

            if done:
                break

            state = next_state

        return episode

    @abstractmethod
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        episode: list[EpisodeStep],
        action_values: CliffWalkingActionValueFunction,
        visit_counts: CliffWalkingVisitCounts,
    ) -> CliffWalkingActionValueFunction:
        """Estimate q_pi(s, a) from one sampled episode."""

    @abstractmethod
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingPolicy:
        """Return an epsilon-greedy policy using the current action values."""


class SarsaBase(SampledControlBase, ABC):
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        alpha: float = 0.5,
        exploration: ExplorationBase | None = None,
        episodes: int = 10_000,
        max_steps_per_episode: int = 1_000,
        evaluation_interval: int = 100,
        seed: int = 0,
        show_progress: bool = True,
    ) -> None:
        super().__init__(
            gamma=gamma,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=show_progress,
        )
        self.alpha: float = alpha
        self.exploration: ExplorationBase = exploration or GreedyExploration(
            ExplorationConfig(epsilon_start=0.0, epsilon_end=0.0)
        )

    def solve(self, env: CliffWalkingModel) -> CliffWalkingPolicy:
        rng = np.random.default_rng(self.seed)
        action_values = self._zero_action_values(env)
        self.episode_returns: list[float] = []
        self.evaluation_returns: list[tuple[int, float]] = []

        for episode_index in self._episode_range(description=type(self).__name__):
            self.exploration.set_iteration(episode_index)
            state = env.reset(seed=self.seed + episode_index)
            episode_return = 0.0
            action = self.exploration.select_action(
                env,
                action_values,
                state,
                rng,
            )

            for _ in range(self.max_steps_per_episode):
                next_state, reward, done = env.step(action)
                next_action = (
                    None
                    if done
                    else self.exploration.select_action(
                        env,
                        action_values,
                        next_state,
                        rng,
                    )
                )
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_action=next_action,
                )

                action_values = self.policy_evaluation(
                    env,
                    transition,
                    action_values,
                )
                episode_return += reward

                if done:
                    break

                assert next_action is not None
                state = next_state
                action = next_action

            self.episode_returns.append(episode_return)
            self._record_greedy_policy_return(
                env,
                action_values,
                episode_index=episode_index,
            )

        self.action_values: CliffWalkingActionValueFunction = action_values
        policy = self._greedy_policy(env, action_values)
        self._validate_policy(env, policy)
        return policy

    @abstractmethod
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingActionValueFunction:
        """Apply one SARSA action-value update from (s, a, r, s', a')."""

    @abstractmethod
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingPolicy:
        """Return the policy used for visualization or inspection."""


class QLearningBase(SampledControlBase, ABC):
    def __init__(
        self,
        *,
        gamma: float = 1.0,
        alpha: float = 0.5,
        exploration: ExplorationBase | None = None,
        episodes: int = 10_000,
        max_steps_per_episode: int = 1_000,
        evaluation_interval: int = 100,
        seed: int = 0,
        show_progress: bool = True,
    ) -> None:
        super().__init__(
            gamma=gamma,
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
            evaluation_interval=evaluation_interval,
            seed=seed,
            show_progress=show_progress,
        )
        self.alpha: float = alpha
        self.exploration: ExplorationBase = exploration or GreedyExploration(
            ExplorationConfig(epsilon_start=0.0, epsilon_end=0.0)
        )

    def solve(self, env: CliffWalkingModel) -> CliffWalkingPolicy:
        rng = np.random.default_rng(self.seed)
        action_values = self._zero_action_values(env)
        self.episode_returns: list[float] = []
        self.evaluation_returns: list[tuple[int, float]] = []

        for episode_index in self._episode_range(description=type(self).__name__):
            self.exploration.set_iteration(episode_index)
            state = env.reset(seed=self.seed + episode_index)
            episode_return = 0.0

            for _ in range(self.max_steps_per_episode):
                action = self.exploration.select_action(
                    env,
                    action_values,
                    state,
                    rng,
                )
                next_state, reward, done = env.step(action)
                transition = Transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                action_values = self.policy_evaluation(
                    env,
                    transition,
                    action_values,
                )
                episode_return += reward

                if done:
                    break

                state = next_state

            self.episode_returns.append(episode_return)
            self._record_greedy_policy_return(
                env,
                action_values,
                episode_index=episode_index,
            )

        self.action_values: CliffWalkingActionValueFunction = action_values
        policy = self._greedy_policy(env, action_values)
        self._validate_policy(env, policy)
        return policy

    @abstractmethod
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingActionValueFunction:
        """Apply one Q-learning update from (s, a, r, s')."""

    @abstractmethod
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingPolicy:
        """Return the policy used for visualization or inspection."""


class ExpectedSarsaBase(QLearningBase, ABC):
    @abstractmethod
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: CliffWalkingActionValueFunction,
    ) -> CliffWalkingActionValueFunction:
        """Apply one Expected SARSA update from (s, a, r, s')."""
