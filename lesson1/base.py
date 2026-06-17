from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from typing import TypeAlias, TypeVar

from frozen_lake import FrozenLakeModel

State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)
ValueFunction: TypeAlias = dict[State, float]
Policy: TypeAlias = dict[State, dict[Action, float]]

FrozenLakeState: TypeAlias = int
FrozenLakeAction: TypeAlias = int
FrozenLakeValueFunction: TypeAlias = ValueFunction[FrozenLakeState]
FrozenLakePolicy: TypeAlias = Policy[FrozenLakeState, FrozenLakeAction]


class ImplementationBase(ABC):
    def __init__(
        self,
        *,
        gamma: float = 0.9,
        max_iterations: int = 1_000,
        eps: float = 1e-8,
    ) -> None:
        self.gamma: float = gamma
        self.max_iterations: int = max_iterations
        self.eps: float = eps

    def _zero_values(self, env: FrozenLakeModel) -> FrozenLakeValueFunction:
        return {state: 0.0 for state in env.states()}

    def _uniform_policy(self, env: FrozenLakeModel) -> FrozenLakePolicy:
        probability = 1.0 / len(env.actions())
        return {
            state: {action: probability for action in env.actions()}
            for state in env.non_terminal_states()
        }

    def _deterministic_policy(
        self,
        env: FrozenLakeModel,
        best_actions: Mapping[FrozenLakeState, FrozenLakeAction],
    ) -> FrozenLakePolicy:
        return {
            state: {
                action: 1.0 if action == best_actions[state] else 0.0
                for action in env.actions()
            }
            for state in env.non_terminal_states()
        }

    def _action_value(
        self,
        env: FrozenLakeModel,
        state: FrozenLakeState,
        action: FrozenLakeAction,
        values: Mapping[FrozenLakeState, float],
    ) -> float:
        return sum(
            probability * (reward if done else reward + self.gamma * values[next_state])
            for probability, next_state, reward, done in env.transitions(state, action)
        )

    def _max_delta(
        self,
        left: Mapping[FrozenLakeState, float],
        right: Mapping[FrozenLakeState, float],
    ) -> float:
        return max(abs(left[state] - right[state]) for state in left)

    def _validate_policy(self, env: FrozenLakeModel, policy: FrozenLakePolicy) -> None:
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


class PolicyIterationBase(ImplementationBase, ABC):
    def __init__(
        self,
        *,
        gamma: float = 0.9,
        max_iterations: int = 1_000,
        max_evaluation_iterations: int = 1_000,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.gamma: float = gamma
        self.max_iterations: int = max_iterations
        self.eps: float = eps
        self.max_evaluation_iterations: int = max_evaluation_iterations

    def solve(self, env: FrozenLakeModel) -> FrozenLakePolicy:
        values = self._zero_values(env)
        policy = self._uniform_policy(env)

        for _ in range(self.max_iterations):
            values = self.policy_evaluation(env, policy, values)
            new_policy = self.policy_improvement(env, values)

            self._validate_policy(env, new_policy)
            if new_policy == policy:
                return new_policy
            policy = new_policy

        return policy

    @abstractmethod
    def policy_evaluation(
        self,
        env: FrozenLakeModel,
        policy: FrozenLakePolicy,
        values: FrozenLakeValueFunction,
    ) -> FrozenLakeValueFunction:
        """Estimate v_pi(s) for the current policy."""

    @abstractmethod
    def policy_improvement(
        self,
        env: FrozenLakeModel,
        values: FrozenLakeValueFunction,
    ) -> FrozenLakePolicy:
        """Return a greedier policy distribution using the current values."""


class ValueIterationBase(ImplementationBase, ABC):
    def __init__(
        self,
        *,
        gamma: float = 0.9,
        max_iterations: int = 1_000,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.gamma: float = gamma
        self.max_iterations: int = max_iterations
        self.eps: float = eps

    def solve(self, env: FrozenLakeModel) -> FrozenLakePolicy:
        values = self._zero_values(env)

        for _ in range(self.max_iterations):
            new_values = self.value_update(env, values)
            if self._max_delta(values, new_values) < self.eps:
                values = new_values
                break
            values = new_values

        policy = self.policy_extraction(env, values)
        self._validate_policy(env, policy)
        return policy

    @abstractmethod
    def value_update(
        self,
        env: FrozenLakeModel,
        values: FrozenLakeValueFunction,
    ) -> FrozenLakeValueFunction:
        """Apply one Bellman optimality update."""

    @abstractmethod
    def policy_extraction(
        self,
        env: FrozenLakeModel,
        values: FrozenLakeValueFunction,
    ) -> FrozenLakePolicy:
        """Extract a greedy policy distribution from the value function."""
