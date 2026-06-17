from typing import override

from base import FrozenLakePolicy as PolicyDistribution
from base import FrozenLakeValueFunction as ValueFunction
from base import PolicyIterationBase, ValueIterationBase
from frozen_lake import FrozenLakeModel


class PolicyIteration(PolicyIterationBase):
    @override
    def policy_evaluation(
        self,
        env: FrozenLakeModel,
        policy: PolicyDistribution,
        values: ValueFunction,
    ) -> ValueFunction:
        for _ in range(self.max_evaluation_iterations):
            new_values = self._zero_values(env)

            for state in env.non_terminal_states():
                new_values[state] = sum(
                    action_probability * self._action_value(env, state, action, values)
                    for action, action_probability in policy[state].items()
                )

            if self._max_delta(values, new_values) < self.eps:
                return new_values

            values = new_values

        return values

    @override
    def policy_improvement(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> PolicyDistribution:
        best_actions = {
            state: max(
                env.actions(),
                key=lambda action: self._action_value(env, state, action, values),
            )
            for state in env.non_terminal_states()
        }
        return self._deterministic_policy(env, best_actions)


class ValueIteration(ValueIterationBase):
    @override
    def value_update(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> ValueFunction:
        new_values = self._zero_values(env)

        for state in env.non_terminal_states():
            new_values[state] = max(
                self._action_value(env, state, action, values)
                for action in env.actions()
            )

        return new_values

    @override
    def policy_extraction(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> PolicyDistribution:
        best_actions = {
            state: max(
                env.actions(),
                key=lambda action: self._action_value(env, state, action, values),
            )
            for state in env.non_terminal_states()
        }
        return self._deterministic_policy(env, best_actions)
