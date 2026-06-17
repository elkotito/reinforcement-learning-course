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
        # TODO:
        # 1. Repeatedly apply the Bellman expectation equation.
        # 2. Stop when the value function changes by less than self.eps.
        # 3. Return the estimated value function v_pi.
        raise NotImplementedError

    @override
    def policy_improvement(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> PolicyDistribution:
        # TODO:
        # 1. For every non-terminal state compute q(s, a).
        # 2. Pick the best action.
        # 3. Return a deterministic policy distribution.
        raise NotImplementedError


class ValueIteration(ValueIterationBase):
    @override
    def value_update(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> ValueFunction:
        # TODO:
        # 1. Apply one Bellman optimality update.
        # 2. Return the updated value function.
        raise NotImplementedError

    @override
    def policy_extraction(
        self,
        env: FrozenLakeModel,
        values: ValueFunction,
    ) -> PolicyDistribution:
        # TODO:
        # 1. Use the final value function to compute q(s, a).
        # 2. Return the greedy deterministic policy distribution.
        raise NotImplementedError
