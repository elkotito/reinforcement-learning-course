from typing import override

from base import CliffWalkingActionDistribution as ActionDistribution
from base import CliffWalkingActionValueFunction as ActionValueFunction
from base import CliffWalkingPolicy as PolicyDistribution
from base import CliffWalkingState as State
from base import CliffWalkingVisitCounts as VisitCounts
from base import (
    EpisodeStep,
    ExpectedSarsaBase,
    ExplorationBase,
    MonteCarloControlBase,
    QLearningBase,
    SarsaBase,
    Transition,
)
from cliff_walking import CliffWalkingModel


class Exploration(ExplorationBase):
    @override
    def epsilon_greedy_action_probabilities(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
        state: State,
        *,
        epsilon: float,
    ) -> ActionDistribution:
        # TODO:
        # 1. Validate that epsilon is in [0, 1].
        # 2. Give every action probability epsilon / number_of_actions.
        # 3. Add 1 - epsilon to the greedy action probability.
        # 4. Break ties deterministically the same way as _greedy_action:
        #    prefer the smaller action id when values are equal.
        raise NotImplementedError


class MonteCarloControl(MonteCarloControlBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        episode: list[EpisodeStep],
        action_values: ActionValueFunction,
        visit_counts: VisitCounts,
    ) -> ActionValueFunction:
        # TODO:
        # 1. Walk backward through the sampled episode and compute the return G.
        # 2. Use first-visit Monte Carlo updates for each (state, action).
        # 3. Update q(s, a) with an incremental sample average.
        raise NotImplementedError

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        # TODO:
        # 1. For every non-terminal state choose the greedy action from q(s, a).
        # 2. Return an epsilon-greedy policy so future episodes keep sampling.
        raise NotImplementedError


class Sarsa(SarsaBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: ActionValueFunction,
    ) -> ActionValueFunction:
        # TODO:
        # 1. Compute the SARSA target r + gamma * q(s', a'), or r when done.
        # 2. Compute the TD error delta.
        # 3. Update only q(s, a) by alpha * delta.
        raise NotImplementedError

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        return self._greedy_policy(env, action_values)


class QLearning(QLearningBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: ActionValueFunction,
    ) -> ActionValueFunction:
        # TODO:
        # 1. Compute the Q-learning target r + gamma * max_a q(s', a), or r when done.
        # 2. Compute the TD error delta.
        # 3. Update only q(s, a) by alpha * delta.
        raise NotImplementedError

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        return self._greedy_policy(env, action_values)


class ExpectedSarsa(ExpectedSarsaBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: ActionValueFunction,
    ) -> ActionValueFunction:
        # TODO:
        # 1. Get the current epsilon-greedy probabilities for s'.
        # 2. Compute expected_q = sum_a pi(a|s') * q(s', a), or 0 when done.
        # 3. Compute the target r + gamma * expected_q.
        # 4. Update only q(s, a) by alpha * delta.
        raise NotImplementedError

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        return self._greedy_policy(env, action_values)
