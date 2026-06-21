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
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        actions = list(env.actions())
        best_action = max(
            actions,
            key=lambda action: (action_values[state][action], -action),
        )
        exploration_probability = epsilon / len(actions)
        probabilities = {action: exploration_probability for action in actions}
        probabilities[best_action] += 1.0 - epsilon
        return probabilities


class MonteCarloControl(MonteCarloControlBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        episode: list[EpisodeStep],
        action_values: ActionValueFunction,
        visit_counts: VisitCounts,
    ) -> ActionValueFunction:
        visited: set[tuple[int, int]] = set()
        returns_from_step: list[tuple[EpisodeStep, float]] = []
        episode_return = 0.0

        for step in reversed(episode):
            episode_return = step.reward + self.gamma * episode_return
            returns_from_step.append((step, episode_return))

        for step, discounted_return in reversed(returns_from_step):
            state_action = (step.state, step.action)
            if state_action in visited:
                continue

            visited.add(state_action)
            visit_counts[step.state][step.action] += 1
            count = visit_counts[step.state][step.action]
            current_value = action_values[step.state][step.action]
            action_values[step.state][step.action] = (
                current_value + (discounted_return - current_value) / count
            )

        return action_values

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        return self._epsilon_greedy_policy(env, action_values, self.epsilon)


class Sarsa(SarsaBase):
    @override
    def policy_evaluation(
        self,
        env: CliffWalkingModel,
        transition: Transition,
        action_values: ActionValueFunction,
    ) -> ActionValueFunction:
        if transition.done:
            target = transition.reward
        else:
            assert transition.next_action is not None
            target = (
                transition.reward
                + self.gamma
                * action_values[transition.next_state][transition.next_action]
            )

        current_value = action_values[transition.state][transition.action]
        td_error = target - current_value
        action_values[transition.state][transition.action] = (
            current_value + self.alpha * td_error
        )
        return action_values

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
        if transition.done:
            target = transition.reward
        else:
            next_action_value = max(
                action_values[transition.next_state][action] for action in env.actions()
            )
            target = transition.reward + self.gamma * next_action_value

        current_value = action_values[transition.state][transition.action]
        td_error = target - current_value
        action_values[transition.state][transition.action] = (
            current_value + self.alpha * td_error
        )
        return action_values

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
        if transition.done:
            target = transition.reward
        else:
            action_probabilities = self.exploration.action_probabilities(
                env,
                action_values,
                transition.next_state,
            )
            expected_next_action_value = sum(
                probability * action_values[transition.next_state][action]
                for action, probability in action_probabilities.items()
            )
            target = transition.reward + self.gamma * expected_next_action_value

        current_value = action_values[transition.state][transition.action]
        td_error = target - current_value
        action_values[transition.state][transition.action] = (
            current_value + self.alpha * td_error
        )
        return action_values

    @override
    def policy_improvement(
        self,
        env: CliffWalkingModel,
        action_values: ActionValueFunction,
    ) -> PolicyDistribution:
        return self._greedy_policy(env, action_values)
