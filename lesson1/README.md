# Lesson 1: Model-Based Reinforcement Learning

This lesson covers the core ideas from
[Reinforcement Learning 101](https://elkotito.github.io/posts/reinforcement-learning-101/):
Bellman expectation updates, Bellman optimality updates, policy iteration, and value iteration.

## Your Task

Open `lesson1/algorithms.py` and implement the four methods below. Each method has a `# TODO` comment with exact steps.

- A deterministic policy assigns probability `1.0` to one action, `0.0` to the rest.
- Actions: `0` = left, `1` = down, `2` = right, `3` = up.

### Policy Iteration (`PolicyIteration` class)

**`policy_evaluation(env, policy, values) -> ValueFunction`**
**`policy_improvement(env, values) -> Policy`**

### Value Iteration (`ValueIteration` class)

**`value_update(env, values) -> ValueFunction`**
**`policy_extraction(env, values) -> Policy`**

## Verification

Verify your implementation against the reference solution first:

```bash
uv run python lesson1/main.py --implementation solution --algorithm policy_iteration
uv run python lesson1/main.py --implementation solution --algorithm value_iteration
```

Then run your own:

```bash
uv run python lesson1/main.py --implementation student --algorithm policy_iteration
uv run python lesson1/main.py --implementation student --algorithm value_iteration
```

The script saves a policy visualization to `lesson1/outputs/frozen_lake_policy.png`.

**Useful flags:**

```bash
# Smaller 4x4 grid (easier to debug)
uv run python lesson1/main.py --implementation student --algorithm policy_iteration --map-name 4x4

# Stochastic transitions (slippery ice)
uv run python lesson1/main.py --implementation student --algorithm policy_iteration --slippery

# Show the matplotlib window instead of saving to file
uv run python lesson1/main.py --implementation student --algorithm policy_iteration --show
```

## Helpers

These are available if you need them. You don't have to use all of them.

**`FrozenLakeModel` methods** (the `env` argument):

- `env.states()` — all states
- `env.actions()` — all possible actions
- `env.non_terminal_states()` — states that are not terminal
- `env.transitions(state, action)` — list of `(probability, next_state, reward, done)` tuples

**`self` methods** (defined in `base.py`):

- `self._action_value(env, state, action, values)` — computes q(s, a) from the Bellman equation
- `self._deterministic_policy(env, best_actions)` — builds a policy dict from a `{state: best_action}` map
- `self._max_delta(left, right)` — max absolute difference between two value functions
- `self.gamma`, `self.eps`, `self.max_iterations`
