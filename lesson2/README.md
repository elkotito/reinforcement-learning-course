# Lesson 2: Sample-Based Control

This lesson uses Gymnasium's Cliff Walking environment to move from exact
model-based planning to General Policy Iteration from samples.

Both algorithms keep the same high-level loop:

1. Estimate action values from sampled experience.
2. Improve the policy using the current action values.
3. Repeat.

Monte Carlo Control evaluates from complete sampled episodes. SARSA evaluates
sampled `(s, a, r, s', a')` transitions on-policy. Q-learning evaluates sampled
`(s, a, r, s')` transitions with the off-policy greedy target. Expected SARSA
uses the expected next action value under the behavior policy. The TD algorithms
share an epsilon-greedy behavior policy whose epsilon can decay over training.

## Your Task

Open `lesson2/algorithms.py` and implement the exploration strategy plus the
sampled-control update methods below.

```python
class Exploration:
    def epsilon_greedy_action_probabilities(self, env, q, state, *, epsilon) -> dict: ...

class MonteCarloControl:
    def policy_evaluation(self, env, episode, action_values, visit_counts) -> Q: ...
    def policy_improvement(self, env, action_values) -> Policy: ...

class Sarsa:
    def policy_evaluation(self, env, transition, action_values) -> Q: ...

class QLearning:
    def policy_evaluation(self, env, transition, action_values) -> Q: ...

class ExpectedSarsa:
    def policy_evaluation(self, env, transition, action_values) -> Q: ...
```

Actions in Cliff Walking:

- `0` = up
- `1` = right
- `2` = down
- `3` = left

Cliff cells are not decision states. Stepping into the cliff gives reward `-100`
and sends the agent back to the start without terminating the episode. The goal
terminates the episode.

## Algorithms

### Monte Carlo Control

Use first-visit Monte Carlo evaluation:

- Generate an episode using the current epsilon-greedy policy.
- Walk backward through the episode to compute returns.
- For the first visit to each `(state, action)` pair in that episode, update
  `q(s, a)` with an incremental sample average.
- Improve to a new epsilon-greedy policy.

### SARSA

Use one-step SARSA for policy evaluation:

- Sample a transition `(s, a, r, s', a')` using epsilon-greedy exploration.
- Compute `delta = r + gamma * q(s', a') - q(s, a)`.
- Update only `q(s, a)` by `alpha * delta`.
- Return the deterministic greedy policy learned from the final action values.

SARSA is on-policy: the `a'` used in the target is sampled by the same behavior
strategy that will be used from the next state.

### Q-learning

Use one-step Q-learning for policy evaluation:

- Sample a transition `(s, a, r, s')` using epsilon-greedy exploration.
- Compute `delta = r + gamma * max_a q(s', a) - q(s, a)`.
- Update only `q(s, a)` by `alpha * delta`.
- Return the deterministic greedy policy learned from the final action values.

Q-learning is off-policy: exploration changes which transitions are sampled, but
the target still uses the greedy action value of the next state.

### Expected SARSA

Use one-step Expected SARSA for policy evaluation:

- Sample a transition `(s, a, r, s')` using epsilon-greedy exploration.
- Compute `expected_q = sum_a pi(a|s') * q(s', a)` under the current
  epsilon-greedy behavior policy.
- Compute `delta = r + gamma * expected_q - q(s, a)`.
- Update only `q(s, a)` by `alpha * delta`.
- Return the deterministic greedy policy learned from the final action values.

Expected SARSA is on-policy in expectation: its target uses the action
probabilities of the same epsilon-greedy policy that samples behavior.

### Exploration

The sampled-control algorithms share `Exploration`.

- Epsilon-greedy mostly chooses the greedy action, but spreads probability
  `epsilon` uniformly across all actions.
- Epsilon decays linearly from `--epsilon` to `--epsilon-end` over
  `--epsilon-decay-episodes`. With the defaults, both values are `0.1`, matching
  the classic fixed-epsilon cliff-walking setup.

## Verification

Verify the reference solution first:

```bash
uv run python lesson2/main.py --implementation solution --algorithm monte_carlo_control
uv run python lesson2/main.py --implementation solution --algorithm sarsa
uv run python lesson2/main.py --implementation solution --algorithm q_learning
uv run python lesson2/main.py --implementation solution --algorithm expected_sarsa
```

Then run your own implementation:

```bash
uv run python lesson2/main.py --implementation student --algorithm monte_carlo_control
uv run python lesson2/main.py --implementation student --algorithm sarsa
uv run python lesson2/main.py --implementation student --algorithm q_learning
uv run python lesson2/main.py --implementation student --algorithm expected_sarsa
```

The script saves a greedy path map under `lesson2/outputs/`. This is the main
path picture: starting from the initial state, it follows the deterministic
greedy policy induced by `argmax_a Q(s, a)` and draws only the visited path.

It also saves a learning curve with:

- raw episode return from the behavior used during training
- a moving average of episode return
- the final greedy return and optimal return in the title

The Y axis is return, meaning the sum of rewards in an episode. In Cliff Walking
higher is better because all rewards are negative. The shortest safe
deterministic path is around `-13`; falling into the cliff adds a `-100` penalty.
Very poor returns are visually clipped below an adaptive lower bound so the
useful learning range near the final policy remains readable.

### Useful flags

```bash
# Faster smoke test
uv run python lesson2/main.py --implementation solution --algorithm sarsa --episodes 500

# Tune the sample-based algorithms
uv run python lesson2/main.py --implementation solution --algorithm monte_carlo_control --epsilon 0.2 --epsilon-end 0.01 --epsilon-decay-episodes 20000 --episodes 20000
uv run python lesson2/main.py --implementation solution --algorithm sarsa --alpha 0.25 --epsilon 0.2 --epsilon-end 0.01 --epsilon-decay-episodes 20000
uv run python lesson2/main.py --implementation solution --algorithm q_learning --alpha 0.5 --epsilon 0.2 --epsilon-end 0.01 --epsilon-decay-episodes 20000
uv run python lesson2/main.py --implementation solution --algorithm expected_sarsa --alpha 0.5 --epsilon 0.2 --epsilon-end 0.01 --epsilon-decay-episodes 20000

# Show the matplotlib window instead of saving to file
uv run python lesson2/main.py --implementation solution --algorithm monte_carlo_control --show

# Disable tqdm progress output
uv run python lesson2/main.py --implementation solution --algorithm q_learning --no-progress
```

## Helpers

`CliffWalkingModel` methods:

- `env.states()` - all tabular states
- `env.actions()` - all possible actions
- `env.non_terminal_states()` - states where the policy chooses actions
- `env.reset(seed=...)` - start a sampled episode
- `env.step(action)` - sample one transition

`self` methods from `base.py`:

- `self._epsilon_greedy_policy(env, q, epsilon)` - epsilon-greedy policy helper
- `self._greedy_policy(env, q)` - deterministic greedy policy for TD control
- `self._greedy_action(env, q, state)` - best action with deterministic tie-breaking
- `self.exploration` - decayed epsilon-greedy behavior strategy
- `self.gamma`, `self.alpha`, `self.epsilon`
