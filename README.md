# Q-LEARN

Tabular **Q-learning** on a 7×7 GridWorld, built to compare how different
**reward-shaping strategies** affect learning speed and the final policy.

## Files

| File | Purpose |
|------|---------|
| `gridworld.py` | The GridWorld environment: grid layout, `reset()`/`step()` API, and four reward modes. |
| `glearn.py` | `QLearningAgent` (ε-greedy, tabular Q-table) and a `train()` loop. |
| `experiments.py` | *(empty)* — intended for running/comparing reward modes. |
| `visualize.py` | *(empty)* — intended for plotting learning curves / policies. |

## The environment

A 7×7 grid with a start `S`, a goal `G`, walls `#`, and re-collectible coins `C`.
The agent has 4 actions (up, down, left, right); bumping a wall or edge keeps it
in place. State is the agent's `(row, col)` position.

### Reward modes

Set via `GridWorld(reward_mode=...)`. All share a per-step penalty and a large
reward at the goal; they differ in the shaping signal added on top:

- **`sparse`** — step penalty + goal reward only (hardest to learn).
- **`distance`** — bonus for reducing Manhattan distance to the goal.
- **`coin`** — bonus for stepping on a coin cell.
- **`potential`** — potential-based shaping `γ·Φ(s') − Φ(s)`, which provably
  preserves the optimal policy.

## Setup

This project targets Python 3.12 (managed by `uv`), so use a virtual
environment rather than installing packages globally.

```bash
cd "Q-LEARN"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Usage

Run any script with the venv's Python:

```bash
.venv/bin/python gridworld.py    # defines the env (no output on its own)
```

Or activate the environment first:

```bash
source .venv/bin/activate
python glearn.py
```

### Example: train an agent

```python
from gridworld import GridWorld
from glearn import QLearningAgent, train

env = GridWorld(reward_mode="potential")
agent = QLearningAgent(env.n_rows, env.n_cols, gamma=env.gamma)

history = train(env, agent, n_episodes=1000)

# Inspect the learned policy as arrows on the grid
for line in env.render_policy_grid(agent.greedy_policy_fn()):
    print(line)
```

## Dependencies

- `numpy` (see `requirements.txt`)
- `random` — Python standard library, no install needed
