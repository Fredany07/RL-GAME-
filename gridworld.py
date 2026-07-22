

import numpy as np

WALL = "#"
COIN = "C"
GOAL = "G"
START = "S"
EMPTY = "."

GRID_TEMPLATE = [
    list(". . . . . . ."),
    list(". # # . . . ."),
    list(". . . C . . ."),
    list("S . # # . . ."),
    list(". C . . . . ."),
    list(". . . . . . ."),
    list(". . . . . . G"),
]
# The list(str) trick above includes spaces as characters; strip them out.
GRID_TEMPLATE = [[c for c in row if c != " "] for row in GRID_TEMPLATE]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]


class GridWorld:
    def __init__(self, reward_mode="sparse", max_steps=100, gamma=0.95,
                 step_penalty=-0.1, goal_reward=10.0, coin_reward=5.0):
        self.reward_mode = reward_mode
        self.max_steps = max_steps
        self.gamma = gamma
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.coin_reward = coin_reward

        self.grid = GRID_TEMPLATE
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])

        self.walls = set()
        self.coins = set()
        self.start = None
        self.goal = None
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                cell = self.grid[r][c]
                if cell == WALL:
                    self.walls.add((r, c))
                elif cell == COIN:
                    self.coins.add((r, c))
                elif cell == START:
                    self.start = (r, c)
                elif cell == GOAL:
                    self.goal = (r, c)

        assert self.start is not None and self.goal is not None

        self.reset()

    def manhattan(self, pos):
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def potential(self, pos):
        return -self.manhattan(pos)

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.coins_collected_this_episode = set()
        return self.state()

    def state(self):
        # State = position only (coins are infinitely re-collectible,
        # so we don't need to track "already collected" in the state).
        return self.pos

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, action_idx):
        self.steps += 1
        dr, dc = ACTIONS[action_idx]
        new_pos = (self.pos[0] + dr, self.pos[1] + dc)

        if not self.in_bounds(new_pos) or new_pos in self.walls:
            new_pos = self.pos  # bump into wall/edge, stay in place

        prev_dist = self.manhattan(self.pos)
        prev_potential = self.potential(self.pos)

        reward = self.step_penalty
        done = False

        if new_pos == self.goal:
            reward += self.goal_reward
            done = True

        if self.reward_mode == "distance":
            new_dist = self.manhattan(new_pos)
            reward += (prev_dist - new_dist)

        elif self.reward_mode == "coin":
            if new_pos in self.coins:
                reward += self.coin_reward

        elif self.reward_mode == "potential":
            new_potential = self.potential(new_pos)
            reward += (self.gamma * new_potential - prev_potential)

        # "sparse" mode adds nothing extra

        self.pos = new_pos
        if self.steps >= self.max_steps:
            done = True

        return self.state(), reward, done, {}

    def render_policy_grid(self, policy_fn):
        """Return a list of strings showing the greedy action at every free cell."""
        arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                pos = (r, c)
                if pos in self.walls:
                    row_chars.append("#")
                elif pos == self.goal:
                    row_chars.append("G")
                elif pos == self.start:
                    row_chars.append("S")
                elif pos in self.coins:
                    row_chars.append("C")
                else:
                    a = policy_fn(pos)
                    row_chars.append(arrows[a])
            lines.append(" ".join(row_chars))
        return lines