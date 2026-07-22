"""Tabular Q-learning agent for the GridWorld environment."""

import numpy as np
import random


class QLearningAgent:
    def __init__(self, n_rows, n_cols, n_actions=4,
                 alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_rows, n_cols, n_actions))

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        r, c = state
        return int(np.argmax(self.Q[r, c]))

    def update(self, state, action, reward, next_state, done):
        r, c = state
        nr, nc = next_state
        best_next = 0.0 if done else np.max(self.Q[nr, nc])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[r, c, action]
        self.Q[r, c, action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy_fn(self):
        def policy(pos):
            r, c = pos
            return int(np.argmax(self.Q[r, c]))
        return policy


def train(env, agent, n_episodes=1000):
    """Train agent on env, return per-episode: total true-goal reward achieved,
    total env reward, steps taken, and whether goal was reached."""
    history = {
        "episode_reward": [],   # reward according to env.reward_mode (what agent optimizes)
        "goal_reached": [],     # 1 if agent reached true goal this episode
        "steps": [],
        "coins_collected": [],  # count of coin-cell visits this episode
    }

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        coins_this_ep = 0
        reached_goal = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            if next_state in env.coins:
                coins_this_ep += 1
            if next_state == env.goal:
                reached_goal = 1

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        history["episode_reward"].append(total_reward)
        history["goal_reached"].append(reached_goal)
        history["steps"].append(env.steps)
        history["coins_collected"].append(coins_this_ep)

    return history