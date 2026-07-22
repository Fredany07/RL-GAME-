"""Generate plots for the reward-hacking case study."""

import pickle
import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld

MODE_LABELS = {
    "sparse": "Sparse (baseline)",
    "distance": "Naive distance-shaped",
    "coin": "Proxy coin-shaped (HACKED)",
    "potential": "Potential-based (fixed)",
}
MODE_COLORS = {
    "sparse": "#4C72B0",
    "distance": "#55A868",
    "coin": "#C44E52",
    "potential": "#8172B2",
}


def smooth(x, window=25):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_goal_reach_rate(results, out_path="goal_reach_rate.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, r in results.items():
        y = smooth(r["history"]["goal_reached"], window=25)
        x = np.arange(len(y))
        ax.plot(x, y, label=MODE_LABELS[mode], color=MODE_COLORS[mode], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Goal-reach rate (25-episode rolling avg)")
    ax.set_title("Does the agent actually reach the true goal?")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coins_collected(results, out_path="coins_collected.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, r in results.items():
        y = smooth(r["history"]["coins_collected"], window=25)
        x = np.arange(len(y))
        ax.plot(x, y, label=MODE_LABELS[mode], color=MODE_COLORS[mode], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Coins collected per episode (rolling avg)")
    ax.set_title("Coin-farming behavior by reward scheme")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_episode_reward(results, out_path="episode_reward.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, r in results.items():
        y = smooth(r["history"]["episode_reward"], window=25)
        x = np.arange(len(y))
        ax.plot(x, y, label=MODE_LABELS[mode], color=MODE_COLORS[mode], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward (as optimized, rolling avg)")
    ax.set_title("Training reward under each scheme (not comparable in scale across schemes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_policy_grids(results, out_path="policy_grids.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    modes = ["sparse", "distance", "coin", "potential"]

    for ax, mode in zip(axes.flat, modes):
        r = results[mode]
        env = GridWorld(reward_mode=mode)
        Q = r["Q"]

        # Draw grid
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                pos = (row, col)
                if pos in env.walls:
                    ax.add_patch(plt.Rectangle((col, env.n_rows - 1 - row), 1, 1,
                                                 facecolor="#333333"))
                elif pos in env.coins:
                    ax.add_patch(plt.Rectangle((col, env.n_rows - 1 - row), 1, 1,
                                                 facecolor="#FFD700", alpha=0.5))
                elif pos == env.goal:
                    ax.add_patch(plt.Rectangle((col, env.n_rows - 1 - row), 1, 1,
                                                 facecolor="#90EE90", alpha=0.7))
                elif pos == env.start:
                    ax.add_patch(plt.Rectangle((col, env.n_rows - 1 - row), 1, 1,
                                                 facecolor="#ADD8E6", alpha=0.7))

        # Draw greedy policy arrows on free cells
        arrow_map = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}  # dx, dy in plot coords
        for row in range(env.n_rows):
            for col in range(env.n_cols):
                pos = (row, col)
                if pos in env.walls or pos == env.goal:
                    continue
                action = int(np.argmax(Q[row, col]))
                dx, dy = arrow_map[action]
                cx, cy = col + 0.5, env.n_rows - 1 - row + 0.5
                ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        # Overlay the actual evaluated path
        path = r["eval_path"]
        xs = [p[1] + 0.5 for p in path]
        ys = [env.n_rows - 1 - p[0] + 0.5 for p in path]
        ax.plot(xs, ys, color="red", linewidth=2, alpha=0.6, zorder=5)

        ax.set_xlim(0, env.n_cols)
        ax.set_ylim(0, env.n_rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        goal_status = "reached goal" if r["eval_reached_goal"] else "NEVER reached goal"
        ax.set_title(f"{MODE_LABELS[mode]}\n({goal_status}, path shown in red)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    plot_goal_reach_rate(results)
    plot_coins_collected(results)
    plot_episode_reward(results)
    plot_policy_grids(results)

    print("Saved: goal_reach_rate.png, coins_collected.png, episode_reward.png, policy_grids.png")