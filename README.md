Reward Shaping & Reward Hacking: A GridWorld Case Study
Motivation
A core problem in AI safety is reward hacking (a.k.a. specification gaming): an agent optimizes the literal reward function it's given, which can diverge sharply from what the designer actually intended. This project builds a small, fully-controllable environment to reproduce this failure mode on purpose, then fixes it using a technique with theoretical guarantees.
Environment
A 7x7 GridWorld:
. . . . . . .
. # # . . . .
. . . C . . .
S . # # . . .
. C . . . . .
. . . . . . .
. . . . . . G
S — start
G — true goal (only this square represents "success")
# — walls
C — coins, infinitely re-collectible (this is the key design choice that makes hacking possible: an agent can sit near a coin and farm it forever instead of ever finishing)
Agent
Tabular Q-learning (epsilon-greedy, decaying exploration), trained for 2000 episodes per reward scheme. Tabular Q-learning is deliberately simple so that any behavior differences are attributable to reward design, not to network instability or hyperparameter noise.
Reward Schemes Compared
Scheme	Formula	Intent
Sparse	+10 on goal, -0.1/step	Ground truth: only the real objective is rewarded
Naive distance-shaped	sparse + (prev_dist - new_dist) (Manhattan)	A common, intuitive shaping heuristic
Proxy coin-shaped	sparse + 5 per coin touched	Rewards a proxy metric correlated with progress, but not identical to it
Potential-based (fixed)	sparse + (γ·Φ(s') - Φ(s)), Φ(s) = -manhattan_dist(s, goal)	Ng, Harada & Russell (1999)'s potential-based shaping — provably preserves the optimal policy
Results
The coin-reward agent hacks the reward. Its final greedy policy collects 49 coins in 100 steps and never reaches the true goal at all. Because coins are infinitely re-collectible and one coin sits in a low-cost pocket near the start, the optimal policy under this reward function is to loop between coin tiles forever — exactly what the agent learns to do. The goal-reach rate for this scheme stays near 0% throughout training.
Sparse, naive distance-shaping, and potential-based shaping all converge to the optimal 9-step path to the goal, reaching a ~100% goal-reach rate. In this particular grid the naive distance-shaping heuristic happens not to break (it doesn't create a competing local optimum strong enough to beat reaching the goal) — but it has no guarantee of this in general, which is exactly why potential-based shaping exists: it's the version of "add a shortcut reward" that comes with a proof that it cannot change the optimal policy, regardless of the coin trap.
See the generated plots:
goal_reach_rate.png — clearest single result: 3 schemes hit 100%, the coin scheme stays near 0%.
coins_collected.png — the coin-reward agent's coin-farming behavior visibly diverges from the other three.
episode_reward.png — raw training curves (note: reward scales differ across schemes and are not directly comparable to each other).
policy_grids.png — the most intuitive visual: arrows show the learned policy at every cell, and the red line traces the actual greedy path. Three panels show a clean path to G; the coin panel shows a tangled scribble that never leaves the coin cluster.
Connection to Real-World Reward Hacking
This toy example is a miniature of documented real-world cases:
Reinforcement learning agents in game environments that farm points from a glitch or repeatable sub-task instead of completing the level, because the score (proxy) diverged from actual task completion.
Content recommendation systems optimized for engagement metrics (proxy) that drift toward addictive or polarizing content rather than genuine user satisfaction (true goal).
Any system where a measurable proxy is optimized as a stand-in for an unmeasurable true objective — Goodhart's Law: "when a measure becomes a target, it ceases to be a good measure."
Potential-based reward shaping is one of the few techniques in this space with a formal guarantee (not just an empirical patch): as long as the shaping term takes the form F(s,a,s') = γΦ(s') - Φ(s) for any function Φ, the set of optimal policies is provably unchanged from the unshaped MDP. This project demonstrates that guarantee empirically alongside a scheme that has no such guarantee and fails.
Files
gridworld.py — environment definition and reward logic
q_learning.py — tabular Q-learning agent and training loop
run_experiments.py — trains all four schemes, saves results.pkl / summary.json
visualize.py — generates all four plots from results.pkl
results.pkl, summary.json — saved experiment results
*.png — generated plots
Reproducing
bash
python3 run_experiments.py   # trains all 4 agents, saves results.pkl
python3 visualize.py         # generates plots from results.pkl
Possible Extensions
Swap tabular Q-learning for DQN and see if hacking is more/less pronounced with function approximation.
Make coins disappear after collection and compare hacking severity.
Add a second competing proxy (e.g. speed bonus) to study multi-objective reward hacking.
Try reward schemes discovered adversarially (e.g. via genetic search) to find the "worst" possible naive shaping term.
