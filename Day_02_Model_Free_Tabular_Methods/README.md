# Day 2: Model-Free Tabular Methods

## 🎯 Goal
Master temporal difference learning algorithms that don't require environment models.

## 📋 Task
- Implement SARSA (on-policy TD control)
- Implement Q-learning (off-policy TD control)
- Compare performance on CliffWalking or FrozenLake environments
- Analyze exploration vs exploitation tradeoffs

## 🔑 Key Concepts
- **SARSA**: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **Q-Learning**: Q(s,a) ← Q(s,a) + α[r + γmax_a'Q(s',a') - Q(s,a)]
- **ε-greedy exploration**: Balance between exploitation and exploration
- **Temporal Difference**: Learning from one-step lookahead

## 📚 Learning Objectives
1. Understand difference between on-policy and off-policy learning
2. Implement and compare SARSA vs Q-learning
3. Analyze the role of exploration strategies
4. Visualize learning curves and convergence behavior

## 🛠️ Implementation Guidelines
- Use OpenAI Gym's CliffWalking-v0 or FrozenLake-v1
- Implement both algorithms with ε-greedy exploration
- Track episode rewards and Q-table evolution
- Create comparison plots and analysis

## 📖 Resources
- Sutton & Barto Chapter 6: Temporal-Difference Learning
- Comparison of TD learning algorithms 