# Day 14: Implicit Q-Learning (IQL)

## 🎯 Goal
Understand why IQL works and implement it on offline datasets.

## 📋 Task
- Implement Implicit Q-Learning algorithm
- Test on offline dataset (D4RL or custom)
- Compare with CQL and other offline methods
- Analyze implicit policy extraction and performance

## 🔑 Key Concepts
- **IQL**: Avoid explicit policy constraints through implicit Q-learning
- **Expectile Regression**: Alternative to quantile regression for value learning
- **Policy Extraction**: Extract policy from learned Q-function
- **Avoiding Distribution Shift**: Implicit approach to conservative learning

## 📚 Learning Objectives
1. Understand IQL's approach to offline RL
2. Implement expectile regression for value learning
3. Compare implicit vs explicit policy constraints
4. Analyze when IQL outperforms other methods

## 🛠️ Implementation Guidelines
- Implement expectile loss for value function learning
- Use separate networks for state value and Q-function
- Extract policy using advantage weighted regression
- Compare performance across different offline datasets

## 📖 Resources
- Implicit Q-Learning (Kostrikov et al.)
- Offline RL via Implicit Q-Learning 