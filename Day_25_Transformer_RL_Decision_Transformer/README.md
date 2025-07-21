# Day 25: Transformer in RL (Decision Transformer)

## 🎯 Goal
Train Decision Transformer on D4RL and compare with CQL/BC.

## 📋 Task
- Implement Decision Transformer for offline RL
- Train on D4RL dataset with return-to-go conditioning
- Compare with Conservative Q-Learning and Behavior Cloning
- Analyze sequence modeling vs value-based approaches

## 🔑 Key Concepts
- **Decision Transformer**: Frame RL as sequence modeling problem
- **Return-to-Go**: Condition on desired future returns
- **Autoregressive Modeling**: Predict next action given history
- **Offline Sequence Learning**: Learn from static trajectory datasets

## 📚 Learning Objectives
1. Understand sequence modeling approach to RL
2. Implement transformer architecture for decisions
3. Compare with traditional RL algorithms
4. Analyze scaling properties and limitations

## 🛠️ Implementation Guidelines
- Implement GPT-style transformer for (state, action, return) sequences
- Use return-to-go conditioning for goal specification
- Train on D4RL MuJoCo environments
- Compare performance with CQL and behavioral cloning

## 📖 Resources
- Decision Transformer: Reinforcement Learning via Sequence Modeling (Chen et al.)
- Trajectory Transformer: Sequence Modeling for RL 