# Day 8: Exploration: Count-Based, RND, UCB

## 🎯 Goal
Implement advanced exploration strategies and analyze exploration quality.

## 📋 Task
- Implement Random Network Distillation (RND) or UCB exploration
- Test on sparse-reward maze environments
- Analyze exploration quality and coverage
- Compare with ε-greedy baseline

## 🔑 Key Concepts
- **Count-Based Exploration**: Use visitation counts for bonuses
- **RND**: Train predictor network on random target network
- **UCB**: Upper Confidence Bound for action selection
- **Intrinsic Motivation**: Internal reward signals for exploration

## 📚 Learning Objectives
1. Understand limitations of simple exploration strategies
2. Implement sophisticated exploration bonuses
3. Analyze exploration-exploitation tradeoff
4. Measure state visitation and coverage

## 🛠️ Implementation Guidelines
- Create maze environment with sparse rewards
- Implement RND with prediction error as bonus
- Track state visitation heatmaps
- Compare final performance and exploration efficiency

## 📖 Resources
- Exploration by Random Network Distillation (Burda et al.)
- Count-Based Exploration with Neural Density Models (Ostrovski et al.) 