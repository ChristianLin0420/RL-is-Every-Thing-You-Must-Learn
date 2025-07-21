# Day 6: Actor-Critic Architectures

## 🎯 Goal
Implement A2C and understand entropy regularization for exploration.

## 📋 Task
- Implement Advantage Actor-Critic (A2C) algorithm
- Add entropy regularization to encourage exploration
- Compare synchronous vs asynchronous implementations
- Test on both discrete and continuous action spaces

## 🔑 Key Concepts
- **Actor-Critic**: Combined policy (actor) and value function (critic)
- **A2C**: Synchronous version of A3C with advantage estimation
- **Entropy Regularization**: H(π) = -Σ π(a|s) log π(a|s)
- **Shared vs Separate Networks**: Architecture design choices

## 📚 Learning Objectives
1. Understand actor-critic framework advantages
2. Implement stable gradient updates for both networks
3. Analyze role of entropy bonus in exploration
4. Compare with pure policy gradient methods

## 🛠️ Implementation Guidelines
- Implement shared network with separate heads
- Use parallel environments for sample efficiency
- Add entropy loss term to policy loss
- Track both policy and value losses

## 📖 Resources
- Asynchronous Methods for Deep RL (Mnih et al.)
- Actor-Critic Algorithms (Konda & Tsitsiklis) 