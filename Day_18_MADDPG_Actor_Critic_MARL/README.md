# Day 18: MADDPG & Actor-Critic in MARL

## 🎯 Goal
Implement MADDPG on cooperative task using centralized training.

## 📋 Task
- Implement Multi-Agent Deep Deterministic Policy Gradient
- Test on cooperative particle environment
- Compare centralized vs decentralized training
- Analyze coordination and communication benefits

## 🔑 Key Concepts
- **MADDPG**: Centralized training, decentralized execution
- **Centralized Critic**: Access to all agents' observations during training
- **Decentralized Actor**: Each agent acts based on local observations
- **Cooperative Tasks**: Shared rewards and coordination requirements

## 📚 Learning Objectives
1. Understand centralized training paradigm
2. Implement multi-agent actor-critic algorithms
3. Analyze benefits of centralized information during training
4. Study coordination mechanisms in MARL

## 🛠️ Implementation Guidelines
- Use particle environment or simple cooperative task
- Implement centralized critic with global state information
- Keep actors decentralized for execution
- Track individual and team performance metrics

## 📖 Resources
- Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (Lowe et al.)
- OpenAI Multi-Agent Particle Environments 