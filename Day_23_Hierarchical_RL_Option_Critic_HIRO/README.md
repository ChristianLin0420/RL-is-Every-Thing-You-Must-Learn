# Day 23: Hierarchical RL (Option Critic / HIRO)

## 🎯 Goal
Implement option-critic or HIRO for goal-conditioned tasks.

## 📋 Task
- Implement Option-Critic algorithm with hierarchical policies
- Create goal-conditioned navigation environment
- Compare with flat RL on long-horizon tasks
- Analyze learned option discovery and temporal abstraction

## 🔑 Key Concepts
- **Options**: Temporally extended actions (sub-policies)
- **Option-Critic**: End-to-end learning of options and meta-policy
- **HIRO**: Hierarchical RL with off-policy correction
- **Temporal Abstraction**: Learning at multiple time scales

## 📚 Learning Objectives
1. Understand hierarchical learning in RL
2. Implement option discovery algorithms
3. Analyze temporal abstraction benefits
4. Study goal-conditioned hierarchical policies

## 🛠️ Implementation Guidelines
- Implement option-critic with termination functions
- Create multi-room navigation or manipulation task
- Add goal conditioning to environments
- Visualize learned options and their usage

## 📖 Resources
- The Option-Critic Architecture (Bacon et al.)
- Data-Efficient Hierarchical RL (Nachum et al.) 