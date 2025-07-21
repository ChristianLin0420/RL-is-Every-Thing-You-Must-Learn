# Day 5: Trust Region & PPO

## 🎯 Goal
Derive TRPO's constraint and implement PPO for stable policy updates.

## 📋 Task
- Derive Trust Region Policy Optimization (TRPO) constraint
- Implement Proximal Policy Optimization (PPO) from scratch
- Test on continuous control environments (e.g., Pendulum, MuJoCo)
- Compare clipped vs KL-penalty versions

## 🔑 Key Concepts
- **Trust Region**: Constrain policy updates to prevent performance collapse
- **KL Divergence**: KL(π_old || π_new) ≤ δ
- **PPO Clipping**: clip(r_t(θ), 1-ε, 1+ε) where r_t(θ) = π(a|s)/π_old(a|s)
- **Surrogate Objective**: Conservative policy improvement

## 📚 Learning Objectives
1. Understand motivation for trust region methods
2. Derive mathematical foundation of TRPO
3. Implement PPO as practical approximation to TRPO
4. Analyze clipping mechanism and its effects

## 🛠️ Implementation Guidelines
- Use continuous control environment (Pendulum-v1)
- Implement both actor and critic networks
- Use GAE from previous day for advantage estimation
- Track KL divergence and clipping statistics

## 📖 Resources
- Trust Region Policy Optimization (Schulman et al.)
- Proximal Policy Optimization Algorithms (Schulman et al.) 