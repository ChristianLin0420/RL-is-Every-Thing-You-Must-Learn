# Day 4: Advantage Estimation (GAE)

## 🎯 Goal
Understand bias-variance tradeoff in advantage estimation and implement GAE.

## 📋 Task
- Implement Generalized Advantage Estimation (GAE)
- Integrate GAE into a policy gradient agent
- Compare different λ values and their effect on learning
- Analyze bias-variance tradeoff empirically

## 🔑 Key Concepts
- **Advantage Function**: A(s,a) = Q(s,a) - V(s)
- **GAE**: A^GAE(γ,λ) = Σ(γλ)^l δ_{t+l}
- **TD Error**: δ_t = r_t + γV(s_{t+1}) - V(s_t)
- **Bias-Variance Tradeoff**: λ=0 (high bias, low variance) vs λ=1 (low bias, high variance)

## 📚 Learning Objectives
1. Understand why advantage estimation is crucial for policy gradients
2. Derive and implement GAE formula
3. Analyze effect of λ parameter on learning stability
4. Compare with simple advantage estimation methods

## 🛠️ Implementation Guidelines
- Extend previous policy gradient implementation
- Implement value function network for baseline
- Experiment with different λ values (0.0, 0.9, 0.95, 1.0)
- Plot learning curves and advantage estimates

## 📖 Resources
- High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al.)
- Actor-Critic methods with GAE 