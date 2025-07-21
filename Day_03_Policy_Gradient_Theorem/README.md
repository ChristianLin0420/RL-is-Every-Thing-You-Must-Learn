# Day 3: Policy Gradient Theorem

## 🎯 Goal
Derive and implement the fundamental policy gradient theorem from first principles.

## 📋 Task
- Derive the REINFORCE algorithm from the policy gradient theorem
- Implement basic policy gradient on CartPole environment
- Understand the high variance problem and baseline techniques
- Visualize policy parameter updates

## 🔑 Key Concepts
- **Policy Gradient Theorem**: ∇J(θ) = E[∇log π(a|s,θ) Q^π(s,a)]
- **REINFORCE**: Monte Carlo policy gradient with return as baseline
- **Score Function**: ∇log π(a|s,θ) for gradient estimation
- **Baseline**: Variance reduction techniques (e.g., state-value baseline)

## 📚 Learning Objectives
1. Derive policy gradient theorem step by step
2. Understand why policy gradients have high variance
3. Implement REINFORCE with and without baseline
4. Compare with value-based methods

## 🛠️ Implementation Guidelines
- Use CartPole-v1 environment from OpenAI Gym
- Implement neural network policy (softmax for discrete actions)
- Use PyTorch or TensorFlow for automatic differentiation
- Track gradient norms and policy entropy

## 📖 Resources
- Sutton & Barto Chapter 13: Policy Gradient Methods
- Policy Gradient Methods for RL with Function Approximation (Sutton et al.) 