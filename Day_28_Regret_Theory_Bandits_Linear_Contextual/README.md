# Day 28: Regret Theory & Bandits (Linear, Contextual)

## 🎯 Goal
Implement LinUCB and Thompson Sampling on contextual bandits.

## 📋 Task
- Implement Linear Upper Confidence Bound (LinUCB) algorithm
- Implement Thompson Sampling for linear bandits
- Compare regret bounds and empirical performance
- Test on contextual bandit problems with linear rewards

## 🔑 Key Concepts
- **Regret**: Difference between optimal and achieved cumulative reward
- **LinUCB**: Confidence bounds for linear bandit problems
- **Thompson Sampling**: Bayesian approach with posterior sampling
- **Contextual Bandits**: Actions depend on context/side information

## 📚 Learning Objectives
1. Understand regret minimization in bandit problems
2. Implement confidence bound and Bayesian algorithms
3. Analyze theoretical regret bounds
4. Compare exploration strategies empirically

## 🛠️ Implementation Guidelines
- Create linear contextual bandit environment
- Implement LinUCB with confidence ellipsoids
- Implement Thompson sampling with Gaussian priors
- Track cumulative regret and confidence bounds

## 📖 Resources
- A Contextual-Bandit Approach to Personalized News Article Recommendation (Li et al.)
- Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal & Goyal) 