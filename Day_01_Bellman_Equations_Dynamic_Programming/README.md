# Day 1: Bellman Equations & Dynamic Programming

## 🎯 Goal
Ensure fluency in key RL concepts and derive fundamental equations for value-based methods.

## 📋 Task
- Derive Bellman optimality equations from first principles
- Implement Value Iteration algorithm in a gridworld environment
- Compare convergence properties and computational complexity

## 🔑 Key Concepts
- **Bellman Optimality Equation**: V*(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV*(s')]
- **Value Iteration**: Iterative algorithm to find optimal value function
- **Policy Iteration**: Alternative approach using policy evaluation and improvement
- **Dynamic Programming**: Solving complex problems by breaking them into subproblems

## 📚 Learning Objectives
1. Understand the mathematical foundation of value-based RL
2. Implement tabular dynamic programming algorithms
3. Analyze convergence guarantees and computational requirements
4. Visualize value function evolution during learning

## 🛠️ Implementation Guidelines
- Create a simple gridworld environment (e.g., 4x4 or 8x8)
- Implement Value Iteration with visualization
- Track and plot value function convergence
- Compare with Policy Iteration if time permits

## 📖 Resources
- Sutton & Barto Chapter 4: Dynamic Programming
- David Silver's RL Course Lecture 3 