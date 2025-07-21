# Day 20: Contrastive & Predictive State Representation

## 🎯 Goal
Implement CPC-style encoder for learning state representations in RL.

## 📋 Task
- Implement Contrastive Predictive Coding for RL
- Learn state representations from pixel observations
- Test representation quality on control tasks
- Compare with other representation learning methods

## 🔑 Key Concepts
- **CPC**: Contrastive Predictive Coding for representation learning
- **InfoNCE Loss**: Contrastive learning objective
- **Temporal Coherence**: Representations should capture temporal structure
- **Self-Supervised Learning**: Learn without explicit labels

## 📚 Learning Objectives
1. Understand contrastive learning in RL context
2. Implement InfoNCE loss for temporal prediction
3. Analyze learned representations and their quality
4. Compare with VAE and other representation methods

## 🛠️ Implementation Guidelines
- Use visual environment (Atari or control with images)
- Implement encoder network with contrastive loss
- Train representation learning alongside RL agent
- Visualize learned representations and their temporal structure

## 📖 Resources
- Data-Efficient Reinforcement Learning with Self-Predictive Representations (Schwarzer et al.)
- Representation Learning with Contrastive Predictive Coding (van den Oord et al.) 