# Day 16: PEARL (Probabilistic Embeddings)

## 🎯 Goal
Study PEARL and implement context encoder with latent policy adaptation.

## 📋 Task
- Implement PEARL with probabilistic context encoder
- Create multi-task environment (e.g., different goal locations)
- Test few-shot adaptation to new tasks
- Compare with RL² and standard multi-task learning

## 🔑 Key Concepts
- **PEARL**: Probabilistic embeddings for actor-critic reinforcement learning
- **Context Encoder**: Neural network that infers task from trajectories
- **Latent Context**: Probabilistic representation of task identity
- **VAE Framework**: Variational inference for context learning

## 📚 Learning Objectives
1. Understand probabilistic approach to meta-learning
2. Implement context inference from experience
3. Design multi-task environments for testing
4. Analyze adaptation quality vs number of context samples

## 🛠️ Implementation Guidelines
- Create family of navigation tasks with different goals
- Implement encoder that maps trajectories to latent context
- Use VAE loss for context learning
- Test adaptation with varying amounts of context data

## 📖 Resources
- Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables (Rakelly et al.)
- PEARL Implementation Guide 