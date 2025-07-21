# Day 10: World Models & Dreamer (Model-Based RL)

## 🎯 Goal
Implement simplified world model and policy learning in latent space.

## 📋 Task
- Implement a world model with VAE encoder/decoder
- Train policy to maximize rewards in imagination
- Compare model-based vs model-free performance
- Analyze model prediction quality

## 🔑 Key Concepts
- **World Model**: Learn environment dynamics in latent space
- **Dreamer**: Policy optimization in imagination
- **VAE**: Variational autoencoder for state representation
- **Dyna-style Planning**: Use model for additional training data

## 📚 Learning Objectives
1. Understand model-based RL advantages and challenges
2. Implement world model learning
3. Train policy using imagined rollouts
4. Analyze model bias and planning horizons

## 🛠️ Implementation Guidelines
- Use simple visual environment (e.g., CarRacing)
- Implement VAE for state encoding
- Train RNN dynamics model in latent space
- Use dreamed trajectories for policy learning

## 📖 Resources
- World Models (Ha & Schmidhuber)
- Dream to Control: Learning Behaviors by Latent Imagination (Hafner et al.) 