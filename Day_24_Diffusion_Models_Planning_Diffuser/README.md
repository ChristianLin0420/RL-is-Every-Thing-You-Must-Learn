# Day 24: Diffusion Models in Planning (e.g., Diffuser)

## 🎯 Goal
Study how diffusion models generate trajectories and implement simplified Diffuser.

## 📋 Task
- Implement diffusion model for trajectory generation
- Train on offline datasets to learn trajectory distributions
- Use diffusion for planning in control tasks
- Compare with model-based and model-free methods

## 🔑 Key Concepts
- **Diffusion Models**: Generative models via denoising process
- **Trajectory Diffusion**: Apply diffusion to sequence generation
- **Planning as Inference**: Frame planning as sampling problem
- **Guided Generation**: Condition generation on rewards/constraints

## 📚 Learning Objectives
1. Understand diffusion models for sequential data
2. Implement trajectory diffusion for planning
3. Compare diffusion planning with other approaches
4. Analyze sample quality and diversity

## 🛠️ Implementation Guidelines
- Implement simplified diffusion model for trajectories
- Train on D4RL or collected trajectory data
- Add reward guidance for directed planning
- Test on navigation or control environments

## 📖 Resources
- Planning with Diffusion for Flexible Behavior Synthesis (Janner et al.)
- Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol) 