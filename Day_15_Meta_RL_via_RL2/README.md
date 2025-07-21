# Day 15: Meta-RL via RL²

## 🎯 Goal
Implement RL² framework for fast adaptation to new tasks.

## 📋 Task
- Implement RL² using GRU for memory
- Test on multi-armed bandit tasks with changing rewards
- Compare few-shot learning performance
- Analyze meta-learning vs standard RL

## 🔑 Key Concepts
- **RL²**: RL algorithm that learns to learn quickly
- **Meta-Learning**: Learning across multiple tasks for fast adaptation
- **Recurrent Policies**: Use LSTM/GRU to maintain task memory
- **Task Distribution**: Train on family of related tasks

## 📚 Learning Objectives
1. Understand meta-learning in RL context
2. Implement recurrent policies for task adaptation
3. Design task distributions for meta-training
4. Analyze adaptation speed and generalization

## 🛠️ Implementation Guidelines
- Create family of multi-armed bandit tasks
- Implement recurrent policy with GRU memory
- Train on task distribution with curriculum
- Test adaptation speed on new tasks

## 📖 Resources
- RL²: Fast Reinforcement Learning via Slow Reinforcement Learning (Duan et al.)
- Learning to Reinforcement Learn (Wang et al.) 