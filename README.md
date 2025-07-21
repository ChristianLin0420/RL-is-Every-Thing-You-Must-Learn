# 🚀 RL is Everything You Must Learn: 30-Day Advanced RL Challenge

A comprehensive 30-day journey through advanced reinforcement learning concepts, from foundational theory to cutting-edge research frontiers.

## 🎯 Challenge Overview

This repository contains a structured 30-day curriculum designed to take you from solid RL foundations to the forefront of RL research. Each day focuses on a specific advanced topic with hands-on implementation tasks, theoretical understanding, and practical applications.

## 📅 Challenge Structure

### 🔰 Week 1: Recap & Solidifying Foundations (Days 1-7)
**Goal**: Ensure fluency in key RL concepts and math prerequisites for deeper theory

| Day | Topic | Focus |
|-----|-------|--------|
| 1 | [Bellman Equations & Dynamic Programming](Day_01_Bellman_Equations_Dynamic_Programming/) | Derive optimality equations, implement Value Iteration |
| 2 | [Model-Free Tabular Methods](Day_02_Model_Free_Tabular_Methods/) | SARSA vs Q-learning comparison |
| 3 | [Policy Gradient Theorem](Day_03_Policy_Gradient_Theorem/) | Derive REINFORCE from first principles |
| 4 | [Advantage Estimation (GAE)](Day_04_Advantage_Estimation_GAE/) | Bias-variance tradeoff in advantage estimation |
| 5 | [Trust Region & PPO](Day_05_Trust_Region_PPO/) | Derive TRPO constraint, implement PPO |
| 6 | [Actor-Critic Architectures](Day_06_Actor_Critic_Architectures/) | A2C with entropy regularization |
| 7 | [Deep Q-Learning & Variants](Day_07_Deep_Q_Learning_Variants/) | DQN, Double DQN, Dueling DQN on Atari |

### 🚀 Week 2: Exploration, Generalization, & Planning (Days 8-14)
**Goal**: Master exploration strategies, model-based RL, and long-horizon reasoning

| Day | Topic | Focus |
|-----|-------|--------|
| 8 | [Exploration: Count-Based, RND, UCB](Day_08_Exploration_Count_Based_RND_UCB/) | Advanced exploration on sparse-reward environments |
| 9 | [Curiosity & Intrinsic Rewards](Day_09_Curiosity_Intrinsic_Rewards/) | ICM and empowerment-based exploration |
| 10 | [World Models & Dreamer](Day_10_World_Models_Dreamer/) | Model-based RL with latent dynamics |
| 11 | [Planning with Value Prediction Networks](Day_11_Planning_Value_Prediction_Networks/) | Value prediction vs model rollout |
| 12 | [MuZero (Planning Without a Model)](Day_12_MuZero_Planning_Without_Model/) | MCTS with learned representations |
| 13 | [Offline RL & Conservative Q-Learning](Day_13_Offline_RL_Conservative_Q_Learning/) | CQL on D4RL benchmark |
| 14 | [Implicit Q-Learning (IQL)](Day_14_Implicit_Q_Learning/) | Implicit approach to offline RL |

### 🧠 Week 3: Meta-RL, Multi-Agent RL, and Representation Learning (Days 15-21)
**Goal**: Learn generalization, few-shot learning, and interactions between agents

| Day | Topic | Focus |
|-----|-------|--------|
| 15 | [Meta-RL via RL²](Day_15_Meta_RL_via_RL2/) | Fast adaptation with recurrent policies |
| 16 | [PEARL (Probabilistic Embeddings)](Day_16_PEARL_Probabilistic_Embeddings/) | Context encoder for meta-learning |
| 17 | [Multi-Agent Basics (Self-Play)](Day_17_Multi_Agent_Basics_Self_Play/) | Independent learners in shared environments |
| 18 | [MADDPG & Actor-Critic in MARL](Day_18_MADDPG_Actor_Critic_MARL/) | Centralized training, decentralized execution |
| 19 | [Mean Field RL or MAPPO](Day_19_Mean_Field_RL_MAPPO/) | Scalable multi-agent policy gradients |
| 20 | [Contrastive & Predictive State Representation](Day_20_Contrastive_Predictive_State_Representation/) | Self-supervised representation learning |
| 21 | [Information Bottleneck in RL](Day_21_Information_Bottleneck_RL/) | Information-theoretic policy constraints |

### 🧪 Week 4: Research & Advanced Topics (Days 22-30)
**Goal**: Engage with the frontier of RL research and prepare to contribute

| Day | Topic | Focus |
|-----|-------|--------|
| 22 | [RLHF (Reinforcement Learning with Human Feedback)](Day_22_RLHF_Human_Feedback/) | Preference-based reward modeling |
| 23 | [Hierarchical RL (Option Critic / HIRO)](Day_23_Hierarchical_RL_Option_Critic_HIRO/) | Temporal abstraction and options |
| 24 | [Diffusion Models in Planning](Day_24_Diffusion_Models_Planning_Diffuser/) | Trajectory generation with diffusion |
| 25 | [Transformer in RL (Decision Transformer)](Day_25_Transformer_RL_Decision_Transformer/) | Sequence modeling for RL |
| 26 | [Imitation Learning: GAIL & D-REX](Day_26_Imitation_Learning_GAIL_D_REX/) | Learning from demonstrations |
| 27 | [Causal RL & Generalization](Day_27_Causal_RL_Generalization/) | Causal inference in RL |
| 28 | [Regret Theory & Bandits](Day_28_Regret_Theory_Bandits_Linear_Contextual/) | LinUCB and Thompson Sampling |
| 29 | [Equivariant RL & Symmetry-Aware Policies](Day_29_Equivariant_RL_Symmetry_Aware_Policies/) | Group equivariance for sample efficiency |
| 30 | [Temporal Abstraction via Diffusion or VLM Planners](Day_30_Temporal_Abstraction_Diffusion_VLM_Planners/) | Modern hierarchical planning approaches |

## 🛠️ Prerequisites

### Mathematical Background
- **Linear Algebra**: Matrix operations, eigenvalues, SVD
- **Probability Theory**: Distributions, expectation, Bayes' theorem
- **Calculus**: Gradients, chain rule, optimization
- **Basic Statistics**: Confidence intervals, hypothesis testing

### Programming Skills
- **Python**: Intermediate to advanced level
- **Deep Learning Frameworks**: PyTorch or TensorFlow
- **Scientific Computing**: NumPy, SciPy, Matplotlib
- **RL Libraries**: OpenAI Gym, Stable-Baselines3 (recommended)

### RL Foundations
- Basic understanding of MDPs, value functions, and policy gradients
- Familiarity with Q-learning and policy gradient methods
- Experience with at least one deep RL algorithm (DQN, PPO, etc.)

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ChristianLin0420/RL-is-Every-Thing-You-Must-Learn.git
cd RL-is-Every-Thing-You-Must-Learn

# Create virtual environment
python -m venv rl_challenge_env
source rl_challenge_env/bin/activate  # On Windows: rl_challenge_env\Scripts\activate

# Install dependencies
pip install torch torchvision gymnasium stable-baselines3 d4rl wandb matplotlib seaborn
```

### 2. Daily Structure
Each day's folder contains:
- **README.md**: Detailed task description, concepts, and resources
- **Implementation space**: Where you'll write your code
- **Evaluation metrics**: How to measure success

### 3. Recommended Approach
1. **Morning (30-45 min)**: Read theory and understand key concepts
2. **Implementation (2-3 hours)**: Code the algorithm from scratch
3. **Experimentation (30-60 min)**: Test, analyze, and compare results
4. **Reflection (15 min)**: Document learnings and insights

## 📊 Progress Tracking

### Weekly Milestones
- **Week 1**: Solid foundation in core RL algorithms
- **Week 2**: Understanding of exploration and planning
- **Week 3**: Multi-agent and meta-learning competency
- **Week 4**: Familiarity with cutting-edge research directions

### Success Metrics
- [ ] Successfully implement and run each day's algorithm
- [ ] Understand mathematical derivations and proofs
- [ ] Achieve reasonable performance on benchmark tasks
- [ ] Write clean, documented, and reusable code
- [ ] Compare results with published benchmarks

## 🎓 Learning Resources

### Essential Textbooks
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2nd Edition)
- **Bertsekas**: "Dynamic Programming and Optimal Control"
- **Puterman**: "Markov Decision Processes"

### Online Courses
- **David Silver's RL Course** (DeepMind/UCL)
- **CS285 Deep RL** (UC Berkeley)
- **OpenAI Spinning Up in Deep RL**

### Research Venues
- **Conferences**: NeurIPS, ICML, ICLR, AAMAS
- **Journals**: JMLR, MLJ, Autonomous Agents and Multi-Agent Systems
- **Workshops**: Deep RL Workshop, Multi-Agent RL

## 🤝 Community & Support

### Discussion and Help
- Open issues for technical questions
- Share implementations and results
- Collaborate on challenging days
- Provide feedback and improvements

### Contributing
- Add alternative implementations
- Improve documentation and explanations
- Share additional resources and references
- Report bugs or suggest enhancements

## 📈 Beyond the Challenge

### Next Steps
1. **Research Projects**: Identify interesting research directions
2. **Open Source Contributions**: Contribute to RL libraries
3. **Paper Implementations**: Reproduce recent research papers
4. **Real-World Applications**: Apply RL to practical problems

### Career Paths
- **Research Scientist**: Academic or industry research
- **ML Engineer**: Production RL systems
- **Data Scientist**: RL for business applications
- **Robotics Engineer**: Embodied AI and control

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the RL research community
- Built upon foundational work by Sutton, Barto, and countless researchers
- Thanks to OpenAI, DeepMind, and other organizations for open-sourcing environments and algorithms

---

**Ready to begin your advanced RL journey? Start with [Day 1: Bellman Equations & Dynamic Programming](Day_01_Bellman_Equations_Dynamic_Programming/)!** 🚀 