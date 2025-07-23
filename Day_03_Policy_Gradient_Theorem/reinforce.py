"""
REINFORCE Algorithm Implementation

This module implements the REINFORCE algorithm (Williams, 1992) from first principles.
REINFORCE is the simplest policy gradient method that uses Monte Carlo sampling
to estimate policy gradients.

The key insight is the Policy Gradient Theorem:
∇_θ J(θ) = E_τ~π_θ [∑_t ∇_θ log π_θ(a_t|s_t) R_t]

Where:
- J(θ) is the expected return under policy π_θ
- R_t is the return (cumulative reward) from time step t
- The expectation is over trajectories τ sampled from the policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from collections import deque
import matplotlib.pyplot as plt

from policy_network import PolicyNetwork


class TrajectoryBuffer:
    """Buffer to store trajectory data for REINFORCE updates."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.action_probs = []
        
    def store(self, state: np.ndarray, action: int, reward: float, 
              log_prob: torch.Tensor, action_prob: torch.Tensor):
        """Store a single transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.action_probs.append(action_prob.detach().clone())
        
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.action_probs.clear()
        
    def __len__(self):
        return len(self.states)


class REINFORCE:
    """
    REINFORCE Algorithm Implementation.
    
    This implements the vanilla REINFORCE algorithm with optional baseline
    for variance reduction. The algorithm:
    1. Collects full episodes (trajectories)
    2. Computes returns for each time step
    3. Updates policy using policy gradient theorem
    """
    
    def __init__(self,
                 policy: PolicyNetwork,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 use_baseline: bool = False,
                 baseline_lr: float = 1e-2,
                 optimizer: str = 'adam',
                 max_grad_norm: Optional[float] = None,
                 seed: Optional[int] = None):
        """
        Initialize REINFORCE agent.
        
        Args:
            policy: Policy network to optimize
            learning_rate: Learning rate for policy optimizer
            gamma: Discount factor for returns
            use_baseline: Whether to use baseline for variance reduction
            baseline_lr: Learning rate for baseline (if used)
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
            max_grad_norm: Maximum gradient norm for clipping
            seed: Random seed for reproducibility
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        self.policy = policy
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.max_grad_norm = max_grad_norm
        
        # Set up optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate)
        
        # Baseline (simple moving average or learned value function)
        if use_baseline:
            self.baseline_values = deque(maxlen=100)  # Moving average baseline
            self.baseline_lr = baseline_lr
            self.running_baseline = 0.0
            
        # Trajectory buffer
        self.buffer = TrajectoryBuffer()
        
        # Training statistics
        self.episode_count = 0
        self.training_stats = {
            'episode_returns': [],
            'episode_lengths': [],
            'policy_losses': [],
            'gradient_norms': [],
            'entropy_values': [],
            'baseline_values': [],
            'action_distributions': []
        }
        
    def _create_optimizer(self, optimizer_type: str, learning_rate: float) -> optim.Optimizer:
        """Create the specified optimizer."""
        optimizers = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        if optimizer_type.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
            
        return optimizers[optimizer_type.lower()](self.policy.parameters(), lr=learning_rate)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Tuple of (action, log_probability, action_probabilities)
        """
        state_tensor = torch.FloatTensor(state)
        return self.policy.sample_action(state_tensor)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        log_prob: torch.Tensor, action_probs: torch.Tensor):
        """Store a transition in the trajectory buffer."""
        self.buffer.store(state, action, reward, log_prob, action_probs)
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """
        Compute discounted returns for a trajectory.
        
        Args:
            rewards: List of rewards for the episode
            
        Returns:
            List of returns (cumulative discounted rewards)
        """
        returns = []
        R = 0
        
        # Compute returns backwards (more numerically stable)
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
            
        return returns
    
    def compute_baseline(self, returns: List[float]) -> float:
        """
        Compute baseline value for variance reduction.
        
        Args:
            returns: Episode returns
            
        Returns:
            Baseline value
        """
        if not self.use_baseline:
            return 0.0
            
        episode_return = returns[0]  # Total episode return
        self.baseline_values.append(episode_return)
        
        # Update running baseline using exponential moving average
        if len(self.baseline_values) == 1:
            self.running_baseline = episode_return
        else:
            self.running_baseline = (1 - self.baseline_lr) * self.running_baseline + \
                                   self.baseline_lr * episode_return
                                   
        return self.running_baseline
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using REINFORCE algorithm.
        
        Returns:
            Dictionary with training statistics
        """
        if len(self.buffer) == 0:
            return {}
            
        # Compute returns
        returns = self.compute_returns(self.buffer.rewards)
        returns_tensor = torch.FloatTensor(returns)
        
        # Compute baseline
        baseline = self.compute_baseline(returns)
        
        # Center returns (variance reduction)
        if self.use_baseline:
            advantages = returns_tensor - baseline
        else:
            advantages = returns_tensor
            
        # Normalize advantages (optional, can help with training stability)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy gradients
        policy_loss = 0
        entropy_loss = 0
        
        for t, (log_prob, advantage) in enumerate(zip(self.buffer.log_probs, advantages)):
            # REINFORCE gradient: ∇_θ log π_θ(a_t|s_t) * R_t
            policy_loss -= log_prob * advantage
            
            # Optional: Add entropy bonus for exploration
            state_tensor = torch.FloatTensor(self.buffer.states[t])
            entropy = self.policy.get_entropy(state_tensor)
            entropy_loss -= 0.01 * entropy  # Small entropy coefficient
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (optional)
        grad_norm = 0.0
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 
                                                      self.max_grad_norm)
        else:
            # Calculate gradient norm for monitoring
            grad_norm = sum(p.grad.norm().item() ** 2 for p in self.policy.parameters() 
                           if p.grad is not None) ** 0.5
        
        # Update parameters
        self.optimizer.step()
        
        # Store statistics
        episode_return = sum(self.buffer.rewards)
        episode_length = len(self.buffer)
        
        stats = {
            'episode_return': episode_return,
            'episode_length': episode_length,
            'policy_loss': policy_loss.item(),
            'gradient_norm': grad_norm,
            'baseline_value': baseline,
            'average_entropy': entropy_loss.item() / len(self.buffer),
        }
        
        # Store action distribution for analysis
        action_dist = torch.stack(self.buffer.action_probs).mean(dim=0)
        stats['action_distribution'] = action_dist.squeeze().detach().numpy()  # Remove any extra dimensions
        
        # Update training statistics
        self.training_stats['episode_returns'].append(episode_return)
        self.training_stats['episode_lengths'].append(episode_length)
        self.training_stats['policy_losses'].append(policy_loss.item())
        self.training_stats['gradient_norms'].append(grad_norm)
        self.training_stats['entropy_values'].append(entropy_loss.item() / len(self.buffer))
        self.training_stats['baseline_values'].append(baseline)
        self.training_stats['action_distributions'].append(action_dist.detach().numpy())
        
        # Clear buffer for next episode
        self.buffer.clear()
        self.episode_count += 1
        
        return stats
    
    def get_training_stats(self) -> Dict[str, List]:
        """Get all training statistics."""
        return self.training_stats.copy()
    
    def save_checkpoint(self, filepath: str):
        """Save agent checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
            'use_baseline': self.use_baseline,
            'running_baseline': getattr(self, 'running_baseline', 0.0)
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        self.episode_count = checkpoint['episode_count']
        self.gamma = checkpoint['gamma']
        self.use_baseline = checkpoint['use_baseline']
        if 'running_baseline' in checkpoint:
            self.running_baseline = checkpoint['running_baseline']
    
    def evaluate_policy(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the current policy.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation statistics
        """
        total_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _, _ = self.select_action(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_length': np.mean(episode_lengths)
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"REINFORCE(episodes={self.episode_count}, " \
               f"baseline={self.use_baseline}, gamma={self.gamma})"


class REINFORCETrainer:
    """High-level trainer for REINFORCE agent."""
    
    def __init__(self, agent: REINFORCE, env):
        self.agent = agent
        self.env = env
        
    def train(self, 
              num_episodes: int,
              max_steps_per_episode: int = 1000,
              eval_freq: int = 50,
              eval_episodes: int = 10,
              verbose: bool = True,
              render: bool = False) -> Dict[str, Any]:
        """
        Train the REINFORCE agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            eval_freq: Frequency of evaluation (episodes)
            eval_episodes: Number of episodes for evaluation
            verbose: Whether to print progress
            render: Whether to render environment
            
        Returns:
            Training results dictionary
        """
        eval_stats = []
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            episode_reward = 0
            
            # Collect trajectory
            for step in range(max_steps_per_episode):
                if render:
                    self.env.render()
                    
                # Select action
                action, log_prob, action_probs = self.agent.select_action(state)
                
                # Take action
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action, reward, log_prob, action_probs)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update policy at end of episode
            update_stats = self.agent.update_policy()
            
            # Evaluation
            if (episode + 1) % eval_freq == 0:
                eval_result = self.agent.evaluate_policy(self.env, eval_episodes)
                eval_stats.append({
                    'episode': episode + 1,
                    **eval_result
                })
                
                if verbose:
                    print(f"Episode {episode + 1}/{num_episodes}")
                    print(f"  Training Reward: {episode_reward:.2f}")
                    print(f"  Eval Mean Reward: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
                    print(f"  Policy Loss: {update_stats.get('policy_loss', 0):.4f}")
                    print(f"  Gradient Norm: {update_stats.get('gradient_norm', 0):.4f}")
                    print()
        
        return {
            'training_stats': self.agent.get_training_stats(),
            'eval_stats': eval_stats,
            'final_performance': eval_stats[-1] if eval_stats else None
        } 