"""
Experience Buffer for PPO
==========================
Implements rollout buffer with GAE (Generalized Advantage Estimation) computation.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


class PPOBuffer:
    """
    Buffer for storing rollout data and computing GAE.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 buffer_size: int,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 device: str = 'cpu',
                 action_type: str = 'continuous'):
        """
        Initialize PPO buffer.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension  
            buffer_size: Maximum buffer size
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            device: Device for tensor operations
            action_type: 'continuous' or 'discrete'
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.device = device
        self.action_type = action_type
        
        # Storage arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        
        # Actions: different shapes for discrete vs continuous
        if action_type == 'discrete':
            self.actions = np.zeros(buffer_size, dtype=np.int64)  # Scalar actions
        else:
            self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)  # Vector actions
            
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.terminals = np.zeros(buffer_size, dtype=np.bool_)
        
        # GAE computation arrays
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Buffer state
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def store(self, 
              obs: np.ndarray, 
              action: np.ndarray, 
              reward: float, 
              value: float,
              log_prob: float,
              terminal: bool = False):
        """
        Store a single step of experience.
        
        Args:
            obs: Observation
            action: Action taken (scalar for discrete, vector for continuous)
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            terminal: Whether episode terminated
        """
        assert self.ptr < self.max_size, "Buffer overflow"
        
        self.observations[self.ptr] = obs
        
        # Handle action storage based on type
        if self.action_type == 'discrete':
            # For discrete actions, store as scalar
            if isinstance(action, np.ndarray):
                self.actions[self.ptr] = action.item() if action.size == 1 else action[0]
            else:
                self.actions[self.ptr] = action
        else:
            # For continuous actions, store as vector
            self.actions[self.ptr] = action
            
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.terminals[self.ptr] = terminal
        
        # Track episode statistics
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if terminal:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0.0
            self.current_episode_length = 0
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Compute GAE advantages and returns for completed path.
        
        Args:
            last_value: Value estimate for final state (0 if terminal)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # Get rewards and values for this path
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        terminals = self.terminals[path_slice]
        
        # Add bootstrap value
        values_with_bootstrap = np.append(values, last_value)
        
        # Compute GAE
        gae = 0
        for step in reversed(range(len(rewards))):
            if terminals[step]:
                # Reset GAE at episode boundaries
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + self.gamma * values_with_bootstrap[step + 1] - values[step]
                gae = delta + self.gamma * self.lambda_gae * gae
            
            self.advantages[self.path_start_idx + step] = gae
        
        # Compute returns (advantages + values)
        self.returns[path_slice] = self.advantages[path_slice] + values
        
        # Update path start for next path
        self.path_start_idx = self.ptr
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored data as tensors.
        
        Returns:
            Dictionary containing all buffer data as tensors
        """
        assert self.ptr == self.max_size, "Buffer not full"
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        # Handle actions based on type
        if self.action_type == 'discrete':
            actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        else:
            actions_tensor = torch.tensor(self.actions, dtype=torch.float32, device=self.device)
        
        data = {
            'observations': torch.tensor(self.observations, device=self.device),
            'actions': actions_tensor,
            'rewards': torch.tensor(self.rewards, device=self.device),
            'values': torch.tensor(self.values, device=self.device),
            'log_probs': torch.tensor(self.log_probs, device=self.device),
            'advantages': torch.tensor(self.advantages, device=self.device),
            'returns': torch.tensor(self.returns, device=self.device),
            'terminals': torch.tensor(self.terminals, device=self.device)
        }
        
        return data
    
    def reset(self):
        """Reset buffer for next collection phase."""
        self.ptr = 0
        self.path_start_idx = 0
        
        # Clear episode tracking (keep history for statistics)
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get episode statistics from recent episodes.
        
        Returns:
            Dictionary with episode statistics
        """
        if len(self.episode_rewards) == 0:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'mean_length': 0.0,
                'num_episodes': 0
            }
        
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'num_episodes': len(self.episode_rewards)
        }
    
    def clear_episode_statistics(self):
        """Clear episode statistics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()


class TrajectoryBuffer:
    """
    Simple trajectory buffer for storing complete episodes.
    Useful for policy gradient methods that require complete episodes.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset the trajectory buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminals = []
    
    def store(self, obs, action, reward, log_prob, value, terminal=False):
        """Store a single step."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.terminals.append(terminal)
    
    def compute_returns_and_advantages(self, gamma=0.99, lambda_gae=0.95):
        """
        Compute returns and GAE advantages for stored trajectory.
        
        Args:
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        terminals = np.array(self.terminals)
        
        # Compute returns using rewards-to-go
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            if terminals[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # Compute GAE advantages
        advantages = np.zeros_like(rewards)
        running_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if terminals[t]:
                running_gae = 0
                next_value = 0
            
            delta = rewards[t] + gamma * next_value - values[t]
            running_gae = delta + gamma * lambda_gae * running_gae
            advantages[t] = running_gae
        
        return returns, advantages
    
    def get_tensors(self):
        """Convert stored data to tensors."""
        return {
            'observations': torch.tensor(np.array(self.observations), dtype=torch.float32, device=self.device),
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(np.array(self.rewards), dtype=torch.float32, device=self.device),
            'log_probs': torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device),
            'values': torch.tensor(np.array(self.values), dtype=torch.float32, device=self.device),
            'terminals': torch.tensor(np.array(self.terminals), dtype=torch.bool, device=self.device)
        }


def compute_gae(rewards: np.ndarray, 
                values: np.ndarray, 
                terminals: np.ndarray,
                gamma: float = 0.99, 
                lambda_gae: float = 0.95,
                last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        values: Array of value estimates
        terminals: Array of terminal flags
        gamma: Discount factor
        lambda_gae: GAE lambda parameter
        last_value: Value estimate for state after last reward
        
    Returns:
        Tuple of (advantages, returns)
    """
    # Add bootstrap value
    values_with_bootstrap = np.append(values, last_value)
    
    # Compute GAE
    advantages = np.zeros_like(rewards)
    gae = 0
    
    for step in reversed(range(len(rewards))):
        if terminals[step]:
            # Reset GAE at episode boundaries
            delta = rewards[step] - values[step]
            gae = delta
        else:
            delta = rewards[step] + gamma * values_with_bootstrap[step + 1] - values[step]
            gae = delta + gamma * lambda_gae * gae
        
        advantages[step] = gae
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sum.
    
    Args:
        x: Input array
        discount: Discount factor
        
    Returns:
        Discounted cumulative sum
    """
    # Simple implementation for clarity
    result = np.zeros_like(x)
    result[-1] = x[-1]
    
    for t in reversed(range(len(x) - 1)):
        result[t] = x[t] + discount * result[t + 1]
    
    return result 