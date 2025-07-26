"""
Actor-Critic Networks for PPO
==============================
Neural network architectures for continuous control with PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from torch.distributions import Normal, Categorical


def init_weights(m):
    """Initialize network weights using orthogonal initialization."""
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0.0)


class Actor(nn.Module):
    """Actor network for continuous action spaces."""
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh',
                 log_std_init: float = 0.0):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
        
        # Mean output layer
        layers.append(nn.Linear(prev_size, action_dim))
        self.mean_net = nn.Sequential(*layers)
        
        # Log standard deviation (learnable parameter)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (mean, std) for action distribution
        """
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action_and_log_prob(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given actions.
        
        Args:
            obs: Observations tensor
            action: Actions tensor
            
        Returns:
            Log probabilities
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
    
    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of action distribution.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Entropy values
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        return dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    """Critic network for value function estimation."""
    
    def __init__(self, 
                 obs_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh'):
        super().__init__()
        
        self.obs_dim = obs_dim
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
        
        # Value output layer
        layers.append(nn.Linear(prev_size, 1))
        self.value_net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Value estimates
        """
        return self.value_net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """Combined Actor-Critic network for PPO."""
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh',
                 log_std_init: float = 0.0,
                 shared_backbone: bool = False):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # Shared backbone layers
            self.backbone = nn.Sequential()
            prev_size = obs_dim
            
            for i, hidden_size in enumerate(hidden_sizes[:-1]):
                self.backbone.add_module(f'fc{i}', nn.Linear(prev_size, hidden_size))
                if activation == 'tanh':
                    self.backbone.add_module(f'tanh{i}', nn.Tanh())
                elif activation == 'relu':
                    self.backbone.add_module(f'relu{i}', nn.ReLU())
                prev_size = hidden_size
            
            # Separate heads
            final_hidden = hidden_sizes[-1]
            
            # Actor head
            self.actor_head = nn.Sequential(
                nn.Linear(prev_size, final_hidden),
                nn.Tanh() if activation == 'tanh' else nn.ReLU(),
                nn.Linear(final_hidden, action_dim)
            )
            
            # Critic head
            self.critic_head = nn.Sequential(
                nn.Linear(prev_size, final_hidden),
                nn.Tanh() if activation == 'tanh' else nn.ReLU(),
                nn.Linear(final_hidden, 1)
            )
            
            # Log std for actor
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
            
        else:
            # Separate networks
            self.actor = Actor(obs_dim, action_dim, hidden_sizes, activation, log_std_init)
            self.critic = Critic(obs_dim, hidden_sizes, activation)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (action_mean, action_std, value)
        """
        if self.shared_backbone:
            features = self.backbone(obs)
            action_mean = self.actor_head(features)
            action_std = torch.exp(self.log_std)
            value = self.critic_head(features).squeeze(-1)
        else:
            action_mean, action_std = self.actor.forward(obs)
            value = self.critic.forward(obs)
        
        return action_mean, action_std, value
    
    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value estimate.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
    
    def get_log_prob_and_entropy(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for given actions.
        
        Args:
            obs: Observations tensor
            action: Actions tensor
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        action_mean, action_std, _ = self.forward(obs)
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate only.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Value estimates
        """
        if self.shared_backbone:
            features = self.backbone(obs)
            value = self.critic_head(features).squeeze(-1)
        else:
            value = self.critic.forward(obs)
        return value


class DiscreteActorCritic(nn.Module):
    """Actor-Critic for discrete action spaces (for comparison/testing)."""
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared backbone
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
        
        self.backbone = nn.Sequential(*layers)
        
        # Actor and critic heads
        self.actor_head = nn.Linear(prev_size, action_dim)
        self.critic_head = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.backbone(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return action_logits, value
    
    def get_action_and_value(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute value estimate.
        
        Args:
            obs: Observations tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def get_log_prob_and_entropy(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log probability and entropy for given actions.
        
        Args:
            obs: Observations tensor
            action: Actions tensor
            
        Returns:
            Tuple of (log_prob, entropy)
        """
        action_logits, _ = self.forward(obs)
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy 