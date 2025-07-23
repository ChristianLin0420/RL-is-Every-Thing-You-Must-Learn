"""
Policy Network Implementation for REINFORCE

This module implements a neural network policy for discrete action spaces
using PyTorch. The policy outputs action probabilities using softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """
    Neural network policy for discrete action spaces.
    
    This network takes observations as input and outputs action probabilities
    using a softmax activation function. It's designed to work with the
    REINFORCE algorithm for policy gradient learning.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: list = [128, 128],
                 activation: str = 'relu',
                 seed: Optional[int] = None):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the observation space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            seed: Random seed for reproducibility
        """
        super(PolicyNetwork, self).__init__()
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Choose activation function
        self.activation_fn = self._get_activation(activation)
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation_fn()
            ])
            input_dim = hidden_dim
            
        # Output layer (no activation - we'll apply softmax later)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function class."""
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU
        }
        return activations.get(activation.lower(), nn.ReLU)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim) or (state_dim,)
            
        Returns:
            Action logits (pre-softmax) of shape (batch_size, action_dim)
        """
        # Ensure input is 2D (batch_size, state_dim)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        logits = self.network(state)
        return logits
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            state: Input state tensor
            
        Returns:
            Action probabilities of shape (batch_size, action_dim)
        """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action, log_prob, action_probabilities)
        """
        # Get action probabilities
        action_probs = self.get_action_probabilities(state)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability of the sampled action
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, action_probs
    
    def get_log_probability(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probability of a specific action given state.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Log probability of the action
        """
        action_probs = self.get_action_probabilities(state)
        dist = torch.distributions.Categorical(action_probs)
        return dist.log_prob(action)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate policy entropy for exploration measurement.
        
        Args:
            state: Input state tensor
            
        Returns:
            Entropy of the policy distribution
        """
        action_probs = self.get_action_probabilities(state)
        dist = torch.distributions.Categorical(action_probs)
        return dist.entropy()
    
    def save_model(self, filepath: str):
        """Save model parameters."""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'network_config': str(self.network)
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model parameters."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])
        
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self) -> str:
        """String representation of the network."""
        param_count = self.get_parameter_count()
        return f"PolicyNetwork(state_dim={self.state_dim}, action_dim={self.action_dim}, " \
               f"parameters={param_count:,})\n{self.network}"


# Utility functions for different policy architectures
def create_cartpole_policy(seed: Optional[int] = None) -> PolicyNetwork:
    """Create a policy network specifically for CartPole-v1."""
    return PolicyNetwork(
        state_dim=4,  # CartPole observation space
        action_dim=2,  # Left/Right actions
        hidden_dims=[128, 128],
        activation='relu',
        seed=seed
    )

def create_small_policy(state_dim: int, action_dim: int, seed: Optional[int] = None) -> PolicyNetwork:
    """Create a smaller policy network for faster training."""
    return PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[64],
        activation='relu',
        seed=seed
    )

def create_large_policy(state_dim: int, action_dim: int, seed: Optional[int] = None) -> PolicyNetwork:
    """Create a larger policy network for complex environments."""
    return PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256, 128],
        activation='relu',
        seed=seed
    ) 