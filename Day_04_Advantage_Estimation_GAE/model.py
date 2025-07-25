"""
Actor-Critic Network Architecture for GAE

Implements separate Actor and Critic networks:
- Actor: Policy network π(a|s) that outputs action probabilities
- Critic: Value network V(s) that estimates state values for GAE computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Optional, Union


class Actor(nn.Module):
    """
    Policy network (Actor) for discrete or continuous action spaces.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh',
                 action_type: str = 'discrete'):
        """
        Initialize Actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: Hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
            action_type: 'discrete' or 'continuous'
        """
        super().__init__()
        
        self.action_type = action_type
        self.action_dim = action_dim
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer depends on action type
        if action_type == 'discrete':
            # For discrete actions: output logits for each action
            self.action_head = nn.Linear(prev_size, action_dim)
        elif action_type == 'continuous':
            # For continuous actions: output mean and log_std
            self.mean_head = nn.Linear(prev_size, action_dim)
            self.log_std_head = nn.Linear(prev_size, action_dim)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
    
    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through actor network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            For discrete: action logits [batch_size, action_dim]
            For continuous: (mean, log_std) each [batch_size, action_dim]
        """
        features = self.shared_layers(state)
        
        if self.action_type == 'discrete':
            logits = self.action_head(features)
            return logits
        else:  # continuous
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            # Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            action: Sampled actions
            log_prob: Log probabilities of sampled actions
        """
        if self.action_type == 'discrete':
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:  # continuous
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum for multi-dimensional actions
            return action, log_prob
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of given actions.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
            
        Returns:
            log_prob: Log probabilities [batch_size]
        """
        if self.action_type == 'discrete':
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.log_prob(action)
        else:  # continuous
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            return dist.log_prob(action).sum(dim=-1)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of policy at given states.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            entropy: Policy entropy [batch_size]
        """
        if self.action_type == 'discrete':
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            return dist.entropy()
        else:  # continuous
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            return dist.entropy().sum(dim=-1)


class Critic(nn.Module):
    """
    Value network (Critic) that estimates V(s) for advantage computation.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh'):
        """
        Initialize Critic network.
        
        Args:
            state_dim: Dimension of state space
            hidden_sizes: Hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'elu')
        """
        super().__init__()
        
        # Choose activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
            
        # Add final value head
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            value: State values [batch_size]
        """
        value = self.network(state).squeeze(-1)  # Remove last dimension
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for GAE-based policy gradients.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 action_type: str = 'discrete',
                 actor_hidden_sizes: Tuple[int, ...] = (64, 64),
                 critic_hidden_sizes: Tuple[int, ...] = (64, 64),
                 activation: str = 'tanh',
                 shared_backbone: bool = False):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_type: 'discrete' or 'continuous'
            actor_hidden_sizes: Hidden sizes for actor network
            critic_hidden_sizes: Hidden sizes for critic network
            activation: Activation function
            shared_backbone: Whether to share feature extraction layers
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.shared_backbone = shared_backbone
        
        if shared_backbone:
            # Shared feature extraction
            if activation == 'tanh':
                act_fn = nn.Tanh()
            elif activation == 'relu':
                act_fn = nn.ReLU()
            elif activation == 'elu':
                act_fn = nn.ELU()
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            backbone_layers = []
            prev_size = state_dim
            shared_sizes = actor_hidden_sizes[:-1]  # All but last layer
            
            for hidden_size in shared_sizes:
                backbone_layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    act_fn
                ])
                prev_size = hidden_size
                
            self.shared_backbone_net = nn.Sequential(*backbone_layers)
            
            # Separate heads
            final_hidden = actor_hidden_sizes[-1] if actor_hidden_sizes else prev_size
            
            if action_type == 'discrete':
                self.actor_head = nn.Sequential(
                    nn.Linear(prev_size, final_hidden),
                    act_fn,
                    nn.Linear(final_hidden, action_dim)
                )
            else:  # continuous
                self.mean_head = nn.Sequential(
                    nn.Linear(prev_size, final_hidden),
                    act_fn,
                    nn.Linear(final_hidden, action_dim)
                )
                self.log_std_head = nn.Sequential(
                    nn.Linear(prev_size, final_hidden),
                    act_fn,
                    nn.Linear(final_hidden, action_dim)
                )
            
            final_critic_hidden = critic_hidden_sizes[-1] if critic_hidden_sizes else prev_size
            self.critic_head = nn.Sequential(
                nn.Linear(prev_size, final_critic_hidden),
                act_fn,
                nn.Linear(final_critic_hidden, 1)
            )
        else:
            # Separate networks
            self.actor = Actor(state_dim, action_dim, actor_hidden_sizes, activation, action_type)
            self.critic = Critic(state_dim, critic_hidden_sizes, activation)
    
    def forward(self, state: torch.Tensor) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through both actor and critic.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            actor_output: Logits (discrete) or (mean, log_std) (continuous)
            critic_output: State values [batch_size]
        """
        if self.shared_backbone:
            features = self.shared_backbone_net(state)
            
            if self.action_type == 'discrete':
                actor_output = self.actor_head(features)
            else:  # continuous
                mean = self.mean_head(features)
                log_std = self.log_std_head(features)
                log_std = torch.clamp(log_std, -20, 2)
                actor_output = (mean, log_std)
                
            critic_output = self.critic_head(features).squeeze(-1)
        else:
            actor_output = self.actor(state)
            critic_output = self.critic(state)
            
        return actor_output, critic_output
    
    def get_action_value_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action, estimate value, and compute log probability in one pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            action: Sampled actions
            value: State values [batch_size]
            log_prob: Log probabilities of sampled actions [batch_size]
        """
        actor_output, value = self.forward(state)
        
        if self.action_type == 'discrete':
            logits = actor_output
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:  # continuous
            mean, log_std = actor_output
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action, value, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given state-action pairs.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions
            
        Returns:
            value: State values [batch_size]
            log_prob: Log probabilities of actions [batch_size]
            entropy: Policy entropy at states [batch_size]
        """
        actor_output, value = self.forward(state)
        
        if self.action_type == 'discrete':
            logits = actor_output
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        else:  # continuous
            mean, log_std = actor_output
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
        return value, log_prob, entropy


def create_actor_critic(env_info: dict, 
                       actor_hidden_sizes: Tuple[int, ...] = (64, 64),
                       critic_hidden_sizes: Tuple[int, ...] = (64, 64),
                       activation: str = 'tanh',
                       shared_backbone: bool = False) -> ActorCritic:
    """
    Factory function to create ActorCritic from environment info.
    
    Args:
        env_info: Environment information dict from env_wrapper.get_env_info()
        actor_hidden_sizes: Hidden sizes for actor
        critic_hidden_sizes: Hidden sizes for critic  
        activation: Activation function
        shared_backbone: Whether to share feature extraction
        
    Returns:
        ActorCritic network
    """
    return ActorCritic(
        state_dim=env_info['obs_dim'],
        action_dim=env_info['action_dim'],
        action_type=env_info['action_type'],
        actor_hidden_sizes=actor_hidden_sizes,
        critic_hidden_sizes=critic_hidden_sizes,
        activation=activation,
        shared_backbone=shared_backbone
    )


if __name__ == "__main__":
    # Test the Actor-Critic networks
    print("🧪 Testing Actor-Critic Networks...")
    
    # Test discrete action space (CartPole)
    print("\n🎮 Testing Discrete Action Space (CartPole-like)")
    state_dim, action_dim = 4, 2
    batch_size = 32
    
    # Create networks
    actor = Actor(state_dim, action_dim, action_type='discrete')
    critic = Critic(state_dim)
    actor_critic = ActorCritic(state_dim, action_dim, action_type='discrete')
    
    # Test with random states
    states = torch.randn(batch_size, state_dim)
    
    # Test individual networks
    logits = actor(states)
    values = critic(states)
    print(f"Actor output shape: {logits.shape}")
    print(f"Critic output shape: {values.shape}")
    
    # Test action sampling
    actions, log_probs = actor.get_action_and_log_prob(states)
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    
    # Test combined network
    actions, values, log_probs = actor_critic.get_action_value_and_log_prob(states)
    print(f"Combined - Actions: {actions.shape}, Values: {values.shape}, Log probs: {log_probs.shape}")
    
    # Test continuous action space (Pendulum)
    print("\n🌙 Testing Continuous Action Space (Pendulum-like)")
    state_dim, action_dim = 3, 1
    
    actor_cont = Actor(state_dim, action_dim, action_type='continuous')
    actor_critic_cont = ActorCritic(state_dim, action_dim, action_type='continuous')
    
    states = torch.randn(batch_size, state_dim)
    
    # Test continuous actions
    mean, log_std = actor_cont(states)
    actions, log_probs = actor_cont.get_action_and_log_prob(states)
    print(f"Continuous - Mean: {mean.shape}, Log std: {log_std.shape}")
    print(f"Continuous - Actions: {actions.shape}, Log probs: {log_probs.shape}")
    
    print("\n✅ Actor-Critic network tests completed!") 