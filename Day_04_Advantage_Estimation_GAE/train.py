"""
Training Loop for Actor-Critic with GAE

Implements:
- Actor-Critic training with GAE advantages
- Policy and value loss computation
- Experiment management with multiple seeds
- Lambda sweep for bias-variance analysis
- Training metrics tracking and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import os

from env_wrapper import make_env, get_env_info
from model import ActorCritic, create_actor_critic
from gae_utils import GAEComputer, Trajectory, create_trajectory_from_episode


@dataclass
class TrainingConfig:
    """Configuration for training run."""
    # Environment settings
    env_name: str = 'CartPole-v1'
    normalize_obs: bool = True
    max_episode_steps: Optional[int] = None
    
    # Network architecture
    actor_hidden_sizes: Tuple[int, ...] = (64, 64)
    critic_hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = 'tanh'
    shared_backbone: bool = False
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    actor_lr: Optional[float] = None  # If None, uses learning_rate
    critic_lr: Optional[float] = None  # If None, uses learning_rate
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training schedule
    max_episodes: int = 1000
    update_frequency: int = 1  # Update every N episodes
    batch_size: int = 32  # For batched updates (if collecting multiple episodes)
    
    # Advantage normalization
    normalize_advantages: bool = True
    normalize_returns: bool = False
    
    # Logging and evaluation
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 5
    save_interval: int = 100
    
    # Other
    seed: int = 42
    device: str = 'cpu'


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    advantages_mean: List[float] = field(default_factory=list)
    advantages_std: List[float] = field(default_factory=list)
    returns_mean: List[float] = field(default_factory=list)
    returns_std: List[float] = field(default_factory=list)
    value_estimates_mean: List[float] = field(default_factory=list)
    value_estimates_std: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    
    def add_episode_result(self, reward: float, length: int):
        """Add episode result."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
    
    def add_training_step(self, 
                         policy_loss: float,
                         value_loss: float,
                         total_loss: float,
                         entropy: float,
                         grad_norm: float,
                         advantages: torch.Tensor,
                         returns: torch.Tensor,
                         values: torch.Tensor,
                         training_time: float):
        """Add training step metrics."""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.total_losses.append(total_loss)
        self.entropies.append(entropy)
        self.grad_norms.append(grad_norm)
        self.advantages_mean.append(float(advantages.mean()))
        self.advantages_std.append(float(advantages.std()))
        self.returns_mean.append(float(returns.mean()))
        self.returns_std.append(float(returns.std()))
        self.value_estimates_mean.append(float(values.mean()))
        self.value_estimates_std.append(float(values.std()))
        self.training_times.append(training_time)


class ActorCriticTrainer:
    """Actor-Critic trainer with GAE."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set device
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Create environment
        self.env = make_env(
            config.env_name, 
            normalize_obs=config.normalize_obs,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed
        )
        self.env_info = get_env_info(self.env)
        
        # Create actor-critic network
        self.actor_critic = create_actor_critic(
            self.env_info,
            config.actor_hidden_sizes,
            config.critic_hidden_sizes,
            config.activation,
            config.shared_backbone
        ).to(self.device)
        
        # Create optimizers
        actor_lr = config.actor_lr if config.actor_lr is not None else config.learning_rate
        critic_lr = config.critic_lr if config.critic_lr is not None else config.learning_rate
        
        if config.shared_backbone or hasattr(self.actor_critic, 'shared_backbone_net'):
            # Single optimizer for shared parameters
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
            self.actor_optimizer = self.optimizer
            self.critic_optimizer = self.optimizer
        else:
            # Separate optimizers for actor and critic
            self.actor_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=critic_lr)
            
        # GAE computer
        self.gae_computer = GAEComputer(
            gamma=config.gamma,
            lambda_gae=config.lambda_gae,
            normalize_advantages=config.normalize_advantages
        )
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        self.current_episode = 0
        
        # Recent performance tracking
        self.recent_rewards = deque(maxlen=100)
        self.best_mean_reward = -float('inf')
        
        print(f"🚀 Initialized Actor-Critic trainer")
        print(f"Environment: {self.env_info['env_name']}")
        print(f"State dim: {self.env_info['obs_dim']}, Action dim: {self.env_info['action_dim']}")
        print(f"Action type: {self.env_info['action_type']}")
        print(f"Network parameters: {sum(p.numel() for p in self.actor_critic.parameters())}")
    
    def collect_episode(self) -> Tuple[Trajectory, float, int]:
        """
        Collect a single episode of data.
        
        Returns:
            trajectory: Episode trajectory
            total_reward: Total episode reward
            episode_length: Episode length
        """
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state, _ = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        
        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, value, log_prob = self.actor_critic.get_action_value_and_log_prob(state_tensor)
                
            # Convert to numpy for environment
            if self.env_info['action_type'] == 'discrete':
                action_np = action.item()
            else:
                action_np = action.cpu().numpy().squeeze()
            
            # Store trajectory data
            states.append(state.copy())
            actions.append(action_np)
            values.append(value.item())
            log_probs.append(log_prob.item())
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            
            total_reward += reward
            episode_length += 1
            
            if done:
                break
                
            state = next_state
        
        # Get next state value for bootstrapping (0 if episode terminated)
        if not terminated and truncated:  # Episode was truncated, not truly done
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_value, _ = self.actor_critic.get_action_value_and_log_prob(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0
        
        # Create trajectory
        trajectory = create_trajectory_from_episode(
            states, actions, rewards, values, log_probs, dones, next_value
        )
        
        return trajectory, total_reward, episode_length
    
    def compute_losses(self, trajectory: Trajectory) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute policy and value losses from trajectory.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            policy_loss: Policy gradient loss
            value_loss: Value function loss  
            entropy_loss: Entropy regularization loss
            total_loss: Combined loss
        """
        # Move trajectory to device
        states = trajectory.states.to(self.device)
        actions = trajectory.actions.to(self.device)
        old_log_probs = trajectory.log_probs.to(self.device)
        
        # Compute GAE advantages and returns
        advantages, returns = self.gae_computer.compute_gae(
            trajectory.rewards, trajectory.values, trajectory.dones, trajectory.next_value
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Optional return normalization
        if self.config.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Get current policy and value estimates
        values, log_probs, entropy = self.actor_critic.evaluate_actions(states, actions)
        
        # Policy loss (negative because we want to maximize expected return)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss (MSE between predicted and target values)
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()
        
        # Combined loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss)
        
        return policy_loss, value_loss, entropy_loss, total_loss
    
    def update_networks(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
        Args:
            trajectory: Episode trajectory
            
        Returns:
            Dictionary of training metrics
        """
        start_time = time.time()
        
        # Compute losses
        policy_loss, value_loss, entropy_loss, total_loss = self.compute_losses(trajectory)
        
        # Backward pass
        if hasattr(self, 'optimizer'):
            # Single optimizer (shared backbone)
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), 
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
        else:
            # Separate optimizers
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            total_loss.backward()
            
            # Gradient clipping
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(), 
                self.config.max_grad_norm
            )
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.actor_critic.critic.parameters(), 
                self.config.max_grad_norm
            )
            grad_norm = max(actor_grad_norm, critic_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        training_time = time.time() - start_time
        
        # Compute advantages and returns for metrics
        advantages, returns = self.gae_computer.compute_gae(
            trajectory.rewards, trajectory.values, trajectory.dones, trajectory.next_value
        )
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
            'entropy': (-entropy_loss).item(),
            'advantages': advantages,
            'returns': returns,
            'values': trajectory.values,
            'training_time': training_time
        }
    
    def train(self) -> TrainingMetrics:
        """
        Run training loop.
        
        Returns:
            Training metrics
        """
        print(f"🏃 Starting training for {self.config.max_episodes} episodes...")
        
        for episode in range(self.config.max_episodes):
            self.current_episode = episode
            
            # Collect episode
            trajectory, total_reward, episode_length = self.collect_episode()
            
            # Update networks
            training_metrics = self.update_networks(trajectory)
            
            # Track metrics
            self.metrics.add_episode_result(total_reward, episode_length)
            self.metrics.add_training_step(
                training_metrics['policy_loss'],
                training_metrics['value_loss'],
                training_metrics['total_loss'],
                training_metrics['entropy'],
                training_metrics['grad_norm'],
                training_metrics['advantages'],
                training_metrics['returns'],
                training_metrics['values'],
                training_metrics['training_time']
            )
            
            # Track recent performance
            self.recent_rewards.append(total_reward)
            mean_recent_reward = np.mean(self.recent_rewards)
            
            if mean_recent_reward > self.best_mean_reward:
                self.best_mean_reward = mean_recent_reward
            
            # Logging
            if episode % self.config.log_interval == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {total_reward:7.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Mean(100): {mean_recent_reward:7.2f} | "
                      f"Policy Loss: {training_metrics['policy_loss']:7.4f} | "
                      f"Value Loss: {training_metrics['value_loss']:7.4f} | "
                      f"Entropy: {training_metrics['entropy']:6.4f}")
        
        print(f"✅ Training completed! Best mean reward: {self.best_mean_reward:.2f}")
        return self.metrics
    
    def evaluate(self, num_episodes: int = 5, render: bool = False) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        # Create evaluation environment with proper render mode if needed
        if render and not hasattr(self.env, '_render_mode'):
            eval_env = make_env(
                self.config.env_name, 
                normalize_obs=self.config.normalize_obs,
                seed=self.config.seed
            )
            # Try to set render mode
            try:
                eval_env = eval_env.unwrapped
                if hasattr(eval_env, 'render_mode'):
                    eval_env.render_mode = 'human'
            except:
                pass
        else:
            eval_env = self.env
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state, _ = eval_env.reset()
            total_reward = 0.0
            episode_length = 0
            
            if render:
                print(f"🎬 Rendering episode {episode + 1}/{num_episodes}")
            
            while True:
                if render:
                    try:
                        eval_env.render()
                    except Exception as e:
                        # If rendering fails, continue without rendering
                        if episode == 0:  # Only print warning once
                            print(f"⚠️  Rendering failed: {e}")
                        pass
                    
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.env_info['action_type'] == 'discrete':
                        logits = self.actor_critic.actor(state_tensor)
                        action = torch.argmax(logits, dim=-1).item()  # Deterministic for evaluation
                    else:
                        mean, _ = self.actor_critic.actor(state_tensor)
                        action = mean.cpu().numpy().squeeze()  # Use mean for evaluation
                
                state, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
            
            if render:
                print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {episode_length}")
        
        # Clean up evaluation environment if we created a separate one
        if render and eval_env != self.env:
            eval_env.close()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'std_length': np.std(eval_lengths)
        }
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict() if hasattr(self, 'critic_optimizer') else None,
            'config': self.config,
            'metrics': self.metrics,
            'episode': self.current_episode
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if checkpoint['critic_optimizer_state_dict'] is not None:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.current_episode = checkpoint['episode']


def run_lambda_sweep(base_config: TrainingConfig, 
                    lambda_values: List[float] = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0],
                    num_seeds: int = 3) -> Dict[float, List[TrainingMetrics]]:
    """
    Run training with different λ values to analyze bias-variance tradeoff.
    
    Args:
        base_config: Base training configuration
        lambda_values: List of λ values to test
        num_seeds: Number of random seeds per λ value
        
    Returns:
        Dictionary mapping λ values to list of training metrics
    """
    results = {}
    
    for lambda_val in lambda_values:
        print(f"\n🧪 Testing λ = {lambda_val}")
        lambda_results = []
        
        for seed in range(num_seeds):
            print(f"  Seed {seed + 1}/{num_seeds}")
            
            # Create config for this run
            config = TrainingConfig(**base_config.__dict__)
            config.lambda_gae = lambda_val
            config.seed = base_config.seed + seed
            
            # Train
            trainer = ActorCriticTrainer(config)
            metrics = trainer.train()
            lambda_results.append(metrics)
            
            # Quick cleanup
            trainer.env.close()
            del trainer
        
        results[lambda_val] = lambda_results
    
    return results


if __name__ == "__main__":
    # Test training
    print("🧪 Testing Actor-Critic Training...")
    
    config = TrainingConfig(
        env_name='CartPole-v1',
        max_episodes=50,  # Quick test
        log_interval=10,
        learning_rate=1e-3,
        lambda_gae=0.95
    )
    
    trainer = ActorCriticTrainer(config)
    
    # Test single episode collection
    trajectory, reward, length = trainer.collect_episode()
    print(f"📊 Test episode: Reward={reward:.2f}, Length={length}")
    
    # Test loss computation
    policy_loss, value_loss, entropy_loss, total_loss = trainer.compute_losses(trajectory)
    print(f"🔍 Losses: Policy={policy_loss:.4f}, Value={value_loss:.4f}, Total={total_loss:.4f}")
    
    # Test short training
    print(f"\n🏃 Running short training...")
    metrics = trainer.train()
    
    print(f"📈 Final metrics:")
    print(f"  Episodes: {len(metrics.episode_rewards)}")
    print(f"  Mean reward: {np.mean(metrics.episode_rewards[-10:]):.2f}")
    print(f"  Final policy loss: {metrics.policy_losses[-1]:.4f}")
    
    trainer.env.close()
    print("\n✅ Training tests completed!") 