"""
Proximal Policy Optimization (PPO) Implementation
=================================================
Implements PPO with clipped surrogate objective for continuous control.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from model import ActorCritic, DiscreteActorCritic
from buffer import PPOBuffer
from env_wrapper import make_env, get_env_info


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Environment
    env_name: str = "Pendulum-v1"
    max_episode_steps: Optional[int] = None
    normalize_obs: bool = True
    normalize_reward: bool = False
    
    # PPO Algorithm
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # None means no value clipping
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10  # Epochs per rollout
    total_timesteps: int = 1000000
    
    # Network
    hidden_sizes: Tuple[int, ...] = (64, 64)
    activation: str = 'tanh'
    shared_backbone: bool = False
    
    # Logging and evaluation
    log_interval: int = 10
    eval_interval: int = 50
    eval_episodes: int = 10
    save_interval: int = 100
    
    # Technical
    device: str = 'auto'
    seed: Optional[int] = None


class PPO:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPO agent.
        
        Args:
            config: PPO configuration
        """
        self.config = config
        
        # Set device
        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)
        
        # Create environment
        self.env = make_env(
            config.env_name,
            normalize_obs=config.normalize_obs,
            normalize_reward=config.normalize_reward,
            max_episode_steps=config.max_episode_steps,
            seed=config.seed
        )
        
        # Get environment info
        self.env_info = get_env_info(self.env)
        print(f"Environment: {config.env_name}")
        print(f"Observation dim: {self.env_info['obs_dim']}")
        print(f"Action dim: {self.env_info['action_dim']}")
        print(f"Action type: {self.env_info['action_type']}")
        
        # Create networks
        if self.env_info['action_type'] == 'continuous':
            self.actor_critic = ActorCritic(
                obs_dim=self.env_info['obs_dim'],
                action_dim=self.env_info['action_dim'],
                hidden_sizes=config.hidden_sizes,
                activation=config.activation,
                shared_backbone=config.shared_backbone
            ).to(self.device)
        else:
            self.actor_critic = DiscreteActorCritic(
                obs_dim=self.env_info['obs_dim'],
                action_dim=self.env_info['action_dim'],
                hidden_sizes=config.hidden_sizes,
                activation=config.activation
            ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Create buffer
        self.buffer = PPOBuffer(
            obs_dim=self.env_info['obs_dim'],
            action_dim=self.env_info['action_dim'],
            buffer_size=config.n_steps,
            gamma=config.gamma,
            lambda_gae=config.lambda_gae,
            device=self.device,
            action_type=self.env_info['action_type']
        )
        
        # Training state
        self.total_steps = 0
        self.num_updates = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Metrics tracking
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': [],
            'learning_rate': []
        }
    
    def collect_rollouts(self) -> Dict[str, float]:
        """
        Collect rollout data for training.
        
        Returns:
            Dictionary with rollout statistics
        """
        self.actor_critic.eval()
        
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        num_episodes = 0
        
        for step in range(self.config.n_steps):
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                if self.env_info['action_type'] == 'continuous':
                    action, log_prob, value = self.actor_critic.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy().squeeze()
                else:
                    action, log_prob, value = self.actor_critic.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy().item()
                
                log_prob = log_prob.cpu().numpy().item()
                value = value.cpu().numpy().item()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            self.buffer.store(obs, action, reward, value, log_prob, done)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # Handle episode end
            if done:
                # Finish path in buffer
                self.buffer.finish_path(last_value=0.0)
                
                # Track episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                num_episodes += 1
                
                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
        
        # If episode didn't finish, bootstrap value
        if not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if self.env_info['action_type'] == 'continuous':
                    _, _, last_value = self.actor_critic.get_action_and_value(obs_tensor)
                else:
                    _, last_value = self.actor_critic.forward(obs_tensor)
                last_value = last_value.cpu().numpy().item()
            
            self.buffer.finish_path(last_value=last_value)
        
        # Get rollout statistics
        rollout_stats = self.buffer.get_episode_statistics()
        self.buffer.clear_episode_statistics()
        
        return rollout_stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using collected rollout data.
        
        Returns:
            Dictionary with training metrics
        """
        self.actor_critic.train()
        
        # Get buffer data
        data = self.buffer.get()
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divergences = []
        clip_fractions = []
        
        # Multiple epochs over the data
        for epoch in range(self.config.n_epochs):
            # Create minibatches
            indices = torch.randperm(self.config.n_steps, device=self.device)
            
            for start in range(0, self.config.n_steps, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = data['observations'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                batch_old_values = data['values'][batch_indices]
                
                # Forward pass
                if self.env_info['action_type'] == 'continuous':
                    new_log_probs, entropy = self.actor_critic.get_log_prob_and_entropy(batch_obs, batch_actions)
                    new_values = self.actor_critic.get_value(batch_obs)
                else:
                    new_log_probs, entropy = self.actor_critic.get_log_prob_and_entropy(batch_obs, batch_actions.long())
                    _, new_values = self.actor_critic.forward(batch_obs)
                
                # Compute ratio (π_θ / π_θ_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.clip_range_vf is not None:
                    # Clipped value loss
                    value_pred_clipped = batch_old_values + torch.clamp(
                        new_values - batch_old_values, 
                        -self.config.clip_range_vf, 
                        self.config.clip_range_vf
                    )
                    value_loss1 = F.mse_loss(new_values, batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.value_coef * value_loss + 
                       self.config.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
                
                # Store metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                kl_divergences.append(kl_div.item())
                clip_fractions.append(clip_fraction.item())
        
        # Reset buffer for next rollout
        self.buffer.reset()
        self.num_updates += 1
        
        # Compute explained variance
        with torch.no_grad():
            y_true = data['returns'].cpu().numpy()
            y_pred = data['values'].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'kl_divergence': np.mean(kl_divergences),
            'clip_fraction': np.mean(clip_fractions),
            'explained_variance': explained_var,
            'learning_rate': self.config.learning_rate
        }
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        self.actor_critic.eval()
        
        # Create evaluation environment with proper render mode if needed
        if render:
            print("🎬 Creating render-enabled environment...")
            try:
                from env_wrapper import make_env
                eval_env = make_env(
                    self.config.env_name,
                    normalize_obs=self.config.normalize_obs,
                    normalize_reward=False,  # Don't normalize rewards for evaluation
                    render_mode='human'
                )
            except Exception as e:
                print(f"⚠️  Failed to create render environment: {e}")
                print("🔄 Using standard environment without rendering...")
                eval_env = self.env
                render = False
        else:
            eval_env = self.env
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            
            if render:
                print(f"🎬 Rendering episode {episode + 1}/{num_episodes}")
            
            while True:
                if render:
                    try:
                        eval_env.render()
                    except Exception as e:
                        if episode == 0:  # Only warn once
                            print(f"⚠️  Rendering issue: {e}")
                
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    if self.env_info['action_type'] == 'continuous':
                        action_mean, action_std, _ = self.actor_critic.forward(obs_tensor)
                        action = action_mean.cpu().numpy().squeeze()  # Use mean for evaluation
                    else:
                        action_logits, _ = self.actor_critic.forward(obs_tensor)
                        action = torch.argmax(action_logits, dim=-1).cpu().numpy().item()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            
            if render:
                print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Clean up evaluation environment if we created a separate one
        if render and eval_env != self.env:
            eval_env.close()
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'mean_length': np.mean(eval_lengths)
        }
    
    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training timesteps (overrides config)
            
        Returns:
            Training metrics history
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
        
        print(f"🚀 Starting PPO training for {total_timesteps} timesteps")
        print(f"📊 Rollout steps: {self.config.n_steps}")
        print(f"🔄 Update epochs: {self.config.n_epochs}")
        print(f"📦 Batch size: {self.config.batch_size}")
        print(f"✂️  Clip range: {self.config.clip_range}")
        print(f"🎲 Entropy coef: {self.config.entropy_coef}")
        
        start_time = time.time()
        
        while self.total_steps < total_timesteps:
            # Collect rollouts
            rollout_stats = self.collect_rollouts()
            
            # Update policy
            update_stats = self.update_policy()
            
            # Log metrics
            for key, value in update_stats.items():
                self.training_metrics[key].append(value)
            
            # Periodic logging
            if self.num_updates % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = self.total_steps / elapsed_time
                
                print(f"\nUpdate {self.num_updates} | Steps {self.total_steps}/{total_timesteps}")
                print(f"  🎮 Episodes: {rollout_stats.get('num_episodes', 0)}")
                print(f"  🏆 Mean Reward: {rollout_stats.get('mean_reward', 0.0):.2f}")
                print(f"  📊 Policy Loss: {update_stats['policy_loss']:.4f}")
                print(f"  📈 Value Loss: {update_stats['value_loss']:.4f}")
                print(f"  🎲 Entropy: {update_stats['entropy']:.4f}")
                print(f"  📏 KL Div: {update_stats['kl_divergence']:.6f}")
                print(f"  ✂️  Clip Frac: {update_stats['clip_fraction']:.3f}")
                print(f"  🔍 Explained Var: {update_stats['explained_variance']:.3f}")
                print(f"  ⚡ FPS: {fps:.0f}")
            
            # Periodic evaluation
            if self.num_updates % self.config.eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=self.config.eval_episodes)
                print(f"  🎯 Eval Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        
        print(f"\n✅ Training completed in {time.time() - start_time:.1f} seconds")
        return self.training_metrics
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'num_updates': self.num_updates,
            'training_metrics': self.training_metrics
        }, filepath, _use_new_zipfile_serialization=False)
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.num_updates = checkpoint.get('num_updates', 0)
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']


class PPOComparison:
    """Compare different PPO configurations."""
    
    def __init__(self, base_config: PPOConfig):
        self.base_config = base_config
        self.results = {}
    
    def run_experiment(self, name: str, config_overrides: Dict, timesteps: int = 100000):
        """Run a single experiment with configuration overrides."""
        config = PPOConfig(**{**self.base_config.__dict__, **config_overrides})
        
        print(f"\n🔬 Running experiment: {name}")
        print(f"   Overrides: {config_overrides}")
        
        agent = PPO(config)
        metrics = agent.train(total_timesteps=timesteps)
        
        # Evaluate final policy
        final_eval = agent.evaluate(num_episodes=20)
        
        self.results[name] = {
            'config': config,
            'metrics': metrics,
            'final_eval': final_eval,
            'agent': agent
        }
        
        print(f"   Final reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        
        return self.results[name]
    
    def compare_clipping(self, timesteps: int = 100000):
        """Compare PPO with different clipping settings."""
        experiments = [
            ("No Clipping", {'clip_range': 1e6}),  # Very large clip range = no clipping
            ("Standard Clipping (0.2)", {'clip_range': 0.2}),
            ("Tight Clipping (0.1)", {'clip_range': 0.1}),
            ("Very Tight Clipping (0.05)", {'clip_range': 0.05})
        ]
        
        for name, overrides in experiments:
            self.run_experiment(name, overrides, timesteps)
    
    def compare_entropy(self, timesteps: int = 100000):
        """Compare PPO with different entropy coefficients."""
        experiments = [
            ("No Entropy", {'entropy_coef': 0.0}),
            ("Low Entropy", {'entropy_coef': 0.001}),
            ("Standard Entropy", {'entropy_coef': 0.01}),
            ("High Entropy", {'entropy_coef': 0.1})
        ]
        
        for name, overrides in experiments:
            self.run_experiment(name, overrides, timesteps) 