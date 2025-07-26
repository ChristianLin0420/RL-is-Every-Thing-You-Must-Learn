"""
Trust Region Policy Optimization (TRPO) Implementation
======================================================
Implements TRPO with KL divergence constraint and natural gradients.
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
class TRPOConfig:
    """Configuration for TRPO training."""
    
    # Environment
    env_name: str = "Pendulum-v1"
    max_episode_steps: Optional[int] = None
    normalize_obs: bool = True
    normalize_reward: bool = False
    
    # TRPO Algorithm
    gamma: float = 0.99
    lambda_gae: float = 0.95
    max_kl: float = 0.01  # KL divergence constraint
    damping: float = 0.1  # Damping for conjugate gradient
    cg_iters: int = 10    # Conjugate gradient iterations
    backtrack_iters: int = 10  # Line search iterations
    backtrack_coeff: float = 0.8  # Line search coefficient
    
    # Value function
    value_lr: float = 3e-4
    value_iters: int = 5  # Value function update iterations
    
    # Training
    n_steps: int = 2048  # Steps per rollout
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


class TRPO:
    """Trust Region Policy Optimization agent."""
    
    def __init__(self, config: TRPOConfig):
        """
        Initialize TRPO agent.
        
        Args:
            config: TRPO configuration
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
        
        # Create value function optimizer (actor uses natural gradients)
        self.value_optimizer = optim.Adam(
            [p for name, p in self.actor_critic.named_parameters() if 'critic' in name or 'value' in name],
            lr=config.value_lr
        )
        
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
            'kl_divergence': [],
            'explained_variance': [],
            'step_size': [],
            'backtrack_steps': []
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
    
    def compute_fisher_vector_product(self, data: Dict[str, torch.Tensor], vector: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher Information Matrix times vector using Hessian-vector product.
        
        Args:
            data: Buffer data
            vector: Vector to multiply with Fisher matrix
            
        Returns:
            Fisher matrix times vector
        """
        # Create a fresh computational graph for this computation
        observations = data['observations'].detach().requires_grad_(False)
        
        # Get policy parameters for which we want gradients
        policy_params = [p for name, p in self.actor_critic.named_parameters() 
                        if 'actor' in name and p.requires_grad]
        
        # Clear any existing gradients
        for param in policy_params:
            if param.grad is not None:
                param.grad.zero_()
        
        # Compute KL divergence with fresh graph
        if self.env_info['action_type'] == 'continuous':
            action_mean, action_std, _ = self.actor_critic.forward(observations)
            # Use a simple surrogate for KL - variance of action distribution
            kl_div = torch.mean(action_std.log())
        else:
            action_logits, _ = self.actor_critic.forward(observations)
            action_probs = torch.softmax(action_logits, dim=-1)
            # Use entropy as KL surrogate for discrete case
            kl_div = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1))
        
        # Compute first-order gradients
        first_grads = torch.autograd.grad(kl_div, policy_params, create_graph=True, retain_graph=True)
        first_grads_flat = self.flatten_gradients(first_grads)
        
        # Compute vector-gradient product
        gvp = torch.sum(first_grads_flat * vector)
        
        # Compute second-order gradients (Hessian-vector product)
        hvp_grads = torch.autograd.grad(gvp, policy_params, retain_graph=False)
        hvp_flat = self.flatten_gradients(hvp_grads)
        
        return hvp_flat + self.config.damping * vector
    
    def conjugate_gradient(self, data: Dict[str, torch.Tensor], b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b using conjugate gradient method with numerical stability.
        
        Args:
            data: Buffer data
            b: Right-hand side vector
            
        Returns:
            Solution vector x
        """
        # Normalize the input vector
        b_norm = torch.norm(b)
        if b_norm < 1e-8:
            return torch.zeros_like(b)
        
        b = b / b_norm
        
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        
        for i in range(self.config.cg_iters):
            try:
                # Compute Ap with completely fresh computation
                Ap = self.compute_fisher_vector_product(data, p.detach())
                
                # Check for numerical issues
                if torch.isnan(Ap).any() or torch.isinf(Ap).any():
                    print(f"⚠️  NaN/Inf in Fisher-vector product at CG iteration {i}")
                    break
                
                # Standard CG update
                pAp = torch.dot(p, Ap)
                if abs(pAp) < 1e-10:
                    print(f"⚠️  Near-zero denominator in CG at iteration {i}")
                    break
                
                alpha = rsold / (pAp + 1e-10)
                
                # Clip alpha to prevent extreme steps
                alpha = torch.clamp(alpha, -1.0, 1.0)
                
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = torch.dot(r, r)
                
                # Check convergence
                if torch.sqrt(rsnew) < 1e-10:
                    break
                    
                beta = rsnew / (rsold + 1e-10)
                beta = torch.clamp(beta, -10.0, 10.0)  # Clip beta
                
                p = r + beta * p
                rsold = rsnew
                
            except Exception as e:
                print(f"⚠️  Error in CG iteration {i}: {e}")
                break
        
        # Restore original scale and clip result
        result = x * b_norm
        result = torch.clamp(result, -5.0, 5.0)  # Clip final result
        return result.detach()
    
    def compute_policy_gradient(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute policy gradient.
        
        Args:
            data: Buffer data
            
        Returns:
            Flattened policy gradient
        """
        observations = data['observations']
        actions = data['actions']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        
        # Clear gradients
        self.actor_critic.zero_grad()
        
        # Get new log probabilities
        if self.env_info['action_type'] == 'continuous':
            new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
        else:
            new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
        
        # Compute importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Policy objective (maximize expected advantage)
        policy_objective = torch.mean(ratio * advantages)
        
        # Compute gradient (negative because autograd computes gradients for minimization)
        policy_params = [p for name, p in self.actor_critic.named_parameters() 
                        if 'actor' in name and p.requires_grad]
        
        grads = torch.autograd.grad(policy_objective, policy_params, retain_graph=False)
        return self.flatten_gradients(grads)
    
    def compute_kl_divergence(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence between old and new policies.
        
        Args:
            data: Buffer data
            
        Returns:
            Mean KL divergence
        """
        observations = data['observations']
        actions = data['actions']
        old_log_probs = data['log_probs']
        
        # Get new log probabilities
        if self.env_info['action_type'] == 'continuous':
            new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
        else:
            new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
        
        # KL divergence: E[log(π_old) - log(π_new)]
        kl_div = torch.mean(old_log_probs - new_log_probs)
        return kl_div
    
    def flatten_gradients(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """Flatten list of gradients into single tensor with numerical stability."""
        flat_grads = torch.cat([g.view(-1) for g in grads])
        # Clip gradients to prevent numerical instability
        flat_grads = torch.clamp(flat_grads, -10.0, 10.0)
        return flat_grads
    
    def apply_update(self, update: torch.Tensor):
        """Apply flattened update to policy parameters with numerical stability."""
        update = torch.clamp(update, -1.0, 1.0)  # Clip updates
        pointer = 0
        for name, param in self.actor_critic.named_parameters():
            if 'actor' in name and param.requires_grad:
                num_params = param.numel()
                param_update = update[pointer:pointer + num_params].view_as(param)
                param.data += param_update
                # Clamp parameters to prevent extreme values
                param.data = torch.clamp(param.data, -10.0, 10.0)
                pointer += num_params
    
    def restore_parameters(self, old_params: List[torch.Tensor]):
        """Restore parameters from saved values."""
        pointer = 0
        for param in self.actor_critic.parameters():
            if param.requires_grad:
                param.data.copy_(old_params[pointer])
                pointer += 1
    
    def check_network_stability(self, data: Dict[str, torch.Tensor]) -> bool:
        """Check if network produces valid outputs."""
        try:
            observations = data['observations'][:10]  # Test with small batch
            with torch.no_grad():
                if self.env_info['action_type'] == 'continuous':
                    action_mean, action_std, value = self.actor_critic.forward(observations)
                    return not (torch.isnan(action_mean).any() or torch.isnan(action_std).any() or torch.isnan(value).any())
                else:
                    action_logits, value = self.actor_critic.forward(observations)
                    return not (torch.isnan(action_logits).any() or torch.isnan(value).any())
        except:
            return False
    
    def line_search(self, data: Dict[str, torch.Tensor], 
                   fullstep: torch.Tensor, 
                   expected_improve: float) -> Tuple[float, int]:
        """
        Perform line search to ensure KL constraint satisfaction with numerical stability.
        
        Args:
            data: Buffer data
            fullstep: Full natural gradient step
            expected_improve: Expected improvement
            
        Returns:
            Tuple of (step size, number of backtrack steps)
        """
        # Save old parameters
        old_params = [p.clone().detach() for p in self.actor_critic.parameters() if p.requires_grad]
        
        # Check initial network stability
        if not self.check_network_stability(data):
            print("⚠️  Network unstable before line search, skipping update")
            return 0.0, self.config.backtrack_iters
        
        # Compute old policy loss
        try:
            with torch.no_grad():
                old_loss = self.compute_policy_loss(data).item()
        except:
            print("⚠️  Cannot compute initial policy loss, skipping update")
            return 0.0, self.config.backtrack_iters
        
        for i in range(self.config.backtrack_iters):
            step_size = (self.config.backtrack_coeff ** i)
            
            # Apply step
            self.apply_update(fullstep * step_size)
            
            # Check network stability after update
            if not self.check_network_stability(data):
                print(f"⚠️  Network unstable after step {i}, restoring parameters")
                self.restore_parameters(old_params)
                continue
            
            try:
                # Check KL constraint
                with torch.no_grad():
                    kl_div = self.compute_kl_divergence(data).item()
                
                if not np.isfinite(kl_div):
                    print(f"⚠️  Non-finite KL divergence: {kl_div}")
                    self.restore_parameters(old_params)
                    continue
                
                if kl_div <= self.config.max_kl:
                    # Check improvement
                    with torch.no_grad():
                        new_loss = self.compute_policy_loss(data).item()
                    
                    if not np.isfinite(new_loss):
                        print(f"⚠️  Non-finite policy loss: {new_loss}")
                        self.restore_parameters(old_params)
                        continue
                    
                    actual_improve = old_loss - new_loss
                    improve_ratio = actual_improve / (abs(expected_improve) + 1e-8)
                    
                    if improve_ratio > 0.1:  # Accept step
                        print(f"✅ Line search step {i}: step_size={step_size:.6f}, KL={kl_div:.6f}, improve_ratio={improve_ratio:.3f}")
                        return step_size, i
                
                # Restore old parameters for next iteration
                self.restore_parameters(old_params)
                
            except Exception as e:
                print(f"⚠️  Error in line search step {i}: {e}")
                self.restore_parameters(old_params)
                continue
        
        # If no acceptable step found, don't update
        print("⚠️  Line search failed, no update applied")
        return 0.0, self.config.backtrack_iters
    
    def compute_policy_loss(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute policy loss for line search with numerical stability."""
        observations = data['observations']
        actions = data['actions']
        advantages = data['advantages']
        old_log_probs = data['log_probs']
        
        try:
            # Get new log probabilities
            if self.env_info['action_type'] == 'continuous':
                new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
            else:
                new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
            
            # Check for numerical issues
            if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                return torch.tensor(float('inf'))
            
            # Compute importance sampling ratio with clipping
            log_ratio = new_log_probs - old_log_probs
            log_ratio = torch.clamp(log_ratio, -10.0, 10.0)  # Prevent extreme ratios
            ratio = torch.exp(log_ratio)
            
            # Check ratio for numerical issues
            if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                return torch.tensor(float('inf'))
            
            # Policy loss (negative because we want to maximize)
            loss = -(ratio * advantages).mean()
            
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(float('inf'))
            
            return loss
            
        except Exception as e:
            print(f"⚠️  Error computing policy loss: {e}")
            return torch.tensor(float('inf'))
    
    def update_value_function(self, data: Dict[str, torch.Tensor]) -> float:
        """
        Update value function.
        
        Args:
            data: Buffer data
            
        Returns:
            Average value loss
        """
        observations = data['observations']
        returns = data['returns']
        
        value_losses = []
        
        for _ in range(self.config.value_iters):
            # Get value predictions
            if self.env_info['action_type'] == 'continuous':
                values = self.actor_critic.get_value(observations)
            else:
                _, values = self.actor_critic.forward(observations)
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Update value function
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            value_losses.append(value_loss.item())
        
        return np.mean(value_losses)
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using TRPO.
        
        Returns:
            Dictionary with training metrics
        """
        self.actor_critic.train()
        
        # Get buffer data
        data = self.buffer.get()
        
        # Compute policy gradient
        policy_grad = self.compute_policy_gradient(data)
        
        # Solve for natural gradient using conjugate gradient
        natural_grad = self.conjugate_gradient(data, policy_grad)
        
        # Compute step size
        quad_form = torch.dot(natural_grad, self.compute_fisher_vector_product(data, natural_grad.detach()))
        step_size_scale = torch.sqrt(2 * self.config.max_kl / (quad_form + 1e-8))
        fullstep = step_size_scale * natural_grad
        
        # Expected improvement
        expected_improve = torch.dot(policy_grad, fullstep).item()
        
        # Line search
        step_size, backtrack_steps = self.line_search(data, fullstep.detach(), expected_improve)
        
        # Update value function
        value_loss = self.update_value_function(data)
        
        # Compute final metrics
        with torch.no_grad():
            final_kl = self.compute_kl_divergence(data).item()
            
            # Explained variance
            y_true = data['returns'].cpu().numpy()
            y_pred = data['values'].cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
            
            # Policy loss for logging
            observations = data['observations']
            actions = data['actions']
            advantages = data['advantages']
            old_log_probs = data['log_probs']
            
            if self.env_info['action_type'] == 'continuous':
                new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
            else:
                new_log_probs, _ = self.actor_critic.get_log_prob_and_entropy(observations, actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            policy_loss = -(ratio * advantages).mean().item()
        
        # Reset buffer
        self.buffer.reset()
        self.num_updates += 1
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'kl_divergence': final_kl,
            'explained_variance': explained_var,
            'step_size': step_size,
            'backtrack_steps': backtrack_steps
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
        Train TRPO agent.
        
        Args:
            total_timesteps: Total training timesteps (overrides config)
            
        Returns:
            Training metrics history
        """
        if total_timesteps is None:
            total_timesteps = self.config.total_timesteps
        
        print(f"🚀 Starting TRPO training for {total_timesteps} timesteps")
        print(f"📊 Rollout steps: {self.config.n_steps}")
        print(f"🎯 Max KL: {self.config.max_kl}")
        print(f"🔄 CG iterations: {self.config.cg_iters}")
        
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
                print(f"  📏 KL Div: {update_stats['kl_divergence']:.6f}")
                print(f"  📐 Step Size: {update_stats['step_size']:.6f}")
                print(f"  🔄 Backtrack Steps: {update_stats['backtrack_steps']}")
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
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'num_updates': self.num_updates,
            'training_metrics': self.training_metrics
        }, filepath, _use_new_zipfile_serialization=False)
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.num_updates = checkpoint.get('num_updates', 0)
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics'] 