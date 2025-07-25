"""
Generalized Advantage Estimation (GAE) Implementation

Implements GAE-λ for bias-variance tradeoff in advantage estimation:
- GAE computation with configurable λ
- Multiple advantage estimation methods for comparison
- Bias-variance analysis tools
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class Trajectory:
    """Container for episode trajectory data."""
    states: torch.Tensor          # [T, state_dim]
    actions: torch.Tensor         # [T] or [T, action_dim]
    rewards: torch.Tensor         # [T]
    values: torch.Tensor          # [T]
    log_probs: torch.Tensor       # [T]
    dones: torch.Tensor           # [T] - whether episode ended at this step
    next_value: float = 0.0       # V(s_{T+1}) for bootstrapping


class GAEComputer:
    """
    Computes Generalized Advantage Estimation (GAE) and related metrics.
    """
    
    def __init__(self, 
                 gamma: float = 0.99, 
                 lambda_gae: float = 0.95,
                 normalize_advantages: bool = True,
                 epsilon: float = 1e-8):
        """
        Initialize GAE computer.
        
        Args:
            gamma: Discount factor
            lambda_gae: GAE parameter (0 = high bias/low variance, 1 = low bias/high variance)
            normalize_advantages: Whether to normalize advantages
            epsilon: Small constant for numerical stability
        """
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.normalize_advantages = normalize_advantages
        self.epsilon = epsilon
    
    def compute_gae(self, 
                   rewards: torch.Tensor, 
                   values: torch.Tensor, 
                   dones: torch.Tensor,
                   next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state after trajectory
            
        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute TD errors (delta_t)
        deltas = torch.zeros_like(rewards)
        for t in range(T):
            if t == T - 1:
                # Last step: use next_value for bootstrapping
                next_val = next_value if not dones[t] else 0.0
            else:
                # Use next state value, but zero if episode ended
                next_val = values[t + 1] if not dones[t] else 0.0
                
            deltas[t] = rewards[t] + self.gamma * next_val - values[t]
        
        # Compute GAE advantages using reverse iteration
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.lambda_gae * gae * (1 - dones[t].float())
            advantages[t] = gae
        
        # Compute returns as advantages + values
        returns = advantages + values
        
        # Normalize advantages if requested
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)
        
        return advantages, returns
    
    def compute_multiple_advantages(self, 
                                   rewards: torch.Tensor, 
                                   values: torch.Tensor, 
                                   dones: torch.Tensor,
                                   next_value: float = 0.0,
                                   lambdas: List[float] = [0.0, 0.5, 0.9, 0.95, 1.0]) -> Dict[str, torch.Tensor]:
        """
        Compute advantages with multiple λ values for comparison.
        
        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state after trajectory
            lambdas: List of λ values to test
            
        Returns:
            Dictionary mapping λ values to advantages
        """
        results = {}
        original_lambda = self.lambda_gae
        
        for lambda_val in lambdas:
            self.lambda_gae = lambda_val
            advantages, _ = self.compute_gae(rewards, values, dones, next_value)
            results[f'gae_lambda_{lambda_val}'] = advantages.clone()
            
        # Restore original lambda
        self.lambda_gae = original_lambda
        
        # Also compute Monte Carlo returns for comparison
        mc_advantages = self._compute_monte_carlo_advantages(rewards, values, dones)
        results['monte_carlo'] = mc_advantages
        
        # And n-step TD advantages
        for n in [1, 3, 5]:
            nstep_advantages = self._compute_nstep_advantages(rewards, values, dones, next_value, n)
            results[f'{n}_step_td'] = nstep_advantages
            
        return results
    
    def _compute_monte_carlo_advantages(self, 
                                      rewards: torch.Tensor, 
                                      values: torch.Tensor, 
                                      dones: torch.Tensor) -> torch.Tensor:
        """Compute Monte Carlo advantages (no bootstrapping)."""
        T = len(rewards)
        mc_returns = torch.zeros_like(rewards)
        
        # Compute discounted returns
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t].float())
            mc_returns[t] = running_return
            
        advantages = mc_returns - values
        
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)
            
        return advantages
    
    def _compute_nstep_advantages(self, 
                                 rewards: torch.Tensor, 
                                 values: torch.Tensor, 
                                 dones: torch.Tensor,
                                 next_value: float,
                                 n: int) -> torch.Tensor:
        """Compute n-step TD advantages."""
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        
        for t in range(T):
            # Compute n-step return
            nstep_return = 0.0
            gamma_power = 1.0
            
            for k in range(min(n, T - t)):
                nstep_return += gamma_power * rewards[t + k]
                gamma_power *= self.gamma
                
                if dones[t + k]:
                    break
            
            # Add bootstrapped value if we didn't reach episode end
            if t + n < T and not any(dones[t:t+n]):
                nstep_return += gamma_power * values[t + n]
            elif t + n >= T and not dones[-1]:
                nstep_return += gamma_power * next_value
                
            advantages[t] = nstep_return - values[t]
        
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)
            
        return advantages
    
    def analyze_bias_variance(self, 
                             trajectories: List[Trajectory],
                             true_values: Optional[torch.Tensor] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze bias-variance tradeoff for different λ values.
        
        Args:
            trajectories: List of trajectory data
            true_values: True value function (if available for synthetic envs)
            
        Returns:
            Dictionary with bias/variance analysis for each method
        """
        lambdas = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        all_advantages = {f'gae_lambda_{lam}': [] for lam in lambdas}
        all_advantages['monte_carlo'] = []
        
        # Collect advantages from all trajectories
        for traj in trajectories:
            advantages_dict = self.compute_multiple_advantages(
                traj.rewards, traj.values, traj.dones, traj.next_value, lambdas
            )
            
            for method, advantages in advantages_dict.items():
                if method in all_advantages:
                    all_advantages[method].append(advantages)
        
        # Compute statistics
        analysis = {}
        for method, advantage_lists in all_advantages.items():
            if len(advantage_lists) == 0:
                continue
                
            # Stack all advantages
            stacked_advantages = torch.cat(advantage_lists, dim=0)
            
            analysis[method] = {
                'mean': float(stacked_advantages.mean()),
                'std': float(stacked_advantages.std()),
                'variance': float(stacked_advantages.var()),
                'min': float(stacked_advantages.min()),
                'max': float(stacked_advantages.max()),
                'q25': float(torch.quantile(stacked_advantages, 0.25)),
                'q50': float(torch.quantile(stacked_advantages, 0.50)),
                'q75': float(torch.quantile(stacked_advantages, 0.75))
            }
            
            # If true values available, compute bias
            if true_values is not None and len(true_values) == len(stacked_advantages):
                bias = stacked_advantages - true_values
                analysis[method]['bias_mean'] = float(bias.mean())
                analysis[method]['bias_std'] = float(bias.std())
                analysis[method]['mse'] = float((bias ** 2).mean())
        
        return analysis


def compute_gae_trajectory(trajectory: Trajectory, 
                          gamma: float = 0.99, 
                          lambda_gae: float = 0.95,
                          normalize_advantages: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to compute GAE for a single trajectory.
    
    Args:
        trajectory: Trajectory data
        gamma: Discount factor
        lambda_gae: GAE parameter
        normalize_advantages: Whether to normalize advantages
        
    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    gae_computer = GAEComputer(gamma, lambda_gae, normalize_advantages)
    return gae_computer.compute_gae(
        trajectory.rewards, 
        trajectory.values, 
        trajectory.dones, 
        trajectory.next_value
    )


def create_trajectory_from_episode(states: List[np.ndarray],
                                  actions: List[Union[int, np.ndarray]],
                                  rewards: List[float],
                                  values: List[float],
                                  log_probs: List[float],
                                  dones: List[bool],
                                  next_value: float = 0.0) -> Trajectory:
    """
    Create Trajectory object from episode data.
    
    Args:
        states: List of state observations
        actions: List of actions taken
        rewards: List of rewards received
        values: List of value estimates
        log_probs: List of log probabilities
        dones: List of done flags
        next_value: Value of state after episode
        
    Returns:
        Trajectory object
    """
    return Trajectory(
        states=torch.tensor(np.array(states), dtype=torch.float32),
        actions=torch.tensor(np.array(actions), dtype=torch.long if isinstance(actions[0], int) else torch.float32),
        rewards=torch.tensor(np.array(rewards), dtype=torch.float32),
        values=torch.tensor(np.array(values), dtype=torch.float32),
        log_probs=torch.tensor(np.array(log_probs), dtype=torch.float32),
        dones=torch.tensor(np.array(dones), dtype=torch.bool),
        next_value=next_value
    )


if __name__ == "__main__":
    # Test GAE computation
    print("🧪 Testing GAE Computation...")
    
    # Create synthetic trajectory
    T = 20
    rewards = torch.randn(T) * 0.5 + 1.0  # Rewards around 1.0
    values = torch.randn(T) * 0.3 + 2.0   # Values around 2.0
    dones = torch.zeros(T, dtype=torch.bool)
    dones[-1] = True  # Episode ends at last step
    
    print(f"📊 Trajectory length: {T}")
    print(f"Mean reward: {rewards.mean():.3f}, Mean value: {values.mean():.3f}")
    
    # Test GAE with different λ values
    gae_computer = GAEComputer(gamma=0.99, lambda_gae=0.95)
    
    # Test single GAE computation
    advantages, returns = gae_computer.compute_gae(rewards, values, dones, next_value=0.0)
    print(f"\n🎯 GAE Results (λ=0.95):")
    print(f"Advantages - Mean: {advantages.mean():.3f}, Std: {advantages.std():.3f}")
    print(f"Returns - Mean: {returns.mean():.3f}, Std: {returns.std():.3f}")
    
    # Test multiple λ values
    multiple_advantages = gae_computer.compute_multiple_advantages(
        rewards, values, dones, next_value=0.0
    )
    
    print(f"\n📈 Comparison of Different Methods:")
    for method, advs in multiple_advantages.items():
        print(f"{method:15} - Mean: {advs.mean():6.3f}, Std: {advs.std():6.3f}")
    
    # Test bias-variance analysis with multiple trajectories
    trajectories = []
    for i in range(10):
        traj_rewards = torch.randn(T) * 0.5 + 1.0
        traj_values = torch.randn(T) * 0.3 + 2.0
        traj_dones = torch.zeros(T, dtype=torch.bool)
        traj_dones[-1] = True
        
        trajectory = Trajectory(
            states=torch.randn(T, 4),  # Dummy states
            actions=torch.randint(0, 2, (T,)),  # Dummy actions
            rewards=traj_rewards,
            values=traj_values,
            log_probs=torch.randn(T),  # Dummy log probs
            dones=traj_dones,
            next_value=0.0
        )
        trajectories.append(trajectory)
    
    analysis = gae_computer.analyze_bias_variance(trajectories)
    print(f"\n🔍 Bias-Variance Analysis:")
    for method, stats in analysis.items():
        print(f"{method:15} - Variance: {stats['variance']:.4f}, Std: {stats['std']:.4f}")
    
    print("\n✅ GAE computation tests completed!") 