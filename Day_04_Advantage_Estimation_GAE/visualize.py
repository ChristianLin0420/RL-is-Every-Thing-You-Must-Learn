"""
Visualization Tools for GAE Analysis

Provides comprehensive plotting capabilities:
- Learning curves and training progress
- Advantage vs return comparisons
- Lambda sweep analysis
- Bias-variance tradeoff visualization
- Interactive training monitoring
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
import pandas as pd
from collections import defaultdict

from train import TrainingMetrics, TrainingConfig
from gae_utils import GAEComputer, Trajectory

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class GAEVisualizer:
    """Comprehensive visualization tools for GAE analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    def plot_training_progress(self, 
                             metrics: TrainingMetrics, 
                             config: TrainingConfig,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive training progress dashboard.
        
        Args:
            metrics: Training metrics
            config: Training configuration
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle(f'GAE Training Progress - {config.env_name} (λ={config.lambda_gae})', 
                    fontsize=16, fontweight='bold')
        
        episodes = range(len(metrics.episode_rewards))
        
        # 1. Episode Rewards
        ax = axes[0, 0]
        ax.plot(episodes, metrics.episode_rewards, alpha=0.6, linewidth=1, label='Episode Reward')
        
        # Moving average
        if len(metrics.episode_rewards) > 10:
            window = min(50, len(metrics.episode_rewards) // 4)
            rewards_smooth = pd.Series(metrics.episode_rewards).rolling(window, min_periods=1).mean()
            ax.plot(episodes, rewards_smooth, linewidth=2, label=f'Moving Avg ({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Policy and Value Losses
        ax = axes[0, 1]
        ax.plot(episodes, metrics.policy_losses, label='Policy Loss', color=self.colors[0])
        ax.plot(episodes, metrics.value_losses, label='Value Loss', color=self.colors[1])
        ax.plot(episodes, metrics.total_losses, label='Total Loss', color=self.colors[2])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Gradient Norms and Entropy
        ax = axes[0, 2]
        ax2 = ax.twinx()
        
        line1 = ax.plot(episodes, metrics.grad_norms, label='Grad Norm', color=self.colors[3])
        line2 = ax2.plot(episodes, metrics.entropies, label='Entropy', color=self.colors[4])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm', color=self.colors[3])
        ax2.set_ylabel('Entropy', color=self.colors[4])
        ax.set_title('Gradients & Exploration')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        # 4. Advantage Statistics
        ax = axes[1, 0]
        ax.fill_between(episodes, 
                       np.array(metrics.advantages_mean) - np.array(metrics.advantages_std),
                       np.array(metrics.advantages_mean) + np.array(metrics.advantages_std),
                       alpha=0.3, label='Std Range')
        ax.plot(episodes, metrics.advantages_mean, label='Mean Advantage', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Advantage')
        ax.set_title('GAE Advantages')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Returns vs Value Estimates
        ax = axes[1, 1]
        ax.plot(episodes, metrics.returns_mean, label='Returns', color=self.colors[0])
        ax.plot(episodes, metrics.value_estimates_mean, label='Value Estimates', color=self.colors[1])
        
        # Add error bars for some points
        sample_episodes = episodes[::max(1, len(episodes)//20)]
        sample_returns_std = [metrics.returns_std[i] for i in sample_episodes]
        sample_values_std = [metrics.value_estimates_std[i] for i in sample_episodes]
        
        ax.errorbar(sample_episodes, 
                   [metrics.returns_mean[i] for i in sample_episodes],
                   yerr=sample_returns_std,
                   fmt='o', alpha=0.6, capsize=3, color=self.colors[0])
        ax.errorbar(sample_episodes, 
                   [metrics.value_estimates_mean[i] for i in sample_episodes],
                   yerr=sample_values_std,
                   fmt='s', alpha=0.6, capsize=3, color=self.colors[1])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.set_title('Returns vs Value Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Training Time and Episode Length
        ax = axes[1, 2]
        ax2 = ax.twinx()
        
        line1 = ax.plot(episodes, metrics.training_times, label='Training Time', color=self.colors[5])
        line2 = ax2.plot(episodes, metrics.episode_lengths, label='Episode Length', color=self.colors[2])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Training Time (s)', color=self.colors[5])
        ax2.set_ylabel('Episode Length', color=self.colors[2])
        ax.set_title('Training Efficiency')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_advantage_comparison(self, 
                                trajectories: List[Trajectory],
                                lambdas: List[float] = [0.0, 0.5, 0.9, 0.95, 1.0],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare advantages computed with different λ values.
        
        Args:
            trajectories: List of trajectories
            lambdas: Lambda values to compare
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('GAE Advantage Comparison Across λ Values', fontsize=16, fontweight='bold')
        
        # Collect advantages for all lambda values
        all_advantages = {lam: [] for lam in lambdas}
        gae_computer = GAEComputer(gamma=0.99, normalize_advantages=False)
        
        for traj in trajectories[:20]:  # Use subset for visualization
            for lam in lambdas:
                gae_computer.lambda_gae = lam
                advantages, _ = gae_computer.compute_gae(
                    traj.rewards, traj.values, traj.dones, traj.next_value
                )
                all_advantages[lam].extend(advantages.tolist())
        
        # 1. Advantage Distributions
        ax = axes[0, 0]
        advantage_data = [all_advantages[lam] for lam in lambdas]
        box_plot = ax.boxplot(advantage_data, labels=[f'λ={lam}' for lam in lambdas], patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Advantage')
        ax.set_title('Advantage Distributions')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Variance vs Lambda
        ax = axes[0, 1]
        variances = [np.var(all_advantages[lam]) for lam in lambdas]
        ax.plot(lambdas, variances, 'o-', linewidth=2, markersize=8, color=self.colors[0])
        ax.set_xlabel('λ')
        ax.set_ylabel('Advantage Variance')
        ax.set_title('Variance vs λ (Bias-Variance Tradeoff)')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        min_var_idx = np.argmin(variances)
        ax.annotate(f'Min Variance\nλ={lambdas[min_var_idx]}', 
                   xy=(lambdas[min_var_idx], variances[min_var_idx]),
                   xytext=(lambdas[min_var_idx] + 0.1, variances[min_var_idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10)
        
        # 3. Advantage Time Series (sample trajectory)
        ax = axes[1, 0]
        if len(trajectories) > 0:
            sample_traj = trajectories[0]
            timesteps = range(len(sample_traj.rewards))
            
            for i, lam in enumerate(lambdas):
                gae_computer.lambda_gae = lam
                advantages, _ = gae_computer.compute_gae(
                    sample_traj.rewards, sample_traj.values, sample_traj.dones, sample_traj.next_value
                )
                ax.plot(timesteps, advantages, label=f'λ={lam}', 
                       color=self.colors[i % len(self.colors)], linewidth=2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Advantage')
            ax.set_title('Advantage Evolution (Sample Episode)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Statistics Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create statistics table
        stats_data = []
        for lam in lambdas:
            advs = all_advantages[lam]
            stats_data.append([
                f'λ={lam}',
                f'{np.mean(advs):.3f}',
                f'{np.std(advs):.3f}',
                f'{np.var(advs):.3f}',
                f'{np.min(advs):.3f}',
                f'{np.max(advs):.3f}'
            ])
        
        headers = ['Lambda', 'Mean', 'Std', 'Variance', 'Min', 'Max']
        table = ax.table(cellText=stats_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold')
        
        ax.set_title('Statistical Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_lambda_sweep_results(self, 
                                 sweep_results: Dict[float, List[TrainingMetrics]],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot results from lambda sweep experiment.
        
        Args:
            sweep_results: Results from run_lambda_sweep
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize, dpi=self.dpi)
        fig.suptitle('Lambda Sweep Analysis - Bias-Variance Tradeoff', fontsize=16, fontweight='bold')
        
        lambdas = sorted(sweep_results.keys())
        
        # Extract final performance for each lambda
        final_rewards = []
        reward_stds = []
        convergence_episodes = []
        final_value_losses = []
        final_advantage_stds = []
        
        for lam in lambdas:
            metrics_list = sweep_results[lam]
            
            # Final rewards (last 20% of episodes)
            seed_final_rewards = []
            seed_conv_episodes = []
            seed_value_losses = []
            seed_adv_stds = []
            
            for metrics in metrics_list:
                last_20_percent = len(metrics.episode_rewards) // 5
                final_reward = np.mean(metrics.episode_rewards[-last_20_percent:])
                seed_final_rewards.append(final_reward)
                
                # Convergence episode (when reward reaches 80% of final)
                target_reward = final_reward * 0.8
                conv_episode = next((i for i, r in enumerate(metrics.episode_rewards) if r >= target_reward), 
                                  len(metrics.episode_rewards))
                seed_conv_episodes.append(conv_episode)
                
                # Final losses and advantage statistics
                seed_value_losses.append(metrics.value_losses[-1])
                seed_adv_stds.append(np.mean(metrics.advantages_std[-last_20_percent:]))
            
            final_rewards.append(np.mean(seed_final_rewards))
            reward_stds.append(np.std(seed_final_rewards))
            convergence_episodes.append(np.mean(seed_conv_episodes))
            final_value_losses.append(np.mean(seed_value_losses))
            final_advantage_stds.append(np.mean(seed_adv_stds))
        
        # 1. Final Performance vs Lambda
        ax = axes[0, 0]
        ax.errorbar(lambdas, final_rewards, yerr=reward_stds, 
                   fmt='o-', linewidth=2, markersize=8, capsize=5, color=self.colors[0])
        ax.set_xlabel('λ')
        ax.set_ylabel('Final Reward')
        ax.set_title('Performance vs λ')
        ax.grid(True, alpha=0.3)
        
        # Highlight best lambda
        best_idx = np.argmax(final_rewards)
        ax.annotate(f'Best: λ={lambdas[best_idx]:.2f}', 
                   xy=(lambdas[best_idx], final_rewards[best_idx]),
                   xytext=(lambdas[best_idx] + 0.1, final_rewards[best_idx] + 10),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, fontweight='bold')
        
        # 2. Convergence Speed vs Lambda
        ax = axes[0, 1]
        ax.plot(lambdas, convergence_episodes, 'o-', linewidth=2, markersize=8, color=self.colors[1])
        ax.set_xlabel('λ')
        ax.set_ylabel('Episodes to Convergence')
        ax.set_title('Convergence Speed vs λ')
        ax.grid(True, alpha=0.3)
        
        # 3. Advantage Variance vs Lambda
        ax = axes[1, 0]
        ax.plot(lambdas, final_advantage_stds, 'o-', linewidth=2, markersize=8, color=self.colors[2])
        ax.set_xlabel('λ')
        ax.set_ylabel('Advantage Std')
        ax.set_title('Advantage Variance vs λ')
        ax.grid(True, alpha=0.3)
        
        # Add theoretical annotations
        ax.annotate('High Bias\nLow Variance', xy=(0.0, final_advantage_stds[0]),
                   xytext=(0.1, final_advantage_stds[0] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                   fontsize=9, ha='center')
        ax.annotate('Low Bias\nHigh Variance', xy=(1.0, final_advantage_stds[-1]),
                   xytext=(0.9, final_advantage_stds[-1] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                   fontsize=9, ha='center')
        
        # 4. Learning Curves Comparison
        ax = axes[1, 1]
        
        # Plot learning curves for selected lambdas
        selected_lambdas = [0.0, 0.5, 0.95, 1.0]
        
        for i, lam in enumerate(selected_lambdas):
            if lam in sweep_results:
                # Average across seeds
                all_rewards = []
                max_length = max(len(metrics.episode_rewards) for metrics in sweep_results[lam])
                
                for episode in range(max_length):
                    episode_rewards = []
                    for metrics in sweep_results[lam]:
                        if episode < len(metrics.episode_rewards):
                            episode_rewards.append(metrics.episode_rewards[episode])
                    if episode_rewards:
                        all_rewards.append(np.mean(episode_rewards))
                
                # Smooth the curve
                if len(all_rewards) > 10:
                    window = min(20, len(all_rewards) // 5)
                    smoothed = pd.Series(all_rewards).rolling(window, min_periods=1).mean()
                    ax.plot(range(len(smoothed)), smoothed, 
                           label=f'λ={lam}', linewidth=2, color=self.colors[i])
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def create_interactive_training_monitor(self, 
                                          metrics: TrainingMetrics,
                                          config: TrainingConfig) -> plt.Figure:
        """
        Create interactive training monitor with sliders.
        
        Args:
            metrics: Training metrics
            config: Training configuration
            
        Returns:
            Figure with interactive elements
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=self.dpi)
        fig.suptitle(f'Interactive Training Monitor - {config.env_name}', fontsize=16, fontweight='bold')
        
        # Make room for slider
        plt.subplots_adjust(bottom=0.15)
        
        # Initial plot range
        max_episodes = len(metrics.episode_rewards)
        current_episode = max_episodes
        
        def update_plots(episode_limit):
            """Update all plots based on episode limit."""
            episodes = range(min(episode_limit, len(metrics.episode_rewards)))
            
            # Clear all axes
            for ax in axes.flat:
                ax.clear()
            
            # Episode Rewards
            ax = axes[0, 0]
            ax.plot(episodes, metrics.episode_rewards[:len(episodes)], alpha=0.6)
            if len(episodes) > 10:
                window = min(20, len(episodes) // 4)
                rewards_smooth = pd.Series(metrics.episode_rewards[:len(episodes)]).rolling(window, min_periods=1).mean()
                ax.plot(episodes, rewards_smooth, linewidth=2, label=f'Moving Avg ({window})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Losses
            ax = axes[0, 1]
            ax.plot(episodes, metrics.policy_losses[:len(episodes)], label='Policy Loss')
            ax.plot(episodes, metrics.value_losses[:len(episodes)], label='Value Loss')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            # Advantages
            ax = axes[1, 0]
            ax.plot(episodes, metrics.advantages_mean[:len(episodes)], label='Mean Advantage')
            ax.fill_between(episodes, 
                           np.array(metrics.advantages_mean[:len(episodes)]) - np.array(metrics.advantages_std[:len(episodes)]),
                           np.array(metrics.advantages_mean[:len(episodes)]) + np.array(metrics.advantages_std[:len(episodes)]),
                           alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Advantage')
            ax.set_title('GAE Advantages')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Performance Summary
            ax = axes[1, 1]
            ax.axis('off')
            
            if len(episodes) > 0:
                recent_rewards = metrics.episode_rewards[max(0, len(episodes)-20):len(episodes)]
                stats_text = f"""
                Episode: {len(episodes)}/{max_episodes}
                
                Recent Performance (last 20):
                Mean Reward: {np.mean(recent_rewards):.2f}
                Std Reward: {np.std(recent_rewards):.2f}
                Max Reward: {np.max(recent_rewards):.2f}
                
                Current Metrics:
                Policy Loss: {metrics.policy_losses[len(episodes)-1]:.4f}
                Value Loss: {metrics.value_losses[len(episodes)-1]:.4f}
                Entropy: {metrics.entropies[len(episodes)-1]:.4f}
                Grad Norm: {metrics.grad_norms[len(episodes)-1]:.4f}
                
                GAE Statistics:
                λ = {config.lambda_gae}
                γ = {config.gamma}
                Advantage Mean: {metrics.advantages_mean[len(episodes)-1]:.3f}
                Advantage Std: {metrics.advantages_std[len(episodes)-1]:.3f}
                """
                ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=11,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.draw()
        
        # Create slider
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, 'Episode', 1, max_episodes, valinit=max_episodes, valfmt='%d')
        
        def on_slider_change(val):
            update_plots(int(val))
        
        slider.on_changed(on_slider_change)
        
        # Initial plot
        update_plots(current_episode)
        
        return fig


def compare_gae_methods(trajectories: List[Trajectory], 
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Compare different advantage estimation methods.
    
    Args:
        trajectories: List of trajectories
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    visualizer = GAEVisualizer()
    return visualizer.plot_advantage_comparison(trajectories, save_path=save_path)


if __name__ == "__main__":
    # Test visualization with synthetic data
    print("🧪 Testing GAE Visualization...")
    
    # Create synthetic training metrics
    episodes = 200
    metrics = TrainingMetrics()
    
    # Simulate training progress
    for i in range(episodes):
        # Simulated learning curve
        base_reward = 50 + 200 * (1 - np.exp(-i / 100)) + np.random.normal(0, 20)
        metrics.add_episode_result(base_reward, int(100 + 50 * np.random.random()))
        
        # Simulated losses (decreasing)
        policy_loss = 2.0 * np.exp(-i / 80) + np.random.normal(0, 0.1)
        value_loss = 1.0 * np.exp(-i / 60) + np.random.normal(0, 0.05)
        total_loss = policy_loss + value_loss
        entropy = 0.5 + 0.3 * np.exp(-i / 50) + np.random.normal(0, 0.02)
        grad_norm = 1.0 * np.exp(-i / 70) + np.random.normal(0, 0.05)
        
        # Simulated advantages and returns
        advantages = torch.randn(50) * 0.5
        returns = torch.randn(50) * 2 + 5
        values = returns + advantages
        
        metrics.add_training_step(
            policy_loss, value_loss, total_loss, entropy, grad_norm,
            advantages, returns, values, 0.1
        )
    
    config = TrainingConfig(env_name='CartPole-v1', lambda_gae=0.95)
    
    # Test training progress plot
    visualizer = GAEVisualizer()
    fig1 = visualizer.plot_training_progress(metrics, config)
    plt.title('Test Training Progress')
    plt.show()
    
    print("✅ Visualization tests completed!") 