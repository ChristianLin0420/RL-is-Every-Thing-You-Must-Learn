"""
Visualization Tools for REINFORCE

This module provides comprehensive visualization tools for analyzing
REINFORCE training progress, including reward curves, action distributions,
gradient analysis, and policy behavior over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import torch

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass  # Use default style if seaborn not available
        
try:
    sns.set_palette("husl")
except:
    pass  # Use default palette if seaborn not available


class REINFORCEVisualizer:
    """
    Comprehensive visualization suite for REINFORCE algorithm analysis.
    
    Provides static and interactive visualizations for:
    - Training progress (rewards, losses, gradients)
    - Action distribution analysis
    - Policy behavior over time
    - Comparison between different configurations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def plot_training_progress(self, 
                             training_stats: Dict[str, List], 
                             eval_stats: Optional[List[Dict]] = None,
                             save_path: Optional[str] = None,
                             show_baseline: bool = True) -> plt.Figure:
        """
        Plot comprehensive training progress including rewards, losses, and gradients.
        
        Args:
            training_stats: Training statistics from REINFORCE agent
            eval_stats: Optional evaluation statistics
            save_path: Path to save the figure
            show_baseline: Whether to show baseline values
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('REINFORCE Training Progress Analysis', fontsize=16, fontweight='bold')
        
        episodes = range(1, len(training_stats['episode_returns']) + 1)
        
        # 1. Episode Returns
        ax = axes[0, 0]
        ax.plot(episodes, training_stats['episode_returns'], 
               color=self.colors[0], alpha=0.7, linewidth=1)
        
        # Moving average for smoother visualization
        if len(training_stats['episode_returns']) > 10:
            window = min(50, len(training_stats['episode_returns']) // 10)
            moving_avg = self._moving_average(training_stats['episode_returns'], window)
            ax.plot(episodes[:len(moving_avg)], moving_avg, 
                   color=self.colors[1], linewidth=2, label=f'Moving Avg ({window})')
            ax.legend()
            
        ax.set_title('Episode Returns', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.grid(True, alpha=0.3)
        
        # Add evaluation results if available
        if eval_stats:
            eval_episodes = [stat['episode'] for stat in eval_stats]
            eval_rewards = [stat['mean_reward'] for stat in eval_stats]
            eval_errors = [stat['std_reward'] for stat in eval_stats]
            ax.errorbar(eval_episodes, eval_rewards, yerr=eval_errors, 
                       color=self.colors[2], marker='o', linestyle='--', 
                       label='Evaluation', capsize=5)
            ax.legend()
        
        # 2. Policy Loss
        ax = axes[0, 1]
        ax.plot(episodes, training_stats['policy_losses'], 
               color=self.colors[3], alpha=0.8)
        ax.set_title('Policy Loss', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
        
        # 3. Gradient Norms
        ax = axes[0, 2]
        ax.plot(episodes, training_stats['gradient_norms'], 
               color=self.colors[4], alpha=0.8)
        ax.set_title('Gradient Norms', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Gradient Norm')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Episode Lengths
        ax = axes[1, 0]
        ax.plot(episodes, training_stats['episode_lengths'], 
               color=self.colors[5], alpha=0.7)
        if len(training_stats['episode_lengths']) > 10:
            window = min(50, len(training_stats['episode_lengths']) // 10)
            moving_avg = self._moving_average(training_stats['episode_lengths'], window)
            ax.plot(episodes[:len(moving_avg)], moving_avg, 
                   color=self.colors[0], linewidth=2, label=f'Moving Avg ({window})')
            ax.legend()
        ax.set_title('Episode Lengths', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
        
        # 5. Entropy (Exploration)
        ax = axes[1, 1]
        ax.plot(episodes, training_stats['entropy_values'], 
               color=self.colors[1], alpha=0.8)
        ax.set_title('Policy Entropy (Exploration)', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax.grid(True, alpha=0.3)
        
        # 6. Baseline Values (if using baseline)
        ax = axes[1, 2]
        if show_baseline and 'baseline_values' in training_stats:
            ax.plot(episodes, training_stats['baseline_values'], 
                   color=self.colors[2], alpha=0.8)
            ax.set_title('Baseline Values', fontweight='bold')
        else:
            # Show reward distribution instead
            ax.hist(training_stats['episode_returns'], bins=30, 
                   color=self.colors[3], alpha=0.7, edgecolor='black')
            ax.set_title('Reward Distribution', fontweight='bold')
            ax.set_ylabel('Frequency')
        ax.set_xlabel('Episode' if show_baseline else 'Reward')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_action_distribution_evolution(self, 
                                         training_stats: Dict[str, List],
                                         action_names: List[str] = None,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot how action distribution evolves during training.
        
        Args:
            training_stats: Training statistics containing action distributions
            action_names: Names for actions (e.g., ['Left', 'Right'])
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Handle potential shape issues with action distributions
        action_distributions = np.array(training_stats['action_distributions'])
        
        # Squeeze out any extra dimensions (e.g., from (2, 1, 500) to (2, 500))
        while action_distributions.ndim > 2:
            action_distributions = np.squeeze(action_distributions)
        
        # Ensure we have shape (num_episodes, num_actions)
        if action_distributions.ndim == 2 and action_distributions.shape[0] < action_distributions.shape[1]:
            action_distributions = action_distributions.T
            
        n_actions = action_distributions.shape[1]
        episodes = range(1, len(action_distributions) + 1)
        
        if action_names is None:
            action_names = [f'Action {i}' for i in range(n_actions)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Action Distribution Evolution', fontsize=16, fontweight='bold')
        
        # 1. Action probabilities over time
        for action_idx in range(n_actions):
            ax1.plot(episodes, action_distributions[:, action_idx], 
                    label=action_names[action_idx], 
                    color=self.colors[action_idx % len(self.colors)],
                    linewidth=2)
        
        ax1.set_title('Action Probabilities Over Time', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Probability')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Heatmap of action distribution evolution
        im = ax2.imshow(action_distributions.T, aspect='auto', cmap='YlOrRd',
                       extent=[1, len(episodes), n_actions-0.5, -0.5])
        ax2.set_title('Action Distribution Heatmap', fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Action')
        ax2.set_yticks(range(n_actions))
        ax2.set_yticklabels(action_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Probability', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_policy_comparison(self, 
                              results_dict: Dict[str, Dict],
                              metric: str = 'episode_returns',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Compare multiple REINFORCE configurations.
        
        Args:
            results_dict: Dictionary with configuration names as keys and training stats as values
            metric: Metric to compare ('episode_returns', 'policy_losses', etc.)
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Policy Comparison: {metric.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # Plot learning curves
        for i, (name, stats) in enumerate(results_dict.items()):
            episodes = range(1, len(stats[metric]) + 1)
            color = self.colors[i % len(self.colors)]
            
            # Raw data
            ax1.plot(episodes, stats[metric], alpha=0.3, color=color)
            
            # Moving average
            if len(stats[metric]) > 10:
                window = min(50, len(stats[metric]) // 10)
                moving_avg = self._moving_average(stats[metric], window)
                ax1.plot(episodes[:len(moving_avg)], moving_avg, 
                        label=name, color=color, linewidth=2)
        
        ax1.set_title('Learning Curves', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot of final performance
        final_performances = []
        labels = []
        for name, stats in results_dict.items():
            if len(stats[metric]) > 100:  # Only if we have enough data
                final_performances.append(stats[metric][-50:])  # Last 50 episodes
                labels.append(name)
        
        if final_performances:
            ax2.boxplot(final_performances, labels=labels)
            ax2.set_title('Final Performance Distribution', fontweight='bold')
            ax2.set_ylabel(metric.replace('_', ' ').title())
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_training_visualization(self, 
                                                training_stats: Dict[str, List],
                                                env_name: str = "CartPole") -> plt.Figure:
        """
        Create an interactive visualization of the training process.
        
        Args:
            training_stats: Training statistics from REINFORCE agent
            env_name: Name of the environment for display
            
        Returns:
            Matplotlib figure with interactive controls
        """
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'Interactive REINFORCE Training Analysis - {env_name}', 
                    fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Main reward plot
        ax_reward = fig.add_subplot(gs[0, :])
        ax_loss = fig.add_subplot(gs[1, 0])
        ax_grad = fig.add_subplot(gs[1, 1])
        ax_entropy = fig.add_subplot(gs[1, 2])
        ax_actions = fig.add_subplot(gs[2, :2])
        ax_info = fig.add_subplot(gs[2, 2])
        
        # Store data and axes for interaction
        self.interactive_data = {
            'training_stats': training_stats,
            'axes': {
                'reward': ax_reward,
                'loss': ax_loss, 
                'grad': ax_grad,
                'entropy': ax_entropy,
                'actions': ax_actions,
                'info': ax_info
            },
            'current_episode': len(training_stats['episode_returns'])
        }
        
        # Initial plots
        self._update_interactive_plots()
        
        # Add slider for episode selection
        ax_slider = plt.axes([0.2, 0.02, 0.5, 0.03])
        self.episode_slider = Slider(ax_slider, 'Episode', 1, 
                                   len(training_stats['episode_returns']),
                                   valinit=len(training_stats['episode_returns']),
                                   valfmt='%d')
        self.episode_slider.on_changed(self._update_episode)
        
        return fig
    
    def _update_interactive_plots(self):
        """Update all interactive plots based on current episode."""
        stats = self.interactive_data['training_stats']
        current_ep = self.interactive_data['current_episode']
        axes = self.interactive_data['axes']
        
        episodes = range(1, current_ep + 1)
        
        # Clear all axes
        for ax in axes.values():
            ax.clear()
        
        # Reward plot
        axes['reward'].plot(episodes, stats['episode_returns'][:current_ep], 
                          color=self.colors[0], alpha=0.7)
        if current_ep > 10:
            window = min(20, current_ep // 5)
            moving_avg = self._moving_average(stats['episode_returns'][:current_ep], window)
            axes['reward'].plot(episodes[:len(moving_avg)], moving_avg, 
                              color=self.colors[1], linewidth=2)
        axes['reward'].set_title('Episode Returns')
        axes['reward'].grid(True, alpha=0.3)
        
        # Loss plot
        axes['loss'].plot(episodes, stats['policy_losses'][:current_ep], 
                        color=self.colors[2])
        axes['loss'].set_title('Policy Loss')
        axes['loss'].grid(True, alpha=0.3)
        
        # Gradient plot
        axes['grad'].plot(episodes, stats['gradient_norms'][:current_ep], 
                        color=self.colors[3])
        axes['grad'].set_title('Gradient Norm')
        axes['grad'].set_yscale('log')
        axes['grad'].grid(True, alpha=0.3)
        
        # Entropy plot
        axes['entropy'].plot(episodes, stats['entropy_values'][:current_ep], 
                           color=self.colors[4])
        axes['entropy'].set_title('Policy Entropy')
        axes['entropy'].grid(True, alpha=0.3)
        
        # Action distribution
        if 'action_distributions' in stats:
            action_dists = np.array(stats['action_distributions'][:current_ep])
            if len(action_dists) > 0:
                for i in range(action_dists.shape[1]):
                    axes['actions'].plot(episodes, action_dists[:, i], 
                                       label=f'Action {i}', 
                                       color=self.colors[i % len(self.colors)])
                axes['actions'].set_title('Action Probabilities')
                axes['actions'].legend()
                axes['actions'].grid(True, alpha=0.3)
        
        # Info panel
        axes['info'].text(0.1, 0.8, f'Episode: {current_ep}', fontsize=12, fontweight='bold')
        axes['info'].text(0.1, 0.6, f'Last Reward: {stats["episode_returns"][current_ep-1]:.1f}', fontsize=10)
        if current_ep >= 10:
            avg_reward = np.mean(stats['episode_returns'][max(0, current_ep-10):current_ep])
            axes['info'].text(0.1, 0.4, f'Avg (last 10): {avg_reward:.1f}', fontsize=10)
        axes['info'].text(0.1, 0.2, f'Policy Loss: {stats["policy_losses"][current_ep-1]:.3f}', fontsize=10)
        axes['info'].axis('off')
        
        plt.draw()
    
    def _update_episode(self, val):
        """Update episode for interactive visualization."""
        self.interactive_data['current_episode'] = int(val)
        self._update_interactive_plots()
    
    def _moving_average(self, data: List[float], window: int) -> List[float]:
        """Compute moving average with given window size."""
        if len(data) < window:
            return data
        return [np.mean(data[i:i+window]) for i in range(len(data) - window + 1)]
    
    def save_all_visualizations(self, 
                              training_stats: Dict[str, List],
                              eval_stats: Optional[List[Dict]] = None,
                              save_dir: str = "figures/",
                              prefix: str = "reinforce") -> List[str]:
        """
        Save all standard visualizations to files.
        
        Args:
            training_stats: Training statistics
            eval_stats: Optional evaluation statistics
            save_dir: Directory to save figures
            prefix: Prefix for figure filenames
            
        Returns:
            List of saved file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Training progress
        fig1 = self.plot_training_progress(training_stats, eval_stats)
        path1 = os.path.join(save_dir, f"{prefix}_training_progress.png")
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        saved_files.append(path1)
        
        # Action distribution
        if 'action_distributions' in training_stats:
            fig2 = self.plot_action_distribution_evolution(training_stats)
            path2 = os.path.join(save_dir, f"{prefix}_action_distribution.png")
            fig2.savefig(path2, dpi=300, bbox_inches='tight')
            plt.close(fig2)
            saved_files.append(path2)
        
        return saved_files


# Utility function for quick visualization
def quick_plot_results(training_stats: Dict[str, List], 
                      title: str = "REINFORCE Training Results") -> plt.Figure:
    """
    Quick utility function to plot basic training results.
    
    Args:
        training_stats: Training statistics dictionary
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    visualizer = REINFORCEVisualizer()
    fig = visualizer.plot_training_progress(training_stats)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    return fig 