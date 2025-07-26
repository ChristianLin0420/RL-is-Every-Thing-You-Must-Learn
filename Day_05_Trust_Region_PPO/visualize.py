"""
Visualization Tools for PPO Analysis
====================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    pass

class PPOVisualizer:
    """Visualization suite for PPO analysis."""
    
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.figsize = (12, 8)
        self.dpi = 300
    
    def plot_training_progress(self, metrics, save_name="training_progress"):
        """Plot comprehensive training progress."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PPO Training Progress', fontsize=16, fontweight='bold')
        
        def smooth(y, window=10):
            if len(y) < window:
                return y
            return pd.Series(y).rolling(window=window, min_periods=1).mean().values
        
        # Policy Loss
        if 'policy_loss' in metrics:
            axes[0, 0].plot(smooth(metrics['policy_loss']), color=self.colors[0])
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        if 'value_loss' in metrics:
            axes[0, 1].plot(smooth(metrics['value_loss']), color=self.colors[1])
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy
        if 'entropy' in metrics:
            axes[0, 2].plot(smooth(metrics['entropy']), color=self.colors[2])
            axes[0, 2].set_title('Policy Entropy')
            axes[0, 2].grid(True, alpha=0.3)
        
        # KL Divergence
        if 'kl_divergence' in metrics:
            axes[1, 0].plot(smooth(metrics['kl_divergence']), color=self.colors[3])
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Clip Fraction
        if 'clip_fraction' in metrics:
            axes[1, 1].plot(smooth(metrics['clip_fraction']), color=self.colors[4])
            axes[1, 1].set_title('Clip Fraction')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Explained Variance
        if 'explained_variance' in metrics:
            axes[1, 2].plot(smooth(metrics['explained_variance']), color=self.colors[0])
            axes[1, 2].set_title('Explained Variance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"📊 Training progress saved: {save_path}")
        return fig
    
    def plot_reward_curve(self, episode_rewards, window=100, save_name="reward_curve"):
        """Plot episode reward curve."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if len(episode_rewards) == 0:
            ax.text(0.5, 0.5, 'No episode data', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        episodes = np.arange(len(episode_rewards))
        ax.plot(episodes, episode_rewards, alpha=0.3, color=self.colors[0])
        
        if len(episode_rewards) >= window:
            smoothed = pd.Series(episode_rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, smoothed, color=self.colors[0], linewidth=2)
        
        mean_reward = np.mean(episode_rewards)
        ax.axhline(y=mean_reward, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_reward:.2f}')
        
        ax.set_title('Episode Rewards Over Time', fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"📈 Reward curve saved: {save_path}")
        return fig
    
    def plot_comparison(self, results, metric='mean_reward', save_name="comparison"):
        """Compare multiple experiments."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        names = list(results.keys())
        values = []
        stds = []
        
        for name in names:
            result = results[name]
            if metric in result['final_eval']:
                val = result['final_eval'][metric]
                std_key = metric.replace('mean_', 'std_')
                std = result['final_eval'].get(std_key, 0)
            else:
                val = 0
                std = 0
            values.append(val)
            stds.append(std)
        
        bars = ax.bar(names, values, yerr=stds, capsize=5, 
                     color=self.colors[:len(names)], alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(values),
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'Experiment Comparison: {metric.replace("_", " ").title()}')
        ax.set_ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}_{metric}.png")
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"📊 Comparison saved: {save_path}")
        return fig
    
    def plot_learning_curves_comparison(self, results, save_name="learning_curves"):
        """Compare learning curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['policy_loss', 'value_loss', 'entropy']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for j, (name, result) in enumerate(results.items()):
                if metric in result['metrics']:
                    values = result['metrics'][metric]
                    smoothed = pd.Series(values).rolling(window=10, min_periods=1).mean()
                    ax.plot(smoothed, label=name, color=self.colors[j % len(self.colors)])
            
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Update')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{save_name}.png")
        fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"📊 Learning curves saved: {save_path}")
        return fig
    
    def save_all_plots(self, agent, timestamp=None):
        """Save all visualization plots."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"📊 Generating PPO visualizations...")
        
        if hasattr(agent, 'training_metrics'):
            self.plot_training_progress(agent.training_metrics, f"training_{timestamp}")
        
        if hasattr(agent, 'episode_rewards'):
            self.plot_reward_curve(agent.episode_rewards, save_name=f"rewards_{timestamp}")
        
        print(f"✅ Visualizations saved to: {self.save_dir}") 