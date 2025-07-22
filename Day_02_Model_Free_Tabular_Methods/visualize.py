"""
Visualization Functions for SARSA vs Q-Learning Analysis
Author: Day 2 - Model-Free Tabular Methods Challenge

This module provides comprehensive visualization capabilities for comparing
SARSA and Q-Learning algorithms across different metrics and environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import os
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class InteractiveTrainingAnimation:
    """
    Interactive training animation showing SARSA vs Q-Learning side by side.
    """
    
    def __init__(self, sarsa_agent, qlearning_agent, env, max_episodes: int = 500):
        """
        Initialize interactive training animation.
        
        Args:
            sarsa_agent: SARSA agent instance
            qlearning_agent: Q-Learning agent instance
            env: Environment instance
            max_episodes: Maximum episodes to train
        """
        self.sarsa_agent = sarsa_agent
        self.qlearning_agent = qlearning_agent
        self.env = env
        self.max_episodes = max_episodes
        
        # Store original hyperparameters for reset
        self.sarsa_agent.original_epsilon = sarsa_agent.epsilon
        self.qlearning_agent.original_epsilon = qlearning_agent.epsilon
        
        # Animation state
        self.current_episode = 0
        self.is_playing = False
        self.animation_speed = 100  # milliseconds
        
        # Training data storage
        self.sarsa_rewards = []
        self.qlearning_rewards = []
        self.sarsa_policies = []
        self.qlearning_policies = []
        self.sarsa_q_tables = []
        self.qlearning_q_tables = []
        
        # Colors
        self.colors = {
            'sarsa': '#2E86AB',
            'q_learning': '#A23B72'
        }
        
        # Action symbols
        self.action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        
        # Grid shape detection
        self.grid_shape = self._detect_grid_shape()
        
    def _detect_grid_shape(self):
        """Detect grid shape for visualization."""
        n_states = self.env.observation_space.n
        if hasattr(self.env.spec, 'id'):
            env_name = self.env.spec.id.lower()
            if 'cliff' in env_name and n_states == 48:
                return (4, 12)
            elif 'frozenlake' in env_name:
                if n_states == 16:
                    return (4, 4)
                elif n_states == 64:
                    return (8, 8)
        return None
    
    def train_one_episode(self):
        """Train both agents for one episode and store results."""
        if self.current_episode >= self.max_episodes:
            return False
        
        # Train SARSA
        sarsa_info = self.sarsa_agent.train_episode(self.env, max_steps=200)
        self.sarsa_rewards.append(sarsa_info['reward'])
        self.sarsa_policies.append(self.sarsa_agent.get_policy().copy())
        self.sarsa_q_tables.append(self.sarsa_agent.Q.copy())
        
        # Train Q-Learning
        qlearning_info = self.qlearning_agent.train_episode(self.env, max_steps=200)
        self.qlearning_rewards.append(qlearning_info['reward'])
        self.qlearning_policies.append(self.qlearning_agent.get_policy().copy())
        self.qlearning_q_tables.append(self.qlearning_agent.Q.copy())
        
        self.current_episode += 1
        return True
    
    def create_interactive_figure(self):
        """Create interactive training visualization."""
        # Create figure with subplots - larger and better spaced
        self.fig = plt.figure(figsize=(24, 14))
        self.fig.suptitle('Interactive SARSA vs Q-Learning Training Animation', 
                         fontsize=16, fontweight='bold', y=0.98)
        
        # Create grid layout with more comfortable spacing
        gs = self.fig.add_gridspec(4, 6, hspace=0.6, wspace=0.4, 
                                  top=0.93, bottom=0.12, left=0.05, right=0.97)
        
        # Learning curves (top row) - make it wider
        self.ax_rewards = self.fig.add_subplot(gs[0, :4])
        self.ax_rewards.set_title('Learning Curves', fontweight='bold', fontsize=14)
        self.ax_rewards.set_xlabel('Episode', fontsize=12)
        self.ax_rewards.set_ylabel('Reward', fontsize=12)
        self.ax_rewards.grid(True, alpha=0.3)
        self.ax_rewards.legend(['SARSA', 'Q-Learning'], fontsize=11)
        
        # Episode info (top right) - smaller but sufficient
        self.ax_info = self.fig.add_subplot(gs[0, 4:])
        self.ax_info.axis('off')
        
        if self.grid_shape:
            # Policy visualizations (second row) - better spacing
            self.ax_sarsa_policy = self.fig.add_subplot(gs[1, 0:2])
            self.ax_sarsa_policy.set_title('SARSA Policy', fontweight='bold', 
                                          color=self.colors['sarsa'], fontsize=13)
            
            self.ax_qlearning_policy = self.fig.add_subplot(gs[1, 2:4])
            self.ax_qlearning_policy.set_title('Q-Learning Policy', fontweight='bold', 
                                              color=self.colors['q_learning'], fontsize=13)
            
            self.ax_policy_diff = self.fig.add_subplot(gs[2, 4:6])
            self.ax_policy_diff.set_title('Policy Differences', fontweight='bold', fontsize=13)
            
            # Q-Value visualizations (third row) - better spacing
            self.ax_sarsa_q = self.fig.add_subplot(gs[2, 0:2])
            self.ax_sarsa_q.set_title('SARSA Q-Values (Best Action)', fontweight='bold', 
                                     color=self.colors['sarsa'], fontsize=13)
            
            self.ax_qlearning_q = self.fig.add_subplot(gs[2, 2:4])
            self.ax_qlearning_q.set_title('Q-Learning Q-Values (Best Action)', fontweight='bold', 
                                         color=self.colors['q_learning'], fontsize=13)
        
        # Controls (bottom row)
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Initialize plots
        self._init_plots()
        self._create_controls()
        
        return self.fig
    
    def _init_plots(self):
        """Initialize plot elements."""
        # Initialize learning curves
        self.sarsa_line, = self.ax_rewards.plot([], [], color=self.colors['sarsa'], 
                                               linewidth=2, label='SARSA')
        self.qlearning_line, = self.ax_rewards.plot([], [], color=self.colors['q_learning'], 
                                                   linewidth=2, label='Q-Learning')
        self.ax_rewards.legend()
        
        # Initialize policy plots if grid environment
        if self.grid_shape:
            rows, cols = self.grid_shape
            
            # Create empty policy grids
            self.sarsa_policy_im = self.ax_sarsa_policy.imshow(
                np.zeros(self.grid_shape), cmap='Greys', alpha=0.1)
            self.qlearning_policy_im = self.ax_qlearning_policy.imshow(
                np.zeros(self.grid_shape), cmap='Greys', alpha=0.1)
            self.policy_diff_im = self.ax_policy_diff.imshow(
                np.zeros(self.grid_shape), cmap='RdYlBu_r', alpha=0.7)
            
            # Set up grid properties
            for ax in [self.ax_sarsa_policy, self.ax_qlearning_policy, self.ax_policy_diff]:
                ax.set_xlim(-0.5, cols-0.5)
                ax.set_ylim(-0.5, rows-0.5)
                ax.set_xticks(range(cols))
                ax.set_yticks(range(rows))
                ax.grid(True, alpha=0.3)
            
            # Initialize Q-value plots
            self.sarsa_q_im = self.ax_sarsa_q.imshow(
                np.zeros(self.grid_shape), cmap='RdYlBu_r', vmin=-100, vmax=0)
            self.qlearning_q_im = self.ax_qlearning_q.imshow(
                np.zeros(self.grid_shape), cmap='RdYlBu_r', vmin=-100, vmax=0)
            
            # Add colorbar for Q-values
            self.fig.colorbar(self.sarsa_q_im, ax=self.ax_sarsa_q, fraction=0.046, pad=0.04)
            self.fig.colorbar(self.qlearning_q_im, ax=self.ax_qlearning_q, fraction=0.046, pad=0.04)
    
    def _create_controls(self):
        """Create interactive controls."""
        # Control buttons - better positioning and spacing
        ax_play = plt.axes([0.08, 0.03, 0.12, 0.05])
        ax_step = plt.axes([0.22, 0.03, 0.12, 0.05])
        ax_reset = plt.axes([0.36, 0.03, 0.12, 0.05])
        
        self.btn_play = Button(ax_play, '▶️ Play/Pause')
        self.btn_step = Button(ax_step, '⏭️ Step')
        self.btn_reset = Button(ax_reset, '🔄 Reset')
        
        # Speed slider - larger and better positioned
        ax_speed = plt.axes([0.55, 0.03, 0.15, 0.05])
        self.slider_speed = Slider(ax_speed, '🎚️ Speed', 1, 10, valinit=5, valfmt='%d')
        
        # Episode slider - larger and better positioned
        ax_episode = plt.axes([0.8, 0.03, 0.15, 0.05])
        self.slider_episode = Slider(ax_episode, '📊 Episode', 0, self.max_episodes, 
                                   valinit=0, valfmt='%d')
        
        # Connect callbacks
        self.btn_play.on_clicked(self._toggle_play)
        self.btn_step.on_clicked(self._step_forward)
        self.btn_reset.on_clicked(self._reset_animation)
        self.slider_speed.on_changed(self._update_speed)
        self.slider_episode.on_changed(self._jump_to_episode)
    
    def _toggle_play(self, event):
        """Toggle play/pause."""
        if self.is_playing:
            self._pause_animation()
        else:
            self._start_animation()
    
    def _start_animation(self):
        """Start automatic training animation."""
        self.is_playing = True
        self.animation = FuncAnimation(
            self.fig, self._animate_step, interval=self.animation_speed,
            repeat=False, cache_frame_data=False
        )
    
    def _pause_animation(self):
        """Pause animation."""
        self.is_playing = False
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
    
    def _animate_step(self, frame):
        """Animation step function."""
        if not self.is_playing:
            return []
        
        # Train one episode
        success = self.train_one_episode()
        if not success:
            self._pause_animation()
            return []
        
        # Update plots
        self._update_plots()
        return []
    
    def _step_forward(self, event):
        """Manual step forward."""
        if self.current_episode < self.max_episodes:
            self.train_one_episode()
            self._update_plots()
    
    def _reset_animation(self, event):
        """Reset animation to beginning."""
        self._pause_animation()
        
        # Store original hyperparameters
        original_epsilon = getattr(self.sarsa_agent, 'original_epsilon', 0.2)
        
        # Reset agents to initial state
        self.sarsa_agent.Q = self.sarsa_agent._initialize_q_table('zeros')
        self.qlearning_agent.Q = self.qlearning_agent._initialize_q_table('zeros')
        self.sarsa_agent.epsilon = original_epsilon
        self.qlearning_agent.epsilon = original_epsilon
        
        # Reset training info
        self.sarsa_agent.training_info = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_table_history': [],
            'epsilon_history': [],
            'td_errors': []
        }
        self.qlearning_agent.training_info = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_table_history': [],
            'epsilon_history': [],
            'td_errors': []
        }
        
        # Clear animation data
        self.current_episode = 0
        self.sarsa_rewards = []
        self.qlearning_rewards = []
        self.sarsa_policies = []
        self.qlearning_policies = []
        self.sarsa_q_tables = []
        self.qlearning_q_tables = []
        
        # Clear and update plots
        self._clear_all_plots()
        self._update_plots()
        if hasattr(self, 'slider_episode'):
            self.slider_episode.reset()
    
    def _clear_all_plots(self):
        """Clear all plot data for fresh start."""
        # Clear learning curve
        self.sarsa_line.set_data([], [])
        self.qlearning_line.set_data([], [])
        self.ax_rewards.set_xlim(0, 1)
        self.ax_rewards.set_ylim(-1, 1)
        
        # Clear policy plots if they exist
        if self.grid_shape:
            self.ax_sarsa_policy.clear()
            self.ax_qlearning_policy.clear()
            self.ax_policy_diff.clear()
            
            # Reset axes properties
            rows, cols = self.grid_shape
            for ax in [self.ax_sarsa_policy, self.ax_qlearning_policy, self.ax_policy_diff]:
                ax.set_xlim(-0.5, cols-0.5)
                ax.set_ylim(-0.5, rows-0.5)
                ax.set_xticks(range(cols))
                ax.set_yticks(range(rows))
                ax.grid(True, alpha=0.3)
            
            # Reset titles with consistent font sizes
            self.ax_sarsa_policy.set_title('🔵 SARSA Policy', fontweight='bold', 
                                          color=self.colors['sarsa'], fontsize=13)
            self.ax_qlearning_policy.set_title('🟣 Q-Learning Policy', fontweight='bold', 
                                              color=self.colors['q_learning'], fontsize=13)
            self.ax_policy_diff.set_title('🔍 Policy Differences', fontweight='bold', fontsize=13)
            
            # Clear Q-value plots
            self.sarsa_q_im.set_array(np.zeros(self.grid_shape))
            self.qlearning_q_im.set_array(np.zeros(self.grid_shape))
            self.sarsa_q_im.set_clim(-1, 0)
            self.qlearning_q_im.set_clim(-1, 0)
    
    def _update_speed(self, val):
        """Update animation speed."""
        self.animation_speed = int(1000 / val)  # Convert to milliseconds
    
    def _jump_to_episode(self, episode):
        """Jump to specific episode."""
        target_episode = int(episode)
        if target_episode < self.current_episode:
            # Need to reset and retrain (call with dummy event)
            self._reset_animation(None)
        
        # Train up to target episode
        while self.current_episode < target_episode and self.current_episode < self.max_episodes:
            self.train_one_episode()
        
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        if len(self.sarsa_rewards) == 0:
            return
        
        # Update learning curves
        episodes = range(1, len(self.sarsa_rewards) + 1)
        self.sarsa_line.set_data(episodes, self.sarsa_rewards)
        self.qlearning_line.set_data(episodes, self.qlearning_rewards)
        
        # Adjust axes
        if len(self.sarsa_rewards) > 1:
            self.ax_rewards.set_xlim(0, len(self.sarsa_rewards))
            all_rewards = self.sarsa_rewards + self.qlearning_rewards
            self.ax_rewards.set_ylim(min(all_rewards) * 1.1, max(all_rewards) * 1.1)
        
        # Update info text with better formatting
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Create more readable info display
        info_lines = [
            f"Episode: {self.current_episode}/{self.max_episodes}",
            "",
            "Current Performance:",
            f"   SARSA: {self.sarsa_rewards[-1]:.1f}",
            f"   Q-Learning: {self.qlearning_rewards[-1]:.1f}",
            "",
            "Average (last 10):",
            f"   SARSA: {np.mean(self.sarsa_rewards[-10:]):.1f}",
            f"   Q-Learning: {np.mean(self.qlearning_rewards[-10:]):.1f}",
            "",
            "Exploration (ε):",
            f"   SARSA: {self.sarsa_agent.epsilon:.3f}",
            f"   Q-Learning: {self.qlearning_agent.epsilon:.3f}"
        ]
        
        info_text = "\n".join(info_lines)
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Update policy and Q-value plots if grid environment
        if self.grid_shape and len(self.sarsa_policies) > 0:
            self._update_policy_plots()
            self._update_q_value_plots()
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def _update_policy_plots(self):
        """Update policy visualization plots."""
        if len(self.sarsa_policies) == 0:
            return
        
        rows, cols = self.grid_shape
        
        # Get current policies
        sarsa_policy = self.sarsa_policies[-1].reshape(self.grid_shape)
        qlearning_policy = self.qlearning_policies[-1].reshape(self.grid_shape)
        
        # Clear and redraw policy arrows
        self.ax_sarsa_policy.clear()
        self.ax_qlearning_policy.clear()
        self.ax_policy_diff.clear()
        
        # SARSA policy
        self.ax_sarsa_policy.imshow(np.ones_like(sarsa_policy), cmap='Greys', alpha=0.1)
        self._draw_policy_arrows(self.ax_sarsa_policy, sarsa_policy, self.colors['sarsa'])
        self.ax_sarsa_policy.set_title('SARSA Policy', fontweight='bold', color=self.colors['sarsa'])
        
        # Q-Learning policy
        self.ax_qlearning_policy.imshow(np.ones_like(qlearning_policy), cmap='Greys', alpha=0.1)
        self._draw_policy_arrows(self.ax_qlearning_policy, qlearning_policy, self.colors['q_learning'])
        self.ax_qlearning_policy.set_title('Q-Learning Policy', fontweight='bold', color=self.colors['q_learning'])
        
        # Policy differences with better visibility
        diff = (sarsa_policy != qlearning_policy).astype(int)
        self.ax_policy_diff.imshow(diff, cmap='RdYlBu_r', alpha=0.7)
        for i in range(rows):
            for j in range(cols):
                symbol = '✗' if diff[i, j] == 1 else '✓'
                color = 'red' if diff[i, j] == 1 else 'darkgreen'
                self.ax_policy_diff.text(j, i, symbol, ha='center', va='center',
                                       fontsize=16, color=color, fontweight='bold')
        self.ax_policy_diff.set_title('🔍 Policy Differences', fontweight='bold')
        
        # Set up grids
        for ax in [self.ax_sarsa_policy, self.ax_qlearning_policy, self.ax_policy_diff]:
            ax.set_xlim(-0.5, cols-0.5)
            ax.set_ylim(-0.5, rows-0.5)
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.grid(True, alpha=0.3)
    
    def _draw_policy_arrows(self, ax, policy_grid, color):
        """Draw policy arrows on grid."""
        rows, cols = policy_grid.shape
        for i in range(rows):
            for j in range(cols):
                action = policy_grid[i, j]
                if action in self.action_symbols:
                    # Larger, more visible arrows
                    ax.text(j, i, self.action_symbols[action],
                           ha='center', va='center', fontsize=16,
                           color=color, fontweight='bold')
    
    def _update_q_value_plots(self):
        """Update Q-value visualization plots."""
        if len(self.sarsa_q_tables) == 0:
            return
        
        # Get current Q-tables and extract state values
        sarsa_q = self.sarsa_q_tables[-1]
        qlearning_q = self.qlearning_q_tables[-1]
        
        sarsa_values = np.max(sarsa_q, axis=1).reshape(self.grid_shape)
        qlearning_values = np.max(qlearning_q, axis=1).reshape(self.grid_shape)
        
        # Update Q-value heatmaps
        vmin = min(sarsa_values.min(), qlearning_values.min())
        vmax = max(sarsa_values.max(), qlearning_values.max())
        
        self.sarsa_q_im.set_array(sarsa_values)
        self.sarsa_q_im.set_clim(vmin, vmax)
        
        self.qlearning_q_im.set_array(qlearning_values)
        self.qlearning_q_im.set_clim(vmin, vmax)
    
    def run(self):
        """Run the interactive training animation."""
        print("🎬 Starting Interactive SARSA vs Q-Learning Training Animation")
        print("=" * 60)
        print("🎮 Controls:")
        print("   ▶️ Play/Pause: Start/stop automatic training")
        print("   ⏭️ Step: Train one episode manually")
        print("   🔄 Reset: Restart training from beginning")
        print("   🎚️ Speed: Control training speed")
        print("   📊 Episode: Jump to specific episode")
        print("=" * 60)
        
        fig = self.create_interactive_figure()
        self._update_plots()  # Initial empty plot
        plt.show()
        
        return fig
    
    def save_trained_agents(self, filepath_prefix: str = "trained_agents"):
        """Save the trained agents for later analysis."""
        if self.current_episode > 0:
            # Save Q-tables
            np.save(f"{filepath_prefix}_sarsa_qtable.npy", self.sarsa_agent.Q)
            np.save(f"{filepath_prefix}_qlearning_qtable.npy", self.qlearning_agent.Q)
            
            # Save training history
            history = {
                'sarsa_rewards': self.sarsa_rewards,
                'qlearning_rewards': self.qlearning_rewards,
                'episodes_trained': self.current_episode,
                'sarsa_final_policy': self.sarsa_agent.get_policy(),
                'qlearning_final_policy': self.qlearning_agent.get_policy()
            }
            
            import json
            with open(f"{filepath_prefix}_history.json", 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_history = {}
                for key, value in history.items():
                    if isinstance(value, np.ndarray):
                        serializable_history[key] = value.tolist()
                    else:
                        serializable_history[key] = value
                json.dump(serializable_history, f, indent=2)
            
            print(f"💾 Trained agents saved:")
            print(f"   Q-tables: {filepath_prefix}_*_qtable.npy")
            print(f"   History: {filepath_prefix}_history.json")
            print(f"   Episodes trained: {self.current_episode}")


class VisualizationSuite:
    """
    Comprehensive visualization suite for RL experiment analysis.
    """
    
    def __init__(self, results_dir: str = "results", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization suite.
        
        Args:
            results_dir: Directory to save visualizations
            figsize: Default figure size
        """
        self.results_dir = results_dir
        self.figsize = figsize
        os.makedirs(results_dir, exist_ok=True)
        
        # Color schemes for algorithms
        self.colors = {
            'sarsa': '#2E86AB',      # Blue
            'q_learning': '#A23B72',  # Purple/Pink
            'comparison': '#F18F01'   # Orange
        }
        
        # Action symbols for policy visualization
        self.action_symbols = {
            0: "↑", 1: "↓", 2: "←", 3: "→"
        }
    
    def plot_learning_curves(
        self, 
        sarsa_results: List[Dict], 
        qlearning_results: List[Dict],
        env_name: str = "",
        window_size: int = 50,
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot learning curves comparing SARSA and Q-Learning.
        
        Args:
            sarsa_results: List of SARSA experiment results
            qlearning_results: List of Q-Learning experiment results
            env_name: Environment name for title
            window_size: Moving average window size
            save_path: Path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Learning Curves Comparison: SARSA vs Q-Learning in {env_name}', 
                     fontsize=16, fontweight='bold')
        
        # Extract episode rewards for all seeds
        def extract_and_smooth(results, window_size):
            all_curves = []
            for result in results:
                rewards = result['training_info']['episode_rewards']
                # Apply moving average
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                all_curves.append(smoothed)
            return all_curves
        
        sarsa_curves = extract_and_smooth(sarsa_results, window_size)
        qlearning_curves = extract_and_smooth(qlearning_results, window_size)
        
        # 1. Episode Rewards (with confidence intervals)
        self._plot_curves_with_confidence(
            ax1, sarsa_curves, qlearning_curves, 
            "Episode Rewards (Moving Average)", "Episode", "Reward"
        )
        
        # 2. Cumulative Rewards
        sarsa_cumulative = [np.cumsum(result['training_info']['episode_rewards']) 
                          for result in sarsa_results]
        qlearning_cumulative = [np.cumsum(result['training_info']['episode_rewards']) 
                              for result in qlearning_results]
        
        self._plot_curves_with_confidence(
            ax2, sarsa_cumulative, qlearning_cumulative,
            "Cumulative Rewards", "Episode", "Cumulative Reward"
        )
        
        # 3. TD Errors
        sarsa_td_errors = [result['training_info']['td_errors'] for result in sarsa_results]
        qlearning_td_errors = [result['training_info']['td_errors'] for result in qlearning_results]
        
        sarsa_td_smooth = extract_and_smooth([{'training_info': {'episode_rewards': td}} 
                                            for td in sarsa_td_errors], window_size)
        qlearning_td_smooth = extract_and_smooth([{'training_info': {'episode_rewards': td}} 
                                                for td in qlearning_td_errors], window_size)
        
        self._plot_curves_with_confidence(
            ax3, sarsa_td_smooth, qlearning_td_smooth,
            "TD Error (Moving Average)", "Episode", "TD Error"
        )
        
        # 4. Epsilon Decay
        sarsa_epsilon = [result['training_info']['epsilon_history'] for result in sarsa_results]
        qlearning_epsilon = [result['training_info']['epsilon_history'] for result in qlearning_results]
        
        self._plot_curves_with_confidence(
            ax4, sarsa_epsilon, qlearning_epsilon,
            "Exploration Rate (ε)", "Episode", "Epsilon"
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning curves saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_curves_with_confidence(
        self, 
        ax: plt.Axes, 
        sarsa_curves: List[np.ndarray], 
        qlearning_curves: List[np.ndarray],
        title: str, 
        xlabel: str, 
        ylabel: str
    ):
        """Helper function to plot curves with confidence intervals."""
        # Ensure all curves have the same length
        min_length = min([min([len(c) for c in sarsa_curves]), 
                         min([len(c) for c in qlearning_curves])])
        
        sarsa_array = np.array([curve[:min_length] for curve in sarsa_curves])
        qlearning_array = np.array([curve[:min_length] for curve in qlearning_curves])
        
        episodes = np.arange(min_length)
        
        # SARSA statistics
        sarsa_mean = np.mean(sarsa_array, axis=0)
        sarsa_std = np.std(sarsa_array, axis=0)
        
        # Q-Learning statistics
        qlearning_mean = np.mean(qlearning_array, axis=0)
        qlearning_std = np.std(qlearning_array, axis=0)
        
        # Plot means
        ax.plot(episodes, sarsa_mean, color=self.colors['sarsa'], 
                linewidth=2, label='SARSA', alpha=0.8)
        ax.plot(episodes, qlearning_mean, color=self.colors['q_learning'], 
                linewidth=2, label='Q-Learning', alpha=0.8)
        
        # Plot confidence intervals
        ax.fill_between(episodes, sarsa_mean - sarsa_std, sarsa_mean + sarsa_std,
                       color=self.colors['sarsa'], alpha=0.2)
        ax.fill_between(episodes, qlearning_mean - qlearning_std, qlearning_mean + qlearning_std,
                       color=self.colors['q_learning'], alpha=0.2)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_policy_comparison(
        self,
        sarsa_result: Dict,
        qlearning_result: Dict,
        env_name: str = "",
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot side-by-side policy comparison.
        
        Args:
            sarsa_result: Single SARSA experiment result
            qlearning_result: Single Q-Learning experiment result
            env_name: Environment name
            save_path: Path to save figure
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Policy Comparison: SARSA vs Q-Learning\n{env_name}', 
                     fontsize=16, fontweight='bold')
        
        # Extract policies
        sarsa_policy = sarsa_result['final_policy']
        qlearning_policy = qlearning_result['final_policy']
        
        # Determine grid shape based on environment
        grid_shape = self._get_grid_shape(len(sarsa_policy), env_name)
        
        if grid_shape:
            # Plot SARSA policy
            self._plot_single_policy(
                ax1, sarsa_policy.reshape(grid_shape), 
                "SARSA Policy", env_name, self.colors['sarsa']
            )
            
            # Plot Q-Learning policy
            self._plot_single_policy(
                ax2, qlearning_policy.reshape(grid_shape),
                "Q-Learning Policy", env_name, self.colors['q_learning']
            )
            
            # Plot policy differences
            self._plot_policy_difference(
                ax3, sarsa_policy.reshape(grid_shape), 
                qlearning_policy.reshape(grid_shape), "Policy Differences"
            )
        else:
            # For non-grid environments, show as bar plots
            self._plot_policy_bars(ax1, sarsa_policy, "SARSA Policy")
            self._plot_policy_bars(ax2, qlearning_policy, "Q-Learning Policy")
            self._plot_policy_agreement(ax3, sarsa_policy, qlearning_policy)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🎯 Policy comparison saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_q_value_heatmaps(
        self,
        sarsa_result: Dict,
        qlearning_result: Dict,
        env_name: str = "",
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot Q-value heatmaps for both algorithms.
        
        Args:
            sarsa_result: SARSA experiment result
            qlearning_result: Q-Learning experiment result
            env_name: Environment name
            save_path: Path to save figure
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Q-Value Heatmaps: SARSA vs Q-Learning\n{env_name}', 
                     fontsize=16, fontweight='bold')
        
        # Extract Q-tables
        sarsa_q = sarsa_result['final_q_table']
        qlearning_q = qlearning_result['final_q_table']
        
        # Determine grid shape
        grid_shape = self._get_grid_shape(sarsa_q.shape[0], env_name)
        
        if grid_shape:
            # Plot Q-values for each action
            actions = ["↑ (Up)", "↓ (Down)", "← (Left)", "→ (Right)"]
            
            # Determine common colormap range
            vmin = min(sarsa_q.min(), qlearning_q.min())
            vmax = max(sarsa_q.max(), qlearning_q.max())
            
            for action in range(4):
                # SARSA Q-values
                sarsa_q_action = sarsa_q[:, action].reshape(grid_shape)
                im1 = axes[0, action].imshow(sarsa_q_action, cmap='RdYlBu_r', 
                                           vmin=vmin, vmax=vmax)
                axes[0, action].set_title(f'SARSA: {actions[action]}')
                self._add_grid_values(axes[0, action], sarsa_q_action)
                
                # Q-Learning Q-values
                qlearning_q_action = qlearning_q[:, action].reshape(grid_shape)
                im2 = axes[1, action].imshow(qlearning_q_action, cmap='RdYlBu_r',
                                           vmin=vmin, vmax=vmax)
                axes[1, action].set_title(f'Q-Learning: {actions[action]}')
                self._add_grid_values(axes[1, action], qlearning_q_action)
            
            # Add colorbar
            fig.colorbar(im1, ax=axes, orientation='horizontal', 
                        fraction=0.05, pad=0.05, label='Q-Value')
        
        else:
            # For non-grid environments, plot as bar charts
            for action in range(min(4, sarsa_q.shape[1])):
                axes[0, action].bar(range(sarsa_q.shape[0]), sarsa_q[:, action], 
                                  color=self.colors['sarsa'], alpha=0.7)
                axes[0, action].set_title(f'SARSA: Action {action}')
                axes[0, action].set_xlabel('State')
                axes[0, action].set_ylabel('Q-Value')
                
                axes[1, action].bar(range(qlearning_q.shape[0]), qlearning_q[:, action],
                                  color=self.colors['q_learning'], alpha=0.7)
                axes[1, action].set_title(f'Q-Learning: Action {action}')
                axes[1, action].set_xlabel('State')
                axes[1, action].set_ylabel('Q-Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔥 Q-value heatmaps saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_performance_comparison(
        self,
        comparison_results: Dict,
        env_names: List[str] = None,
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot performance comparison across multiple metrics.
        
        Args:
            comparison_results: Results from comparison experiments
            env_names: List of environment names
            save_path: Path to save figure
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        if env_names is None:
            env_names = list(comparison_results.keys())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison: SARSA vs Q-Learning', 
                     fontsize=16, fontweight='bold')
        
        # Extract metrics for each environment
        metrics = ['final_reward', 'success_rate', 'training_time', 'learning_curve']
        metric_labels = ['Final Reward', 'Success Rate', 'Training Time (s)', 'Learning Curve']
        axes = [ax1, ax2, ax3, ax4]
        
        for i, (metric, label, ax) in enumerate(zip(metrics, metric_labels, axes)):
            sarsa_means = []
            sarsa_stds = []
            qlearning_means = []
            qlearning_stds = []
            
            for env_name in env_names:
                if env_name in comparison_results and metric in comparison_results[env_name]:
                    comp = comparison_results[env_name][metric]
                    sarsa_means.append(comp['sarsa_mean'])
                    sarsa_stds.append(comp['sarsa_std'])
                    qlearning_means.append(comp['qlearning_mean'])
                    qlearning_stds.append(comp['qlearning_std'])
                else:
                    sarsa_means.append(0)
                    sarsa_stds.append(0)
                    qlearning_means.append(0)
                    qlearning_stds.append(0)
            
            # Create bar plot
            x = np.arange(len(env_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, sarsa_means, width, yerr=sarsa_stds,
                          color=self.colors['sarsa'], alpha=0.8, label='SARSA',
                          capsize=5)
            bars2 = ax.bar(x + width/2, qlearning_means, width, yerr=qlearning_stds,
                          color=self.colors['q_learning'], alpha=0.8, label='Q-Learning',
                          capsize=5)
            
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Environment')
            ax.set_ylabel(label)
            ax.set_xticks(x)
            ax.set_xticklabels([name.replace('-v0', '').replace('-v1', '') for name in env_names], 
                              rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            self._add_bar_labels(ax, bars1)
            self._add_bar_labels(ax, bars2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Performance comparison saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_summary_dashboard(
        self,
        sarsa_results: List[Dict],
        qlearning_results: List[Dict],
        comparison_analysis: Dict,
        env_name: str = "",
        save_path: str = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create comprehensive summary dashboard.
        
        Args:
            sarsa_results: SARSA experiment results
            qlearning_results: Q-Learning experiment results
            comparison_analysis: Statistical comparison analysis
            env_name: Environment name
            save_path: Path to save figure
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle(f'SARSA vs Q-Learning: Complete Analysis Dashboard\n{env_name}', 
                     fontsize=18, fontweight='bold')
        
        # 1. Learning curves (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        sarsa_rewards = [np.convolve(r['training_info']['episode_rewards'], 
                                    np.ones(20)/20, mode='valid') for r in sarsa_results]
        qlearning_rewards = [np.convolve(r['training_info']['episode_rewards'], 
                                        np.ones(20)/20, mode='valid') for r in qlearning_results]
        self._plot_curves_with_confidence(ax1, sarsa_rewards, qlearning_rewards,
                                        "Learning Curves (20-episode moving average)", 
                                        "Episode", "Reward")
        
        # 2. Performance metrics comparison (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_metrics_radar(ax2, comparison_analysis)
        
        # 3. Policy visualization (middle row)
        if sarsa_results and qlearning_results:
            sarsa_policy = sarsa_results[0]['final_policy']
            qlearning_policy = qlearning_results[0]['final_policy']
            grid_shape = self._get_grid_shape(len(sarsa_policy), env_name)
            
            if grid_shape:
                ax3 = fig.add_subplot(gs[1, 0])
                self._plot_single_policy(ax3, sarsa_policy.reshape(grid_shape),
                                       "SARSA Policy", env_name, self.colors['sarsa'])
                
                ax4 = fig.add_subplot(gs[1, 1])
                self._plot_single_policy(ax4, qlearning_policy.reshape(grid_shape),
                                       "Q-Learning Policy", env_name, self.colors['q_learning'])
                
                ax5 = fig.add_subplot(gs[1, 2])
                self._plot_policy_difference(ax5, sarsa_policy.reshape(grid_shape),
                                           qlearning_policy.reshape(grid_shape),
                                           "Policy Differences")
        
        # 4. Statistical analysis (bottom row)
        ax6 = fig.add_subplot(gs[2:, :2])
        self._plot_statistical_comparison(ax6, comparison_analysis)
        
        # 5. Q-value distribution (bottom right)
        ax7 = fig.add_subplot(gs[2:, 2:])
        self._plot_q_value_distributions(ax7, sarsa_results, qlearning_results)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📋 Summary dashboard saved to: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    # Helper methods
    
    def _get_grid_shape(self, n_states: int, env_name: str) -> Optional[Tuple[int, int]]:
        """Determine grid shape for environment."""
        env_name_lower = env_name.lower()
        if 'cliffwalking' in env_name_lower and n_states == 48:
            return (4, 12)
        elif 'frozenlake' in env_name_lower:
            if n_states == 16:
                return (4, 4)
            elif n_states == 64:
                return (8, 8)
        return None
    
    def _plot_single_policy(self, ax, policy_grid, title, env_name, color):
        """Plot a single policy as arrows on grid."""
        rows, cols = policy_grid.shape
        
        # Create background
        ax.imshow(np.ones_like(policy_grid), cmap='Greys', alpha=0.1)
        
        # Add arrows for each cell
        for i in range(rows):
            for j in range(cols):
                action = policy_grid[i, j]
                if action in self.action_symbols:
                    ax.text(j, i, self.action_symbols[action], 
                           ha='center', va='center', fontsize=14, 
                           color=color, fontweight='bold')
        
        # Special states for CliffWalking
        if 'cliff' in env_name.lower():
            # Mark cliff states (bottom row, excluding start and goal)
            for j in range(1, cols-1):
                ax.add_patch(Rectangle((j-0.4, rows-1-0.4), 0.8, 0.8, 
                                     facecolor='red', alpha=0.3))
            
            # Mark start and goal
            ax.add_patch(Rectangle((-0.4, rows-1-0.4), 0.8, 0.8, 
                                 facecolor='green', alpha=0.5))  # Start
            ax.add_patch(Rectangle((cols-1-0.4, rows-1-0.4), 0.8, 0.8, 
                                 facecolor='gold', alpha=0.5))  # Goal
        
        ax.set_xlim(-0.5, cols-0.5)
        ax.set_ylim(-0.5, rows-0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_policy_difference(self, ax, policy1, policy2, title):
        """Plot differences between two policies."""
        diff = (policy1 != policy2).astype(int)
        im = ax.imshow(diff, cmap='RdYlBu_r', alpha=0.7)
        
        rows, cols = diff.shape
        for i in range(rows):
            for j in range(cols):
                if diff[i, j] == 1:
                    ax.text(j, i, '✗', ha='center', va='center', 
                           fontsize=16, color='red', fontweight='bold')
                else:
                    ax.text(j, i, '✓', ha='center', va='center', 
                           fontsize=12, color='green', fontweight='bold')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Different'),
                          Patch(facecolor='blue', alpha=0.7, label='Same')]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _add_grid_values(self, ax, values):
        """Add value annotations to grid."""
        rows, cols = values.shape
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, f'{values[i, j]:.2f}', 
                       ha='center', va='center', fontsize=8, 
                       color='white' if abs(values[i, j]) > np.max(np.abs(values))*0.5 else 'black')
    
    def _add_bar_labels(self, ax, bars):
        """Add value labels on top of bars."""
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_metrics_radar(self, ax, comparison_analysis):
        """Plot radar chart of performance metrics."""
        if not comparison_analysis:
            ax.text(0.5, 0.5, 'No comparison data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        metrics = list(comparison_analysis.keys())
        if not metrics:
            return
        
        # Normalize metrics for radar plot
        sarsa_values = []
        qlearning_values = []
        
        for metric in metrics:
            if metric in comparison_analysis:
                sarsa_val = comparison_analysis[metric]['sarsa_mean']
                qlearning_val = comparison_analysis[metric]['qlearning_mean']
                
                # Normalize (simple approach)
                max_val = max(abs(sarsa_val), abs(qlearning_val))
                if max_val > 0:
                    sarsa_values.append(sarsa_val / max_val)
                    qlearning_values.append(qlearning_val / max_val)
                else:
                    sarsa_values.append(0)
                    qlearning_values.append(0)
        
        if not sarsa_values:
            return
        
        # Create simple bar comparison instead of radar (simpler implementation)
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, sarsa_values, width, label='SARSA', 
               color=self.colors['sarsa'], alpha=0.7)
        ax.bar(x + width/2, qlearning_values, width, label='Q-Learning',
               color=self.colors['q_learning'], alpha=0.7)
        
        ax.set_title('Normalized Performance Metrics', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_comparison(self, ax, comparison_analysis):
        """Plot statistical comparison of metrics."""
        if not comparison_analysis:
            ax.text(0.5, 0.5, 'No comparison data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        metrics = list(comparison_analysis.keys())
        differences = [comparison_analysis[m]['difference'] for m in metrics if m in comparison_analysis]
        std_errors = [comparison_analysis[m]['pooled_std'] for m in metrics if m in comparison_analysis]
        
        if not differences:
            return
        
        # Create horizontal bar plot of differences
        y_pos = np.arange(len(metrics))
        
        bars = ax.barh(y_pos, differences, xerr=std_errors, 
                      color=[self.colors['q_learning'] if d > 0 else self.colors['sarsa'] 
                            for d in differences], alpha=0.7, capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_xlabel('Difference (Q-Learning - SARSA)')
        ax.set_title('Statistical Comparison\n(Positive = Q-Learning Better)', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (diff, std_err) in enumerate(zip(differences, std_errors)):
            if abs(diff) > std_err:
                ax.text(diff + (0.1 if diff > 0 else -0.1), i, 
                       '***' if abs(diff) > 2*std_err else '*',
                       ha='left' if diff > 0 else 'right', va='center',
                       fontweight='bold', color='red')
    
    def _plot_q_value_distributions(self, ax, sarsa_results, qlearning_results):
        """Plot Q-value distributions."""
        if not sarsa_results or not qlearning_results:
            ax.text(0.5, 0.5, 'No Q-value data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Collect all Q-values
        sarsa_q_values = np.concatenate([result['final_q_table'].flatten() 
                                        for result in sarsa_results])
        qlearning_q_values = np.concatenate([result['final_q_table'].flatten() 
                                           for result in qlearning_results])
        
        # Plot histograms
        ax.hist(sarsa_q_values, bins=30, alpha=0.6, label='SARSA', 
               color=self.colors['sarsa'], density=True)
        ax.hist(qlearning_q_values, bins=30, alpha=0.6, label='Q-Learning',
               color=self.colors['q_learning'], density=True)
        
        ax.set_xlabel('Q-Value')
        ax.set_ylabel('Density')
        ax.set_title('Q-Value Distributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        sarsa_mean = np.mean(sarsa_q_values)
        qlearning_mean = np.mean(qlearning_q_values)
        ax.axvline(sarsa_mean, color=self.colors['sarsa'], linestyle='--', alpha=0.8)
        ax.axvline(qlearning_mean, color=self.colors['q_learning'], linestyle='--', alpha=0.8)
        
        ax.text(0.02, 0.98, f'SARSA mean: {sarsa_mean:.3f}\nQ-Learning mean: {qlearning_mean:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_policy_bars(self, ax, policy, title):
        """Plot policy as bar chart for non-grid environments."""
        ax.bar(range(len(policy)), policy, color=self.colors['sarsa'], alpha=0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('State')
        ax.set_ylabel('Action')
        ax.set_xticks(range(0, len(policy), max(1, len(policy)//10)))
    
    def _plot_policy_agreement(self, ax, policy1, policy2):
        """Plot policy agreement for non-grid environments."""
        agreement = (policy1 == policy2).astype(int)
        ax.bar(range(len(agreement)), agreement, 
               color=self.colors['comparison'], alpha=0.7)
        ax.set_title('Policy Agreement', fontweight='bold')
        ax.set_xlabel('State')
        ax.set_ylabel('Agreement (1=Same, 0=Different)')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(0, len(agreement), max(1, len(agreement)//10)))


def save_all_visualizations(
    sarsa_results: List[Dict],
    qlearning_results: List[Dict],
    comparison_analysis: Dict,
    env_name: str,
    results_dir: str = "results"
) -> Dict[str, str]:
    """
    Generate and save all visualization types.
    
    Args:
        sarsa_results: SARSA experiment results
        qlearning_results: Q-Learning experiment results  
        comparison_analysis: Statistical comparison
        env_name: Environment name
        results_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization type to file path
    """
    vis = VisualizationSuite(results_dir)
    saved_files = {}
    
    env_clean = env_name.replace('-v0', '').replace('-v1', '')
    
    # Learning curves
    fig1 = vis.plot_learning_curves(sarsa_results, qlearning_results, env_name, show=False)
    path1 = os.path.join(results_dir, f'learning_curves_{env_clean}.png')
    fig1.savefig(path1, dpi=300, bbox_inches='tight')
    saved_files['learning_curves'] = path1
    plt.close(fig1)
    
    # Policy comparison
    if sarsa_results and qlearning_results:
        fig2 = vis.plot_policy_comparison(sarsa_results[0], qlearning_results[0], env_name, show=False)
        path2 = os.path.join(results_dir, f'policy_comparison_{env_clean}.png')
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        saved_files['policy_comparison'] = path2
        plt.close(fig2)
        
        # Q-value heatmaps
        fig3 = vis.plot_q_value_heatmaps(sarsa_results[0], qlearning_results[0], env_name, show=False)
        path3 = os.path.join(results_dir, f'q_value_heatmaps_{env_clean}.png')
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        saved_files['q_value_heatmaps'] = path3
        plt.close(fig3)
        
        # Summary dashboard
        fig4 = vis.create_summary_dashboard(sarsa_results, qlearning_results, comparison_analysis, env_name, show=False)
        path4 = os.path.join(results_dir, f'summary_dashboard_{env_clean}.png')
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        saved_files['summary_dashboard'] = path4
        plt.close(fig4)
    
    print(f"📊 All visualizations saved to: {results_dir}")
    return saved_files


if __name__ == "__main__":
    print("🎨 Testing Visualization Module...")
    
    # Create dummy data for testing
    dummy_sarsa = [{
        'training_info': {
            'episode_rewards': np.random.normal(-20, 10, 500).cumsum(),
            'td_errors': np.random.exponential(0.1, 500),
            'epsilon_history': np.linspace(0.1, 0.01, 500)
        },
        'final_policy': np.random.randint(0, 4, 48),
        'final_q_table': np.random.normal(0, 1, (48, 4))
    }]
    
    dummy_qlearning = [{
        'training_info': {
            'episode_rewards': np.random.normal(-18, 8, 500).cumsum(),
            'td_errors': np.random.exponential(0.12, 500),
            'epsilon_history': np.linspace(0.1, 0.01, 500)
        },
        'final_policy': np.random.randint(0, 4, 48),
        'final_q_table': np.random.normal(0, 1, (48, 4))
    }]
    
    dummy_comparison = {
        'final_reward': {'sarsa_mean': -120, 'qlearning_mean': -110, 'difference': 10, 'pooled_std': 5},
        'success_rate': {'sarsa_mean': 0.8, 'qlearning_mean': 0.85, 'difference': 0.05, 'pooled_std': 0.02}
    }
    
    vis = VisualizationSuite()
    vis.plot_learning_curves(dummy_sarsa, dummy_qlearning, "CliffWalking-v0", show=False)
    
    print("✅ Visualization module test completed!") 