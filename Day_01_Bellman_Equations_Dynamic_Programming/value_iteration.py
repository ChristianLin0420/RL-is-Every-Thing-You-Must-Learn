"""
Value Iteration Algorithm Implementation
Author: Day 1 - Bellman Equations & Dynamic Programming Challenge

Implements the Bellman Optimality Equation:
V*(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from typing import Tuple, Dict, List
from gridworld import GridWorld


class ValueIteration:
    """
    Value Iteration algorithm for solving MDPs.
    
    The algorithm iteratively applies the Bellman optimality operator:
    V_{k+1}(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
    """
    
    def __init__(self, env: GridWorld, threshold: float = 1e-6, max_iterations: int = 1000):
        """
        Initialize Value Iteration solver.
        
        Args:
            env: GridWorld environment
            threshold: Convergence threshold (ε)
            max_iterations: Maximum number of iterations
        """
        self.env = env
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        # Initialize value function to zeros
        self.V = np.zeros(env.n_states)
        self.policy = np.zeros(env.n_states, dtype=int)
        
        # Metrics tracking
        self.convergence_history = []
        self.value_history = []
        self.iterations = 0
    
    def bellman_update(self, state_idx: int) -> float:
        """
        Apply Bellman optimality operator for a single state.
        
        V(s) = max_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
        
        Args:
            state_idx: Linear index of the state
            
        Returns:
            Updated value for the state
        """
        state = self.env.index_to_state(state_idx)
        
        # If terminal state, value is 0
        if self.env.is_terminal(state):
            return 0.0
        
        max_value = float('-inf')
        
        # Try all actions and find the maximum
        for action in range(self.env.n_actions):
            action_value = 0.0
            
            # Sum over all possible next states
            for next_state_idx in range(self.env.n_states):
                # Get transition probability P(s'|s,a)
                prob = self.env.P[state_idx, action, next_state_idx]
                
                if prob > 0:  # Only consider reachable states
                    # Get reward R(s,a,s')
                    reward = self.env.R[state_idx, action, next_state_idx]
                    
                    # Bellman equation: R + γV(s')
                    action_value += prob * (reward + self.env.gamma * self.V[next_state_idx])
            
            max_value = max(max_value, action_value)
        
        return max_value
    
    def extract_policy(self) -> np.ndarray:
        """
        Extract optimal policy from value function.
        
        π*(s) = argmax_a ∑_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
        
        Returns:
            Optimal policy array where policy[s] = optimal action for state s
        """
        policy = np.zeros(self.env.n_states, dtype=int)
        
        for state_idx in range(self.env.n_states):
            state = self.env.index_to_state(state_idx)
            
            # Terminal states have no meaningful action
            if self.env.is_terminal(state):
                policy[state_idx] = 0  # Arbitrary action
                continue
            
            best_action = 0
            best_value = float('-inf')
            
            # Find action that maximizes expected value
            for action in range(self.env.n_actions):
                action_value = 0.0
                
                for next_state_idx in range(self.env.n_states):
                    prob = self.env.P[state_idx, action, next_state_idx]
                    
                    if prob > 0:
                        reward = self.env.R[state_idx, action, next_state_idx]
                        action_value += prob * (reward + self.env.gamma * self.V[next_state_idx])
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            policy[state_idx] = best_action
        
        return policy
    
    def solve(self, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Run Value Iteration until convergence.
        
        Args:
            verbose: Whether to print iteration progress
            
        Returns:
            Tuple of (optimal_value_function, optimal_policy, metrics)
        """
        if verbose:
            print("Starting Value Iteration...")
            print(f"Convergence threshold: {self.threshold}")
            print(f"Max iterations: {self.max_iterations}")
            print("-" * 50)
        
        for iteration in range(self.max_iterations):
            # Store previous value function
            V_old = self.V.copy()
            
            # Update all states
            for state_idx in range(self.env.n_states):
                self.V[state_idx] = self.bellman_update(state_idx)
            
            # Calculate convergence metric: max |V_k+1(s) - V_k(s)|
            delta = np.max(np.abs(self.V - V_old))
            self.convergence_history.append(delta)
            self.value_history.append(self.V.copy())
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:3d}: Δ = {delta:.6f}")
            
            # Check convergence
            if delta < self.threshold:
                self.iterations = iteration + 1
                if verbose:
                    print(f"\nConverged after {self.iterations} iterations!")
                    print(f"Final Δ = {delta:.8f}")
                break
        else:
            self.iterations = self.max_iterations
            if verbose:
                print(f"\nReached maximum iterations ({self.max_iterations})")
                print(f"Final Δ = {delta:.8f}")
        
        # Extract optimal policy
        self.policy = self.extract_policy()
        
        # Prepare metrics
        metrics = {
            'iterations': self.iterations,
            'final_delta': delta,
            'convergence_history': self.convergence_history,
            'value_history': self.value_history,
            'converged': delta < self.threshold
        }
        
        return self.V.copy(), self.policy.copy(), metrics
    
    def get_value_grid(self) -> np.ndarray:
        """
        Convert linear value function to 2D grid for visualization.
        
        Returns:
            2D array representing value function on the grid
        """
        value_grid = np.zeros((self.env.size, self.env.size))
        
        for state_idx in range(self.env.n_states):
            row, col = self.env.index_to_state(state_idx)
            value_grid[row, col] = self.V[state_idx]
        
        return value_grid
    
    def get_policy_grid(self) -> np.ndarray:
        """
        Convert linear policy to 2D grid for visualization.
        
        Returns:
            2D array representing policy on the grid
        """
        policy_grid = np.zeros((self.env.size, self.env.size), dtype=int)
        
        for state_idx in range(self.env.n_states):
            row, col = self.env.index_to_state(state_idx)
            policy_grid[row, col] = self.policy[state_idx]
        
        return policy_grid
    
    def solve_interactive(self, animation_speed: float = 500, figsize: Tuple[int, int] = (15, 8)):
        """
        Run Value Iteration with interactive visualization showing real-time updates.
        
        Args:
            animation_speed: Delay between iterations in milliseconds
            figsize: Figure size for the interactive plot
        """
        # Prepare for interactive solving
        self.V = np.zeros(self.env.n_states)
        self.convergence_history = []
        self.value_history = []
        self.iteration_data = []
        self.current_iteration = 0
        self.is_playing = False
        self.animation_speed = animation_speed
        
        print("🎬 Starting Interactive Value Iteration Visualization...")
        print("🎮 Controls: Play/Pause, Step Forward/Back, Speed Control")
        print("📊 Watch values propagate from goal to start state!")
        print("-" * 60)
        
        # Pre-compute all iterations for smooth playback
        self._precompute_iterations()
        
        # Create interactive figure
        self._create_interactive_figure(figsize)
        
        return self.V.copy(), self.policy.copy(), {
            'iterations': len(self.iteration_data),
            'final_delta': self.convergence_history[-1] if self.convergence_history else 0,
            'convergence_history': self.convergence_history,
            'value_history': self.value_history,
            'converged': True
        }
    
    def _precompute_iterations(self):
        """Pre-compute all Value Iteration steps for smooth animation."""
        V = np.zeros(self.env.n_states)
        
        for iteration in range(self.max_iterations):
            V_old = V.copy()
            
            # Update all states
            for state_idx in range(self.env.n_states):
                V[state_idx] = self._bellman_update_static(state_idx, V)
            
            # Calculate convergence
            delta = np.max(np.abs(V - V_old))
            
            # Store iteration data
            self.iteration_data.append({
                'values': V.copy(),
                'delta': delta,
                'iteration': iteration + 1
            })
            
            self.convergence_history.append(delta)
            self.value_history.append(V.copy())
            
            # Check convergence
            if delta < self.threshold:
                break
        
        # Extract final policy
        self.V = V.copy()
        self.policy = self.extract_policy()
        
        print(f"✅ Pre-computed {len(self.iteration_data)} iterations")
        print(f"🎯 Converged with final Δ = {self.convergence_history[-1]:.2e}")
    
    def _bellman_update_static(self, state_idx: int, V: np.ndarray) -> float:
        """Static version of Bellman update for pre-computation."""
        state = self.env.index_to_state(state_idx)
        
        if self.env.is_terminal(state):
            return 0.0
        
        max_value = float('-inf')
        
        for action in range(self.env.n_actions):
            action_value = 0.0
            
            for next_state_idx in range(self.env.n_states):
                prob = self.env.P[state_idx, action, next_state_idx]
                
                if prob > 0:
                    reward = self.env.R[state_idx, action, next_state_idx]
                    action_value += prob * (reward + self.env.gamma * V[next_state_idx])
            
            max_value = max(max_value, action_value)
        
        return max_value
    
    def _create_interactive_figure(self, figsize):
        """Create the interactive matplotlib figure with controls."""
        # Create figure and subplots
        self.fig = plt.figure(figsize=figsize)
        
        # Main value function plot
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Policy plot
        self.ax_policy = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
        
        # Convergence plot
        self.ax_conv = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        
        # Info panel
        self.ax_info = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        self.ax_info.axis('off')
        
        # Control buttons area
        plt.subplots_adjust(bottom=0.15)
        
        # Initialize plots
        self._init_plots()
        
        # Create control widgets
        self._create_controls()
        
        # Set initial state
        self._update_plots(0)
        
        # Show the interactive figure
        plt.suptitle('🎬 Interactive Value Iteration - Real-time Algorithm Visualization', 
                    fontsize=16, fontweight='bold')
        plt.show()
    
    def _init_plots(self):
        """Initialize the plot elements."""
        # Value function heatmap
        self.value_im = self.ax_main.imshow(np.zeros((self.env.size, self.env.size)), 
                                          cmap='viridis', vmin=0, vmax=1)
        self.ax_main.set_title('Value Function V(s)')
        self.ax_main.set_xticks(range(self.env.size))
        self.ax_main.set_yticks(range(self.env.size))
        
        # Add colorbar
        self.cbar = plt.colorbar(self.value_im, ax=self.ax_main, shrink=0.7)
        self.cbar.set_label('Value V(s)')
        
        # Value text annotations
        self.value_texts = []
        for row in range(self.env.size):
            text_row = []
            for col in range(self.env.size):
                text = self.ax_main.text(col, row, '0.000', ha='center', va='center',
                                       color='white', fontsize=8, weight='bold')
                text_row.append(text)
            self.value_texts.append(text_row)
        
        # Mark special states
        start_row, start_col = self.env.start_state
        goal_row, goal_col = self.env.goal_state
        
        start_circle = patches.Circle((start_col, start_row), 0.4, 
                                    linewidth=2, edgecolor='red', facecolor='none')
        self.ax_main.add_patch(start_circle)
        
        goal_circle = patches.Circle((goal_col, goal_row), 0.4, 
                                   linewidth=2, edgecolor='gold', facecolor='none')
        self.ax_main.add_patch(goal_circle)
        
        # Policy arrows plot
        self.ax_policy.set_title('Current Policy π(s)')
        self.ax_policy.set_xticks(range(self.env.size))
        self.ax_policy.set_yticks(range(self.env.size))
        self.ax_policy.set_xlim(-0.5, self.env.size - 0.5)
        self.ax_policy.set_ylim(-0.5, self.env.size - 0.5)
        self.ax_policy.invert_yaxis()
        
        # Initialize empty arrows list
        self.policy_arrows = []
        
        # Convergence plot
        self.conv_line, = self.ax_conv.plot([], [], 'b-', linewidth=2, marker='o', markersize=4)
        self.ax_conv.set_xlabel('Iteration')
        self.ax_conv.set_ylabel('Max |ΔV|')
        self.ax_conv.set_yscale('log')
        self.ax_conv.grid(True, alpha=0.3)
        self.ax_conv.set_title('Convergence Progress')
    
    def _create_controls(self):
        """Create interactive control widgets."""
        # Play/Pause button
        ax_play = plt.axes([0.1, 0.02, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)
        
        # Step buttons
        ax_prev = plt.axes([0.22, 0.02, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, '◀ Step')
        self.btn_prev.on_clicked(self._step_backward)
        
        ax_next = plt.axes([0.32, 0.02, 0.08, 0.04])
        self.btn_next = Button(ax_next, 'Step ▶')
        self.btn_next.on_clicked(self._step_forward)
        
        # Reset button
        ax_reset = plt.axes([0.42, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_animation)
        
        # Speed slider
        ax_speed = plt.axes([0.55, 0.02, 0.3, 0.04])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)
        self.speed_slider.on_changed(self._update_speed)
        
        # Iteration slider
        ax_iter = plt.axes([0.1, 0.08, 0.75, 0.04])
        self.iter_slider = Slider(ax_iter, 'Iteration', 0, len(self.iteration_data)-1, 
                                valinit=0, valfmt='%d')
        self.iter_slider.on_changed(self._slider_update)
    
    def _toggle_play(self, event):
        """Toggle play/pause animation."""
        if not hasattr(self, 'timer'):
            self._start_animation()
        else:
            if self.is_playing:
                self._pause_animation()
            else:
                self._resume_animation()
    
    def _start_animation(self):
        """Start the animation timer."""
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        
        def animate():
            if self.is_playing and self.current_iteration < len(self.iteration_data) - 1:
                self.current_iteration += 1
                self._update_plots(self.current_iteration)
                self.iter_slider.set_val(self.current_iteration)
                self.fig.canvas.draw_idle()
                
                # Schedule next frame
                self.timer = self.fig.canvas.new_timer(interval=self.animation_speed)
                self.timer.single_shot = True
                self.timer.add_callback(animate)
                self.timer.start()
            else:
                self.is_playing = False
                self.btn_play.label.set_text('Play')
        
        animate()
    
    def _pause_animation(self):
        """Pause the animation."""
        self.is_playing = False
        self.btn_play.label.set_text('Play')
    
    def _resume_animation(self):
        """Resume the animation."""
        self.is_playing = True
        self.btn_play.label.set_text('Pause')
        self._start_animation()
    
    def _step_forward(self, event):
        """Step forward one iteration."""
        self._pause_animation()
        if self.current_iteration < len(self.iteration_data) - 1:
            self.current_iteration += 1
            self._update_plots(self.current_iteration)
            self.iter_slider.set_val(self.current_iteration)
    
    def _step_backward(self, event):
        """Step backward one iteration."""
        self._pause_animation()
        if self.current_iteration > 0:
            self.current_iteration -= 1
            self._update_plots(self.current_iteration)
            self.iter_slider.set_val(self.current_iteration)
    
    def _reset_animation(self, event):
        """Reset animation to beginning."""
        self._pause_animation()
        self.current_iteration = 0
        self._update_plots(self.current_iteration)
        self.iter_slider.set_val(self.current_iteration)
    
    def _update_speed(self, val):
        """Update animation speed."""
        self.animation_speed = int(500 / val)  # Inverse relationship
    
    def _slider_update(self, val):
        """Update display based on iteration slider."""
        self._pause_animation()
        self.current_iteration = int(val)
        self._update_plots(self.current_iteration)
    
    def _update_plots(self, iteration_idx):
        """Update all plots for the given iteration."""
        if iteration_idx >= len(self.iteration_data):
            return
        
        data = self.iteration_data[iteration_idx]
        values = data['values']
        delta = data['delta']
        iteration = data['iteration']
        
        # Convert to grid
        value_grid = np.zeros((self.env.size, self.env.size))
        for state_idx in range(self.env.n_states):
            row, col = self.env.index_to_state(state_idx)
            value_grid[row, col] = values[state_idx]
        
        # Update value function heatmap
        max_val = np.max([np.max(data['values']) for data in self.iteration_data])
        self.value_im.set_array(value_grid)
        self.value_im.set_clim(0, max_val)
        
        # Update value text annotations
        for row in range(self.env.size):
            for col in range(self.env.size):
                value = value_grid[row, col]
                color = 'white' if value < max_val * 0.5 else 'black'
                self.value_texts[row][col].set_text(f'{value:.3f}')
                self.value_texts[row][col].set_color(color)
        
        # Update policy arrows
        self._update_policy_arrows(values)
        
        # Update convergence plot
        conv_data = self.convergence_history[:iteration_idx+1]
        self.conv_line.set_data(range(1, len(conv_data)+1), conv_data)
        
        if conv_data:
            self.ax_conv.set_xlim(0, max(10, len(conv_data)))
            self.ax_conv.set_ylim(min(conv_data[-1], 1e-8), max(conv_data) * 1.1)
        
        # Update info panel
        self._update_info_panel(iteration, delta, values)
        
        self.fig.canvas.draw_idle()
    
    def _update_policy_arrows(self, values):
        """Update policy arrows based on current values."""
        # Clear existing arrows
        for arrow in self.policy_arrows:
            arrow.remove()
        self.policy_arrows.clear()
        
        # Extract current policy
        current_policy = self._extract_policy_from_values(values)
        
        action_vectors = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        action_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
        
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                
                if self.env.is_terminal(state):
                    # Draw goal marker
                    circle = patches.Circle((col, row), 0.4, 
                                          linewidth=3, edgecolor='gold', facecolor='gold', alpha=0.8)
                    self.ax_policy.add_patch(circle)
                    text = self.ax_policy.text(col, row, 'G', ha='center', va='center',
                                             color='black', fontsize=12, weight='bold')
                    self.policy_arrows.append(circle)
                    self.policy_arrows.append(text)
                    continue
                
                state_idx = self.env.state_to_index(state)
                action = current_policy[state_idx]
                dx, dy = action_vectors[action]
                color = action_colors[action]
                
                # Draw arrow
                arrow = self.ax_policy.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1,
                                           fc=color, ec=color, linewidth=2, alpha=0.8)
                self.policy_arrows.append(arrow)
                
                # Mark start state
                if state == self.env.start_state:
                    circle = patches.Circle((col, row), 0.15,
                                          linewidth=2, edgecolor='red', facecolor='white')
                    self.ax_policy.add_patch(circle)
                    text = self.ax_policy.text(col, row, 'S', ha='center', va='center',
                                             color='red', fontsize=8, weight='bold')
                    self.policy_arrows.append(circle)
                    self.policy_arrows.append(text)
    
    def _extract_policy_from_values(self, values):
        """Extract policy from given value function."""
        policy = np.zeros(self.env.n_states, dtype=int)
        
        for state_idx in range(self.env.n_states):
            state = self.env.index_to_state(state_idx)
            
            if self.env.is_terminal(state):
                policy[state_idx] = 0
                continue
            
            best_action = 0
            best_value = float('-inf')
            
            for action in range(self.env.n_actions):
                action_value = 0.0
                
                for next_state_idx in range(self.env.n_states):
                    prob = self.env.P[state_idx, action, next_state_idx]
                    
                    if prob > 0:
                        reward = self.env.R[state_idx, action, next_state_idx]
                        action_value += prob * (reward + self.env.gamma * values[next_state_idx])
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            policy[state_idx] = best_action
        
        return policy
    
    def _update_info_panel(self, iteration, delta, values):
        """Update the information panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Get current state values
        start_value = values[self.env.state_to_index(self.env.start_state)]
        goal_value = values[self.env.state_to_index(self.env.goal_state)]
        max_value = np.max(values)
        
        info_text = f"""
ITERATION {iteration}

Convergence: Δ = {delta:.2e}
{'✅ CONVERGED' if delta < self.threshold else '🔄 UPDATING'}

Current Values:
• Start State: {start_value:.6f}
• Goal State:  {goal_value:.6f}
• Max Value:   {max_value:.6f}

Controls:
🎮 Play/Pause: Auto-play iterations
⏮️⏭️ Step: Manual step forward/back
🔄 Reset: Return to iteration 0
🎚️ Speed: Control animation speed
📊 Slider: Jump to any iteration
        """
        
        self.ax_info.text(0.05, 0.95, info_text.strip(), transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def print_results(self):
        """Print formatted results of Value Iteration."""
        print("\n" + "="*60)
        print("VALUE ITERATION RESULTS")
        print("="*60)
        
        print(f"Iterations to convergence: {self.iterations}")
        print(f"Final convergence delta: {self.convergence_history[-1]:.8f}")
        
        print("\nValue Function (2D Grid):")
        value_grid = self.get_value_grid()
        for row in range(self.env.size):
            for col in range(self.env.size):
                print(f"{value_grid[row, col]:6.3f}", end=" ")
            print()
        
        print("\nOptimal Policy (Actions):")
        policy_grid = self.get_policy_grid()
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        
        for row in range(self.env.size):
            for col in range(self.env.size):
                state = (row, col)
                if state == self.env.goal_state:
                    print("  G", end=" ")
                elif state == self.env.start_state:
                    print(f"S{action_symbols[policy_grid[row, col]]}", end=" ")
                else:
                    print(f" {action_symbols[policy_grid[row, col]]}", end=" ")
            print()
        
        print("\nLegend: ↑=UP, ↓=DOWN, ←=LEFT, →=RIGHT, S=Start, G=Goal")
        print("="*60) 