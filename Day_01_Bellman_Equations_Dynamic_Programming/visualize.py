"""
Visualization functions for Value Iteration results
Author: Day 1 - Bellman Equations & Dynamic Programming Challenge
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List
from gridworld import GridWorld
from value_iteration import ValueIteration


def plot_value_function(vi: ValueIteration, save_path: str = None, show: bool = True):
    """
    Plot value function as a heatmap.
    
    Args:
        vi: ValueIteration object with solved values
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    value_grid = vi.get_value_grid()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(value_grid, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value Function V*(s)', rotation=270, labelpad=20)
    
    # Add grid lines
    for i in range(vi.env.size + 1):
        ax.axhline(i - 0.5, color='white', linewidth=0.5)
        ax.axvline(i - 0.5, color='white', linewidth=0.5)
    
    # Add value text annotations
    for row in range(vi.env.size):
        for col in range(vi.env.size):
            value = value_grid[row, col]
            color = 'white' if value < np.max(value_grid) * 0.5 else 'black'
            ax.text(col, row, f'{value:.3f}', ha='center', va='center', 
                   color=color, fontsize=10, weight='bold')
    
    # Mark special states
    start_row, start_col = vi.env.start_state
    goal_row, goal_col = vi.env.goal_state
    
    # Add start marker
    start_circle = patches.Circle((start_col, start_row), 0.3, 
                                linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(start_circle)
    ax.text(start_col, start_row - 0.45, 'START', ha='center', va='center', 
           color='red', fontsize=8, weight='bold')
    
    # Add goal marker
    goal_circle = patches.Circle((goal_col, goal_row), 0.3, 
                               linewidth=2, edgecolor='gold', facecolor='none')
    ax.add_patch(goal_circle)
    ax.text(goal_col, goal_row - 0.45, 'GOAL', ha='center', va='center', 
           color='gold', fontsize=8, weight='bold')
    
    # Set labels and title
    ax.set_title('Value Function Heatmap\nV*(s) from Value Iteration', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Set ticks
    ax.set_xticks(range(vi.env.size))
    ax.set_yticks(range(vi.env.size))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Value function plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_policy(vi: ValueIteration, save_path: str = None, show: bool = True):
    """
    Plot optimal policy as arrows on the grid.
    
    Args:
        vi: ValueIteration object with solved policy
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    policy_grid = vi.get_policy_grid()
    value_grid = vi.get_value_grid()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create background heatmap (lighter)
    im = ax.imshow(value_grid, cmap='viridis', alpha=0.3, interpolation='nearest')
    
    # Action direction mappings
    action_vectors = {
        0: (0, 0.3),   # UP
        1: (0, -0.3),  # DOWN
        2: (-0.3, 0),  # LEFT
        3: (0.3, 0)    # RIGHT
    }
    
    action_colors = {
        0: 'red',     # UP
        1: 'blue',    # DOWN
        2: 'green',   # LEFT
        3: 'orange'   # RIGHT
    }
    
    # Draw policy arrows
    for row in range(vi.env.size):
        for col in range(vi.env.size):
            state = (row, col)
            
            # Skip terminal state
            if vi.env.is_terminal(state):
                # Draw goal marker
                goal_circle = patches.Circle((col, row), 0.4, 
                                           linewidth=3, edgecolor='gold', facecolor='gold', alpha=0.8)
                ax.add_patch(goal_circle)
                ax.text(col, row, 'G', ha='center', va='center', 
                       color='black', fontsize=16, weight='bold')
                continue
            
            action = policy_grid[row, col]
            dx, dy = action_vectors[action]
            color = action_colors[action]
            
            # Draw arrow
            ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, 
                    fc=color, ec=color, linewidth=2, alpha=0.8)
            
            # Mark start state
            if state == vi.env.start_state:
                start_circle = patches.Circle((col, row), 0.15, 
                                            linewidth=2, edgecolor='red', facecolor='white')
                ax.add_patch(start_circle)
                ax.text(col, row, 'S', ha='center', va='center', 
                       color='red', fontsize=10, weight='bold')
    
    # Add grid lines
    for i in range(vi.env.size + 1):
        ax.axhline(i - 0.5, color='black', linewidth=1, alpha=0.3)
        ax.axvline(i - 0.5, color='black', linewidth=1, alpha=0.3)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=3, label='UP'),
        plt.Line2D([0], [0], color='blue', lw=3, label='DOWN'),
        plt.Line2D([0], [0], color='green', lw=3, label='LEFT'),
        plt.Line2D([0], [0], color='orange', lw=3, label='RIGHT')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Set labels and title
    ax.set_title('Optimal Policy π*(s)\nArrows show optimal actions', fontsize=14, fontweight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    
    # Set ticks
    ax.set_xticks(range(vi.env.size))
    ax.set_yticks(range(vi.env.size))
    
    # Invert y-axis to match matrix convention
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Policy plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_convergence(metrics: Dict, save_path: str = None, show: bool = True):
    """
    Plot convergence history.
    
    Args:
        metrics: Dictionary containing convergence history
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    convergence_history = metrics['convergence_history']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Convergence delta over iterations
    iterations = range(1, len(convergence_history) + 1)
    ax1.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Max |V_{k+1}(s) - V_k(s)|')
    ax1.set_title('Value Function Convergence')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add convergence threshold line
    if 'threshold' in metrics:
        ax1.axhline(y=metrics['threshold'], color='red', linestyle='--', 
                   label=f'Threshold = {metrics["threshold"]}')
        ax1.legend()
    
    # Plot 2: Value evolution for selected states
    if 'value_history' in metrics:
        value_history = metrics['value_history']
        
        # Plot value evolution for a few key states
        n_states = len(value_history[0])
        
        # Select interesting states to plot
        selected_states = [0, n_states//2, n_states-1]  # Start, middle, goal
        state_labels = ['Start State', 'Middle State', 'Goal State']
        colors = ['red', 'blue', 'green']
        
        for i, (state_idx, label, color) in enumerate(zip(selected_states, state_labels, colors)):
            values = [v[state_idx] for v in value_history]
            ax2.plot(iterations, values, color=color, linewidth=2, 
                    marker='o', markersize=3, label=label)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Value V(s)')
        ax2.set_title('Value Evolution for Selected States')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_combined_results(vi: ValueIteration, metrics: Dict, save_path: str = None, show: bool = True):
    """
    Create a combined plot showing all results.
    
    Args:
        vi: ValueIteration object with solved values and policy
        metrics: Dictionary containing convergence metrics
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Value function heatmap
    ax1 = plt.subplot(2, 3, 1)
    value_grid = vi.get_value_grid()
    im1 = ax1.imshow(value_grid, cmap='viridis', interpolation='nearest')
    
    # Add value annotations
    for row in range(vi.env.size):
        for col in range(vi.env.size):
            value = value_grid[row, col]
            color = 'white' if value < np.max(value_grid) * 0.5 else 'black'
            ax1.text(col, row, f'{value:.2f}', ha='center', va='center', 
                   color=color, fontsize=8, weight='bold')
    
    ax1.set_title('Value Function V*(s)')
    ax1.set_xticks(range(vi.env.size))
    ax1.set_yticks(range(vi.env.size))
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    
    # Policy visualization
    ax2 = plt.subplot(2, 3, 2)
    policy_grid = vi.get_policy_grid()
    
    # Background
    ax2.imshow(value_grid, cmap='viridis', alpha=0.2, interpolation='nearest')
    
    action_vectors = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    action_colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange'}
    
    for row in range(vi.env.size):
        for col in range(vi.env.size):
            state = (row, col)
            if not vi.env.is_terminal(state):
                action = policy_grid[row, col]
                dx, dy = action_vectors[action]
                color = action_colors[action]
                ax2.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1, 
                         fc=color, ec=color, linewidth=2)
            else:
                ax2.text(col, row, 'G', ha='center', va='center', 
                        color='gold', fontsize=12, weight='bold')
    
    ax2.set_title('Optimal Policy π*(s)')
    ax2.set_xticks(range(vi.env.size))
    ax2.set_yticks(range(vi.env.size))
    ax2.invert_yaxis()
    
    # Convergence plot
    ax3 = plt.subplot(2, 3, 3)
    convergence_history = metrics['convergence_history']
    iterations = range(1, len(convergence_history) + 1)
    ax3.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Max Δ')
    ax3.set_title('Convergence')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # GridWorld layout
    ax4 = plt.subplot(2, 3, 4)
    layout = np.zeros((vi.env.size, vi.env.size))
    ax4.imshow(layout, cmap='gray', alpha=0.1)
    
    for row in range(vi.env.size):
        for col in range(vi.env.size):
            state = (row, col)
            if state == vi.env.start_state:
                ax4.text(col, row, 'START', ha='center', va='center', 
                        color='red', fontsize=10, weight='bold')
            elif state == vi.env.goal_state:
                ax4.text(col, row, 'GOAL', ha='center', va='center', 
                        color='green', fontsize=10, weight='bold')
            else:
                ax4.text(col, row, f'({row},{col})', ha='center', va='center', 
                        color='black', fontsize=8)
    
    # Add grid
    for i in range(vi.env.size + 1):
        ax4.axhline(i - 0.5, color='black', linewidth=1)
        ax4.axvline(i - 0.5, color='black', linewidth=1)
    
    ax4.set_title('GridWorld Layout')
    ax4.set_xticks(range(vi.env.size))
    ax4.set_yticks(range(vi.env.size))
    ax4.invert_yaxis()
    
    # Metrics summary
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    metrics_text = f"""
    RESULTS SUMMARY
    
    Iterations: {metrics['iterations']}
    Converged: {metrics['converged']}
    Final Δ: {metrics['final_delta']:.2e}
    
    Environment:
    • Grid Size: {vi.env.size}×{vi.env.size}
    • Discount γ: {vi.env.gamma}
    • Start: {vi.env.start_state}
    • Goal: {vi.env.goal_state}
    
    Algorithm:
    • Threshold: {vi.threshold:.2e}
    • Max Iterations: {vi.max_iterations}
    """
    
    ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    # Value evolution plot
    ax6 = plt.subplot(2, 3, 6)
    if 'value_history' in metrics:
        value_history = metrics['value_history']
        start_idx = vi.env.state_to_index(vi.env.start_state)
        goal_idx = vi.env.state_to_index(vi.env.goal_state)
        
        start_values = [v[start_idx] for v in value_history]
        goal_values = [v[goal_idx] for v in value_history]
        
        ax6.plot(iterations, start_values, 'r-', linewidth=2, label='Start State', marker='o', markersize=3)
        ax6.plot(iterations, goal_values, 'g-', linewidth=2, label='Goal State', marker='s', markersize=3)
        
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Value V(s)')
        ax6.set_title('Value Evolution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Value Iteration Results - Day 1 Challenge', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined results plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close() 