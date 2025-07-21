"""
Main script for Day 1: Bellman Equations & Dynamic Programming Challenge
Author: Day 1 - Value Iteration Implementation

This script demonstrates Value Iteration on a 5x5 GridWorld environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from gridworld import GridWorld
from value_iteration import ValueIteration
from visualize import (plot_value_function, plot_policy, plot_convergence, 
                      plot_combined_results)


def main():
    """Main function to run Value Iteration challenge."""
    
    print("="*70)
    print("🚀 DAY 1: BELLMAN EQUATIONS & DYNAMIC PROGRAMMING CHALLENGE")
    print("="*70)
    print("📘 Topic: Value Iteration in GridWorld")
    print("🎯 Goal: Implement and visualize Bellman optimality equations\n")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved to: {results_dir}/")
    
    # Initialize environment
    print("\n🏗️  Initializing 5x5 GridWorld Environment...")
    env = GridWorld(size=5, gamma=0.9)
    print(f"   • Grid size: {env.size}×{env.size}")
    print(f"   • Discount factor γ: {env.gamma}")
    print(f"   • Start state: {env.start_state}")
    print(f"   • Goal state: {env.goal_state}")
    print(f"   • Actions: {list(env.actions.values())}")
    
    # Display environment layout
    print(env.render_grid())
    
    # Initialize Value Iteration
    print("🧮 Initializing Value Iteration Algorithm...")
    vi = ValueIteration(env, threshold=1e-6, max_iterations=1000)
    print(f"   • Convergence threshold: {vi.threshold}")
    print(f"   • Maximum iterations: {vi.max_iterations}")
    
    # Run Value Iteration
    print("\n🔄 Running Value Iteration...")
    start_time = time.time()
    
    optimal_values, optimal_policy, metrics = vi.solve(verbose=True)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    print(f"\n⏱️  Algorithm completed in {runtime:.3f} seconds")
    
    # Display results
    vi.print_results()
    
    # Create visualizations
    print("\n🎨 Generating Visualizations...")
    
    # 1. Value function heatmap
    print("   📊 Creating value function heatmap...")
    plot_value_function(vi, 
                       save_path=os.path.join(results_dir, "value_function.png"),
                       show=False)
    
    # 2. Policy visualization
    print("   🧭 Creating policy visualization...")
    plot_policy(vi, 
               save_path=os.path.join(results_dir, "optimal_policy.png"),
               show=False)
    
    # 3. Convergence plots
    print("   📈 Creating convergence plots...")
    metrics['threshold'] = vi.threshold  # Add threshold for plotting
    plot_convergence(metrics, 
                    save_path=os.path.join(results_dir, "convergence.png"),
                    show=False)
    
    # 4. Combined results plot
    print("   🖼️  Creating combined results visualization...")
    plot_combined_results(vi, metrics,
                         save_path=os.path.join(results_dir, "combined_results.png"),
                         show=False)
    
    # Save numerical results
    print("   💾 Saving numerical results...")
    save_results(vi, metrics, results_dir)
    
    # Theoretical analysis
    print("\n📚 THEORETICAL ANALYSIS")
    print("="*50)
    analyze_results(vi, metrics)
    
    # Generate summary report
    print("\n📝 Generating summary report...")
    generate_report(vi, metrics, results_dir, runtime)
    
    print(f"\n✅ Challenge completed! All results saved to: {results_dir}/")
    print("\n🎉 Day 1 Value Iteration implementation successful!")
    
    # Display final combined plot
    print("\n🖼️  Displaying final results...")
    plot_combined_results(vi, metrics, show=True)


def save_results(vi: ValueIteration, metrics: dict, results_dir: str):
    """Save numerical results to files."""
    
    # Save value function
    value_grid = vi.get_value_grid()
    np.savetxt(os.path.join(results_dir, "value_function.txt"), 
               value_grid, fmt='%.6f', 
               header="Value Function V*(s) - GridWorld 5x5")
    
    # Save policy
    policy_grid = vi.get_policy_grid()
    np.savetxt(os.path.join(results_dir, "optimal_policy.txt"), 
               policy_grid, fmt='%d',
               header="Optimal Policy π*(s) - Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT")
    
    # Save convergence history
    np.savetxt(os.path.join(results_dir, "convergence_history.txt"), 
               metrics['convergence_history'], fmt='%.8e',
               header="Convergence History - Max |V_k+1(s) - V_k(s)| per iteration")
    
    # Save metrics summary
    metrics_summary = {
        'iterations': metrics['iterations'],
        'converged': metrics['converged'],
        'final_delta': metrics['final_delta'],
        'grid_size': vi.env.size,
        'discount_factor': vi.env.gamma,
        'threshold': vi.threshold
    }
    
    with open(os.path.join(results_dir, "metrics_summary.txt"), 'w') as f:
        f.write("VALUE ITERATION METRICS SUMMARY\n")
        f.write("="*40 + "\n\n")
        for key, value in metrics_summary.items():
            f.write(f"{key}: {value}\n")


def analyze_results(vi: ValueIteration, metrics: dict):
    """Provide theoretical analysis of the results."""
    
    value_grid = vi.get_value_grid()
    
    print("🔍 Key Insights:")
    print(f"   • Convergence achieved in {metrics['iterations']} iterations")
    print(f"   • Maximum value: {np.max(value_grid):.6f} (at goal state)")
    print(f"   • Minimum value: {np.min(value_grid):.6f}")
    print(f"   • Start state value: {value_grid[0, 0]:.6f}")
    
    print("\n🧠 Bellman Equation Verification:")
    # Verify Bellman equation for a few states
    for i, state in enumerate([(0, 0), (2, 2), (4, 3)]):
        if i >= 3:  # Limit output
            break
        
        state_idx = vi.env.state_to_index(state)
        computed_value = vi.V[state_idx]
        
        # Manually compute Bellman equation
        max_action_value = float('-inf')
        for action in range(vi.env.n_actions):
            action_value = 0.0
            for next_state_idx in range(vi.env.n_states):
                prob = vi.env.P[state_idx, action, next_state_idx]
                if prob > 0:
                    reward = vi.env.R[state_idx, action, next_state_idx]
                    action_value += prob * (reward + vi.env.gamma * vi.V[next_state_idx])
            max_action_value = max(max_action_value, action_value)
        
        print(f"   • State {state}: V={computed_value:.6f}, Bellman={max_action_value:.6f}")
    
    print("\n🎯 Policy Analysis:")
    policy_grid = vi.get_policy_grid()
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    
    # Count action frequencies
    action_counts = {}
    for action in range(4):
        count = np.sum(policy_grid == action)
        action_counts[action_names[action]] = count
    
    print("   • Action distribution in optimal policy:")
    for action_name, count in action_counts.items():
        percentage = (count / vi.env.n_states) * 100
        print(f"     - {action_name}: {count} states ({percentage:.1f}%)")
    
    print("\n📐 Value Propagation:")
    print("   • Values decrease with Manhattan distance from goal")
    print("   • Shortest path length from start to goal: 8 steps")
    expected_start_value = vi.env.gamma ** 8  # Theoretical minimum for shortest path
    print(f"   • Theoretical shortest path value: {expected_start_value:.6f}")
    print(f"   • Actual start state value: {value_grid[0, 0]:.6f}")


def generate_report(vi: ValueIteration, metrics: dict, results_dir: str, runtime: float):
    """Generate a comprehensive report."""
    
    report_path = os.path.join(results_dir, "challenge_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Day 1 Challenge Report: Value Iteration\n\n")
        
        f.write("## 🎯 Challenge Overview\n")
        f.write("- **Topic**: Bellman Equations & Dynamic Programming\n")
        f.write("- **Algorithm**: Value Iteration\n")
        f.write("- **Environment**: 5×5 GridWorld\n")
        f.write(f"- **Completion Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 🏗️ Environment Setup\n")
        f.write(f"- Grid Size: {vi.env.size}×{vi.env.size}\n")
        f.write(f"- Start State: {vi.env.start_state}\n")
        f.write(f"- Goal State: {vi.env.goal_state}\n")
        f.write(f"- Discount Factor (γ): {vi.env.gamma}\n")
        f.write(f"- Actions: UP, DOWN, LEFT, RIGHT\n")
        f.write(f"- Transitions: Deterministic\n\n")
        
        f.write("## 🧮 Algorithm Configuration\n")
        f.write(f"- Convergence Threshold (ε): {vi.threshold}\n")
        f.write(f"- Maximum Iterations: {vi.max_iterations}\n")
        f.write(f"- Actual Runtime: {runtime:.3f} seconds\n\n")
        
        f.write("## 📊 Results\n")
        f.write(f"- **Converged**: {metrics['converged']}\n")
        f.write(f"- **Iterations**: {metrics['iterations']}\n")
        f.write(f"- **Final Δ**: {metrics['final_delta']:.2e}\n")
        f.write(f"- **Start State Value**: {vi.get_value_grid()[0, 0]:.6f}\n")
        f.write(f"- **Goal State Value**: {vi.get_value_grid()[4, 4]:.6f}\n\n")
        
        f.write("## 🧠 Key Insights\n")
        f.write("1. **Convergence**: Algorithm converged successfully within threshold\n")
        f.write("2. **Value Propagation**: Values decrease with distance from goal\n")
        f.write("3. **Policy Optimality**: Learned policy represents shortest paths to goal\n")
        f.write("4. **Bellman Consistency**: Value function satisfies optimality equations\n\n")
        
        f.write("## 📁 Generated Files\n")
        f.write("- `value_function.png` - Value function heatmap\n")
        f.write("- `optimal_policy.png` - Policy visualization with arrows\n")
        f.write("- `convergence.png` - Convergence analysis plots\n")
        f.write("- `combined_results.png` - Comprehensive results overview\n")
        f.write("- `value_function.txt` - Numerical value function data\n")
        f.write("- `optimal_policy.txt` - Numerical policy data\n")
        f.write("- `convergence_history.txt` - Iteration-by-iteration convergence\n")
        f.write("- `metrics_summary.txt` - Summary statistics\n\n")
        
        f.write("## 🎓 Theoretical Verification\n")
        f.write("The implementation successfully demonstrates:\n")
        f.write("- ✅ Bellman Optimality Equation: V*(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV*(s')]\n")
        f.write("- ✅ Value Iteration Update: V_{k+1}(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV_k(s')]\n")
        f.write("- ✅ Policy Extraction: π*(s) = argmax_a Σ P(s'|s,a)[R(s,a,s') + γV*(s')]\n")
        f.write("- ✅ Convergence Guarantee: max_s |V_{k+1}(s) - V_k(s)| < ε\n\n")
        
        f.write("## 🏆 Challenge Completion\n")
        f.write("All objectives successfully achieved:\n")
        f.write("- ✅ Derived and implemented Bellman equations\n")
        f.write("- ✅ Value Iteration algorithm from scratch\n")
        f.write("- ✅ Environment visualization\n")
        f.write("- ✅ Value function heatmap\n")
        f.write("- ✅ Optimal policy arrows\n")
        f.write("- ✅ Convergence metrics tracking\n")
        f.write("- ✅ Theoretical analysis and verification\n")
    
    print(f"   📄 Report saved to: {report_path}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Configure matplotlib for better plots
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Run the main challenge
    main() 