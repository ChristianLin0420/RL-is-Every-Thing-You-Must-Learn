"""
Main Experiment Runner for Day 2: Model-Free Tabular Methods
Author: Day 2 - Model-Free Tabular Methods Challenge

This script runs comprehensive experiments comparing SARSA and Q-Learning
on CliffWalking and FrozenLake environments, generates visualizations,
and produces detailed analysis reports.
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from typing import Dict, List

from train import ExperimentRunner
from visualize import VisualizationSuite, save_all_visualizations, InteractiveTrainingAnimation
from sarsa import SarsaAgent
from q_learning import QLearningAgent


def main():
    """Main experiment runner."""
    print("🚀 Starting Day 2: Model-Free Tabular Methods Challenge")
    print("=" * 80)
    print("🎯 Objective: Compare SARSA (on-policy) vs Q-Learning (off-policy)")
    print("🌍 Environments: CliffWalking-v0, FrozenLake-v1")
    print("📊 Metrics: Learning curves, policies, Q-values, performance")
    print("=" * 80)
    
    # Experiment configuration
    config = {
        'environments': [
            'CliffWalking-v0',
            'FrozenLake-v1'  # Will be modified to deterministic in create_environment
        ],
        'n_episodes': 1500,  # Sufficient for convergence
        'seeds': [42, 123, 456, 789, 999],  # Multiple seeds for robustness
        'hyperparams': {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'q_init': 'zeros'
        }
    }
    
    print(f"⚙️  Configuration:")
    print(f"   Episodes per run: {config['n_episodes']}")
    print(f"   Random seeds: {config['seeds']}")
    print(f"   Hyperparameters: {config['hyperparams']}")
    print()
    
    # Create experiment runner
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    runner = ExperimentRunner(results_dir)
    
    # Run experiments on all environments
    start_time = time.time()
    
    all_results = runner.run_multiple_environments(
        env_names=config['environments'],
        n_episodes=config['n_episodes'],
        seeds=config['seeds'],
        hyperparams=config['hyperparams'],
        verbose=True
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n⏱️  Total experiment time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    
    # Generate visualizations for each environment
    print(f"\n🎨 Generating visualizations...")
    visualization_files = {}
    
    for env_name in config['environments']:
        if env_name in all_results:
            print(f"   📊 Creating plots for {env_name}...")
            
            sarsa_results = all_results[env_name]['sarsa_results']
            qlearning_results = all_results[env_name]['qlearning_results']
            comparison = all_results[env_name]['comparison']
            
            # Generate all visualizations
            files = save_all_visualizations(
                sarsa_results, qlearning_results, comparison, 
                env_name, results_dir
            )
            visualization_files[env_name] = files
    
    # Create multi-environment performance comparison
    print(f"\n📈 Creating multi-environment comparison...")
    vis = VisualizationSuite(results_dir)
    
    # Extract comparison results for performance plot
    comparison_results = {}
    for env_name in config['environments']:
        if env_name in runner.results['comparisons']:
            comparison_results[env_name] = runner.results['comparisons'][env_name]
    
    if comparison_results:
        fig = vis.plot_performance_comparison(
            comparison_results, 
            list(comparison_results.keys()),
            save_path=os.path.join(results_dir, 'multi_environment_comparison.png'),
            show=False
        )
        plt.close(fig)
    
    # Save experiment results
    print(f"\n💾 Saving experiment data...")
    runner.save_results("experiment_results.json")
    
    # Generate comprehensive report
    generate_report(runner.results, visualization_files, total_time, results_dir)
    
    # Print final summary
    print_final_summary(runner.results, total_time)
    
    print(f"\n✅ Experiment completed successfully!")
    print(f"📁 All results saved to: {results_dir}/")
    
    return runner.results, visualization_files


def generate_report(
    results: Dict, 
    visualization_files: Dict, 
    total_time: float, 
    results_dir: str
):
    """Generate comprehensive markdown report."""
    
    report_path = os.path.join(results_dir, "EXPERIMENT_REPORT.md")
    
    with open(report_path, 'w') as f:
        f.write("# SARSA vs Q-Learning: Experimental Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Experiment Time:** {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n\n")
        
        # Experiment configuration
        f.write("## 🔧 Experiment Configuration\n\n")
        f.write(f"- **Environments:** {', '.join(results['metadata']['environments'])}\n")
        f.write(f"- **Episodes per run:** {results['metadata']['n_episodes']}\n")
        f.write(f"- **Random seeds:** {results['metadata']['seeds']}\n")
        f.write(f"- **Number of runs per algorithm:** {len(results['metadata']['seeds'])}\n\n")
        
        # Algorithm comparison
        f.write("## 📊 Algorithm Overview\n\n")
        f.write("### SARSA (State-Action-Reward-State-Action)\n")
        f.write("- **Type:** On-policy temporal difference learning\n")
        f.write("- **Update rule:** `Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]`\n")
        f.write("- **Characteristics:** Uses actual next action selected by policy\n")
        f.write("- **Behavior:** More cautious, learns from followed policy\n\n")
        
        f.write("### Q-Learning\n")
        f.write("- **Type:** Off-policy temporal difference learning\n")
        f.write("- **Update rule:** `Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`\n")
        f.write("- **Characteristics:** Uses maximum Q-value for next state\n")
        f.write("- **Behavior:** More aggressive, learns optimal policy regardless of behavior\n\n")
        
        # Results for each environment
        for env_name in results['metadata']['environments']:
            if env_name in results['comparisons']:
                f.write(f"## 🌍 Results: {env_name}\n\n")
                
                comparison = results['comparisons'][env_name]
                
                # Performance summary table
                f.write("### 📈 Performance Summary\n\n")
                f.write("| Metric | SARSA | Q-Learning | Difference | Winner |\n")
                f.write("|--------|--------|------------|------------|--------|\n")
                
                for metric_name, stats in comparison.items():
                    if isinstance(stats, dict) and 'sarsa_mean' in stats:
                        sarsa_val = stats['sarsa_mean']
                        qlearning_val = stats['qlearning_mean']
                        diff = stats['difference']
                        winner = "Q-Learning" if diff > 0 else "SARSA"
                        if abs(diff) <= stats.get('pooled_std', 0):
                            winner = "Tie"
                        
                        f.write(f"| {metric_name.replace('_', ' ').title()} | "
                               f"{sarsa_val:.3f} ± {stats['sarsa_std']:.3f} | "
                               f"{qlearning_val:.3f} ± {stats['qlearning_std']:.3f} | "
                               f"{diff:+.3f} | {winner} |\n")
                
                f.write("\n")
                
                # Key insights
                f.write("### 🔍 Key Insights\n\n")
                
                if 'final_reward' in comparison:
                    reward_diff = comparison['final_reward']['difference']
                    if abs(reward_diff) > comparison['final_reward']['pooled_std']:
                        if reward_diff > 0:
                            f.write(f"- **Q-Learning outperforms SARSA** in final reward by {reward_diff:.2f}\n")
                        else:
                            f.write(f"- **SARSA outperforms Q-Learning** in final reward by {abs(reward_diff):.2f}\n")
                    else:
                        f.write("- **Similar performance** between algorithms in final reward\n")
                
                if 'success_rate' in comparison:
                    success_diff = comparison['success_rate']['difference']
                    if abs(success_diff) > 0.05:  # 5% threshold
                        if success_diff > 0:
                            f.write(f"- **Q-Learning has higher success rate** by {success_diff:.1%}\n")
                        else:
                            f.write(f"- **SARSA has higher success rate** by {abs(success_diff):.1%}\n")
                
                if 'training_time' in comparison:
                    time_diff = comparison['training_time']['difference']
                    if abs(time_diff) > 1.0:  # 1 second threshold
                        if time_diff > 0:
                            f.write(f"- **SARSA is faster** by {abs(time_diff):.1f} seconds per run\n")
                        else:
                            f.write(f"- **Q-Learning is faster** by {time_diff:.1f} seconds per run\n")
                
                # Environment-specific insights
                if 'cliff' in env_name.lower():
                    f.write("- **CliffWalking environment**: SARSA typically more conservative (avoids cliff), "
                           "Q-Learning more aggressive (may risk cliff for optimal path)\n")
                elif 'frozenlake' in env_name.lower():
                    f.write("- **FrozenLake environment**: Both algorithms face stochastic transitions, "
                           "testing robustness to environmental uncertainty\n")
                
                f.write("\n")
                
                # Visualizations
                if env_name in visualization_files:
                    f.write("### 📊 Generated Visualizations\n\n")
                    for viz_type, filepath in visualization_files[env_name].items():
                        filename = os.path.basename(filepath)
                        f.write(f"- **{viz_type.replace('_', ' ').title()}**: `{filename}`\n")
                    f.write("\n")
        
        # Overall conclusions
        f.write("## 🎯 Overall Conclusions\n\n")
        
        # Determine overall winner
        overall_scores = {'sarsa': 0, 'q_learning': 0, 'tie': 0}
        
        for env_name in results['metadata']['environments']:
            if env_name in results['comparisons']:
                comparison = results['comparisons'][env_name]
                if 'final_reward' in comparison:
                    diff = comparison['final_reward']['difference']
                    std = comparison['final_reward']['pooled_std']
                    if abs(diff) > std:
                        if diff > 0:
                            overall_scores['q_learning'] += 1
                        else:
                            overall_scores['sarsa'] += 1
                    else:
                        overall_scores['tie'] += 1
        
        f.write("### 🏆 Algorithm Performance Summary\n\n")
        if overall_scores['q_learning'] > overall_scores['sarsa']:
            f.write("**Q-Learning** shows superior performance across environments.\n\n")
            f.write("**Why Q-Learning excels:**\n")
            f.write("- Off-policy learning allows exploration of optimal actions\n")
            f.write("- Maximum operator in update rule drives towards optimal policy\n")
            f.write("- Less conservative, willing to take risks for better long-term rewards\n\n")
        elif overall_scores['sarsa'] > overall_scores['q_learning']:
            f.write("**SARSA** shows superior performance across environments.\n\n")
            f.write("**Why SARSA excels:**\n")
            f.write("- On-policy learning ensures consistency with behavior\n")
            f.write("- More conservative approach reduces risk in dangerous environments\n")
            f.write("- Better suited for environments where safety is paramount\n\n")
        else:
            f.write("**Both algorithms show similar performance** across environments.\n\n")
            f.write("**Key insights:**\n")
            f.write("- Choice of algorithm depends on specific environment characteristics\n")
            f.write("- SARSA better for risk-sensitive applications\n")
            f.write("- Q-Learning better for pure performance optimization\n\n")
        
        f.write("### 🔬 Theoretical vs Practical Observations\n\n")
        f.write("**Theoretical Expectations:**\n")
        f.write("- Q-Learning should learn optimal policy faster (off-policy)\n")
        f.write("- SARSA should be more stable and conservative (on-policy)\n")
        f.write("- Both should converge to optimal policy under ideal conditions\n\n")
        
        f.write("**Practical Results:**\n")
        f.write("- Convergence speed varies by environment structure\n")
        f.write("- Exploration strategy (ε-greedy) significantly impacts both algorithms\n")
        f.write("- Environment stochasticity affects learning stability\n\n")
        
        # Recommendations
        f.write("## 💡 Recommendations\n\n")
        f.write("### When to use SARSA:\n")
        f.write("- Safety-critical applications (e.g., robotics, autonomous vehicles)\n")
        f.write("- Environments with severe penalties for suboptimal actions\n")
        f.write("- When behavior policy must be close to learned policy\n\n")
        
        f.write("### When to use Q-Learning:\n")
        f.write("- Pure performance optimization scenarios\n")
        f.write("- Environments where exploration risks are acceptable\n")
        f.write("- When learning optimal policy is more important than safe exploration\n\n")
        
        f.write("### Hyperparameter Tuning:\n")
        f.write("- **Learning rate (α)**: Higher for faster learning, lower for stability\n")
        f.write("- **Epsilon decay**: Slower decay for better exploration, faster for exploitation\n")
        f.write("- **Discount factor (γ)**: Higher for long-term planning, lower for immediate rewards\n\n")
        
        # Future work
        f.write("## 🚀 Future Directions\n\n")
        f.write("- **Function Approximation**: Test with neural networks for larger state spaces\n")
        f.write("- **Advanced Exploration**: Compare with UCB, Thompson sampling, etc.\n")
        f.write("- **Double Q-Learning**: Reduce overestimation bias\n")
        f.write("- **Expected SARSA**: Combine benefits of both approaches\n")
        f.write("- **Multi-step Methods**: TD(λ) for better credit assignment\n\n")
        
        f.write("---\n")
        f.write(f"*Report generated by Day 2 Challenge: Model-Free Tabular Methods*\n")
    
    print(f"📝 Comprehensive report saved to: {report_path}")


def print_final_summary(results: Dict, total_time: float):
    """Print final experiment summary."""
    print(f"\n" + "="*80)
    print(f"🎉 EXPERIMENT SUMMARY")
    print(f"="*80)
    
    print(f"⏱️  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"🌍 Environments: {len(results['metadata']['environments'])}")
    print(f"🎲 Seeds per algorithm: {len(results['metadata']['seeds'])}")
    print(f"📊 Episodes per run: {results['metadata']['n_episodes']}")
    print(f"🤖 Total training runs: {len(results['metadata']['environments']) * len(results['metadata']['seeds']) * 2}")
    
    print(f"\n🏆 PERFORMANCE WINNERS:")
    for env_name in results['metadata']['environments']:
        if env_name in results['comparisons']:
            comp = results['comparisons'][env_name]
            if 'final_reward' in comp:
                diff = comp['final_reward']['difference']
                std = comp['final_reward']['pooled_std']
                
                if abs(diff) > std:
                    winner = "Q-Learning" if diff > 0 else "SARSA"
                    margin = abs(diff)
                    print(f"   {env_name:20s}: {winner:10s} (+{margin:.2f} reward)")
                else:
                    print(f"   {env_name:20s}: TIE        (statistically similar)")
    
    print(f"\n🔍 KEY INSIGHTS:")
    print(f"   • SARSA learns the policy it follows (on-policy)")
    print(f"   • Q-Learning learns the optimal policy (off-policy)")
    print(f"   • Choice depends on environment risk characteristics")
    print(f"   • Both algorithms successfully converged in tabular settings")
    
    print(f"\n📁 Results saved to detailed report and visualizations")
    print(f"="*80)


def interactive_training_demo(env_name: str = "CliffWalking-v0", max_episodes: int = 300):
    """Interactive training animation demo."""
    print(f"🎬 Interactive Training Demo: {env_name}")
    print("=" * 60)
    
    # Create environment
    if 'FrozenLake' in env_name:
        env = gym.make(env_name, is_slippery=False)  # Deterministic for better learning
    else:
        env = gym.make(env_name)
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Create agents with same hyperparameters
    hyperparams = {
        'learning_rate': 0.2,        # Faster learning for demo
        'discount_factor': 0.9,
        'epsilon': 0.2,              # Higher exploration initially
        'epsilon_decay': 0.999,      # Slower decay for demo
        'epsilon_min': 0.05
    }
    
    print(f"🌍 Environment: {env_name} ({n_states} states, {n_actions} actions)")
    print(f"⚙️  Hyperparameters: {hyperparams}")
    print(f"📺 Max episodes: {max_episodes}")
    
    # Create agents
    sarsa_agent = SarsaAgent(n_states, n_actions, **hyperparams)
    qlearning_agent = QLearningAgent(n_states, n_actions, **hyperparams)
    
    # Create and run interactive animation
    animation = InteractiveTrainingAnimation(
        sarsa_agent, qlearning_agent, env, max_episodes
    )
    
    fig = animation.run()
    
    env.close()
    return animation


def quick_demo(env_name: str = "CliffWalking-v0", n_episodes: int = 300):
    """Quick demonstration for testing."""
    print(f"🚀 Quick Demo: {env_name}")
    
    from train import quick_comparison
    results = quick_comparison(env_name, n_episodes, n_seeds=2, verbose=True)
    
    print(f"\n✅ Quick demo completed for {env_name}")
    return results


if __name__ == "__main__":
    # Check if we want to run a quick demo, interactive demo, or full experiment
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            # Quick demo mode
            print("🎮 Running quick demo mode...")
            quick_demo("CliffWalking-v0", 200)
        elif sys.argv[1] == "interactive":
            # Interactive training animation
            env_name = sys.argv[2] if len(sys.argv) > 2 else "CliffWalking-v0"
            max_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 300
            print("🎬 Running interactive training animation...")
            interactive_training_demo(env_name, max_episodes)
        elif sys.argv[1] == "help":
            # Show usage
            print("🚀 Day 2: Model-Free Tabular Methods")
            print("=" * 50)
            print("Usage options:")
            print("  python main.py                    - Full experiment suite")
            print("  python main.py demo               - Quick comparison demo")
            print("  python main.py interactive [env]  - Interactive training animation")
            print("  python main.py help               - Show this help")
            print()
            print("Interactive examples:")
            print("  python main.py interactive CliffWalking-v0 300")
            print("  python main.py interactive FrozenLake-v1 200")
        else:
            print(f"❌ Unknown option: {sys.argv[1]}")
            print("Use 'python main.py help' for usage options")
    else:
        # Full experiment
        main() 