"""
Main Script for REINFORCE Experiments

This script provides a command-line interface to run various REINFORCE experiments:
- Single training runs
- Hyperparameter sweeps
- Baseline comparisons
- Interactive training visualization
"""

import argparse
import os
import sys
import time
from typing import Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt

from train import ExperimentRunner, TrainingConfig, quick_cartpole_experiment, get_baseline_configs, get_hyperparameter_grid
from visualize import REINFORCEVisualizer
from policy_network import create_cartpole_policy
from reinforce import REINFORCE


def run_basic_training(args):
    """Run basic REINFORCE training."""
    print("🎯 Running Basic REINFORCE Training")
    print("=" * 50)
    
    config = TrainingConfig(
        env_name=args.env,
        num_episodes=args.episodes,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        use_baseline=args.use_baseline,
        seed=args.seed,
        verbose=True,
        save_checkpoints=True
    )
    
    runner = ExperimentRunner()
    results = runner.run_single_experiment(
        config=config,
        experiment_name=f"reinforce_{args.env}_{int(time.time())}",
        save_dir=args.save_dir
    )
    
    # Display final results
    print("\n" + "=" * 50)
    print("🎉 Training Completed!")
    print("=" * 50)
    
    if results['final_performance']:
        final_perf = results['final_performance']
        print(f"Final Performance: {final_perf['mean_reward']:.2f} ± {final_perf['std_reward']:.2f}")
        print(f"Max Reward: {final_perf['max_reward']:.2f}")
        print(f"Episode Length: {final_perf['mean_length']:.1f}")
    
    print(f"Training Time: {results['training_time']:.2f}s")
    print(f"Policy Parameters: {results['policy_parameters']:,}")
    
    if 'saved_figures' in results:
        print(f"Figures saved: {len(results['saved_figures'])}")
        for fig_path in results['saved_figures']:
            print(f"  - {fig_path}")
    
    return results


def run_hyperparameter_sweep(args):
    """Run hyperparameter sweep experiment."""
    print("🔬 Running Hyperparameter Sweep")
    print("=" * 50)
    
    if args.quick:
        # Quick sweep for testing
        param_grid = {
            'learning_rate': [1e-3, 5e-3],
            'use_baseline': [True, False]
        }
        num_episodes = 200
    else:
        # Full sweep
        param_grid = get_hyperparameter_grid()
        num_episodes = args.episodes
    
    # Update base config
    base_config = TrainingConfig(
        env_name=args.env,
        num_episodes=num_episodes,
        seed=args.seed,
        verbose=False  # Reduce verbosity for sweeps
    )
    
    runner = ExperimentRunner(base_config)
    
    print(f"Parameter grid: {param_grid}")
    total_experiments = np.prod([len(values) for values in param_grid.values()])
    print(f"Total experiments: {total_experiments}")
    
    if total_experiments > 20 and not args.force:
        response = input(f"This will run {total_experiments} experiments. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Sweep cancelled.")
            return None
    
    results = runner.run_hyperparameter_sweep(
        param_grid=param_grid,
        base_name="hp_sweep",
        save_dir=os.path.join(args.save_dir, "hyperparameter_sweep")
    )
    
    # Display best results
    print("\n" + "=" * 50)
    print("🏆 Hyperparameter Sweep Results")
    print("=" * 50)
    print(f"Best Performance: {results['best_performance']:.2f}")
    print("Best Configuration:")
    if results['best_config']:
        for key, value in results['best_config'].items():
            print(f"  {key}: {value}")
    
    return results


def run_baseline_comparison(args):
    """Run baseline comparison experiment."""
    print("📊 Running Baseline Comparison")
    print("=" * 50)
    
    configs = get_baseline_configs()
    
    # Update episode count for all configs
    for name, config in configs:
        config.num_episodes = args.episodes
        config.env_name = args.env
        config.seed = args.seed
    
    print(f"Comparing {len(configs)} configurations:")
    for name, _ in configs:
        print(f"  - {name}")
    
    runner = ExperimentRunner()
    results = runner.run_baseline_comparison(
        configs=configs,
        save_dir=os.path.join(args.save_dir, "baseline_comparison")
    )
    
    # Display comparison results
    print("\n" + "=" * 50)
    print("📈 Comparison Results")
    print("=" * 50)
    
    performances = []
    for name, result in results.items():
        if result['final_performance']:
            perf = result['final_performance']['mean_reward']
            std = result['final_performance']['std_reward']
            performances.append((name, perf, std))
            print(f"{name:25s}: {perf:6.2f} ± {std:5.2f}")
    
    # Find best performing configuration
    if performances:
        best_config = max(performances, key=lambda x: x[1])
        print(f"\n🥇 Best: {best_config[0]} ({best_config[1]:.2f})")
    
    return results


def run_interactive_demo(args):
    """Run interactive training demonstration."""
    print("🎮 Running Interactive Training Demo")
    print("=" * 50)
    
    config = TrainingConfig(
        env_name=args.env,
        num_episodes=min(args.episodes, 300),  # Limit for interactive demo
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        seed=args.seed,
        verbose=True
    )
    
    # Run training
    runner = ExperimentRunner()
    results = runner.run_single_experiment(
        config=config,
        experiment_name="interactive_demo",
        save_dir=args.save_dir
    )
    
    # Create interactive visualization
    visualizer = REINFORCEVisualizer()
    fig = visualizer.create_interactive_training_visualization(
        results['training_stats'],
        env_name=args.env
    )
    
    print("\n🎨 Interactive visualization created!")
    print("Use the slider to explore training progress.")
    print("Close the plot window to continue...")
    
    plt.show()
    
    return results


def run_quick_test(args):
    """Run quick test for debugging."""
    print("⚡ Running Quick Test")
    print("=" * 30)
    
    results = quick_cartpole_experiment(
        num_episodes=100,
        use_baseline=args.use_baseline,
        learning_rate=args.learning_rate,
        seed=args.seed
    )
    
    print("✅ Quick test completed!")
    if results['final_performance']:
        print(f"Final reward: {results['final_performance']['mean_reward']:.2f}")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="REINFORCE Algorithm Training and Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode basic --episodes 500
  python main.py --mode sweep --quick
  python main.py --mode comparison --episodes 300
  python main.py --mode interactive --episodes 200
  python main.py --mode test
        """
    )
    
    # Main arguments
    parser.add_argument('--mode', type=str, default='basic',
                       choices=['basic', 'sweep', 'comparison', 'interactive', 'test'],
                       help='Experiment mode to run')
    
    # Environment and training settings
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Gymnasium environment name')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate for policy optimizer')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--use-baseline', action='store_true',
                       help='Use baseline for variance reduction')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-dir', type=str, default='results/',
                       help='Directory to save results')
    
    # Special flags
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version (for testing)')
    parser.add_argument('--force', action='store_true',
                       help='Force run large experiments without confirmation')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Setup
    if args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    print("🧠 REINFORCE Policy Gradient Implementation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Use Baseline: {args.use_baseline}")
    print(f"Seed: {args.seed}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() and not args.no_cuda else 'CPU'}")
    print("=" * 60)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run selected mode
    start_time = time.time()
    
    try:
        if args.mode == 'basic':
            results = run_basic_training(args)
        elif args.mode == 'sweep':
            results = run_hyperparameter_sweep(args)
        elif args.mode == 'comparison':
            results = run_baseline_comparison(args)
        elif args.mode == 'interactive':
            results = run_interactive_demo(args)
        elif args.mode == 'test':
            results = run_quick_test(args)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
            
        total_time = time.time() - start_time
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")
        
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results is not None:
        print("\n🎊 Experiment completed successfully!")
        print("Check the results directory for saved outputs.")
    else:
        sys.exit(1) 