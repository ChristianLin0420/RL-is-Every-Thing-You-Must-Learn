"""
Training Module for REINFORCE

This module provides comprehensive training functionality for REINFORCE,
including hyperparameter experiments, baseline comparisons, and analysis tools.
"""

import gymnasium as gym
import torch
import numpy as np
import os
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt

from policy_network import PolicyNetwork, create_cartpole_policy
from reinforce import REINFORCE, REINFORCETrainer
from visualize import REINFORCEVisualizer


@dataclass
class TrainingConfig:
    """Configuration for REINFORCE training."""
    
    # Environment settings
    env_name: str = "CartPole-v1"
    max_steps_per_episode: int = 500
    
    # Training settings
    num_episodes: int = 1000
    eval_freq: int = 50
    eval_episodes: int = 10
    
    # Agent hyperparameters
    learning_rate: float = 1e-3
    gamma: float = 0.99
    use_baseline: bool = False
    baseline_lr: float = 1e-2
    optimizer: str = 'adam'
    max_grad_norm: Optional[float] = None
    
    # Network architecture
    hidden_dims: List[int] = None
    activation: str = 'relu'
    
    # Experiment settings
    seed: Optional[int] = 42
    verbose: bool = True
    save_checkpoints: bool = True
    save_frequency: int = 100
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 128]


class ExperimentRunner:
    """
    High-level experiment runner for REINFORCE studies.
    
    Supports:
    - Single training runs
    - Hyperparameter sweeps
    - Baseline comparisons
    - Multi-seed experiments
    """
    
    def __init__(self, base_config: TrainingConfig = None):
        """
        Initialize experiment runner.
        
        Args:
            base_config: Base configuration for experiments
        """
        self.base_config = base_config or TrainingConfig()
        self.visualizer = REINFORCEVisualizer()
        self.results = {}
        
    def run_single_experiment(self, 
                            config: TrainingConfig,
                            experiment_name: str = "experiment",
                            save_dir: str = "results/") -> Dict[str, Any]:
        """
        Run a single REINFORCE experiment.
        
        Args:
            config: Training configuration
            experiment_name: Name for this experiment
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing training results and statistics
        """
        print(f"🚀 Starting experiment: {experiment_name}")
        print(f"   Environment: {config.env_name}")
        print(f"   Episodes: {config.num_episodes}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Baseline: {config.use_baseline}")
        print(f"   Seed: {config.seed}")
        
        # Set seeds for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # Create environment
        env = gym.make(config.env_name)
        env.action_space.seed(config.seed)
        
        # Create policy network
        if config.env_name == "CartPole-v1":
            policy = create_cartpole_policy(seed=config.seed)
        else:
            # Generic setup for other environments
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = PolicyNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.hidden_dims,
                activation=config.activation,
                seed=config.seed
            )
        
        # Create REINFORCE agent
        agent = REINFORCE(
            policy=policy,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            use_baseline=config.use_baseline,
            baseline_lr=config.baseline_lr,
            optimizer=config.optimizer,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed
        )
        
        # Create trainer
        trainer = REINFORCETrainer(agent, env)
        
        # Training
        start_time = time.time()
        
        results = trainer.train(
            num_episodes=config.num_episodes,
            max_steps_per_episode=config.max_steps_per_episode,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes,
            verbose=config.verbose
        )
        
        training_time = time.time() - start_time
        
        # Add experiment metadata
        results['experiment_name'] = experiment_name
        results['config'] = config.__dict__
        results['training_time'] = training_time
        results['policy_parameters'] = policy.get_parameter_count()
        
        # Save results if requested
        if config.save_checkpoints:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save agent checkpoint
            agent_path = os.path.join(save_dir, f"{experiment_name}_agent.pth")
            agent.save_checkpoint(agent_path)
            
            # Save results
            results_path = os.path.join(save_dir, f"{experiment_name}_results.json")
            self._save_results(results, results_path)
            
            # Save visualizations
            fig_dir = os.path.join(save_dir, "figures")
            saved_figs = self.visualizer.save_all_visualizations(
                results['training_stats'],
                results.get('eval_stats'),
                save_dir=fig_dir,
                prefix=experiment_name
            )
            results['saved_figures'] = saved_figs
        
        env.close()
        
        print(f"✅ Experiment completed in {training_time:.2f}s")
        if results['final_performance']:
            final_reward = results['final_performance']['mean_reward']
            print(f"   Final performance: {final_reward:.2f} ± {results['final_performance']['std_reward']:.2f}")
        
        self.results[experiment_name] = results
        return results
    
    def run_hyperparameter_sweep(self, 
                                param_grid: Dict[str, List],
                                base_name: str = "hp_sweep",
                                save_dir: str = "results/hyperparameter_sweep/") -> Dict[str, Any]:
        """
        Run hyperparameter sweep experiment.
        
        Args:
            param_grid: Dictionary with parameter names as keys and lists of values
            base_name: Base name for experiments
            save_dir: Directory to save results
            
        Returns:
            Dictionary with sweep results
        """
        print(f"🔬 Starting hyperparameter sweep with {len(param_grid)} parameters")
        
        # Generate all combinations
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        sweep_results = {}
        best_performance = -float('inf')
        best_config = None
        
        for i, combo in enumerate(combinations):
            # Create config for this combination
            config = TrainingConfig(**self.base_config.__dict__)
            
            for param_name, param_value in zip(param_names, combo):
                setattr(config, param_name, param_value)
            
            # Create experiment name
            param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, combo)])
            experiment_name = f"{base_name}_{i:03d}_{param_str}"
            
            # Run experiment
            try:
                results = self.run_single_experiment(
                    config=config,
                    experiment_name=experiment_name,
                    save_dir=save_dir
                )
                
                sweep_results[experiment_name] = results
                
                # Track best performance
                if results['final_performance']:
                    performance = results['final_performance']['mean_reward']
                    if performance > best_performance:
                        best_performance = performance
                        best_config = config
                        
            except Exception as e:
                print(f"❌ Failed experiment {experiment_name}: {e}")
                continue
        
        # Analyze sweep results
        analysis = self._analyze_hyperparameter_sweep(sweep_results, param_names)
        
        print(f"🎯 Hyperparameter sweep completed!")
        print(f"   Best performance: {best_performance:.2f}")
        print(f"   Best config: {best_config.__dict__ if best_config else 'None'}")
        
        return {
            'sweep_results': sweep_results,
            'analysis': analysis,
            'best_performance': best_performance,
            'best_config': best_config.__dict__ if best_config else None
        }
    
    def run_baseline_comparison(self, 
                              configs: List[Tuple[str, TrainingConfig]],
                              save_dir: str = "results/baseline_comparison/") -> Dict[str, Any]:
        """
        Compare different REINFORCE configurations.
        
        Args:
            configs: List of (name, config) tuples
            save_dir: Directory to save results
            
        Returns:
            Comparison results
        """
        print(f"📊 Running baseline comparison with {len(configs)} configurations")
        
        comparison_results = {}
        
        for name, config in configs:
            print(f"\n--- Running {name} ---")
            try:
                results = self.run_single_experiment(
                    config=config,
                    experiment_name=name,
                    save_dir=save_dir
                )
                comparison_results[name] = results
            except Exception as e:
                print(f"❌ Failed configuration {name}: {e}")
                continue
        
        # Create comparison visualization
        if len(comparison_results) > 1:
            training_stats_dict = {
                name: results['training_stats'] 
                for name, results in comparison_results.items()
            }
            
            # Plot comparison
            fig = self.visualizer.plot_policy_comparison(
                training_stats_dict, 
                metric='episode_returns'
            )
            
            comparison_path = os.path.join(save_dir, "baseline_comparison.png")
            fig.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"💾 Comparison plot saved to {comparison_path}")
        
        return comparison_results
    
    def run_multi_seed_experiment(self, 
                                config: TrainingConfig,
                                seeds: List[int],
                                experiment_name: str = "multi_seed",
                                save_dir: str = "results/multi_seed/") -> Dict[str, Any]:
        """
        Run experiment with multiple seeds for statistical reliability.
        
        Args:
            config: Training configuration
            seeds: List of seeds to use
            experiment_name: Base name for experiments
            save_dir: Directory to save results
            
        Returns:
            Multi-seed results with statistics
        """
        print(f"🎲 Running multi-seed experiment with {len(seeds)} seeds")
        
        seed_results = {}
        final_performances = []
        
        for seed in seeds:
            seed_config = TrainingConfig(**config.__dict__)
            seed_config.seed = seed
            seed_config.verbose = False  # Reduce verbosity for multi-seed
            
            seed_name = f"{experiment_name}_seed_{seed}"
            
            try:
                results = self.run_single_experiment(
                    config=seed_config,
                    experiment_name=seed_name,
                    save_dir=save_dir
                )
                
                seed_results[seed] = results
                
                if results['final_performance']:
                    final_performances.append(results['final_performance']['mean_reward'])
                    
            except Exception as e:
                print(f"❌ Failed seed {seed}: {e}")
                continue
        
        # Compute statistics across seeds
        if final_performances:
            statistics = {
                'mean_performance': np.mean(final_performances),
                'std_performance': np.std(final_performances),
                'min_performance': np.min(final_performances),
                'max_performance': np.max(final_performances),
                'median_performance': np.median(final_performances),
                'num_seeds': len(final_performances)
            }
            
            print(f"📈 Multi-seed statistics:")
            print(f"   Mean: {statistics['mean_performance']:.2f} ± {statistics['std_performance']:.2f}")
            print(f"   Range: [{statistics['min_performance']:.2f}, {statistics['max_performance']:.2f}]")
        else:
            statistics = None
        
        return {
            'seed_results': seed_results,
            'statistics': statistics,
            'config': config.__dict__
        }
    
    def _analyze_hyperparameter_sweep(self, 
                                    sweep_results: Dict[str, Any],
                                    param_names: List[str]) -> Dict[str, Any]:
        """Analyze hyperparameter sweep results."""
        analysis = {}
        
        # Extract performance for each parameter value
        for param_name in param_names:
            param_performance = {}
            
            for exp_name, results in sweep_results.items():
                if results['final_performance']:
                    param_value = results['config'][param_name]
                    performance = results['final_performance']['mean_reward']
                    
                    if param_value not in param_performance:
                        param_performance[param_value] = []
                    param_performance[param_value].append(performance)
            
            # Compute statistics for each parameter value
            param_stats = {}
            for value, performances in param_performance.items():
                param_stats[value] = {
                    'mean': np.mean(performances),
                    'std': np.std(performances),
                    'count': len(performances)
                }
            
            analysis[param_name] = param_stats
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any], filepath: str):
        """Save results to JSON file (handling non-serializable objects)."""
        # Create serializable version
        serializable_results = {}
        
        for key, value in results.items():
            if key == 'training_stats':
                # Convert numpy arrays to lists
                serializable_stats = {}
                for stat_key, stat_value in value.items():
                    if isinstance(stat_value, list) and len(stat_value) > 0:
                        if isinstance(stat_value[0], np.ndarray):
                            serializable_stats[stat_key] = [arr.tolist() for arr in stat_value]
                        else:
                            serializable_stats[stat_key] = stat_value
                    else:
                        serializable_stats[stat_key] = stat_value
                serializable_results[key] = serializable_stats
            elif isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)


# Predefined experiment configurations
def get_baseline_configs() -> List[Tuple[str, TrainingConfig]]:
    """Get standard baseline configurations for comparison."""
    configs = []
    
    # Vanilla REINFORCE
    vanilla_config = TrainingConfig(
        learning_rate=1e-3,
        use_baseline=False,
        num_episodes=800
    )
    configs.append(("Vanilla REINFORCE", vanilla_config))
    
    # REINFORCE with baseline
    baseline_config = TrainingConfig(
        learning_rate=1e-3,
        use_baseline=True,
        baseline_lr=1e-2,
        num_episodes=800
    )
    configs.append(("REINFORCE with Baseline", baseline_config))
    
    # Different learning rates
    high_lr_config = TrainingConfig(
        learning_rate=5e-3,
        use_baseline=True,
        num_episodes=800
    )
    configs.append(("High Learning Rate", high_lr_config))
    
    low_lr_config = TrainingConfig(
        learning_rate=1e-4,
        use_baseline=True,
        num_episodes=800
    )
    configs.append(("Low Learning Rate", low_lr_config))
    
    return configs


def get_hyperparameter_grid() -> Dict[str, List]:
    """Get standard hyperparameter grid for sweeps."""
    return {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'use_baseline': [True, False],
        'gamma': [0.95, 0.99, 0.995],
        'optimizer': ['adam', 'sgd']
    }


# Quick experiment functions
def quick_cartpole_experiment(num_episodes: int = 500, 
                            use_baseline: bool = True,
                            learning_rate: float = 1e-3,
                            seed: int = 42) -> Dict[str, Any]:
    """Quick CartPole experiment for testing."""
    config = TrainingConfig(
        num_episodes=num_episodes,
        use_baseline=use_baseline,
        learning_rate=learning_rate,
        seed=seed,
        verbose=True
    )
    
    runner = ExperimentRunner()
    return runner.run_single_experiment(config, "quick_cartpole")


if __name__ == "__main__":
    # Example usage
    print("🧪 Running example REINFORCE experiments...")
    
    # Quick test
    results = quick_cartpole_experiment(num_episodes=200)
    print(f"Quick test completed with final reward: {results['final_performance']['mean_reward']:.2f}")
    
    # Baseline comparison
    runner = ExperimentRunner()
    baseline_configs = get_baseline_configs()
    
    # Run just the first two for demo
    demo_configs = baseline_configs[:2]
    comparison_results = runner.run_baseline_comparison(demo_configs)
    
    print("🎉 Demo experiments completed!") 