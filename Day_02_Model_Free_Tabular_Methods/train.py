"""
Training and Experiment Management for SARSA vs Q-Learning
Author: Day 2 - Model-Free Tabular Methods Challenge

This module handles training multiple agents across different environments
and random seeds for comprehensive comparison between SARSA and Q-Learning.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any
import time
import json
import os
from datetime import datetime

from sarsa import SarsaAgent
from q_learning import QLearningAgent


class ExperimentRunner:
    """
    Manages experiments comparing SARSA and Q-Learning across environments.
    """
    
    def __init__(self, results_dir: str = None):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to save results
        """
        if results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"results_{timestamp}"
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Store all experiment results
        self.results = {
            'sarsa': {},
            'q_learning': {},
            'comparisons': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'environments': [],
                'seeds': [],
                'n_episodes': 0
            }
        }
    
    def create_agent(
        self, 
        algorithm: str, 
        n_states: int, 
        n_actions: int, 
        **hyperparams
    ) -> Any:
        """
        Create agent based on algorithm type.
        
        Args:
            algorithm: 'sarsa' or 'q_learning'
            n_states: Number of states
            n_actions: Number of actions
            **hyperparams: Agent hyperparameters
            
        Returns:
            Agent instance
        """
        if algorithm.lower() == 'sarsa':
            return SarsaAgent(n_states, n_actions, **hyperparams)
        elif algorithm.lower() == 'q_learning':
            return QLearningAgent(n_states, n_actions, **hyperparams)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def run_single_experiment(
        self,
        env_name: str,
        algorithm: str,
        n_episodes: int = 1000,
        seed: int = 42,
        hyperparams: Dict = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single experiment with one agent on one environment.
        
        Args:
            env_name: Environment name (e.g., 'CliffWalking-v0')
            algorithm: Algorithm name ('sarsa' or 'q_learning')
            n_episodes: Number of training episodes
            seed: Random seed
            hyperparams: Agent hyperparameters
            verbose: Whether to print progress
            
        Returns:
            Experiment results dictionary
        """
        if hyperparams is None:
            hyperparams = {}
        
        # Set random seeds
        np.random.seed(seed)
        
        # Create environment with appropriate configuration
        try:
            if 'FrozenLake' in env_name:
                # Use deterministic version for better learning
                env = gym.make(env_name, is_slippery=False)
            else:
                env = gym.make(env_name)
        except Exception as e:
            print(f"❌ Failed to create environment {env_name}: {e}")
            return {}
        
        # Set environment seed
        env.reset(seed=seed)
        
        # Get environment info
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        if verbose:
            print(f"\n🏁 Starting Experiment")
            print(f"🌍 Environment: {env_name}")
            print(f"🤖 Algorithm: {algorithm.upper()}")
            print(f"🎲 Seed: {seed}")
            print(f"📊 States: {n_states}, Actions: {n_actions}")
            print(f"🔄 Episodes: {n_episodes}")
            print("-" * 50)
        
        # Create agent
        agent = self.create_agent(algorithm, n_states, n_actions, **hyperparams)
        
        # Record start time
        start_time = time.time()
        
        # Train agent
        training_info = agent.train(
            env, 
            n_episodes, 
            verbose=verbose, 
            log_interval=max(1, n_episodes // 10)
        )
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Evaluate final policy
        if verbose:
            print(f"\n🧪 Evaluating final policy...")
        
        eval_results = agent.evaluate_policy(env, n_episodes=100, render=False)
        
        # Compile results
        results = {
            'env_name': env_name,
            'algorithm': algorithm,
            'seed': seed,
            'n_episodes': n_episodes,
            'hyperparams': hyperparams,
            'training_time': training_time,
            'training_info': training_info,
            'evaluation': eval_results,
            'final_q_table': agent.Q.copy(),
            'final_policy': agent.get_policy(),
            'final_state_values': agent.get_state_values()
        }
        
        if verbose:
            print(f"⏱️  Training time: {training_time:.2f} seconds")
            print(f"📈 Final eval reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"🎯 Success rate: {eval_results['success_rate']:.2%}")
        
        env.close()
        return results
    
    def run_comparison_experiment(
        self,
        env_name: str,
        n_episodes: int = 1000,
        seeds: List[int] = None,
        hyperparams: Dict = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run comparison experiment between SARSA and Q-Learning.
        
        Args:
            env_name: Environment name
            n_episodes: Number of training episodes
            seeds: List of random seeds
            hyperparams: Shared hyperparameters for both agents
            verbose: Whether to print progress
            
        Returns:
            Comparison results
        """
        if seeds is None:
            seeds = [42, 123, 456, 789, 999]
        
        if hyperparams is None:
            hyperparams = {
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'epsilon': 0.1,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01
            }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔬 COMPARISON EXPERIMENT: SARSA vs Q-Learning")
            print(f"{'='*60}")
            print(f"🌍 Environment: {env_name}")
            print(f"🎲 Seeds: {seeds}")
            print(f"📊 Episodes per run: {n_episodes}")
            print(f"⚙️  Hyperparameters: {hyperparams}")
        
        # Store results for each algorithm
        sarsa_results = []
        qlearning_results = []
        
        # Run experiments for each seed
        for i, seed in enumerate(seeds):
            if verbose:
                print(f"\n🔄 Running seed {seed} ({i+1}/{len(seeds)})")
            
            # Run SARSA
            sarsa_result = self.run_single_experiment(
                env_name, 'sarsa', n_episodes, seed, hyperparams, verbose=False
            )
            if sarsa_result:
                sarsa_results.append(sarsa_result)
            
            # Run Q-Learning
            qlearning_result = self.run_single_experiment(
                env_name, 'q_learning', n_episodes, seed, hyperparams, verbose=False
            )
            if qlearning_result:
                qlearning_results.append(qlearning_result)
            
            if verbose and sarsa_result and qlearning_result:
                print(f"   SARSA:      {sarsa_result['evaluation']['mean_reward']:7.2f} ± {sarsa_result['evaluation']['std_reward']:5.2f}")
                print(f"   Q-Learning: {qlearning_result['evaluation']['mean_reward']:7.2f} ± {qlearning_result['evaluation']['std_reward']:5.2f}")
        
        # Analyze results
        comparison = self._analyze_comparison(sarsa_results, qlearning_results, verbose)
        
        # Store results
        self.results['sarsa'][env_name] = sarsa_results
        self.results['q_learning'][env_name] = qlearning_results
        self.results['comparisons'][env_name] = comparison
        
        return {
            'sarsa_results': sarsa_results,
            'qlearning_results': qlearning_results,
            'comparison': comparison
        }
    
    def _analyze_comparison(
        self, 
        sarsa_results: List[Dict], 
        qlearning_results: List[Dict],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze comparison between SARSA and Q-Learning results.
        
        Args:
            sarsa_results: List of SARSA experiment results
            qlearning_results: List of Q-Learning experiment results
            verbose: Whether to print analysis
            
        Returns:
            Comparison analysis
        """
        if not sarsa_results or not qlearning_results:
            return {}
        
        # Extract metrics
        def extract_metrics(results):
            return {
                'final_rewards': [r['evaluation']['mean_reward'] for r in results],
                'success_rates': [r['evaluation']['success_rate'] for r in results],
                'training_times': [r['training_time'] for r in results],
                'final_episodes_rewards': [np.mean(r['training_info']['episode_rewards'][-100:]) for r in results]
            }
        
        sarsa_metrics = extract_metrics(sarsa_results)
        qlearning_metrics = extract_metrics(qlearning_results)
        
        # Statistical comparison
        def compare_metric(sarsa_vals, qlearning_vals, metric_name):
            sarsa_mean = np.mean(sarsa_vals)
            sarsa_std = np.std(sarsa_vals)
            qlearning_mean = np.mean(qlearning_vals)
            qlearning_std = np.std(qlearning_vals)
            
            # Simple t-test approximation
            diff = qlearning_mean - sarsa_mean
            pooled_std = np.sqrt((sarsa_std**2 + qlearning_std**2) / 2)
            
            return {
                'sarsa_mean': sarsa_mean,
                'sarsa_std': sarsa_std,
                'qlearning_mean': qlearning_mean,
                'qlearning_std': qlearning_std,
                'difference': diff,
                'difference_pct': (diff / abs(sarsa_mean)) * 100 if sarsa_mean != 0 else 0,
                'pooled_std': pooled_std
            }
        
        analysis = {
            'final_reward': compare_metric(
                sarsa_metrics['final_rewards'], 
                qlearning_metrics['final_rewards'], 
                'final_reward'
            ),
            'success_rate': compare_metric(
                sarsa_metrics['success_rates'], 
                qlearning_metrics['success_rates'], 
                'success_rate'
            ),
            'training_time': compare_metric(
                sarsa_metrics['training_times'], 
                qlearning_metrics['training_times'], 
                'training_time'
            ),
            'learning_curve': compare_metric(
                sarsa_metrics['final_episodes_rewards'], 
                qlearning_metrics['final_episodes_rewards'], 
                'learning_curve'
            )
        }
        
        if verbose:
            print(f"\n📊 COMPARISON ANALYSIS")
            print("=" * 50)
            
            for metric_name, stats in analysis.items():
                print(f"\n{metric_name.replace('_', ' ').title()}:")
                print(f"  SARSA:      {stats['sarsa_mean']:8.3f} ± {stats['sarsa_std']:6.3f}")
                print(f"  Q-Learning: {stats['qlearning_mean']:8.3f} ± {stats['qlearning_std']:6.3f}")
                print(f"  Difference: {stats['difference']:8.3f} ({stats['difference_pct']:+5.1f}%)")
                
                # Determine winner
                if abs(stats['difference']) > stats['pooled_std']:
                    if stats['difference'] > 0:
                        winner = "Q-Learning"
                    else:
                        winner = "SARSA"
                    print(f"  Winner: {winner} 🏆")
                else:
                    print(f"  Winner: Tie (statistically similar)")
        
        return analysis
    
    def run_multiple_environments(
        self,
        env_names: List[str],
        n_episodes: int = 1000,
        seeds: List[int] = None,
        hyperparams: Dict = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run experiments across multiple environments.
        
        Args:
            env_names: List of environment names
            n_episodes: Episodes per experiment
            seeds: Random seeds
            hyperparams: Agent hyperparameters
            verbose: Whether to print progress
            
        Returns:
            All results across environments
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"🌍 MULTI-ENVIRONMENT EXPERIMENT")
            print(f"{'='*80}")
            print(f"Environments: {env_names}")
            print(f"Episodes per run: {n_episodes}")
            print(f"Seeds: {seeds if seeds else 'Default'}")
        
        all_results = {}
        
        for env_name in env_names:
            if verbose:
                print(f"\n{'='*60}")
                print(f"🎮 Environment: {env_name}")
                print(f"{'='*60}")
            
            try:
                env_results = self.run_comparison_experiment(
                    env_name, n_episodes, seeds, hyperparams, verbose
                )
                all_results[env_name] = env_results
                
                # Update metadata
                if env_name not in self.results['metadata']['environments']:
                    self.results['metadata']['environments'].append(env_name)
                
            except Exception as e:
                print(f"❌ Failed to run experiment on {env_name}: {e}")
                continue
        
        # Update metadata
        self.results['metadata']['seeds'] = seeds if seeds else [42]
        self.results['metadata']['n_episodes'] = n_episodes
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ ALL EXPERIMENTS COMPLETED")
            print(f"{'='*80}")
            self._print_summary()
        
        return all_results
    
    def _print_summary(self):
        """Print summary of all experiments."""
        print(f"\n📋 EXPERIMENT SUMMARY")
        print("-" * 40)
        
        for env_name in self.results['metadata']['environments']:
            if env_name in self.results['comparisons']:
                comp = self.results['comparisons'][env_name]
                if 'final_reward' in comp:
                    sarsa_reward = comp['final_reward']['sarsa_mean']
                    qlearning_reward = comp['final_reward']['qlearning_mean']
                    diff = comp['final_reward']['difference']
                    
                    winner = "Q-Learning" if diff > 0 else "SARSA"
                    print(f"{env_name:20s}: {winner:10s} (+{abs(diff):5.2f})")
    
    def save_results(self, filename: str = "experiment_results.json"):
        """Save all results to JSON file."""
        filepath = os.path.join(self.results_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)
        
        print(f"📂 Results loaded from: {filepath}")


def quick_comparison(
    env_name: str = "CliffWalking-v0",
    n_episodes: int = 500,
    n_seeds: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick comparison function for testing.
    
    Args:
        env_name: Environment to test
        n_episodes: Number of episodes
        n_seeds: Number of random seeds
        verbose: Whether to print results
        
    Returns:
        Comparison results
    """
    # Generate random seeds
    seeds = [42 + i * 111 for i in range(n_seeds)]
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run comparison
    results = runner.run_comparison_experiment(
        env_name, n_episodes, seeds, verbose=verbose
    )
    
    return results


if __name__ == "__main__":
    # Quick test of the training module
    print("🧪 Testing Training Module...")
    
    # Test on CliffWalking
    results = quick_comparison("CliffWalking-v0", n_episodes=200, n_seeds=2)
    
    print("\n✅ Training module test completed!") 