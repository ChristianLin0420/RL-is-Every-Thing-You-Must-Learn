"""
Main Script for GAE Training and Analysis

Provides command-line interface for:
- Basic GAE training
- Lambda sweep experiments
- Bias-variance analysis
- Interactive visualization
- Model evaluation
"""

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import pickle
from datetime import datetime

from env_wrapper import make_env, get_env_info
from train import ActorCriticTrainer, TrainingConfig, run_lambda_sweep
from visualize import GAEVisualizer, compare_gae_methods
from gae_utils import GAEComputer, create_trajectory_from_episode


def setup_directories():
    """Create necessary directories for results."""
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)


def save_results(results: Dict, filename: str):
    """Save experiment results."""
    setup_directories()
    with open(f'results/{filename}', 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Results saved to results/{filename}")


def load_results(filename: str) -> Dict:
    """Load experiment results."""
    with open(f'results/{filename}', 'rb') as f:
        return pickle.load(f)


def train_basic(args):
    """Run basic GAE training."""
    print("🚀 Starting Basic GAE Training...")
    
    config = TrainingConfig(
        env_name=args.env,
        max_episodes=args.episodes,
        learning_rate=args.lr,
        lambda_gae=args.lambda_gae,
        gamma=args.gamma,
        normalize_obs=args.normalize_obs,
        shared_backbone=args.shared_backbone,
        log_interval=args.log_interval,
        seed=args.seed
    )
    
    trainer = ActorCriticTrainer(config)
    metrics = trainer.train()
    
    # Save model and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/gae_basic_{args.env}_{timestamp}.pth'
    trainer.save_model(model_path)
    
    # Create visualizations
    visualizer = GAEVisualizer()
    
    # Training progress
    fig1 = visualizer.plot_training_progress(metrics, config)
    plt.savefig(f'figures/gae_training_progress_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Evaluate final policy
    eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes, render=args.render)
    print(f"\n📊 Final Evaluation Results:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Save results
    results = {
        'config': config,
        'metrics': metrics,
        'eval_metrics': eval_metrics,
        'model_path': model_path
    }
    save_results(results, f'basic_training_{timestamp}.pkl')
    
    trainer.env.close()
    print("✅ Basic training completed!")


def lambda_sweep_experiment(args):
    """Run lambda sweep experiment."""
    print("🧪 Starting Lambda Sweep Experiment...")
    
    base_config = TrainingConfig(
        env_name=args.env,
        max_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        normalize_obs=args.normalize_obs,
        shared_backbone=args.shared_backbone,
        log_interval=max(args.log_interval, args.episodes // 10),
        seed=args.seed
    )
    
    # Define lambda values to test
    if args.quick:
        lambda_values = [0.0, 0.5, 0.95, 1.0]
        num_seeds = 2
    else:
        lambda_values = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        num_seeds = args.seeds
    
    print(f"Testing λ values: {lambda_values}")
    print(f"Seeds per λ: {num_seeds}")
    
    # Run sweep
    sweep_results = run_lambda_sweep(base_config, lambda_values, num_seeds)
    
    # Create visualizations
    visualizer = GAEVisualizer()
    
    # Lambda sweep analysis
    fig2 = visualizer.plot_lambda_sweep_results(sweep_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'figures/lambda_sweep_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print(f"\n📈 Lambda Sweep Summary:")
    for lam in lambda_values:
        metrics_list = sweep_results[lam]
        final_rewards = []
        for metrics in metrics_list:
            last_20_percent = len(metrics.episode_rewards) // 5
            final_rewards.append(np.mean(metrics.episode_rewards[-last_20_percent:]))
        
        print(f"  λ={lam:4.2f}: {np.mean(final_rewards):7.2f} ± {np.std(final_rewards):5.2f}")
    
    # Find best lambda
    best_lambda = max(lambda_values, key=lambda lam: np.mean([
        np.mean(metrics.episode_rewards[-len(metrics.episode_rewards)//5:]) 
        for metrics in sweep_results[lam]
    ]))
    print(f"\n🏆 Best λ: {best_lambda}")
    
    # Save results
    results = {
        'base_config': base_config,
        'lambda_values': lambda_values,
        'num_seeds': num_seeds,
        'sweep_results': sweep_results,
        'best_lambda': best_lambda
    }
    save_results(results, f'lambda_sweep_{timestamp}.pkl')
    
    print("✅ Lambda sweep completed!")


def bias_variance_analysis(args):
    """Run detailed bias-variance analysis."""
    print("🔍 Starting Bias-Variance Analysis...")
    
    # Create multiple environments for data collection
    env = make_env(args.env, normalize_obs=args.normalize_obs, seed=args.seed)
    env_info = get_env_info(env)
    
    # Collect trajectories from random policy
    trajectories = []
    print(f"Collecting {args.num_trajectories} trajectories...")
    
    for i in range(args.num_trajectories):
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        
        state, _ = env.reset()
        episode_length = 0
        
        while episode_length < 100:  # Limit episode length
            # Random action
            action = env.action_space.sample()
            
            # Dummy value estimate (random for analysis)
            value = np.random.normal(0, 1)
            log_prob = np.random.normal(0, 1)
            
            states.append(state.copy())
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            dones.append(done)
            
            episode_length += 1
            
            if done:
                break
                
            state = next_state
        
        # Create trajectory
        trajectory = create_trajectory_from_episode(
            states, actions, rewards, values, log_probs, dones, 0.0
        )
        trajectories.append(trajectory)
        
        if (i + 1) % 10 == 0:
            print(f"  Collected {i + 1}/{args.num_trajectories} trajectories")
    
    env.close()
    
    # Analyze bias-variance tradeoff
    gae_computer = GAEComputer(gamma=args.gamma)
    analysis = gae_computer.analyze_bias_variance(trajectories)
    
    print(f"\n📊 Bias-Variance Analysis Results:")
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'Variance':>10}")
    print("-" * 50)
    for method, stats in analysis.items():
        print(f"{method:<20} {stats['mean']:8.3f} {stats['std']:8.3f} {stats['variance']:10.4f}")
    
    # Create advantage comparison visualization
    visualizer = GAEVisualizer()
    fig = compare_gae_methods(trajectories[:20])  # Use subset for visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'figures/bias_variance_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'trajectories': trajectories,
        'analysis': analysis,
        'config': {
            'env': args.env,
            'gamma': args.gamma,
            'num_trajectories': args.num_trajectories
        }
    }
    save_results(results, f'bias_variance_analysis_{timestamp}.pkl')
    
    print("✅ Bias-variance analysis completed!")


def interactive_monitor(args):
    """Run interactive training monitor."""
    print("🎮 Starting Interactive Training Monitor...")
    
    if args.load_results:
        print(f"Loading results from {args.load_results}...")
        results = load_results(args.load_results)
        metrics = results['metrics']
        config = results['config']
    else:
        # Run training with frequent updates for smooth monitoring
        config = TrainingConfig(
            env_name=args.env,
            max_episodes=args.episodes,
            learning_rate=args.lr,
            lambda_gae=args.lambda_gae,
            gamma=args.gamma,
            log_interval=max(1, args.episodes // 50),
            seed=args.seed
        )
        
        trainer = ActorCriticTrainer(config)
        metrics = trainer.train()
        trainer.env.close()
    
    # Create interactive visualization
    visualizer = GAEVisualizer()
    fig = visualizer.create_interactive_training_monitor(metrics, config)
    plt.show()
    
    print("✅ Interactive monitor session completed!")


def evaluate_model(args):
    """Evaluate a trained model."""
    print("🎯 Evaluating Trained Model...")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    # Create trainer and load model
    trainer = ActorCriticTrainer(config)
    trainer.load_model(args.model_path)
    
    print(f"Loaded model from episode {checkpoint['episode']}")
    print(f"Environment: {config.env_name}, λ={config.lambda_gae}")
    
    # For rendering, create a separate environment with proper render mode
    if args.render:
        print("🎬 Creating render-enabled environment...")
        try:
            import gymnasium as gym
            render_env = gym.make(config.env_name, render_mode='human')
            
            # Apply same wrappers as training environment (but skip normalization for consistent viewing)
            from env_wrapper import EpisodeTracker, NormalizeObservation
            render_env = EpisodeTracker(render_env)
            if config.normalize_obs:
                render_env = NormalizeObservation(render_env)
            
            # Evaluate with rendering
            eval_rewards = []
            eval_lengths = []
            
            for episode in range(args.eval_episodes):
                print(f"🎬 Rendering episode {episode + 1}/{args.eval_episodes}")
                state, _ = render_env.reset()
                total_reward = 0.0
                episode_length = 0
                
                while True:
                    try:
                        render_env.render()
                    except Exception as e:
                        if episode == 0:
                            print(f"⚠️  Rendering issue: {e}")
                    
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        if trainer.env_info['action_type'] == 'discrete':
                            logits = trainer.actor_critic.actor(state_tensor)
                            action = torch.argmax(logits, dim=-1).item()
                        else:
                            mean, _ = trainer.actor_critic.actor(state_tensor)
                            action = mean.cpu().numpy().squeeze()
                    
                    state, reward, terminated, truncated, _ = render_env.step(action)
                    total_reward += reward
                    episode_length += 1
                    
                    if terminated or truncated:
                        break
                
                eval_rewards.append(total_reward)
                eval_lengths.append(episode_length)
                print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {episode_length}")
            
            render_env.close()
            
            # Also run normal evaluation for accurate statistics
            eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes, render=False)
            
        except Exception as e:
            print(f"⚠️  Rendering failed: {e}")
            print("Running evaluation without rendering...")
            eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes, render=False)
    else:
        # Standard evaluation without rendering
        eval_metrics = trainer.evaluate(num_episodes=args.eval_episodes, render=False)
    
    print(f"\n📊 Evaluation Results ({args.eval_episodes} episodes):")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Performance interpretation
    mean_reward = eval_metrics['mean_reward']
    if 'CartPole' in config.env_name:
        if mean_reward > 475:
            print("🏆 Excellent! CartPole is considered solved (>475 average reward)")
        elif mean_reward > 400:
            print("✅ Good performance! Close to solving CartPole")
        elif mean_reward > 200:
            print("📈 Decent performance, but room for improvement")
        else:
            print("📉 Low performance, may need more training or hyperparameter tuning")
    elif 'LunarLander' in config.env_name:
        if mean_reward > 200:
            print("🏆 Excellent! LunarLander is considered solved (>200 average reward)")
        elif mean_reward > 100:
            print("✅ Good progress! Getting close to solving LunarLander")
        elif mean_reward > 0:
            print("📈 Learning in progress, positive average reward")
        else:
            print("📉 Still learning, negative average reward indicates need for more training")
    
    trainer.env.close()
    print("✅ Model evaluation completed!")


def main():
    parser = argparse.ArgumentParser(description='GAE Training and Analysis')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['basic', 'sweep', 'analysis', 'interactive', 'evaluate'],
                       help='Training mode')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name')
    parser.add_argument('--normalize-obs', action='store_true', default=True,
                       help='Normalize observations')
    
    # Training settings
    parser.add_argument('--episodes', type=int, default=300,
                       help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--lambda-gae', type=float, default=0.95,
                       help='GAE lambda parameter')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--shared-backbone', action='store_true',
                       help='Use shared backbone for actor-critic')
    
    # Experiment settings
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of seeds for sweep experiments')
    parser.add_argument('--quick', action='store_true',
                       help='Quick experiment with fewer lambda values')
    parser.add_argument('--num-trajectories', type=int, default=50,
                       help='Number of trajectories for bias-variance analysis')
    
    # Evaluation settings
    parser.add_argument('--eval-episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    
    # I/O settings
    parser.add_argument('--model-path', type=str,
                       help='Path to model for evaluation')
    parser.add_argument('--load-results', type=str,
                       help='Load previous results for analysis')
    
    # Other settings
    parser.add_argument('--log-interval', type=int, default=20,
                       help='Logging interval')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup directories
    setup_directories()
    
    # Run appropriate mode
    if args.mode == 'basic':
        train_basic(args)
    elif args.mode == 'sweep':
        lambda_sweep_experiment(args)
    elif args.mode == 'analysis':
        bias_variance_analysis(args)
    elif args.mode == 'interactive':
        interactive_monitor(args)
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("❌ Error: --model-path required for evaluation mode")
            return
        evaluate_model(args)


if __name__ == "__main__":
    # Print banner
    print("=" * 60)
    print("🎯 GAE (Generalized Advantage Estimation) Training Suite")
    print("📘 Day 4: Advanced RL Challenge")
    print("=" * 60)
    
    main() 