"""
Main Training Script for PPO and TRPO
=====================================
Command-line interface for training and evaluating PPO and TRPO agents.
"""

import argparse
import os
import torch
from datetime import datetime

from ppo import PPO, PPOConfig, PPOComparison
from trpo import TRPO, TRPOConfig
from visualize import PPOVisualizer


def train_basic(args):
    """Train basic PPO agent."""
    print("🚀 Training Basic PPO Agent")
    
    config = PPOConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        entropy_coef=args.entropy_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed
    )
    
    agent = PPO(config)
    metrics = agent.train()
    
    # Save model
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/ppo_{args.env}_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model(save_path)
        print(f"💾 Model saved: {save_path}")
    
    # Create visualizations
    if args.plot:
        visualizer = PPOVisualizer("plots")
        visualizer.save_all_plots(agent)
    
    # Final evaluation
    print("\n🎯 Final Evaluation:")
    eval_stats = agent.evaluate(num_episodes=20)
    for key, value in eval_stats.items():
        print(f"  {key}: {value:.3f}")
    
    return agent, metrics


def train_trpo(args):
    """Train TRPO agent."""
    print("🚀 Training TRPO Agent")
    
    config = TRPOConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        max_kl=args.max_kl,
        value_lr=args.value_lr,
        n_steps=args.n_steps,
        seed=args.seed
    )
    
    agent = TRPO(config)
    metrics = agent.train()
    
    # Save model
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"models/trpo_{args.env}_{timestamp}.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model(save_path)
        print(f"💾 Model saved: {save_path}")
    
    # Create visualizations
    if args.plot:
        visualizer = PPOVisualizer("plots")
        visualizer.save_all_plots(agent)
    
    # Final evaluation
    print("\n🎯 Final Evaluation:")
    eval_stats = agent.evaluate(num_episodes=20)
    for key, value in eval_stats.items():
        print(f"  {key}: {value:.3f}")
    
    return agent, metrics


def train_comparison(args):
    """Run PPO comparison experiments."""
    print("🔬 Running PPO Comparison Experiments")
    
    base_config = PPOConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        seed=args.seed
    )
    
    comparison = PPOComparison(base_config)
    
    if args.compare_type == "clipping":
        comparison.compare_clipping(timesteps=args.timesteps)
    elif args.compare_type == "entropy":
        comparison.compare_entropy(timesteps=args.timesteps)
    else:
        # Both
        comparison.compare_clipping(timesteps=args.timesteps//2)
        comparison.compare_entropy(timesteps=args.timesteps//2)
    
    # Visualize results
    if args.plot:
        visualizer = PPOVisualizer("plots")
        visualizer.plot_comparison(comparison.results, metric='mean_reward')
        visualizer.plot_learning_curves_comparison(comparison.results)
    
    return comparison


def train_trpo_vs_ppo(args):
    """Compare TRPO vs PPO performance."""
    print("🔬 Running TRPO vs PPO Comparison")
    
    results = {}
    
    # Train TRPO
    print("\n" + "="*50)
    print("🎯 Training TRPO")
    print("="*50)
    
    trpo_config = TRPOConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        seed=args.seed
    )
    
    trpo_agent = TRPO(trpo_config)
    trpo_metrics = trpo_agent.train()
    trpo_eval = trpo_agent.evaluate(num_episodes=20)
    
    results['TRPO'] = {
        'agent': trpo_agent,
        'metrics': trpo_metrics,
        'final_eval': trpo_eval
    }
    
    # Train PPO
    print("\n" + "="*50)
    print("🎯 Training PPO")
    print("="*50)
    
    ppo_config = PPOConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        seed=args.seed
    )
    
    ppo_agent = PPO(ppo_config)
    ppo_metrics = ppo_agent.train()
    ppo_eval = ppo_agent.evaluate(num_episodes=20)
    
    results['PPO'] = {
        'agent': ppo_agent,
        'metrics': ppo_metrics,
        'final_eval': ppo_eval
    }
    
    # Save models
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("models", exist_ok=True)
        
        trpo_path = f"models/trpo_{args.env}_{timestamp}.pth"
        ppo_path = f"models/ppo_{args.env}_{timestamp}.pth"
        
        trpo_agent.save_model(trpo_path)
        ppo_agent.save_model(ppo_path)
        
        print(f"💾 TRPO model saved: {trpo_path}")
        print(f"💾 PPO model saved: {ppo_path}")
    
    # Create comparison visualizations
    if args.plot:
        visualizer = PPOVisualizer("plots")
        
        # Performance comparison
        visualizer.plot_comparison(results, metric='mean_reward', save_name="trpo_vs_ppo_comparison")
        
        # Learning curves comparison
        metrics_to_compare = ['policy_loss', 'value_loss', 'kl_divergence']
        visualizer.plot_learning_curves_comparison(results, save_name="trpo_vs_ppo_learning")
        
        # Create summary analysis
        create_trpo_ppo_analysis(results, visualizer)
    
    # Print comparison summary
    print("\n" + "="*60)
    print("📊 TRPO vs PPO COMPARISON SUMMARY")
    print("="*60)
    
    trpo_reward = trpo_eval['mean_reward']
    ppo_reward = ppo_eval['mean_reward']
    
    print(f"🎯 Final Performance:")
    print(f"   TRPO: {trpo_reward:.2f} ± {trpo_eval['std_reward']:.2f}")
    print(f"   PPO:  {ppo_reward:.2f} ± {ppo_eval['std_reward']:.2f}")
    
    winner = "TRPO" if trpo_reward > ppo_reward else "PPO"
    diff = abs(trpo_reward - ppo_reward)
    print(f"   Winner: {winner} (+{diff:.2f})")
    
    print(f"\n📈 Training Efficiency:")
    print(f"   TRPO Updates: {trpo_agent.num_updates}")
    print(f"   PPO Updates:  {ppo_agent.num_updates}")
    
    print(f"\n🔍 Key Insights:")
    print(f"   • TRPO uses natural gradients with KL constraint")
    print(f"   • PPO uses clipped surrogate objective (simpler)")
    print(f"   • TRPO: Theoretical guarantees, higher complexity")
    print(f"   • PPO: Practical performance, easier to implement")
    
    return results


def create_trpo_ppo_analysis(results, visualizer):
    """Create detailed TRPO vs PPO analysis visualization."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TRPO vs PPO: Detailed Analysis', fontsize=16, fontweight='bold')
    
    trpo_metrics = results['TRPO']['metrics']
    ppo_metrics = results['PPO']['metrics']
    
    # 1. Policy Loss Comparison
    ax = axes[0, 0]
    if 'policy_loss' in trpo_metrics:
        ax.plot(trpo_metrics['policy_loss'], label='TRPO', color='blue', linewidth=2)
    if 'policy_loss' in ppo_metrics:
        ax.plot(ppo_metrics['policy_loss'], label='PPO', color='orange', linewidth=2)
    ax.set_title('Policy Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Value Loss Comparison
    ax = axes[0, 1]
    if 'value_loss' in trpo_metrics:
        ax.plot(trpo_metrics['value_loss'], label='TRPO', color='blue', linewidth=2)
    if 'value_loss' in ppo_metrics:
        ax.plot(ppo_metrics['value_loss'], label='PPO', color='orange', linewidth=2)
    ax.set_title('Value Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. KL Divergence Comparison
    ax = axes[1, 0]
    if 'kl_divergence' in trpo_metrics:
        ax.plot(trpo_metrics['kl_divergence'], label='TRPO', color='blue', linewidth=2)
    if 'kl_divergence' in ppo_metrics:
        ax.plot(ppo_metrics['kl_divergence'], label='PPO', color='orange', linewidth=2)
    ax.set_title('KL Divergence')
    ax.set_xlabel('Update')
    ax.set_ylabel('KL Divergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Algorithm Complexity Comparison
    ax = axes[1, 1]
    algorithms = ['TRPO', 'PPO']
    complexity_scores = [8, 3]  # Relative complexity (1-10 scale)
    features = ['Natural\nGradients', 'Conjugate\nGradient', 'Line\nSearch', 'KL\nConstraint', 'Simple\nObjective']
    trpo_features = [1, 1, 1, 1, 0]
    ppo_features = [0, 0, 0, 0, 1]
    
    x = np.arange(len(features))
    width = 0.35
    
    ax.bar(x - width/2, trpo_features, width, label='TRPO', color='blue', alpha=0.7)
    ax.bar(x + width/2, ppo_features, width, label='PPO', color='orange', alpha=0.7)
    
    ax.set_title('Algorithm Features')
    ax.set_xlabel('Features')
    ax.set_ylabel('Uses Feature')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(visualizer.save_dir, "trpo_vs_ppo_analysis.png")
    fig.savefig(save_path, dpi=visualizer.dpi, bbox_inches='tight')
    print(f"📊 TRPO vs PPO analysis saved: {save_path}")
    
    plt.close(fig)


def evaluate_model(args):
    """Evaluate a trained model."""
    print("🎯 Evaluating Trained Model")
    
    # Load model
    checkpoint = torch.load(args.model_path, weights_only=False)
    config = checkpoint['config']
    
    # Determine if it's PPO or TRPO model
    if 'ppo' in args.model_path.lower() or isinstance(config, PPOConfig):
        agent = PPO(config)
    else:
        agent = TRPO(config)
    
    agent.load_model(args.model_path)
    
    print(f"Loaded model: {args.model_path}")
    print(f"Environment: {config.env_name}")
    
    # Evaluate
    eval_stats = agent.evaluate(num_episodes=args.eval_episodes, render=args.render)
    
    print(f"\n📊 Evaluation Results ({args.eval_episodes} episodes):")
    for key, value in eval_stats.items():
        print(f"  {key}: {value:.3f}")
    
    return eval_stats


def main():
    parser = argparse.ArgumentParser(description='PPO and TRPO Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'trpo', 'compare', 'trpo-vs-ppo', 'evaluate'],
                       help='Mode: train (PPO), trpo, compare, trpo-vs-ppo, or evaluate')
    
    # Environment settings
    parser.add_argument('--env', type=str, default='Pendulum-v1',
                       help='Environment name')
    
    # Training settings
    parser.add_argument('--timesteps', type=int, default=200000,
                       help='Total training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (PPO only)')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy coefficient (PPO only)')
    parser.add_argument('--max-kl', type=float, default=0.01,
                       help='Max KL divergence (TRPO only)')
    parser.add_argument('--value-lr', type=float, default=3e-4,
                       help='Value function learning rate (TRPO only)')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (PPO only)')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Epochs per update (PPO only)')
    
    # Comparison settings
    parser.add_argument('--compare-type', type=str, default='both',
                       choices=['clipping', 'entropy', 'both'],
                       help='Type of comparison experiment')
    
    # Evaluation settings
    parser.add_argument('--model-path', type=str, 
                       help='Path to model for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation')
    
    # General settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save', action='store_true',
                       help='Save trained model')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    if args.mode == 'trpo':
        print("🎯 TRPO (Trust Region Policy Optimization) Training Suite")
    elif args.mode == 'trpo-vs-ppo':
        print("🔬 TRPO vs PPO Comparison Suite")
    else:
        print("🎯 PPO (Proximal Policy Optimization) Training Suite")
    print("📘 Day 5: Trust Region & PPO Challenge")
    print("=" * 60)
    
    if args.mode == 'train':
        agent, metrics = train_basic(args)
    elif args.mode == 'trpo':
        agent, metrics = train_trpo(args)
    elif args.mode == 'compare':
        comparison = train_comparison(args)
    elif args.mode == 'trpo-vs-ppo':
        results = train_trpo_vs_ppo(args)
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("❌ Error: --model-path required for evaluation mode")
            return
        eval_stats = evaluate_model(args)
    
    print("\n✅ Execution completed!")


if __name__ == "__main__":
    main() 