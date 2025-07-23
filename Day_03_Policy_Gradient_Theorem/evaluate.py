"""
Evaluation Script for REINFORCE

This script loads trained REINFORCE models and evaluates them with environment rendering.
You can watch the trained policy in action and analyze its performance.
"""

import argparse
import os
import glob
import time
import json
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from policy_network import PolicyNetwork, create_cartpole_policy
from reinforce import REINFORCE
from visualize import REINFORCEVisualizer


class PolicyEvaluator:
    """
    Comprehensive evaluation system for trained REINFORCE policies.
    
    Features:
    - Load and evaluate trained models
    - Render environment during evaluation
    - Record evaluation episodes
    - Generate performance statistics
    - Create evaluation videos/GIFs
    """
    
    def __init__(self, model_path: str, env_name: str = "CartPole-v1"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to saved model checkpoint
            env_name: Environment name
        """
        self.env_name = env_name
        self.model_path = model_path
        
        # Load the trained agent
        self.agent = self._load_agent(model_path)
        
        # Create environment
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.eval_env = gym.make(env_name, render_mode="human")  # For visual evaluation
        
        print(f"✅ Loaded model from: {model_path}")
        print(f"🎮 Environment: {env_name}")
        print(f"🧠 Policy parameters: {self.agent.policy.get_parameter_count():,}")
    
    def _load_agent(self, model_path: str) -> REINFORCE:
        """Load trained agent from checkpoint."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create policy network (assuming CartPole for now)
        if self.env_name == "CartPole-v1":
            policy = create_cartpole_policy()
        else:
            # Generic policy creation
            env_temp = gym.make(self.env_name)
            state_dim = env_temp.observation_space.shape[0]
            action_dim = env_temp.action_space.n
            policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
            env_temp.close()
        
        # Create agent
        agent = REINFORCE(
            policy=policy,
            learning_rate=1e-3,  # These don't matter for evaluation
            gamma=checkpoint.get('gamma', 0.99),
            use_baseline=checkpoint.get('use_baseline', False)
        )
        
        # Load the trained parameters
        agent.load_checkpoint(model_path)
        
        return agent
    
    def evaluate_policy(self, 
                       num_episodes: int = 10,
                       render: bool = True,
                       verbose: bool = True,
                       max_steps: int = 1000) -> Dict[str, float]:
        """
        Evaluate the policy performance.
        
        Args:
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            verbose: Whether to print episode results
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary with evaluation statistics
        """
        print(f"\n🎯 Evaluating policy for {num_episodes} episodes...")
        print("=" * 50)
        
        rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            env = self.eval_env if render else self.env
            state, _ = env.reset()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            if verbose:
                print(f"\n📺 Episode {episode + 1}/{num_episodes}")
                if render:
                    print("   (Rendering - close window when done watching)")
            
            while not done and episode_length < max_steps:
                # Get action from policy
                action, _, action_probs = self.agent.select_action(state)
                
                # Take action
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    time.sleep(0.05)  # Slow down for better viewing
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"   Reward: {episode_reward:.1f}, Length: {episode_length} steps")
                if episode_reward >= 200:  # CartPole success threshold
                    print("   ✅ Success!")
                elif episode_reward >= 100:
                    print("   🟡 Good performance")
                else:
                    print("   🔴 Poor performance")
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'median_reward': np.median(rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': np.mean([r >= 195 for r in rewards]),  # CartPole success threshold
            'episodes': num_episodes
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Range: [{stats['min_reward']:.1f}, {stats['max_reward']:.1f}]")
        print(f"Median: {stats['median_reward']:.1f}")
        print(f"Mean Length: {stats['mean_length']:.1f} steps")
        print(f"Success Rate: {stats['success_rate']:.1%} (reward ≥ 195)")
        
        # Performance assessment
        if stats['mean_reward'] >= 195:
            print("🏆 EXCELLENT: Policy has solved the environment!")
        elif stats['mean_reward'] >= 100:
            print("🟡 GOOD: Policy shows solid performance")
        else:
            print("🔴 POOR: Policy needs more training")
        
        return stats
    
    def record_episode(self, save_path: str = "episode_recording.gif") -> List[np.ndarray]:
        """
        Record a single episode and save as GIF.
        
        Args:
            save_path: Path to save the GIF
            
        Returns:
            List of frames from the episode
        """
        print(f"\n🎬 Recording episode to {save_path}...")
        
        frames = []
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 1000:
            # Render and capture frame
            frame = self.env.render()
            frames.append(frame)
            
            # Get action and step
            action, _, _ = self.agent.select_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
        
        print(f"✅ Recorded {len(frames)} frames")
        print(f"   Episode reward: {episode_reward}")
        print(f"   Episode length: {step} steps")
        
        # Save as GIF if requested
        if save_path.endswith('.gif'):
            self._save_frames_as_gif(frames, save_path)
        
        return frames
    
    def _save_frames_as_gif(self, frames: List[np.ndarray], save_path: str):
        """Save frames as animated GIF."""
        try:
            from PIL import Image
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            
            # Save as GIF
            pil_frames[0].save(
                save_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=50,  # 50ms per frame = 20 FPS
                loop=0
            )
            print(f"💾 Saved GIF: {save_path}")
            
        except ImportError:
            print("⚠️  PIL not available - cannot save GIF")
            print("   Install with: pip install Pillow")
    
    def analyze_policy_behavior(self, num_episodes: int = 5) -> Dict[str, any]:
        """
        Analyze detailed policy behavior.
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Analysis results
        """
        print(f"\n🔍 Analyzing policy behavior over {num_episodes} episodes...")
        
        all_states = []
        all_actions = []
        all_action_probs = []
        all_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            
            episode_states = []
            episode_actions = []
            episode_probs = []
            episode_rewards = []
            
            while not done:
                # Get detailed action information
                action, _, action_probs = self.agent.select_action(state)
                
                # Store data
                episode_states.append(state.copy())
                episode_actions.append(action)
                episode_probs.append(action_probs.squeeze().detach().numpy())
                
                # Step environment
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_rewards.append(reward)
            
            all_states.extend(episode_states)
            all_actions.extend(episode_actions)
            all_action_probs.extend(episode_probs)
            all_rewards.extend(episode_rewards)
        
        # Convert to numpy arrays
        states = np.array(all_states)
        actions = np.array(all_actions)
        action_probs = np.array(all_action_probs)
        
        # Analyze action distribution
        action_distribution = np.bincount(actions) / len(actions)
        
        # Analyze action probabilities
        mean_action_probs = np.mean(action_probs, axis=0)
        
        # Analyze state-action relationships (for CartPole)
        if self.env_name == "CartPole-v1":
            # CartPole states: [position, velocity, angle, angular_velocity]
            analysis = {
                'action_distribution': action_distribution,
                'mean_action_probabilities': mean_action_probs,
                'action_entropy': -np.sum(mean_action_probs * np.log(mean_action_probs + 1e-8)),
                'states': {
                    'position_range': [states[:, 0].min(), states[:, 0].max()],
                    'velocity_range': [states[:, 1].min(), states[:, 1].max()],
                    'angle_range': [states[:, 2].min(), states[:, 2].max()],
                    'angular_velocity_range': [states[:, 3].min(), states[:, 3].max()],
                },
                'action_by_angle': self._analyze_action_by_angle(states, actions)
            }
        else:
            analysis = {
                'action_distribution': action_distribution,
                'mean_action_probabilities': mean_action_probs,
                'action_entropy': -np.sum(mean_action_probs * np.log(mean_action_probs + 1e-8)),
            }
        
        # Print analysis
        print("\n📈 POLICY BEHAVIOR ANALYSIS")
        print("=" * 40)
        print(f"Action Distribution: {action_distribution}")
        print(f"Mean Action Probabilities: {mean_action_probs}")
        print(f"Policy Entropy: {analysis['action_entropy']:.3f}")
        
        if 'action_by_angle' in analysis:
            print(f"Action preference by pole angle:")
            for angle_range, action_pref in analysis['action_by_angle'].items():
                print(f"  {angle_range}: {action_pref}")
        
        return analysis
    
    def _analyze_action_by_angle(self, states: np.ndarray, actions: np.ndarray) -> Dict[str, str]:
        """Analyze how actions depend on pole angle (CartPole specific)."""
        angles = states[:, 2]  # Pole angle
        
        # Categorize angles
        left_angles = angles[angles < -0.1]
        center_angles = angles[np.abs(angles) <= 0.1]
        right_angles = angles[angles > 0.1]
        
        # Get corresponding actions
        left_actions = actions[angles < -0.1]
        center_actions = actions[np.abs(angles) <= 0.1]
        right_actions = actions[angles > 0.1]
        
        def get_action_preference(action_list):
            if len(action_list) == 0:
                return "No data"
            left_pct = np.mean(action_list == 0) * 100
            right_pct = np.mean(action_list == 1) * 100
            return f"Left: {left_pct:.1f}%, Right: {right_pct:.1f}%"
        
        return {
            "Pole left (< -0.1)": get_action_preference(left_actions),
            "Pole center (±0.1)": get_action_preference(center_actions),
            "Pole right (> 0.1)": get_action_preference(right_actions)
        }
    
    def compare_with_random_policy(self, num_episodes: int = 10) -> Dict[str, float]:
        """Compare trained policy with random policy."""
        print(f"\n⚔️  Comparing with random policy ({num_episodes} episodes each)...")
        
        # Evaluate trained policy
        trained_stats = self.evaluate_policy(num_episodes, render=False, verbose=False)
        
        # Evaluate random policy
        random_rewards = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = self.env.action_space.sample()  # Random action
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
            
            random_rewards.append(episode_reward)
        
        random_stats = {
            'mean_reward': np.mean(random_rewards),
            'std_reward': np.std(random_rewards)
        }
        
        # Compare
        improvement = trained_stats['mean_reward'] / random_stats['mean_reward']
        
        print("\n🏁 COMPARISON RESULTS")
        print("=" * 30)
        print(f"Trained Policy: {trained_stats['mean_reward']:.2f} ± {trained_stats['std_reward']:.2f}")
        print(f"Random Policy:  {random_stats['mean_reward']:.2f} ± {random_stats['std_reward']:.2f}")
        print(f"Improvement:    {improvement:.2f}x better")
        
        return {
            'trained': trained_stats,
            'random': random_stats,
            'improvement_factor': improvement
        }


def find_latest_model(results_dir: str = "results/") -> Optional[str]:
    """Find the most recent model checkpoint."""
    if not os.path.exists(results_dir):
        return None
    
    # Look for .pth files
    pattern = os.path.join(results_dir, "**", "*agent.pth")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        return None
    
    # Get the most recent one
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def main():
    """Main evaluation function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained REINFORCE policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model results/my_agent.pth --episodes 5 --render
  python evaluate.py --auto --episodes 10 --record episode.gif
  python evaluate.py --model results/my_agent.pth --analyze --compare
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically find the latest model')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                       help='Environment name')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--record', type=str, default=None,
                       help='Record episode to file (e.g., episode.gif)')
    parser.add_argument('--analyze', action='store_true',
                       help='Perform detailed policy behavior analysis')
    parser.add_argument('--compare', action='store_true',
                       help='Compare with random policy')
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directory to search for models (with --auto)')
    
    args = parser.parse_args()
    
    # Find model
    if args.auto:
        model_path = find_latest_model(args.results_dir)
        if model_path is None:
            print("❌ No model files found in results directory")
            print(f"   Searched in: {args.results_dir}")
            print("   Train a model first or specify --model path")
            return
        print(f"🔍 Auto-found model: {model_path}")
    elif args.model:
        model_path = args.model
    else:
        print("❌ Please specify --model path or use --auto to find latest model")
        return
    
    print("🎬 REINFORCE Policy Evaluation")
    print("=" * 50)
    
    try:
        # Create evaluator
        evaluator = PolicyEvaluator(model_path, args.env)
        
        # Basic evaluation
        stats = evaluator.evaluate_policy(
            num_episodes=args.episodes,
            render=args.render,
            verbose=True
        )
        
        # Record episode if requested
        if args.record:
            evaluator.record_episode(args.record)
        
        # Detailed analysis if requested
        if args.analyze:
            evaluator.analyze_policy_behavior(num_episodes=3)
        
        # Comparison if requested
        if args.compare:
            evaluator.compare_with_random_policy(num_episodes=5)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 