"""
Q-Learning Algorithm Implementation
Author: Day 2 - Model-Free Tabular Methods Challenge

Q-Learning is an off-policy temporal difference learning algorithm that learns
the optimal action-value function Q*(s,a) regardless of the policy being followed.

Update Rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
"""

import numpy as np
from typing import Tuple, Dict, Any
import gymnasium as gym
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning (Off-Policy) Agent for tabular environments.
    
    Q-Learning learns the optimal Q-function Q*(s,a) independently of 
    the policy being followed, making it an off-policy method. It uses
    the maximum Q-value over all actions for the next state.
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        q_init: str = "zeros"
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions available
            learning_rate: Learning rate α for Q-value updates
            discount_factor: Discount factor γ for future rewards
            epsilon: Initial exploration rate for ε-greedy policy
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            q_init: Q-table initialization strategy ("zeros", "random", "optimistic")
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.Q = self._initialize_q_table(q_init)
        
        # Tracking metrics
        self.training_info = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_table_history': [],
            'epsilon_history': [],
            'td_errors': []
        }
        
        # Algorithm name for comparison
        self.name = "Q-Learning"
    
    def _initialize_q_table(self, init_type: str) -> np.ndarray:
        """Initialize Q-table with different strategies."""
        if init_type == "zeros":
            return np.zeros((self.n_states, self.n_actions))
        elif init_type == "random":
            return np.random.normal(0, 0.1, (self.n_states, self.n_actions))
        elif init_type == "optimistic":
            return np.ones((self.n_states, self.n_actions)) * 1.0  # Optimistic initialization
        else:
            raise ValueError(f"Unknown initialization type: {init_type}")
    
    def epsilon_greedy_policy(self, state: int) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action according to Q-table
            return np.argmax(self.Q[state])
    
    def update_q_value(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool
    ) -> float:
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state s
            action: Current action a
            reward: Received reward r
            next_state: Next state s'
            done: Whether episode is finished
            
        Returns:
            TD error for monitoring
        """
        # Current Q-value
        current_q = self.Q[state, action]
        
        # Next Q-value (using maximum over all actions - OFF-POLICY)
        if done:
            next_q = 0.0  # Terminal state
        else:
            next_q = np.max(self.Q[next_state])  # max_a' Q(s',a')
        
        # Q-Learning target: r + γ max_a' Q(s',a')
        target = reward + self.gamma * next_q
        
        # TD error
        td_error = target - current_q
        
        # Update Q-value
        self.Q[state, action] += self.alpha * td_error
        
        return abs(td_error)
    
    def train_episode(self, env: gym.Env, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Train agent for one episode using Q-Learning algorithm.
        
        Args:
            env: OpenAI Gym environment
            max_steps: Maximum steps per episode
            
        Returns:
            Episode information dictionary
        """
        state, _ = env.reset()
        
        episode_reward = 0
        episode_length = 0
        td_errors = []
        
        for step in range(max_steps):
            # Select action using ε-greedy policy
            action = self.epsilon_greedy_policy(state)
            
            # Take action and observe result
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update Q-value using Q-Learning rule (off-policy)
            td_error = self.update_q_value(state, action, reward, next_state, done)
            td_errors.append(td_error)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store episode information
        episode_info = {
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': self.epsilon,
            'avg_td_error': np.mean(td_errors) if td_errors else 0,
            'max_td_error': np.max(td_errors) if td_errors else 0
        }
        
        return episode_info
    
    def train(
        self, 
        env: gym.Env, 
        n_episodes: int, 
        max_steps_per_episode: int = 1000,
        verbose: bool = True,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Train Q-Learning agent for multiple episodes.
        
        Args:
            env: OpenAI Gym environment
            n_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Print progress every N episodes
            
        Returns:
            Training history and metrics
        """
        if verbose:
            print(f"🎓 Training Q-Learning Agent for {n_episodes} episodes...")
            print(f"📊 Environment: {env.spec.id}")
            print(f"⚙️ Parameters: α={self.alpha}, γ={self.gamma}, ε₀={self.epsilon}")
            print("-" * 60)
        
        for episode in range(n_episodes):
            # Train one episode
            episode_info = self.train_episode(env, max_steps_per_episode)
            
            # Store metrics
            self.training_info['episode_rewards'].append(episode_info['reward'])
            self.training_info['episode_lengths'].append(episode_info['length'])
            self.training_info['epsilon_history'].append(episode_info['epsilon'])
            self.training_info['td_errors'].append(episode_info['avg_td_error'])
            
            # Periodically store Q-table for analysis
            if episode % 50 == 0:
                self.training_info['q_table_history'].append(self.Q.copy())
            
            # Print progress
            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.training_info['episode_rewards'][-log_interval:])
                avg_length = np.mean(self.training_info['episode_lengths'][-log_interval:])
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"ε: {episode_info['epsilon']:.3f} | "
                      f"TD Error: {episode_info['avg_td_error']:.4f}")
        
        if verbose:
            final_reward = np.mean(self.training_info['episode_rewards'][-100:])
            print(f"\n✅ Training completed!")
            print(f"📈 Final average reward (last 100 episodes): {final_reward:.2f}")
        
        return self.training_info
    
    def get_policy(self) -> np.ndarray:
        """
        Extract deterministic policy from Q-table.
        
        Returns:
            Policy array where policy[s] = optimal action for state s
        """
        return np.argmax(self.Q, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """
        Extract state values from Q-table using V(s) = max_a Q(s,a).
        
        Returns:
            State value array
        """
        return np.max(self.Q, axis=1)
    
    def evaluate_policy(
        self, 
        env: gym.Env, 
        n_episodes: int = 100, 
        max_steps: int = 1000,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the learned policy without exploration.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Use greedy policy (no exploration)
                action = np.argmax(self.Q[state])
                state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if render and episode < 5:  # Render first few episodes
                    env.render()
                
                if terminated or truncated:
                    if terminated and reward > 0:  # Successful completion
                        success_rate += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_rate / n_episodes,
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def save_q_table(self, filepath: str):
        """Save Q-table to file."""
        np.save(filepath, self.Q)
        print(f"💾 Q-table saved to {filepath}")
    
    def load_q_table(self, filepath: str):
        """Load Q-table from file."""
        self.Q = np.load(filepath)
        print(f"📂 Q-table loaded from {filepath}")
    
    def print_policy(self, env_name: str = "", action_symbols: Dict[int, str] = None):
        """
        Print policy in a readable format.
        
        Args:
            env_name: Name of environment for context
            action_symbols: Mapping from action indices to symbols
        """
        if action_symbols is None:
            action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        
        policy = self.get_policy()
        
        print(f"\n🎯 Q-Learning Learned Policy ({env_name})")
        print("=" * 40)
        
        # For grid environments, try to reshape policy
        if env_name.lower() in ['cliffwalking', 'frozenlake']:
            if self.n_states == 48:  # CliffWalking 4x12
                grid_shape = (4, 12)
            elif self.n_states == 16:  # FrozenLake 4x4
                grid_shape = (4, 4)
            elif self.n_states == 64:  # FrozenLake 8x8
                grid_shape = (8, 8)
            else:
                grid_shape = None
            
            if grid_shape:
                policy_grid = policy.reshape(grid_shape)
                for row in policy_grid:
                    print(" ".join([action_symbols.get(action, str(action)) for action in row]))
            else:
                print("Policy array:", policy)
        else:
            print("Policy array:", policy)
        
        print("=" * 40)
    
    def double_q_learning_update(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        done: bool,
        Q2: np.ndarray
    ) -> float:
        """
        Update using Double Q-Learning to reduce overestimation bias.
        
        Alternates between using Q1 and Q2 for action selection and value estimation.
        This is an optional extension to standard Q-Learning.
        
        Args:
            state: Current state
            action: Current action
            reward: Received reward
            next_state: Next state
            done: Whether episode is finished
            Q2: Second Q-table for double Q-learning
            
        Returns:
            TD error
        """
        # Current Q-value from primary table
        current_q = self.Q[state, action]
        
        if done:
            next_q = 0.0
        else:
            # Select action using Q1, evaluate using Q2
            best_action = np.argmax(self.Q[next_state])
            next_q = Q2[next_state, best_action]
        
        # Target using Q2's value
        target = reward + self.gamma * next_q
        
        # TD error
        td_error = target - current_q
        
        # Update Q1
        self.Q[state, action] += self.alpha * td_error
        
        return abs(td_error)
    
    def get_q_value_statistics(self) -> Dict[str, float]:
        """
        Get statistics about Q-values for analysis.
        
        Returns:
            Dictionary with Q-value statistics
        """
        q_values = self.Q.flatten()
        
        return {
            'mean_q': np.mean(q_values),
            'std_q': np.std(q_values),
            'min_q': np.min(q_values),
            'max_q': np.max(q_values),
            'median_q': np.median(q_values),
            'q_range': np.max(q_values) - np.min(q_values)
        } 