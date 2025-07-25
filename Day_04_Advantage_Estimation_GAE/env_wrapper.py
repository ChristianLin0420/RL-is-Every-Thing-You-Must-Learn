"""
Environment Wrapper for GAE Training

Provides standardized interface for gymnasium environments with:
- Observation normalization (running mean/std)
- Episode tracking and statistics
- Reward clipping (optional)
- Frame stacking (for Atari-like envs)
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any
from collections import deque


class RunningMeanStd:
    """Tracks running mean and standard deviation of observations."""
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update from batch moments using Welford's algorithm."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation(gym.ObservationWrapper):
    """Normalizes observations to zero mean and unit variance."""
    
    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        super().__init__(env)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        self.obs_rms.update(np.array([obs]))
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class EpisodeTracker(gym.Wrapper):
    """Tracks episode statistics and provides additional info."""
    
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.reset_stats()
        
    def reset_stats(self) -> None:
        """Reset episode tracking statistics."""
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.reward_history = deque(maxlen=100)  # Last 100 episodes
        self.length_history = deque(maxlen=100)
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and update episode statistics."""
        if self.episode_length > 0:  # Not the first episode
            self.reward_history.append(self.episode_reward)
            self.length_history.append(self.episode_length)
            self.episode_count += 1
            
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment and track statistics."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add episode statistics to info
        info.update({
            'episode_reward': self.episode_reward,
            'episode_length': self.episode_length,
            'episode_count': self.episode_count,
        })
        
        # Add historical statistics if available
        if len(self.reward_history) > 0:
            info.update({
                'mean_episode_reward': np.mean(self.reward_history),
                'std_episode_reward': np.std(self.reward_history),
                'mean_episode_length': np.mean(self.length_history),
            })
            
        # Check for manual truncation based on max_episode_steps
        if self.max_episode_steps and self.episode_length >= self.max_episode_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info
        
    def get_episode_stats(self) -> Dict[str, float]:
        """Get current episode statistics."""
        if len(self.reward_history) == 0:
            return {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_length': 0.0,
                'episodes_completed': 0
            }
            
        return {
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'mean_length': np.mean(self.length_history),
            'episodes_completed': len(self.reward_history)
        }


class RewardClipping(gym.RewardWrapper):
    """Clips rewards to a specified range."""
    
    def __init__(self, env: gym.Env, min_reward: float = -10.0, max_reward: float = 10.0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        
    def reward(self, reward: float) -> float:
        """Clip reward to specified range."""
        return np.clip(reward, self.min_reward, self.max_reward)


def make_env(env_name: str, 
             normalize_obs: bool = True,
             clip_rewards: bool = False,
             max_episode_steps: Optional[int] = None,
             seed: Optional[int] = None,
             render_mode: Optional[str] = None) -> gym.Env:
    """
    Create a standardized environment with optional wrappers.
    
    Args:
        env_name: Name of the gymnasium environment
        normalize_obs: Whether to normalize observations
        clip_rewards: Whether to clip rewards to [-10, 10]
        max_episode_steps: Maximum steps per episode (None = env default)
        seed: Random seed for reproducibility
        render_mode: Render mode ('human', 'rgb_array', None)
        
    Returns:
        Wrapped gymnasium environment
    """
    # Create base environment
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    
    # Set seed if provided
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
    # Add episode tracking (should be innermost wrapper for state tracking)
    env = EpisodeTracker(env, max_episode_steps=max_episode_steps)
    
    # Add reward clipping if requested
    if clip_rewards:
        env = RewardClipping(env)
        
    # Add observation normalization (should be outermost for consistency)
    if normalize_obs:
        env = NormalizeObservation(env)
        
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get comprehensive environment information."""
    obs_space = env.observation_space
    action_space = env.action_space
    
    info = {
        'env_name': getattr(env.unwrapped, 'spec', {}).id if hasattr(env.unwrapped, 'spec') else 'Unknown',
        'obs_shape': obs_space.shape,
        'obs_dim': np.prod(obs_space.shape),
        'action_space_type': type(action_space).__name__,
    }
    
    # Action space specific info
    if isinstance(action_space, gym.spaces.Discrete):
        info.update({
            'action_dim': action_space.n,
            'action_type': 'discrete'
        })
    elif isinstance(action_space, gym.spaces.Box):
        info.update({
            'action_dim': action_space.shape[0] if len(action_space.shape) > 0 else 1,
            'action_type': 'continuous',
            'action_low': action_space.low,
            'action_high': action_space.high
        })
    else:
        info.update({
            'action_dim': None,
            'action_type': 'unknown'
        })
        
    return info


# Convenience functions for common environments
def make_cartpole(normalize_obs: bool = True, seed: Optional[int] = None) -> gym.Env:
    """Create CartPole-v1 environment with standard wrappers."""
    return make_env('CartPole-v1', normalize_obs=normalize_obs, seed=seed)


def make_lunarlander(normalize_obs: bool = True, seed: Optional[int] = None) -> gym.Env:
    """Create LunarLander-v2 environment with standard wrappers."""
    return make_env('LunarLander-v2', normalize_obs=normalize_obs, seed=seed)


def make_pendulum(normalize_obs: bool = True, seed: Optional[int] = None) -> gym.Env:
    """Create Pendulum-v1 environment with standard wrappers."""
    return make_env('Pendulum-v1', normalize_obs=normalize_obs, clip_rewards=True, seed=seed)


if __name__ == "__main__":
    # Test the wrapper functionality
    print("🧪 Testing Environment Wrappers...")
    
    # Test CartPole
    env = make_cartpole(seed=42)
    info = get_env_info(env)
    print(f"\n📊 CartPole Info: {info}")
    
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}, type: {type(obs)}")
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: reward={reward:.3f}, total={total_reward:.3f}, terminated={terminated}")
        
        if terminated or truncated:
            obs, _ = env.reset()
            total_reward = 0
            
    env.close()
    print("\n✅ Environment wrapper tests completed!") 