"""
Environment Wrapper for PPO
============================
Provides standardized environment utilities for PPO training on continuous control tasks.
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Union, Tuple
from collections import deque


class RunningMeanStd:
    """Running statistics for observation normalization."""
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations using running mean and std."""
    
    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.rms = RunningMeanStd(shape=env.observation_space.shape)
        self.epsilon = epsilon
    
    def observation(self, obs):
        self.rms.update(np.array([obs]))
        return (obs - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)


class NormalizeReward(gym.RewardWrapper):
    """Normalize rewards using running mean and std."""
    
    def __init__(self, env, epsilon: float = 1e-8, gamma: float = 0.99):
        super().__init__(env)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = None
        self.epsilon = epsilon
        self.gamma = gamma
    
    def reward(self, reward):
        if self.returns is None:
            self.returns = reward
        else:
            self.returns = self.returns * self.gamma + reward
        
        self.return_rms.update(np.array([self.returns]))
        return reward / np.sqrt(self.return_rms.var + self.epsilon)
    
    def reset(self, **kwargs):
        self.returns = None
        return self.env.reset(**kwargs)


class EpisodeTracker(gym.Wrapper):
    """Track episode statistics."""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        
        if terminated or truncated:
            self.episode_rewards.append(self.episode_reward)
            self.episode_lengths.append(self.episode_length)
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        return self.env.reset(**kwargs)


class ActionClipping(gym.ActionWrapper):
    """Clip actions to valid range for continuous control."""
    
    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


def make_env(env_name: str, 
             normalize_obs: bool = True,
             normalize_reward: bool = False,
             clip_actions: bool = True,
             max_episode_steps: Optional[int] = None,
             seed: Optional[int] = None,
             render_mode: Optional[str] = None) -> gym.Env:
    """
    Create a standardized environment for PPO training.
    
    Args:
        env_name: Name of the gymnasium environment
        normalize_obs: Whether to normalize observations
        normalize_reward: Whether to normalize rewards
        clip_actions: Whether to clip actions to valid range
        max_episode_steps: Maximum steps per episode
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
    
    # Set time limit
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    
    # Add episode tracking
    env = EpisodeTracker(env)
    
    # Normalize observations
    if normalize_obs:
        env = NormalizeObservation(env)
    
    # Normalize rewards
    if normalize_reward:
        env = NormalizeReward(env)
    
    # Clip actions for continuous control
    if clip_actions and isinstance(env.action_space, gym.spaces.Box):
        env = ActionClipping(env)
    
    # Set seed
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def get_env_info(env: gym.Env) -> dict:
    """
    Get environment information for network initialization.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Dictionary with environment info
    """
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Determine observation dimension
    if isinstance(obs_space, gym.spaces.Box):
        obs_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else np.prod(obs_space.shape)
    else:
        obs_dim = obs_space.n
    
    # Determine action space properties
    if isinstance(action_space, gym.spaces.Box):
        action_type = 'continuous'
        action_dim = action_space.shape[0] if len(action_space.shape) == 1 else np.prod(action_space.shape)
        action_low = action_space.low
        action_high = action_space.high
    elif isinstance(action_space, gym.spaces.Discrete):
        action_type = 'discrete'
        action_dim = action_space.n
        action_low = None
        action_high = None
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported")
    
    return {
        'obs_dim': int(obs_dim),
        'action_dim': int(action_dim),
        'action_type': action_type,
        'action_low': action_low,
        'action_high': action_high,
        'obs_space': obs_space,
        'action_space': action_space
    }


class VecEnv:
    """Simple vectorized environment wrapper (for future extensions)."""
    
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # Get environment info from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])
    
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        obs, rewards, terminated, truncated, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(terminated), np.array(truncated), list(infos)
    
    def close(self):
        for env in self.envs:
            env.close()
    
    def render(self, mode='human'):
        return self.envs[0].render()


def make_vec_env(env_name: str, num_envs: int = 1, **kwargs):
    """Create vectorized environment."""
    env_fns = [lambda: make_env(env_name, **kwargs) for _ in range(num_envs)]
    return VecEnv(env_fns) 