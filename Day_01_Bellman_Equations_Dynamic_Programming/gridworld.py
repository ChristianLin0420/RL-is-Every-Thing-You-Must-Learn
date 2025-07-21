"""
GridWorld Environment for Value Iteration Implementation
Author: Day 1 - Bellman Equations & Dynamic Programming Challenge
"""

import numpy as np
from typing import Tuple, List, Dict


class GridWorld:
    """
    5x5 GridWorld environment for testing Value Iteration.
    
    Environment Details:
    - Start state: (0, 0)
    - Goal state: (4, 4) with reward +1
    - Other states: reward 0
    - Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3)
    - Deterministic transitions
    """
    
    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.gamma = gamma
        self.n_states = size * size
        self.n_actions = 4
        
        # Action mappings
        self.actions = {
            0: "UP",
            1: "DOWN", 
            2: "LEFT",
            3: "RIGHT"
        }
        
        # Action vectors (row_delta, col_delta)
        self.action_vectors = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1)    # RIGHT
        }
        
        # Define special states
        self.start_state = (0, 0)
        self.goal_state = (4, 4)
        
        # Initialize transition probabilities and rewards
        self._build_transitions()
        self._build_rewards()
    
    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to linear index."""
        return state[0] * self.size + state[1]
    
    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) state."""
        return (index // self.size, index % self.size)
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if state is within grid boundaries."""
        row, col = state
        return 0 <= row < self.size and 0 <= col < self.size
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get next state given current state and action."""
        row, col = state
        d_row, d_col = self.action_vectors[action]
        next_state = (row + d_row, col + d_col)
        
        # If next state is invalid, stay in current state
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def _build_transitions(self):
        """Build transition probability matrix P[s,a,s'] = P(s'|s,a)."""
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        for s_idx in range(self.n_states):
            state = self.index_to_state(s_idx)
            
            for action in range(self.n_actions):
                next_state = self.get_next_state(state, action)
                next_s_idx = self.state_to_index(next_state)
                
                # Deterministic transition
                self.P[s_idx, action, next_s_idx] = 1.0
    
    def _build_rewards(self):
        """Build reward matrix R[s,a,s'] = R(s,a,s')."""
        self.R = np.zeros((self.n_states, self.n_actions, self.n_states))
        
        goal_idx = self.state_to_index(self.goal_state)
        
        for s_idx in range(self.n_states):
            for action in range(self.n_actions):
                for next_s_idx in range(self.n_states):
                    if self.P[s_idx, action, next_s_idx] > 0:
                        # Reward +1 for reaching goal state
                        if next_s_idx == goal_idx:
                            self.R[s_idx, action, next_s_idx] = 1.0
                        else:
                            self.R[s_idx, action, next_s_idx] = 0.0
    
    def get_transition_prob(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        """Get transition probability P(s'|s,a)."""
        s_idx = self.state_to_index(state)
        next_s_idx = self.state_to_index(next_state)
        return self.P[s_idx, action, next_s_idx]
    
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        """Get reward R(s,a,s')."""
        s_idx = self.state_to_index(state)
        next_s_idx = self.state_to_index(next_state)
        return self.R[s_idx, action, next_s_idx]
    
    def get_all_states(self) -> List[Tuple[int, int]]:
        """Get all possible states in the environment."""
        return [self.index_to_state(i) for i in range(self.n_states)]
    
    def get_all_actions(self) -> List[int]:
        """Get all possible actions."""
        return list(range(self.n_actions))
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal (goal state)."""
        return state == self.goal_state
    
    def render_grid(self, values: np.ndarray = None, policy: np.ndarray = None) -> str:
        """Render the grid with optional values and policy."""
        grid_str = "\nGridWorld Layout:\n"
        grid_str += "S = Start, G = Goal\n\n"
        
        for row in range(self.size):
            for col in range(self.size):
                state = (row, col)
                if state == self.start_state:
                    grid_str += "S "
                elif state == self.goal_state:
                    grid_str += "G "
                else:
                    grid_str += ". "
            grid_str += "\n"
        
        return grid_str 