"""
CartPole Environment Wrapper
Discretizes continuous CartPole state space for Q-learning.
"""

import gymnasium as gym
import numpy as np


class DiscreteCartPoleWrapper:
    """
    Wrapper for CartPole that discretizes the continuous state space.
    """

    def __init__(self, bins=(6, 6, 6, 6)):
        """
        Initialize CartPole wrapper.

        Args:
            bins: Number of bins for each state dimension
                  (cart_position, cart_velocity, pole_angle, pole_velocity)
        """
        self.env = gym.make('CartPole-v1')
        self.bins = bins
        self.action_space = 2  # left or right

        # Define bounds for discretization
        self.bounds = [
            (-4.8, 4.8),      # cart position
            (-3.0, 3.0),      # cart velocity
            (-0.418, 0.418),  # pole angle (~24 degrees)
            (-2.0, 2.0)       # pole velocity
        ]

    def reset(self):
        """Reset environment and return discretized state."""
        state, _ = self.env.reset()
        return self._discretize_state(state)

    def step(self, action):
        """
        Take action and return discretized next state.

        Args:
            action: 0 (left) or 1 (right)

        Returns:
            next_state: Discretized state
            reward: Reward (modified for better learning)
            done: Whether episode is done
            info: Additional info
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Discretize state
        discrete_state = self._discretize_state(next_state)

        # Modify reward for better learning
        # Penalize early termination
        if done and reward < 500:
            reward = -10.0
        else:
            reward = 1.0

        return discrete_state, reward, done, info

    def _discretize_state(self, state):
        """
        Convert continuous state to discrete state.

        Args:
            state: Continuous state [cart_pos, cart_vel, pole_angle, pole_vel]

        Returns:
            Discretized state tuple
        """
        discrete = []
        for i, val in enumerate(state):
            # Clip to bounds
            low, high = self.bounds[i]
            val = np.clip(val, low, high)

            # Discretize
            bins = self.bins[i]
            scaled = (val - low) / (high - low)  # Scale to [0, 1]
            bin_idx = int(scaled * (bins - 1))
            bin_idx = min(bins - 1, max(0, bin_idx))  # Ensure within bounds
            discrete.append(bin_idx)

        return tuple(discrete)

    def close(self):
        """Close environment."""
        self.env.close()


class CartPoleVariant:
    """
    Create different CartPole variants for knowledge transfer experiments.
    """

    @staticmethod
    def create_standard():
        """Create standard CartPole environment."""
        return DiscreteCartPoleWrapper(bins=(6, 6, 6, 6))

    @staticmethod
    def create_fine_grained():
        """Create CartPole with finer discretization."""
        return DiscreteCartPoleWrapper(bins=(8, 8, 8, 8))

    @staticmethod
    def create_coarse_grained():
        """Create CartPole with coarser discretization."""
        return DiscreteCartPoleWrapper(bins=(4, 4, 4, 4))
