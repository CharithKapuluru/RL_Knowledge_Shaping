"""
Q-Learning Agent Implementation
This is the baseline reinforcement learning agent using Q-learning algorithm.
"""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """
    Standard Q-Learning agent for discrete state and action spaces.
    """

    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.

        Args:
            action_space: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table implemented as dictionary for sparse state spaces
        self.q_table = defaultdict(lambda: np.zeros(action_space))

    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy) or evaluation mode (greedy)

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_space)
        else:
            # Exploitation: best known action
            state_key = self._state_to_key(state)
            return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_key])

        # Update Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_to_key(self, state):
        """Convert state to hashable key for Q-table."""
        if isinstance(state, (int, str)):
            return state
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, tuple):
            return state
        else:
            return str(state)

    def get_q_values(self, state):
        """Get Q-values for a given state."""
        state_key = self._state_to_key(state)
        return self.q_table[state_key].copy()

    def set_q_values(self, state, q_values):
        """Set Q-values for a given state."""
        state_key = self._state_to_key(state)
        self.q_table[state_key] = q_values.copy()
