"""
Reinforcement Learning with Knowledge Shaping (RL-KS) Agent
Extends Q-learning with knowledge transfer from source tasks.
"""

import numpy as np
from collections import defaultdict
from q_learning_agent import QLearningAgent


class RLKSAgent(QLearningAgent):
    """
    RL-KS Agent that incorporates prior knowledge from a source task
    into the learning process for a target task.
    """

    def __init__(self, action_space, source_agent=None, shaping_weight=0.5,
                 learning_rate=0.1, discount_factor=0.99, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize RL-KS agent.

        Args:
            action_space: Number of possible actions
            source_agent: Trained agent from source task (provides prior knowledge)
            shaping_weight: Weight for knowledge shaping (0-1), how much to rely on source
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        super().__init__(action_space, learning_rate, discount_factor,
                         epsilon, epsilon_decay, epsilon_min)

        self.source_agent = source_agent
        self.shaping_weight = shaping_weight
        self.use_knowledge_shaping = source_agent is not None

    def get_shaped_q_values(self, state):
        """
        Get Q-values shaped by prior knowledge from source task.

        Args:
            state: Current state

        Returns:
            Shaped Q-values combining learned values and prior knowledge
        """
        state_key = self._state_to_key(state)

        # Get current Q-values
        current_q = self.q_table[state_key].copy()

        if self.use_knowledge_shaping and self.source_agent is not None:
            # Get Q-values from source agent (prior knowledge)
            source_q = self.source_agent.get_q_values(state)

            # Combine using shaping weight: Q_shaped = (1-λ)*Q_current + λ*Q_source
            shaped_q = (1 - self.shaping_weight) * current_q + self.shaping_weight * source_q
            return shaped_q
        else:
            return current_q

    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy with knowledge shaping.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.action_space)
        else:
            # Exploitation: best action based on shaped Q-values
            shaped_q = self.get_shaped_q_values(state)
            return np.argmax(shaped_q)

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning with knowledge shaping.

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

        # Get next state value using shaped Q-values
        if done:
            next_value = 0
        else:
            if self.use_knowledge_shaping:
                # Use shaped Q-values for next state value estimation
                shaped_next_q = self.get_shaped_q_values(next_state)
                next_value = np.max(shaped_next_q)
            else:
                next_value = np.max(self.q_table[next_state_key])

        # Q-learning update with shaping
        target = reward + self.discount_factor * next_value

        # Update Q-value
        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

    def initialize_from_source(self):
        """
        Initialize Q-table with scaled values from source agent.
        This provides a warm start for learning.
        """
        if self.source_agent is not None:
            print("Initializing Q-table from source agent...")
            for state_key, q_values in self.source_agent.q_table.items():
                # Initialize with scaled source Q-values
                self.q_table[state_key] = q_values * self.shaping_weight


class PotentialBasedShaping:
    """
    Potential-based reward shaping for knowledge transfer.
    This is an alternative approach to knowledge shaping.
    """

    def __init__(self, source_agent, discount_factor=0.99, shaping_weight=1.0):
        """
        Initialize potential-based shaping.

        Args:
            source_agent: Trained agent from source task
            discount_factor: Discount factor (gamma)
            shaping_weight: Weight for shaping reward
        """
        self.source_agent = source_agent
        self.discount_factor = discount_factor
        self.shaping_weight = shaping_weight

    def get_potential(self, state):
        """
        Get potential value for a state from source agent.

        Args:
            state: State to evaluate

        Returns:
            Potential value
        """
        if self.source_agent is None:
            return 0.0

        q_values = self.source_agent.get_q_values(state)
        # Potential is the value of the state (max Q-value)
        return np.max(q_values)

    def get_shaped_reward(self, state, next_state, reward):
        """
        Calculate shaped reward: F(s,s') = γ*Φ(s') - Φ(s)

        Args:
            state: Current state
            next_state: Next state
            reward: Original reward

        Returns:
            Shaped reward
        """
        potential_current = self.get_potential(state)
        potential_next = self.get_potential(next_state)

        # Potential-based shaping: F(s,s') = γ*Φ(s') - Φ(s)
        shaping_reward = self.discount_factor * potential_next - potential_current

        # Return original reward plus weighted shaping reward
        return reward + self.shaping_weight * shaping_reward


class AdviceBasedShaping:
    """
    Advice-based knowledge shaping where source agent provides action advice.
    """

    def __init__(self, source_agent, advice_probability=0.3):
        """
        Initialize advice-based shaping.

        Args:
            source_agent: Trained agent from source task
            advice_probability: Probability of following source agent's advice
        """
        self.source_agent = source_agent
        self.advice_probability = advice_probability

    def should_use_advice(self):
        """Determine if advice should be used this step."""
        return np.random.random() < self.advice_probability

    def get_advice(self, state):
        """
        Get action advice from source agent.

        Args:
            state: Current state

        Returns:
            Advised action
        """
        if self.source_agent is None:
            return None

        q_values = self.source_agent.get_q_values(state)
        return np.argmax(q_values)
