"""
Gridworld Environment Implementation
Simple grid navigation task for testing RL algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    """
    Simple GridWorld environment.
    Agent must navigate from start to goal, avoiding obstacles.
    """

    def __init__(self, grid_size=(5, 5), obstacles=None, start=None, goal=None):
        """
        Initialize GridWorld environment.

        Args:
            grid_size: Tuple (height, width) of the grid
            obstacles: List of (row, col) tuples for obstacle positions
            start: Starting position (row, col), defaults to (0, 0)
            goal: Goal position (row, col), defaults to bottom-right corner
        """
        self.height, self.width = grid_size
        self.grid_size = grid_size

        # Default positions
        self.start_pos = start if start else (0, 0)
        self.goal_pos = goal if goal else (self.height - 1, self.width - 1)
        self.obstacles = obstacles if obstacles else []

        # Current agent position
        self.agent_pos = None

        # Action space: 0=up, 1=down, 2=left, 3=right
        self.action_space = 4
        self.action_map = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        # Episode tracking
        self.max_steps = grid_size[0] * grid_size[1] * 2
        self.current_step = 0

    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = list(self.start_pos)
        self.current_step = 0
        return tuple(self.agent_pos)

    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).

        Args:
            action: Action to take (0-3)

        Returns:
            next_state: New position
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1

        # Get action delta
        delta = self.action_map[action]
        new_pos = [self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1]]

        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        # If invalid, agent stays in place (hit wall)

        # Calculate reward
        reward = self._get_reward()

        # Check if done
        done = (tuple(self.agent_pos) == self.goal_pos) or (self.current_step >= self.max_steps)

        return tuple(self.agent_pos), reward, done, {}

    def _is_valid_position(self, pos):
        """Check if position is within bounds and not an obstacle."""
        row, col = pos
        # Check bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        # Check obstacles
        if tuple(pos) in self.obstacles:
            return False
        return True

    def _get_reward(self):
        """Calculate reward for current position."""
        if tuple(self.agent_pos) == self.goal_pos:
            return 100.0  # Large positive reward for reaching goal
        elif tuple(self.agent_pos) in self.obstacles:
            return -10.0  # Penalty for hitting obstacle
        else:
            return -1.0  # Small penalty for each step (encourages shorter paths)

    def render(self, show=True):
        """Visualize the current state of the gridworld."""
        grid = np.zeros((self.height, self.width))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1

        # Mark goal
        grid[self.goal_pos] = 2

        # Mark agent
        grid[tuple(self.agent_pos)] = 1

        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
            plt.colorbar(ticks=[-1, 0, 1, 2], label='Cell Type')
            plt.title('GridWorld State')
            plt.xlabel('Column')
            plt.ylabel('Row')

            # Add grid lines
            for i in range(self.height + 1):
                plt.axhline(i - 0.5, color='black', linewidth=0.5)
            for j in range(self.width + 1):
                plt.axvline(j - 0.5, color='black', linewidth=0.5)

            plt.tight_layout()
            plt.show()

        return grid

    def get_state_space_size(self):
        """Return the size of the state space."""
        return self.height * self.width


class GridWorldVariant:
    """
    Variant GridWorld for testing transfer learning.
    Similar structure but different goal/obstacle positions.
    """

    @staticmethod
    def create_source_task():
        """Create a source task for learning prior knowledge."""
        return GridWorld(
            grid_size=(5, 5),
            obstacles=[(1, 1), (2, 2), (3, 3)],
            start=(0, 0),
            goal=(4, 4)
        )

    @staticmethod
    def create_target_task():
        """Create a target task (similar but different)."""
        return GridWorld(
            grid_size=(5, 5),
            obstacles=[(1, 2), (2, 3), (3, 1)],
            start=(0, 0),
            goal=(4, 4)
        )

    @staticmethod
    def create_simple_task():
        """Create a simple task with no obstacles."""
        return GridWorld(
            grid_size=(5, 5),
            obstacles=[],
            start=(0, 0),
            goal=(4, 4)
        )

    @staticmethod
    def create_complex_task():
        """Create a more complex task with more obstacles."""
        return GridWorld(
            grid_size=(7, 7),
            obstacles=[(1, 1), (1, 2), (2, 1), (3, 4), (4, 3), (4, 4), (5, 2)],
            start=(0, 0),
            goal=(6, 6)
        )
