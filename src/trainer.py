"""
Training and Evaluation Utilities for RL agents
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time


class Trainer:
    """Trainer class for RL agents."""

    def __init__(self, env, agent, verbose=False):
        """
        Initialize trainer.

        Args:
            env: Environment to train on
            agent: Agent to train
            verbose: Whether to print progress
        """
        self.env = env
        self.agent = agent
        self.verbose = verbose
        self.episode_rewards = []
        self.episode_lengths = []

    def train_episode(self):
        """
        Train for one episode.

        Returns:
            total_reward: Total reward for the episode
            episode_length: Number of steps in the episode
        """
        state = self.env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            # Select action
            action = self.agent.get_action(state, training=True)

            # Take action
            next_state, reward, done, _ = self.env.step(action)

            # Update agent
            self.agent.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

        # Decay epsilon after episode
        self.agent.decay_epsilon()

        return total_reward, steps

    def train(self, num_episodes, eval_interval=100):
        """
        Train agent for multiple episodes.

        Args:
            num_episodes: Number of episodes to train
            eval_interval: Evaluate every N episodes

        Returns:
            training_history: Dictionary with training metrics
        """
        print(f"Training for {num_episodes} episodes...")
        start_time = time.time()

        for episode in range(num_episodes):
            reward, length = self.train_episode()
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)

            if self.verbose and (episode + 1) % eval_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-eval_interval:])
                avg_length = np.mean(self.episode_lengths[-eval_interval:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Epsilon: {self.agent.epsilon:.3f}")

        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'final_epsilon': self.agent.epsilon
        }

    def evaluate(self, num_episodes=100):
        """
        Evaluate trained agent.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            evaluation_results: Dictionary with evaluation metrics
        """
        print(f"Evaluating for {num_episodes} episodes...")
        eval_rewards = []
        eval_lengths = []
        success_count = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

            while not done:
                # Select action greedily (no exploration)
                action = self.agent.get_action(state, training=False)

                # Take action
                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward
                state = next_state
                steps += 1

            eval_rewards.append(total_reward)
            eval_lengths.append(steps)

            # Check if successful (reached goal)
            if total_reward > 0:  # Positive reward indicates goal reached
                success_count += 1

        avg_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        avg_length = np.mean(eval_lengths)
        success_rate = success_count / num_episodes

        print(f"Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Length: {avg_length:.2f}")
        print(f"  Success Rate: {success_rate:.2%}")

        return {
            'rewards': eval_rewards,
            'lengths': eval_lengths,
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'avg_length': avg_length,
            'success_rate': success_rate
        }


def compare_agents(results_dict: Dict[str, Dict], save_path=None):
    """
    Compare multiple agents' training results.

    Args:
        results_dict: Dictionary mapping agent names to their training results
        save_path: Path to save comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Episode Rewards
    ax1 = axes[0, 0]
    for name, results in results_dict.items():
        rewards = results['episode_rewards']
        # Smooth rewards with moving average
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smoothed, label=name, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (smoothed)')
    ax1.set_title('Training Rewards Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    ax2 = axes[0, 1]
    for name, results in results_dict.items():
        lengths = results['episode_lengths']
        # Smooth lengths
        window = 50
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, label=name, alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length (smoothed)')
    ax2.set_title('Episode Length Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Rewards
    ax3 = axes[1, 0]
    for name, results in results_dict.items():
        rewards = results['episode_rewards']
        cumulative = np.cumsum(rewards)
        ax3.plot(cumulative, label=name, alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Cumulative Reward')
    ax3.set_title('Cumulative Rewards Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence Speed (episodes to threshold)
    ax4 = axes[1, 1]
    convergence_data = {}
    threshold = -50  # Threshold for "good" performance

    for name, results in results_dict.items():
        rewards = results['episode_rewards']
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

        # Find first episode where smoothed reward exceeds threshold
        converged = np.where(smoothed >= threshold)[0]
        if len(converged) > 0:
            convergence_episode = converged[0]
        else:
            convergence_episode = len(rewards)

        convergence_data[name] = convergence_episode

    names = list(convergence_data.keys())
    episodes = list(convergence_data.values())
    colors = ['steelblue', 'coral'][:len(names)]

    ax4.bar(names, episodes, color=colors, alpha=0.7)
    ax4.set_ylabel('Episodes to Convergence')
    ax4.set_title(f'Convergence Speed (Threshold: {threshold})')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()

    return convergence_data


def print_comparison_summary(results_dict: Dict[str, Dict]):
    """
    Print a summary comparison of agents.

    Args:
        results_dict: Dictionary mapping agent names to their results
    """
    print("\n" + "="*70)
    print("TRAINING COMPARISON SUMMARY")
    print("="*70)

    for name, results in results_dict.items():
        rewards = results['episode_rewards']
        lengths = results['episode_lengths']

        final_100_reward = np.mean(rewards[-100:])
        final_100_length = np.mean(lengths[-100:])
        total_reward = np.sum(rewards)

        print(f"\n{name}:")
        print(f"  Final 100 episodes avg reward: {final_100_reward:.2f}")
        print(f"  Final 100 episodes avg length: {final_100_length:.2f}")
        print(f"  Total cumulative reward: {total_reward:.2f}")

    print("\n" + "="*70)
