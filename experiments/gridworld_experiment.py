"""
GridWorld Experiment: Compare baseline RL vs RL-KS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld, GridWorldVariant
from q_learning_agent import QLearningAgent
from rl_ks_agent import RLKSAgent
from trainer import Trainer, compare_agents, print_comparison_summary
import pickle


def train_source_task():
    """
    Train an agent on a source task to obtain prior knowledge.
    """
    print("\n" + "="*70)
    print("PHASE 1: Training Source Task")
    print("="*70)

    # Create source environment
    env_source = GridWorldVariant.create_source_task()
    print(f"Source task: {env_source.grid_size} grid with {len(env_source.obstacles)} obstacles")

    # Create and train source agent
    agent_source = QLearningAgent(
        action_space=env_source.action_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    trainer_source = Trainer(env_source, agent_source, verbose=True)
    results_source = trainer_source.train(num_episodes=500, eval_interval=100)

    # Evaluate source agent
    eval_source = trainer_source.evaluate(num_episodes=100)

    print(f"\nSource agent trained successfully!")
    print(f"Final performance: {eval_source['avg_reward']:.2f} ± {eval_source['std_reward']:.2f}")

    return agent_source, results_source


def compare_on_target_task(source_agent):
    """
    Compare baseline RL vs RL-KS on target task.
    """
    print("\n" + "="*70)
    print("PHASE 2: Training Target Task (Baseline RL vs RL-KS)")
    print("="*70)

    # Create target environment
    env_target = GridWorldVariant.create_target_task()
    print(f"Target task: {env_target.grid_size} grid with {len(env_target.obstacles)} obstacles")

    # Baseline RL agent (no knowledge transfer)
    print("\nTraining Baseline RL agent...")
    agent_baseline = QLearningAgent(
        action_space=env_target.action_space,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    trainer_baseline = Trainer(env_target, agent_baseline, verbose=True)
    results_baseline = trainer_baseline.train(num_episodes=500, eval_interval=100)

    # RL-KS agent (with knowledge transfer)
    print("\nTraining RL-KS agent...")
    agent_rlks = RLKSAgent(
        action_space=env_target.action_space,
        source_agent=source_agent,
        shaping_weight=0.5,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    trainer_rlks = Trainer(env_target, agent_rlks, verbose=True)
    results_rlks = trainer_rlks.train(num_episodes=500, eval_interval=100)

    # Evaluate both agents
    print("\n" + "="*70)
    print("EVALUATION ON TARGET TASK")
    print("="*70)

    print("\nBaseline RL:")
    eval_baseline = trainer_baseline.evaluate(num_episodes=100)

    print("\nRL-KS:")
    eval_rlks = trainer_rlks.evaluate(num_episodes=100)

    return {
        'Baseline RL': results_baseline,
        'RL-KS': results_rlks
    }, {
        'Baseline RL': eval_baseline,
        'RL-KS': eval_rlks
    }


def run_gridworld_experiment():
    """
    Run complete GridWorld experiment.
    """
    print("\n" + "="*70)
    print("GRIDWORLD EXPERIMENT: RL vs RL-KS")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Phase 1: Train source task
    source_agent, source_results = train_source_task()

    # Phase 2: Compare on target task
    training_results, eval_results = compare_on_target_task(source_agent)

    # Visualization and comparison
    print("\n" + "="*70)
    print("RESULTS VISUALIZATION")
    print("="*70)

    # Print comparison summary
    print_comparison_summary(training_results)

    # Plot comparison
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    save_path = os.path.join(results_dir, 'gridworld_comparison.png')
    convergence = compare_agents(training_results, save_path=save_path)

    # Print convergence comparison
    print("\nConvergence Speed:")
    for name, episodes in convergence.items():
        print(f"  {name}: {episodes} episodes")

    if 'RL-KS' in convergence and 'Baseline RL' in convergence:
        speedup = convergence['Baseline RL'] / convergence['RL-KS']
        print(f"\nSpeedup: {speedup:.2f}x faster convergence with RL-KS")

    # Save results to file
    results_file = os.path.join(results_dir, 'gridworld_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'training_results': training_results,
            'eval_results': eval_results,
            'convergence': convergence
        }, f)
    print(f"\nResults saved to {results_file}")

    return training_results, eval_results


if __name__ == "__main__":
    results_training, results_eval = run_gridworld_experiment()
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
