"""
CartPole Experiment: Compare baseline RL vs RL-KS
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from cartpole_wrapper import DiscreteCartPoleWrapper, CartPoleVariant
from q_learning_agent import QLearningAgent
from rl_ks_agent import RLKSAgent
from trainer import Trainer, compare_agents, print_comparison_summary
import pickle


def train_source_task():
    """
    Train an agent on a source task (coarse-grained CartPole).
    """
    print("\n" + "="*70)
    print("PHASE 1: Training Source Task (Coarse-grained CartPole)")
    print("="*70)

    # Create source environment (coarse discretization)
    env_source = CartPoleVariant.create_coarse_grained()
    print(f"Source task: CartPole with discretization bins = {env_source.bins}")

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
    results_source = trainer_source.train(num_episodes=1000, eval_interval=200)

    # Evaluate source agent
    eval_source = trainer_source.evaluate(num_episodes=100)

    print(f"\nSource agent trained successfully!")
    print(f"Final performance: {eval_source['avg_reward']:.2f} ± {eval_source['std_reward']:.2f}")

    env_source.close()
    return agent_source, results_source


def compare_on_target_task(source_agent):
    """
    Compare baseline RL vs RL-KS on target task (standard CartPole).
    """
    print("\n" + "="*70)
    print("PHASE 2: Training Target Task (Baseline RL vs RL-KS)")
    print("="*70)

    # Create target environment (standard discretization)
    env_target = CartPoleVariant.create_standard()
    print(f"Target task: CartPole with discretization bins = {env_target.bins}")

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
    results_baseline = trainer_baseline.train(num_episodes=1000, eval_interval=200)

    # Create new environment for RL-KS agent
    env_target_rlks = CartPoleVariant.create_standard()

    # RL-KS agent (with knowledge transfer)
    print("\nTraining RL-KS agent...")
    agent_rlks = RLKSAgent(
        action_space=env_target_rlks.action_space,
        source_agent=source_agent,
        shaping_weight=0.3,  # Lower weight due to different discretization
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    trainer_rlks = Trainer(env_target_rlks, agent_rlks, verbose=True)
    results_rlks = trainer_rlks.train(num_episodes=1000, eval_interval=200)

    # Evaluate both agents
    print("\n" + "="*70)
    print("EVALUATION ON TARGET TASK")
    print("="*70)

    print("\nBaseline RL:")
    eval_baseline = trainer_baseline.evaluate(num_episodes=100)

    print("\nRL-KS:")
    eval_rlks = trainer_rlks.evaluate(num_episodes=100)

    env_target.close()
    env_target_rlks.close()

    return {
        'Baseline RL': results_baseline,
        'RL-KS': results_rlks
    }, {
        'Baseline RL': eval_baseline,
        'RL-KS': eval_rlks
    }


def run_cartpole_experiment():
    """
    Run complete CartPole experiment.
    """
    print("\n" + "="*70)
    print("CARTPOLE EXPERIMENT: RL vs RL-KS")
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

    save_path = os.path.join(results_dir, 'cartpole_comparison.png')
    convergence = compare_agents(training_results, save_path=save_path)

    # Print convergence comparison
    print("\nConvergence Speed:")
    for name, episodes in convergence.items():
        print(f"  {name}: {episodes} episodes")

    # Calculate improvement metrics
    baseline_final_reward = np.mean(training_results['Baseline RL']['episode_rewards'][-100:])
    rlks_final_reward = np.mean(training_results['RL-KS']['episode_rewards'][-100:])

    print(f"\nFinal Performance Comparison:")
    print(f"  Baseline RL: {baseline_final_reward:.2f}")
    print(f"  RL-KS: {rlks_final_reward:.2f}")

    if rlks_final_reward > baseline_final_reward:
        improvement = ((rlks_final_reward - baseline_final_reward) / abs(baseline_final_reward)) * 100
        print(f"  Improvement: +{improvement:.2f}%")

    # Save results to file
    results_file = os.path.join(results_dir, 'cartpole_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump({
            'training_results': training_results,
            'eval_results': eval_results,
            'convergence': convergence
        }, f)
    print(f"\nResults saved to {results_file}")

    return training_results, eval_results


if __name__ == "__main__":
    results_training, results_eval = run_cartpole_experiment()
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED!")
    print("="*70)
