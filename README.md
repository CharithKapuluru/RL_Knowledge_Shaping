# Reinforcement Learning with Knowledge Shaping (RL-KS)

A comprehensive implementation and evaluation of knowledge transfer in reinforcement learning through Q-value shaping.

## Project Overview

This project implements and compares two reinforcement learning approaches:
1. **Baseline Q-Learning**: Standard tabular Q-learning
2. **RL-KS (Reinforcement Learning with Knowledge Shaping)**: Q-learning enhanced with knowledge transfer from source tasks

**Course**: CSCE 5214
**Date**: November 2025

### Team Members
- **Charith Kapuluru** - Literature review & problem statement
- **Bhargav Reddy Alimili** - Algorithm implementation
- **Karthik Saraf** - Simulation experiments
- **Sreekanth Taduru** - Performance evaluation
- **Himesh Chander Addiga** - Evaluation & documentation

## Key Features

- ✅ Complete Q-learning implementation with epsilon-greedy exploration
- ✅ Knowledge shaping agent with configurable source task transfer
- ✅ Two benchmark environments (GridWorld and CartPole)
- ✅ Comprehensive training and evaluation framework
- ✅ Automated visualization and comparison tools
- ✅ Detailed analysis notebook and report
- ✅ Fully reproducible experiments

## Project Structure

```
RL_Knowledge_Shaping/
├── src/                           # Source code
│   ├── q_learning_agent.py        # Baseline Q-learning agent
│   ├── rl_ks_agent.py             # RL-KS agent with knowledge shaping
│   ├── gridworld.py               # GridWorld environment
│   ├── cartpole_wrapper.py        # CartPole discretization wrapper
│   └── trainer.py                 # Training and evaluation utilities
├── experiments/                   # Experiment scripts
│   ├── gridworld_experiment.py    # GridWorld comparison experiment
│   └── cartpole_experiment.py     # CartPole comparison experiment
├── results/                       # Experimental results
│   ├── gridworld_comparison.png   # GridWorld visualization
│   ├── cartpole_comparison.png    # CartPole visualization
│   ├── gridworld_results.pkl      # GridWorld data
│   └── cartpole_results.pkl       # CartPole data
├── docs/                          # Documentation
│   ├── Final_Report.md            # Comprehensive project report
│   └── Presentation_Slides.md     # Presentation materials
├── RL_KS_Analysis.ipynb           # Jupyter notebook for analysis
├── README.md                      # This file
└── venv/                          # Virtual environment
```

## Installation

### Requirements
- Python 3.13+
- pip (Python package manager)

### Setup

1. **Clone or navigate to the project directory:**
```bash
cd RL_Knowledge_Shaping
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
```

3. **Activate the virtual environment:**
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

4. **Install dependencies:**
```bash
pip install numpy gymnasium matplotlib torch
```

### Required Packages
- `numpy` (≥1.26.0) - Numerical computations
- `gymnasium` (≥0.29.0) - RL environments
- `matplotlib` (≥3.8.0) - Visualization
- `torch` (≥2.0.0) - Optional, for future extensions

## Usage

### Running Experiments

#### GridWorld Experiment
```bash
python experiments/gridworld_experiment.py
```

**Output:**
- Training progress for source task
- Comparative training of Baseline RL vs RL-KS
- Evaluation results and metrics
- Visualization saved to `results/gridworld_comparison.png`

#### CartPole Experiment
```bash
python experiments/cartpole_experiment.py
```

**Output:**
- Training progress for source task (coarse discretization)
- Comparative training on target task (fine discretization)
- Evaluation results and metrics
- Visualization saved to `results/cartpole_comparison.png`

### Using the Analysis Notebook

```bash
jupyter notebook RL_KS_Analysis.ipynb
```

The notebook provides:
- Detailed results analysis
- Additional visualizations
- Statistical comparisons
- Key insights and findings

## Algorithm Overview

### Baseline Q-Learning

Standard temporal difference learning:

```python
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**Features:**
- Epsilon-greedy exploration
- Dynamic epsilon decay
- Tabular Q-table (dictionary-based for sparse states)

### RL-KS (Knowledge Shaping)

Enhanced Q-learning with source knowledge:

```python
Q_shaped(s,a) = (1-λ)Q_target(s,a) + λQ_source(s,a)
```

**Features:**
- Incorporates pre-trained source agent Q-values
- Configurable shaping weight (λ)
- Uses shaped Q-values for action selection and learning

## Environments

### GridWorld

**Description:** Discrete grid navigation task

**Specifications:**
- Grid size: 5×5
- State space: 25 discrete states
- Action space: 4 actions (up, down, left, right)
- Rewards:
  - +100 for reaching goal
  - -1 per step
  - -10 for obstacles

**Variants:**
- **Source Task**: Obstacles at (1,1), (2,2), (3,3)
- **Target Task**: Obstacles at (1,2), (2,3), (3,1)

### CartPole

**Description:** Classic control task (OpenAI Gymnasium)

**Specifications:**
- State space: 4 continuous dimensions (discretized)
  - Cart position, velocity
  - Pole angle, angular velocity
- Action space: 2 actions (left, right)
- Rewards:
  - +1 per timestep
  - -10 for early termination

**Variants:**
- **Source Task**: Coarse discretization (4×4×4×4 = 256 states)
- **Target Task**: Fine discretization (6×6×6×6 = 1,296 states)

## Results Summary

### GridWorld

| Metric | Baseline RL | RL-KS | Result |
|--------|-------------|-------|--------|
| Final Reward | 91.98 | 91.92 | Tie |
| Success Rate | 100% | 100% | Tie |
| Avg Episode Length | 8 steps | 8 steps | Tie |

**Conclusion:** Both approaches achieved optimal performance on GridWorld.

### CartPole

| Metric | Baseline RL | RL-KS | Result |
|--------|-------------|-------|--------|
| Final Reward | 111.36 | 74.82 | **Baseline wins** |
| Success Rate | 100% | 99% | **Baseline wins** |
| Avg Episode Length | 123.66 | 83.18 | **Baseline wins** |

**Conclusion:** Baseline RL significantly outperformed RL-KS, demonstrating negative transfer due to representation mismatch.

## Key Findings

### Successes ✓
1. Successfully implemented both baseline RL and RL-KS
2. GridWorld showed that knowledge transfer works for similar tasks
3. Created reusable, well-documented framework

### Challenges ✗
1. CartPole revealed negative transfer from incompatible discretizations
2. State representation compatibility is critical for successful transfer
3. Knowledge shaping requires careful hyperparameter tuning

### Insights 💡
1. **Representation matters most**: Compatible state-action spaces are essential
2. **Task similarity is subtle**: Same environment ≠ compatible representations
3. **Baseline is robust**: Learning from scratch is surprisingly effective

## Hyperparameters

### Learning Parameters
```python
learning_rate = 0.1          # α
discount_factor = 0.99       # γ
initial_epsilon = 1.0        # ε
epsilon_decay = 0.995
min_epsilon = 0.01
```

### Knowledge Shaping
```python
shaping_weight_gridworld = 0.5   # λ for GridWorld
shaping_weight_cartpole = 0.3    # λ for CartPole
```

### Training
```python
gridworld_episodes = 500
cartpole_episodes = 1000
eval_episodes = 100
```

## Reproducibility

**Random Seed:** 42 (set via `np.random.seed(42)`)

**To reproduce results:**
```bash
cd RL_Knowledge_Shaping
source venv/bin/activate
python experiments/gridworld_experiment.py
python experiments/cartpole_experiment.py
```

**Expected Runtime:**
- GridWorld: ~5 seconds
- CartPole: ~10 seconds
- Total: <1 minute on modern hardware

## Documentation

### Reports
- **Full Technical Report**: `docs/Final_Report.md` (30+ pages)
  - Comprehensive methodology
  - Detailed results and analysis
  - Discussion and future work

- **Presentation Slides**: `docs/Presentation_Slides.md`
  - 26 slides with backup slides
  - Visual summaries
  - Key takeaways

### Notebooks
- **Analysis Notebook**: `RL_KS_Analysis.ipynb`
  - Interactive visualizations
  - Statistical analysis
  - Code examples

## API Reference

### QLearningAgent

```python
from src.q_learning_agent import QLearningAgent

agent = QLearningAgent(
    action_space=4,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Get action
action = agent.get_action(state, training=True)

# Update Q-values
agent.update(state, action, reward, next_state, done)

# Decay exploration
agent.decay_epsilon()
```

### RLKSAgent

```python
from src.rl_ks_agent import RLKSAgent

# Requires pre-trained source agent
agent = RLKSAgent(
    action_space=4,
    source_agent=source_agent,
    shaping_weight=0.5,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)

# Same interface as QLearningAgent
action = agent.get_action(state, training=True)
agent.update(state, action, reward, next_state, done)
```

### Trainer

```python
from src.trainer import Trainer

trainer = Trainer(env, agent, verbose=True)

# Train for N episodes
results = trainer.train(num_episodes=500, eval_interval=100)

# Evaluate trained agent
eval_results = trainer.evaluate(num_episodes=100)
```

## Future Improvements

### Immediate Extensions
1. **State Mapping Functions**: Handle different discretizations
2. **Adaptive Shaping**: Dynamic λ adjustment
3. **Statistical Robustness**: Multiple random seeds

### Advanced Features
1. **Deep RL**: DQN, Actor-Critic methods
2. **Alternative Transfer**: Policy distillation, meta-learning
3. **More Environments**: MuJoCo, Atari
4. **Multi-Source Transfer**: Combine knowledge from multiple tasks

## Contributing

This is a course project, but suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is created for educational purposes as part of CSCE 5214.

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

2. Taylor, M. E., & Stone, P. (2009). Transfer learning for reinforcement learning domains: A survey. *JMLR*, 10, 1633-1685.

3. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. *ICML*.

4. Brys, T., et al. (2015). Policy transfer using reward shaping. *AAMAS*.

5. OpenAI Gymnasium: https://gymnasium.farama.org/

## Troubleshooting

### Common Issues

**Problem:** `ModuleNotFoundError: No module named 'numpy'`
```bash
# Solution: Activate virtual environment
source venv/bin/activate
pip install numpy gymnasium matplotlib
```

**Problem:** `externally-managed-environment` error
```bash
# Solution: Use virtual environment (already done in setup)
python3 -m venv venv
source venv/bin/activate
```

**Problem:** Plots not displaying
```bash
# Solution: Plots are saved to results/ directory
# View them directly or use Jupyter notebook
```

## Contact

For questions or issues:
- See `docs/Final_Report.md` for detailed explanations
- Review `RL_KS_Analysis.ipynb` for analysis examples
- Check `docs/Presentation_Slides.md` for visual summaries

## Acknowledgments

- OpenAI Gymnasium for environment framework
- Python scientific computing community (NumPy, Matplotlib)
- Course instructors and teaching assistants
- Reinforcement learning research community

---

**Project Status:** ✅ Complete

**Last Updated:** November 2025

**Total Lines of Code:** ~1,134 lines of Python

**Documentation:** 30+ pages of reports and slides
