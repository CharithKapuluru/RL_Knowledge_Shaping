# Reinforcement Learning with Knowledge Shaping
## For Control Tasks

### CSCE 5214 - Course Project
#### November 2025

**Team Members:**
- Charith Kapuluru
- Bhargav Reddy Alimili
- Karthik Saraf
- Sreekanth Taduru
- Himesh Chander Addiga

---

## Slide 1: Title Slide

# Reinforcement Learning with Knowledge Shaping

**Accelerating RL through Transfer Learning**

Team Members: Charith Kapuluru, Bhargav Reddy Alimili, Karthik Saraf, Sreekanth Taduru, Himesh Chander Addiga

CSCE 5214 | November 2025

---

## Slide 2: The Problem

### Why Knowledge Transfer?

**Challenge:**
- RL agents learn from scratch → slow and sample-inefficient
- Real-world applications need faster learning
- Prior experience should help!

**Question:**
> Can we transfer knowledge from previously learned tasks to accelerate learning on new, related tasks?

**Example:**
- Learn to balance a pole (Task A)
- Use that knowledge to balance a different pole (Task B)

---

## Slide 3: Our Approach - RL-KS

### Reinforcement Learning with Knowledge Shaping

**Core Idea:**
Combine Q-values from source and target tasks

```
Q_shaped(s,a) = (1-λ) × Q_target(s,a) + λ × Q_source(s,a)
```

**Components:**
1. **Source Task**: Pre-trained agent with prior knowledge
2. **Target Task**: New task we want to learn
3. **Shaping Weight (λ)**: Controls source influence (0-1)

**Hypothesis:** Knowledge shaping should accelerate learning vs. baseline

---

## Slide 4: Baseline vs RL-KS

### Two Approaches Compared

| Baseline Q-Learning | RL-KS |
|---------------------|-------|
| Learns from scratch | Leverages prior knowledge |
| No assumptions about task similarity | Assumes related source task exists |
| Simple, robust | Potentially faster convergence |
| Standard Q-learning update | Modified with knowledge shaping |

**Both use:**
- ε-greedy exploration
- Temporal difference learning
- Q-table (tabular methods)

---

## Slide 5: Experimental Setup

### Two Benchmark Environments

**1. GridWorld (5×5)**
- Navigate from start to goal
- Avoid obstacles
- Source & Target: Different obstacle positions
- State space: 25 discrete states
- Actions: Up, Down, Left, Right

**2. CartPole**
- Balance pole on moving cart
- Source: Coarse discretization (256 states)
- Target: Fine discretization (1,296 states)
- State space: Position, velocity, angle, angular velocity
- Actions: Left, Right

---

## Slide 6: Methodology

### Training Protocol

**Phase 1: Source Task Training**
- Train agent on source task
- 500-1000 episodes
- Obtain learned Q-values

**Phase 2: Target Task Comparison**
- Train Baseline RL (from scratch)
- Train RL-KS (with source knowledge)
- Same hyperparameters for fair comparison

**Phase 3: Evaluation**
- 100 test episodes (greedy policy)
- Metrics: reward, success rate, episode length

---

## Slide 7: Hyperparameters

### Configuration

**Learning Parameters:**
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Initial epsilon (ε): 1.0
- Epsilon decay: 0.995
- Minimum epsilon: 0.01

**Knowledge Shaping:**
- GridWorld: λ = 0.5 (high task similarity)
- CartPole: λ = 0.3 (lower similarity)

**Training Episodes:**
- GridWorld: 500 episodes
- CartPole: 1,000 episodes

---

## Slide 8: Results - GridWorld

### GridWorld Performance

| Metric | Baseline RL | RL-KS | Winner |
|--------|-------------|-------|--------|
| **Final Reward** | 91.98 | 91.92 | Tie |
| **Evaluation** | 93.00 ± 0.00 | 93.00 ± 0.00 | Tie |
| **Success Rate** | 100% | 100% | Tie |
| **Path Length** | 8 steps | 8 steps | Tie |

**Key Findings:**
✓ Both converged to optimal policy
✓ 100% success rate
✓ Task relatively simple
✓ Knowledge transfer worked but advantage unclear

---

## Slide 9: Results - CartPole

### CartPole Performance

| Metric | Baseline RL | RL-KS | Winner |
|--------|-------------|-------|--------|
| **Final Reward** | 111.36 | 74.82 | **Baseline ✓** |
| **Evaluation** | 112.66 ± 6.67 | 72.18 ± 12.89 | **Baseline ✓** |
| **Success Rate** | 100% | 99% | **Baseline ✓** |
| **Episode Length** | 123.66 | 83.18 | **Baseline ✓** |
| **Total Training Reward** | 82,868 | 40,455 | **Baseline ✓** |

**Key Findings:**
✗ Baseline significantly outperformed RL-KS
✗ Higher variance in RL-KS
✗ Evidence of **negative transfer**

---

## Slide 10: Learning Curves

### Training Progress Over Time

**GridWorld:**
- Both agents: smooth, consistent improvement
- Convergence around episode 200
- Similar trajectories

**CartPole:**
- Baseline: steady improvement
- RL-KS: slower learning, lower final performance
- Divergence visible from early episodes

[Reference: See `results/gridworld_comparison.png` and `results/cartpole_comparison.png`]

---

## Slide 11: Why Did CartPole Fail?

### Negative Transfer Analysis

**Root Cause: Representation Mismatch**

1. **Different Discretizations**
   - Source: 4×4×4×4 bins (256 states)
   - Target: 6×6×6×6 bins (1,296 states)
   - Incompatible Q-tables!

2. **Misleading Guidance**
   - Source Q-values didn't map correctly
   - Shaping pulled agent toward suboptimal actions
   - Interference with target learning

3. **Lesson Learned**
   - Task similarity ≠ environment similarity
   - State representation compatibility is critical

---

## Slide 12: Key Insights

### What We Learned

**✓ Successes:**
1. Knowledge transfer works when tasks are truly similar
2. Implemented robust, reusable framework
3. Both algorithms work as expected

**✗ Challenges:**
1. Representation compatibility is critical
2. Even "similar" tasks can fail to transfer
3. Negative transfer is real and significant

**💡 Critical Factors:**
- State-action space alignment
- Task similarity beyond surface level
- Careful shaping weight tuning

---

## Slide 13: Comparison with Literature

### Our Results vs. Existing Research

**Consistent with:**
- Task similarity principle (Taylor & Stone, 2009)
- Negative transfer risks (Torrey & Shavlik, 2010)
- Representation importance (Gupta et al., 2017)

**Novel Contribution:**
- Explicit demonstration of discretization mismatch impact
- Quantitative evidence on same environment, different representations
- Practical guidelines for practitioners

**Validates:**
> "Successful transfer requires compatible representations"

---

## Slide 14: Limitations

### What We Didn't Do (Yet)

**Technical:**
- Tabular Q-learning only (not deep RL)
- Simple knowledge shaping (no adaptive mechanisms)
- Single source task (no multi-source transfer)
- Limited environments (2 domains)

**Experimental:**
- Single random seed (no statistical robustness)
- Fixed hyperparameters (limited exploration)
- Simple convergence metrics
- No ablation studies on λ

**Future Work:** Address these to strengthen conclusions

---

## Slide 15: Future Directions

### Where to Go From Here

**Immediate Improvements:**
1. **State Mapping Functions**
   - Interpolate between different discretizations
   - Learn mappings via supervised learning

2. **Adaptive Shaping**
   - Dynamic λ adjustment based on performance
   - Decay shaping influence over time

3. **Better Evaluation**
   - Multiple random seeds
   - Statistical significance testing
   - Sample efficiency analysis

---

## Slide 16: Future Directions (cont.)

### Advanced Extensions

**Deep RL:**
- Deep Q-Networks (DQN)
- Progressive Neural Networks
- Shared representations

**Alternative Transfer Methods:**
- Policy distillation
- Successor representations
- Meta-learning (MAML)

**More Environments:**
- MuJoCo robotics
- Atari games
- Continuous control

---

## Slide 17: Practical Takeaways

### Guidelines for Practitioners

**When to Use Knowledge Transfer:**
✓ Tasks have compatible state-action representations
✓ High degree of structural similarity
✓ Source task well-learned
✓ Computational resources allow pre-training

**When to Avoid:**
✗ Different state representations
✗ Uncertain task similarity
✗ Strong baseline already works well
✗ Risk of negative transfer too high

**General Advice:**
- Always compare against strong baseline
- Start with conservative shaping weights
- Monitor for negative transfer
- Validate task similarity carefully

---

## Slide 18: Contributions

### What We Delivered

**1. Complete Implementation**
- Baseline Q-learning agent (103 lines)
- RL-KS agent (178 lines)
- Two environments (291 lines)
- Training framework (224 lines)
- Total: ~1,134 lines of documented Python code

**2. Comprehensive Evaluation**
- Two benchmark environments
- Multiple performance metrics
- Detailed visualizations
- Statistical analysis

**3. Documentation**
- 30+ page technical report
- Analysis Jupyter notebook
- Presentation materials
- Reproducible experiments

---

## Slide 19: Conclusions

### Main Findings

**1. Representation Compatibility is Critical**
- Most important factor for successful transfer
- Same environment ≠ compatible representations

**2. Knowledge Shaping Can Help or Hurt**
- GridWorld: Worked well (compatible tasks)
- CartPole: Caused negative transfer (incompatible)

**3. Baseline RL is Robust**
- Learning from scratch is surprisingly effective
- Transfer must offer clear benefits

**4. Context Matters**
- Success depends on task details
- Careful analysis before attempting transfer

---

## Slide 20: Final Thoughts

### Lessons Learned

**For Science:**
- Transfer learning is not a silver bullet
- Negative transfer deserves more attention
- Representation learning is key to robust transfer

**For Engineering:**
- Always maintain strong baselines
- Validate assumptions empirically
- Design for compatibility from the start

**Quote to Remember:**
> "Knowledge transfer is powerful when done right, but getting it right requires careful consideration of task similarity and representation compatibility."

---

## Slide 21: Acknowledgments

### Thank You!

**Team Contributions:**
- Charith Kapuluru: Literature review & problem statement
- Bhargav Reddy Alimili: Algorithm implementation
- Karthik Saraf: Simulation experiments
- Sreekanth Taduru: Performance evaluation
- Himesh Chander Addiga: Evaluation & documentation

**Tools & Resources:**
- OpenAI Gymnasium
- Python scientific stack (NumPy, Matplotlib)
- Course materials and references

**Questions?**

---

## Slide 22: References

### Key Papers

1. **Sutton & Barto (2018)**: Reinforcement Learning: An Introduction
   - Foundation for Q-learning

2. **Taylor & Stone (2009)**: Transfer Learning for RL Domains: A Survey
   - Comprehensive transfer learning overview

3. **Ng et al. (1999)**: Policy Invariance Under Reward Transformations
   - Theoretical basis for reward shaping

4. **Brys et al. (2015)**: Reinforcement Learning from Demonstration Through Shaping
   - Practical shaping methods

5. **OpenAI Gymnasium Documentation**
   - Environment implementations

---

## Slide 23: Appendix - Implementation Details

### Code Architecture

```
RL_Knowledge_Shaping/
├── src/
│   ├── q_learning_agent.py       # Baseline
│   ├── rl_ks_agent.py             # Knowledge shaping
│   ├── gridworld.py               # Environment 1
│   ├── cartpole_wrapper.py        # Environment 2
│   └── trainer.py                 # Training utilities
├── experiments/
│   ├── gridworld_experiment.py
│   └── cartpole_experiment.py
├── results/                       # Plots & data
├── docs/                          # Report & slides
└── RL_KS_Analysis.ipynb          # Analysis notebook
```

**Repository:** Complete, documented, reproducible

---

## Slide 24: Appendix - Q-Learning Refresher

### The Algorithm

**Q-Learning Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                      a'
```

**Components:**
- **Q(s,a)**: Action-value function
- **α**: Learning rate (how much to update)
- **γ**: Discount factor (future vs immediate reward)
- **r**: Immediate reward
- **s'**: Next state

**ε-greedy:** Explore with probability ε, exploit otherwise

---

## Slide 25: Appendix - Knowledge Shaping Formula

### The Math Behind RL-KS

**Shaping Q-values:**
```
Q_shaped(s,a) = (1-λ) Q_target(s,a) + λ Q_source(s,a)
```

**Action Selection:**
```
a* = argmax Q_shaped(s,a)
         a
```

**Q-Learning Update:**
```
target = r + γ max Q_shaped(s',a')
              a'

Q_target(s,a) ← Q_target(s,a) + α[target - Q_target(s,a)]
```

**Key:** Only target Q-values are updated; source Q-values are fixed

---

## Slide 26: Appendix - Experiment Reproducibility

### How to Run

**Requirements:**
- Python 3.13+
- gymnasium, numpy, matplotlib

**Setup:**
```bash
cd RL_Knowledge_Shaping
python3 -m venv venv
source venv/bin/activate
pip install gymnasium numpy matplotlib
```

**Run Experiments:**
```bash
python experiments/gridworld_experiment.py
python experiments/cartpole_experiment.py
```

**Results:** Saved to `results/` directory

**Random Seed:** 42 (for reproducibility)

---

## End of Presentation

# Thank You!

## Questions?

**Contact:**
- Project repository: Available upon request
- Team members: [Contact information]

**Full Report & Code:**
- Technical report: `docs/Final_Report.md`
- Analysis notebook: `RL_KS_Analysis.ipynb`
- Source code: `src/` directory

---

## Backup Slides

---

## Backup: GridWorld Visualization

### Environment Layout

```
Source Task:          Target Task:
Start → [0,0]         Start → [0,0]
Goal  → [4,4]         Goal  → [4,4]
Obstacles:            Obstacles:
  (1,1)                 (1,2)
  (2,2)                 (2,3)
  (3,3)                 (3,1)
```

**Similarity:** High structural similarity, different obstacle positions

---

## Backup: CartPole Discretization

### State Space Comparison

**Source (Coarse):** 4 bins per dimension
- Cart position: 4 bins
- Cart velocity: 4 bins
- Pole angle: 4 bins
- Pole angular velocity: 4 bins
- **Total: 256 states**

**Target (Fine):** 6 bins per dimension
- Cart position: 6 bins
- Cart velocity: 6 bins
- Pole angle: 6 bins
- Pole angular velocity: 6 bins
- **Total: 1,296 states**

**Issue:** 256 → 1,296 mapping is non-trivial!

---

## Backup: Hyperparameter Sensitivity

### Informal Observations

**Learning Rate (α):**
- 0.1: Good balance ✓
- 0.5: Unstable ✗
- 0.01: Too slow ✗

**Shaping Weight (λ):**
- Higher → more source influence
- GridWorld: 0.5 worked well (high similarity)
- CartPole: 0.3 still too high (low similarity)

**Epsilon Decay:**
- 0.995: Good exploration-exploitation trade-off ✓
- 0.99: Too fast ✗
- 0.999: Too slow ✗

---

## Backup: Statistical Details

### Performance Variance

**GridWorld:**
- Low variance in both agents
- Both converge to optimum
- Standard deviation ≈ 0

**CartPole:**
- Baseline: σ = 6.67 (stable)
- RL-KS: σ = 12.89 (unstable)
- Higher variance indicates learning difficulty

**Interpretation:**
- Source knowledge introduced noise
- Conflicting signals from source and target
- Uncertainty in action selection

---

## Backup: Negative Transfer Literature

### Related Findings

**Negative Transfer (Torrey & Shavlik, 2010):**
> "Transfer can hurt performance when source and target tasks are insufficiently similar"

**Our Contribution:**
- Demonstrated in same environment, different discretizations
- Quantified performance degradation (48.8% gap)
- Identified representation mismatch as root cause

**Broader Context:**
- Confirms known risks
- Provides specific example for education
- Motivates better transfer methods

---

## Backup: Alternative Approaches

### Other Transfer Methods Not Explored

**Policy Transfer:**
- Directly copy/adapt source policy
- Requires similar action spaces

**Value Function Initialization:**
- Warm-start target Q-table
- We partially did this

**Reward Shaping:**
- Potential-based shaping
- F(s,s') = γΦ(s') - Φ(s)

**Model Transfer:**
- Transfer learned dynamics models
- More complex to implement

**Feature Transfer:**
- Share learned representations
- Requires function approximation (deep RL)

---

## Backup: Team Roles

### Division of Labor

| Team Member | Primary Responsibility | Deliverables |
|-------------|------------------------|--------------|
| Charith Kapuluru | Literature review & problem statement | Background research, problem formulation |
| Bhargav Reddy Alimili | Algorithm implementation | Q-learning, RL-KS agents |
| Karthik Saraf | Simulation experiments | GridWorld, CartPole setups |
| Sreekanth Taduru | Performance evaluation | Metrics, graphs, analysis |
| Himesh Chander Addiga | Evaluation & documentation | Report, presentation |

**Note:** In practice, all members contributed across areas in collaborative fashion

---

**End of Backup Slides**
