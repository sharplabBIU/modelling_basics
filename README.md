# Modelling Basics in RL for Sharp Lab

**A Python implementation contrasting Model-Based, Model-Free, and Successor Representation algorithms on sequential decision-making tasks.**

This repository contains a Jupyter Notebook simulation of the classic "Three-Branch" behavioral task (adapted from Momennejad et al., 2017). It is designed to visualize and quantify how different agents (MB, MF, SR) succeed or fail under specific experimental manipulations, specifically **Reward Revaluation** and **Transition Revaluation**.

## 📦 Contents

1.  [Overview](https://www.google.com/search?q=%23-overview)
2.  [Key Concepts](https://www.google.com/search?q=%23-key-concepts)
3.  [Installation](https://www.google.com/search?q=%23-installation)
4.  [Featured Task: The Three-Branch Environment](https://www.google.com/search?q=%23-featured-task-the-three-branch-environment)
5.  [References](https://www.google.com/search?q=%23-references)

-----

## 🧠 Overview

In biological and artificial reinforcement learning, "value" can be computed in multiple ways. This repository provides clean, readable Python code to simulate these strategies side-by-side in a single notebook. It is intended for:

  * Researchers in Computational Psychiatry and Neuroscience.
  * Students learning about the trade-offs between planning and habit.
  * AI practitioners interested in biological inspirations for RL.

## 🔑 Key Concepts

The simulation focuses on three distinct agent architectures:

  * **Model-Free (MF):** Learns policies by caching long-run values ($Q$) directly from reward prediction errors (SARSA/Q-Learning). It is computationally cheap but inflexible to sudden changes in the environment structure.
  * **Model-Based (MB):** Learns a mental map of the world (Transitions $T$ and Rewards $R$). It computes values "on the fly" via planning (Value Iteration). It is flexible but computationally expensive.
  * **Successor Representation (SR):** A hybrid approach. It learns the "predictive map" of future state occupancies ($M$) and combines it with rewards. It solves Reward Revaluation like an MB agent but fails Transition Revaluation like an MF agent.

-----

## 💻 Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/cognitive-rl-benchmarks.git
cd cognitive-rl-benchmarks
pip install numpy pandas matplotlib seaborn jupyter
```

**How to Run:**

Open the notebook in Jupyter Lab or Notebook:

```bash
jupyter notebook simulations_MB_SR_MF.ipynb
```

-----

## 🏆 Featured Task: The Three-Branch Environment

**File:** `simulations_MB_SR_MF.ipynb`

This notebook implements the environment, agents, and plotting logic in one place. It contrasts agents across four distinct phases:

1.  **Latent Learning:** Agents explore the environment without reward to learn the structure (Transition Matrix $T$ or Successor Matrix $M$).
2.  **Reward Learning:** Agents learn to navigate to a specific reward state.
3.  **Reward Revaluation:** Agents are *told* (or shown in isolation) that the reward has moved to a new state.
      * *Result:* MB and SR adapt immediately; MF fails (persists in old habit).
4.  **Transition Revaluation:** The path structure changes (e.g., the path that used to lead to reward now leads to a loss), but the agent only sees the local change, not the full path.
      * *Result:* MB adapts immediately; SR and MF fail (SR caches long-run path predictions which become stale).

### Visualization

The notebook generates:

1.  **Line Plots:** Showing learning curves (probability of optimal choice) during the Reward Learning phase.
2.  **Bar Plots:** Comparing the "zero-shot" performance of agents immediately after Reward and Transition revaluation, highlighting the specific deficits of MF and SR agents.

-----

## 📚 References

1.  **Momennejad, I., et al. (2017).** *The successor representation in human reinforcement learning.* Nature Human Behaviour.
2.  **Daw, N. D., et al. (2011).** *Model-based influences on humans' choices and striatal prediction errors.* Neuron.
3.  **Gershman, S. J. (2018).** *The Successor Representation: Its Computational Logic and Neural Substrates.* Journal of Neuroscience.
4.  **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction.* MIT Press.

-----

*This project is for educational and research purposes.*