# Modelling Basics in RL for Sharp Lab

**A collection of Python implementations contrasting Model-Based, Model-Free, and Successor Representation algorithms on sequential decision-making tasks.**

This repository contains simulations of classic behavioral tasks designed to tease apart different reinforcement learning strategies. The primary goal is to visualize and quantify how different agents (MB, MF, SR) succeed or fail under specific experimental manipulations, such as Reward Revaluation and Transition Revaluation.

## 📦 Contents

1.  [Overview](https://www.google.com/search?q=%23-overview)
2.  [Key Concepts](https://www.google.com/search?q=%23-key-concepts)
3.  [Installation](https://www.google.com/search?q=%23-installation)
4.  [Featured Task: Momennejad et al. (2017)](https://www.google.com/search?q=%23-featured-task-momennejad-et-al-2017)
5.  [Other Algorithms & Tasks](https://www.google.com/search?q=%23-other-algorithms--tasks)
6.  [References](https://www.google.com/search?q=%23-references)

-----

## 🧠 Overview

In biological and artificial reinforcement learning, "value" can be computed in multiple ways. This repository provides clean, readable Python code to simulate these strategies side-by-side. It is intended for:

  * Researchers in Computational Psychiatry and Neuroscience.
  * Students learning about the trade-offs between planning and habit.
  * AI practitioners interested in biological inspirations for RL.

## 🔑 Key Concepts

The repository focuses on three distinct agent architectures:

  * **Model-Free (MF):** Learns policies by caching long-run values ($Q$) directly from reward prediction errors. It is computationally cheap but inflexible to sudden changes in the environment structure.
  * **Model-Based (MB):** Learns a mental map of the world (Transitions $T$ and Rewards $R$). It computes values "on the fly" by simulating the future. It is flexible but computationally expensive.
  * **Successor Representation (SR):** A hybrid approach. It learns the "predictive map" of future state occupancies ($M$) and combines it with rewards. It solves Reward Revaluation like an MB agent but fails Transition Revaluation like an MF agent.

-----

## 💻 Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/cognitive-rl-benchmarks.git
cd cognitive-rl-benchmarks
pip install -r requirements.txt
```

**Requirements:**

  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`

-----

## 🏆 Featured Task: Momennejad et al. (2017)

**File:** `momennejad_2017.py`

This script reproduces the "Two-Step Revaluation Task" logic. It contrasts agents across three phases:

1.  **Learning:** Agents learn the structure and reward locations.
2.  **Reward Revaluation:** Agents are *told* (or shown in isolation) that the reward has moved to a new state.
      * *Prediction:* MB and SR succeed; MF fails (needs experience).
3.  **Transition Revaluation:** The path structure changes (e.g., the state that led to reward now leads to a loss), but the agent only sees the local change, not the full path.
      * *Prediction:* MB succeeds; SR and MF fail (SR caches long-run path predictions which become stale).

### Running the Simulation

```bash
python momennejad_2017.py
```

*Outputs a seaborn point-plot comparing `p(Best Choice)` for all agents across phases.*

-----

## 📂 Other Algorithms & Tasks

*The repository includes templates for other canonical tasks in the literature:*

### `daw_two_step.py`

**The Daw (2011) Two-Step Task.**

  * **Description:** A probabilistic task where transitions are either "Common" (70%) or "Rare" (30%).
  * **Objective:** Distinguishes MB from MF based on how they update after "Rare" transitions. MB agents boost the action that *produced* the outcome (even if via a rare transition), while MF agents just boost the action they *took*.

### `tolman_detour.py`

**Tolman's Latent Learning & Detour.**

  * **Description:** Agents explore a maze without reward. Later, a reward is introduced, or a barrier blocks the main path.
  * **Objective:** Demonstrates the value of learning $T$ (transitions) without immediate $R$ (reward), a hallmark of Model-Based/Cognitive Map learning.

### `linear_track_sr.py`

**SR Fields in a Linear Track.**

  * **Description:** Visualizes the Successor Representation matrix $M$ as "place fields" that skew backwards from the direction of travel.
  * **Objective:** Visualizing how the SR matrix approximates Hippocampal place cell firing.

-----

## 📚 References

1.  **Momennejad, I., et al. (2017).** *The successor representation in human reinforcement learning.* Nature Human Behaviour.
2.  **Daw, N. D., et al. (2011).** *Model-based influences on humans' choices and striatal prediction errors.* Neuron.
3.  **Gershman, S. J. (2018).** *The Successor Representation: Its Computational Logic and Neural Substrates.* Journal of Neuroscience.
4.  **Sutton, R. S., & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction.* MIT Press.

-----

*This project is for educational and research purposes.*