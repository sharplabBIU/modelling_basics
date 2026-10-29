# Reinforcement Learning for Computational Cognitive Science

**A hands-on tutorial course, from "what is a model?" to fitting model-based, successor-representation, Bayesian and distributional reinforcement-learning models to behaviour and relating their parameters to psychopathology.**

Built for newcomers to the Sharp Lab (Bar-Ilan University), but usable by anyone starting out in computational cognitive science or computational psychiatry. No background in model fitting, parameter recovery, model comparison or Bayesian statistics is assumed; basic Python is enough.

The course is a set of 16 Jupyter notebooks. Every lesson teaches by doing: short explanations with the maths, runnable demonstrations, and **auto-checked exercises** of three kinds (conceptual, maths, coding) that give **gradual feedback** (a diagnosis of what went wrong, up to three hints, and finally the worked solution with the usual mistakes).

## Contents

| # | Lesson | You will be able to … |
|---|--------|------------------------|
| **A. Foundations** |||
| 0 | [Welcome and how the exercises work](notebooks/lesson00_welcome.ipynb) | run the course and use the checker |
| 1 | [Thinking like a modeller](notebooks/lesson01_thinking_like_a_modeller.ipynb) | simulate tasks and agents; follow the modelling cycle |
| 2 | [Probability and Bayes for modellers](notebooks/lesson02_probability_and_bayes.ipynb) | compute likelihoods, posteriors, credible intervals |
| **B. Model-free reinforcement learning** |||
| 3 | [Rescorla–Wagner, prediction errors and softmax](notebooks/lesson03_rescorla_wagner.ipynb) | implement and simulate the delta rule and softmax choice |
| 4 | [Temporal-difference learning, control and dopamine](notebooks/lesson04_temporal_difference.ipynb) | implement TD(0), SARSA, Q-learning, TD(λ), actor–critic; reproduce the dopamine results |
| **C. Fitting and comparing models** |||
| 5 | [Fitting models to behaviour: maximum likelihood](notebooks/lesson05_maximum_likelihood.ipynb) | write a likelihood, optimise it, fit many participants |
| 6 | [Trust but verify: parameter and model recovery](notebooks/lesson06_parameter_and_model_recovery.ipynb) | validate a fit before believing it |
| 7 | [Model comparison](notebooks/lesson07_model_comparison.ipynb) | AIC/BIC, cross-validation, Bayesian model selection, posterior predictive checks |
| 8 | [Bayesian estimation and hierarchical models](notebooks/lesson08_bayesian_estimation.ipynb) | MAP, MCMC by hand, empirical-Bayes EM, PyMC |
| **D. Beyond model-free learning** |||
| 9 | [Model-based RL and the two-step task](notebooks/lesson09_model_based_two_step.ipynb) | implement and fit the hybrid MB/MF model of Daw et al. (2011) |
| 10 | [The successor representation](notebooks/lesson10_successor_representation.ipynb) | learn the SR by TD; run revaluation experiments |
| 11 | [Bayesian learners I: uncertainty and exploration](notebooks/lesson11_bayesian_learners.ipynb) | Beta–Bernoulli learner, Kalman filter, UCB/Thompson, change-point filter |
| 12 | [Bayesian learners II: volatility and the VKF](notebooks/lesson12_volatility_vkf.ipynb) | implement and fit the volatile Kalman filter (Piray & Daw 2020) |
| 13 | [Distributional reinforcement learning](notebooks/lesson13_distributional_rl.ipynb) | expectile/quantile TD, decoding reward distributions, asymmetric learning rates |
| **E. Computational psychiatry** |||
| 14 | [Linking parameters to psychopathology](notebooks/lesson14_computational_psychiatry.ipynb) | relate fitted parameters to symptom scores without fooling yourself |
| 15 | [Capstone project](notebooks/lesson15_capstone.ipynb) | run a complete analysis end to end and write it up |

Dependencies between lessons:

```
1 → 2 → 3 → 4
        3 → 5 → 6 → 7 → 8
        4 → 9 → 10
        2 → 11 → 12
        3 → 13
   5–8, 9, 12, 13 → 14 → 15
```

A minimal core is lessons 1–3, 5–7 and 9 (about two working days); the rest can be taken in any order afterwards. The full course is roughly 30–40 hours.

## Getting started

```bash
git clone git@github.com:sharplabBIU/modelling_basics.git
cd modelling_basics
pip install -r requirements.txt      # numpy, scipy, matplotlib, pandas, jupyterlab, ... (+ optional pymc)
jupyter lab                          # then open notebooks/lesson00_welcome.ipynb
```

Open the notebooks from inside the repository (the setup cell of every notebook adds the repository root to the Python path), or run `pip install -e .` once to make the `rlcourse` package importable from anywhere.

**Optional Bayesian sections** (lessons 8 and 14) use PyMC. They are wrapped so that the notebooks run without it. On recent macOS, PyTensor's C compiler may fail; the notebooks therefore set `PYTENSOR_FLAGS=cxx=` and sample with the JAX/numpyro backend (`pip install numpyro jax`).

## How the exercises work

```python
check("3.1", rw_update)      # coding: tests your function against the reference on several cases
check("3.2", 0.48)           # maths: checks your number (with a tolerance)
check("3.3", "b")            # conceptual: multiple choice with feedback per option
check("3.10", "my answer")   # written: keyword feedback, then a model answer
hint("3.1"); reveal("3.1"); reset("3.1"); progress()
```

Feedback is gradual. A wrong answer gets a **diagnosis** of what probably went wrong (the checker recognises common bugs such as a flipped prediction error, a forgotten normalisation or an off-by-one in a trial loop) and the **first hint**; later attempts unlock the remaining hints; after the hints are exhausted the **worked solution** is shown with the mistakes students most commonly make. Cells that depend on an exercise use `use("3.1", rw_update)`, which substitutes the reference implementation when your version is not passing yet, so a stuck exercise never blocks the rest of a lesson. Progress is saved in `notebooks/.rlcourse_progress.json`.

## What is in the repository

```
notebooks/      the 16 lessons (generated from lessons_src/)
lessons_src/    plain-text sources of the notebooks (edit these, then python scripts/nbtools.py)
rlcourse/       the course package
  checker.py      exercise checker with gradual feedback
  exercises/      the exercises of each lesson (tests, hints, solutions, common mistakes)
  models.py       reference implementations: RW, TD, SARSA, Q-learning, actor–critic, two-step
                  hybrid MB/MF, successor representation, Beta/Kalman/change-point learners,
                  Pearce–Hall, volatile Kalman filter, expectile/quantile TD, ...
  fitting.py      MLE with restarts, parameter transforms, AIC/BIC, Bayesian model selection
                  (exceedance and protected exceedance probabilities), MAP, Laplace,
                  Metropolis–Hastings, empirical-Bayes EM, reliability helpers
  tasks.py        bandits, restless bandits, reversal (stable/volatile), the two-step task,
                  small MDPs (chain, grid world, three-branch revaluation graph), Pavlovian trace
  plotting.py     course style and convenience plots
  datasets.py     simulated participant data sets with symptom scores (see data/)
data/           simulated data sets (two-step + compulsivity, reversal + anxiety, bandit + anhedonia)
docs/           AUTHORING.md (how lessons are built), SYLLABUS.md (lesson outlines), READING_LIST.md
scripts/        nbtools.py (build notebooks), run_notebooks.py (execute & test), make_datasets.py
tests/          tests of the checker and of every exercise's reference solution
```

**Every model implementation follows a specific paper** (named in the docstrings and in the lesson text) and is accompanied by the reference: the two-step hybrid model follows Daw et al. (2011), the successor representation Dayan (1993), the volatile Kalman filter Piray & Daw (2020) equation by equation, distributional TD Dabney et al. (2020), Bayesian model selection Stephan et al. (2009) and Rigoux et al. (2014), the empirical-Bayes procedure Huys et al. (2011) and Gershman (2016), and so on. The full list is in [docs/READING_LIST.md](docs/READING_LIST.md).

**The data sets are simulated.** The relations between parameters and symptom scores built into them are *inspired by* published findings (Gillan et al. 2016; Browning et al. 2015; Huys et al. 2013; Lefebvre et al. 2017) so that the computational-psychiatry lessons have something realistic to analyse, and the generating parameters are provided so that every analysis can be checked against the ground truth.

## For instructors and contributors

- Lessons are written as plain-text sources in `lessons_src/` (one file per lesson, a simple cell-marker format) and built with `python scripts/nbtools.py`. Exercises are defined in `rlcourse/exercises/lessonNN.py`.
- `python scripts/run_notebooks.py` executes every notebook twice: once as a student receives it (every cell must run even with the exercises unsolved) and once with the registered solutions substituted (every check must pass). `python tests/test_checker.py` and `python tests/test_exercises.py` test the checker and the exercises.
- [docs/AUTHORING.md](docs/AUTHORING.md) explains the conventions in detail; [docs/SYLLABUS.md](docs/SYLLABUS.md) outlines every lesson.

## Acknowledgements

The course follows the workflow of Wilson & Collins (2019), *Ten simple rules for the computational modeling of behavioral data*, and draws on the tutorial chapter by Daw (2011). The PyMC section adapts the example by Ricardo Vieira (2022, MIT licence). All sources are listed in [docs/READING_LIST.md](docs/READING_LIST.md).
