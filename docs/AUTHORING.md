# Authoring guide: how the lessons are built

This document is the specification every lesson follows.  Read it before
editing or adding a lesson.

## 1. Repository layout

```
lessons_src/      plain-text sources of the notebooks (one file per lesson)   <- edit these
notebooks/        generated notebooks (python scripts/nbtools.py)              <- students open these
rlcourse/         the teaching package
  checker.py      the exercise checker (gradual feedback)
  exercises/      one module per lesson: lessonNN.py defines EXERCISES = [...]
  models.py       reference implementations (= the revealed solutions)
  fitting.py      MLE / MAP / EM / MCMC / model comparison helpers
  tasks.py        simulated tasks (bandits, reversal, two-step, MDPs, Pavlovian)
  plotting.py     course style + convenience plots
  datasets.py     synthetic participant data sets (data/*.csv)
scripts/          nbtools.py (build), run_notebooks.py (test), make_datasets.py
tests/            test_checker.py (feedback flow), test_exercises.py (solutions pass)
docs/             this guide, the reading list
```

Build and test:

```bash
python scripts/nbtools.py                      # lessons_src/*.txt -> notebooks/*.ipynb
python scripts/run_notebooks.py lesson03       # stub pass + solution pass
python tests/test_checker.py && python tests/test_exercises.py
```

## 2. The course map

| # | Lesson | Core content | Key sources |
|---|--------|--------------|-------------|
| 00 | Welcome & how the exercises work | setup, feedback system, demo exercises | — |
| 01 | Thinking like a modeller | tasks, agents, simulation, generative models, the modelling cycle | Wilson & Collins 2019; Palminteri et al. 2017 |
| 02 | Probability & Bayes for modellers | distributions, likelihood, log-likelihood, MLE of a coin, Bayes on a grid, Beta–Bernoulli, priors/posteriors/predictives | Lee & Wagenmakers 2013; McElreath 2020 |
| 03 | Model-free learning I | Rescorla–Wagner, prediction errors, learning rate, softmax, exploration | Rescorla & Wagner 1972; Sutton & Barto 2018; Niv 2009; Daw 2011 |
| 04 | Model-free learning II | TD(0), SARSA, Q-learning, eligibility traces, actor–critic, dopamine RPE | Sutton 1988; Watkins & Dayan 1992; Schultz, Dayan & Montague 1997; Joel, Niv & Ruppin 2002 |
| 05 | Fitting models to behaviour | likelihood of a choice sequence, MLE, optimisation, transforms, multiple starts, likelihood surfaces | Daw 2011; Wilson & Collins 2019 |
| 06 | Trust but verify | parameter recovery, identifiability, model recovery, confusion matrices | Wilson & Collins 2019; Palminteri et al. 2017 |
| 07 | Model comparison | AIC, BIC, cross-validation, random-effects BMS, (protected) exceedance probabilities, posterior predictive checks | Akaike 1974; Schwarz 1978; Stephan et al. 2009; Rigoux et al. 2014; Vehtari et al. 2017 |
| 08 | Bayesian parameter estimation | priors, MAP, Laplace, Metropolis–Hastings, hierarchical models, empirical-Bayes EM, PyMC (optional), HBI | Huys et al. 2011; Gershman 2016; Piray et al. 2019; Ahn et al. 2017 |
| 09 | Model-based RL and the two-step task | planning with a model, the hybrid MB/MF model, stay probabilities, fitting w | Daw et al. 2005, 2011; Gläscher et al. 2010; Gillan et al. 2016; Akam et al. 2015 |
| 10 | The successor representation | SR definition, SR-TD, reward vs transition revaluation, SR in humans | Dayan 1993; Momennejad et al. 2017; Russek et al. 2017; Gershman 2018 |
| 11 | Bayesian learners I | Beta–Bernoulli learner, Kalman filter, uncertainty-driven exploration, change-point filter, Pearce–Hall | Daw et al. 2006; Gershman 2015, 2018; Behrens et al. 2007; Nassar et al. 2010; Pearce & Hall 1980 |
| 12 | Bayesian learners II: volatility | the VKF (binary & continuous), HGF overview, fitting VKF vs RW, learning-rate adaptation | Piray & Daw 2020; Mathys et al. 2011, 2014; Behrens et al. 2007; Browning et al. 2015 |
| 13 | Distributional RL | expectile/quantile TD, asymmetric learning rates, decoding distributions, dopamine evidence, links to mood | Dabney et al. 2020; Bellemare et al. 2017; Dabney et al. 2018; Lowet et al. 2020; Lefebvre et al. 2017 |
| 14 | Computational psychiatry | parameters as phenotypes, regression on fitted parameters, reliability, pitfalls, embedded covariates, power by simulation | Huys et al. 2016; Gillan et al. 2016; Browning et al. 2015; Huys et al. 2013; Eckstein et al. 2021; Brown et al. 2020; Zorowitz & Niv 2023 |
| 15 | Capstone | end-to-end project on a provided data set | all of the above |

## 3. Notebook structure (every lesson)

1. **Title cell**: `# Lesson N — Title`, a 2–4 sentence motivation, then
   *What you will learn* (bullets), *Prerequisites* (lesson numbers),
   *Estimated time*.
2. **Setup cell** (identical in every lesson):
   ```python
   import sys; sys.path.insert(0, "..")      # find the rlcourse package when running from notebooks/
   import numpy as np
   import matplotlib.pyplot as plt
   from rlcourse import check, hint, reveal, progress, use, setup_style
   from rlcourse import models, tasks, fitting, plotting
   setup_style()
   rng = np.random.default_rng(2024)        # one seed per notebook: results are reproducible
   ```
3. **Sections** that alternate *explanation* (markdown with LaTeX), *demonstration*
   (code that runs and plots), and *exercises*.  Explain every symbol the first
   time it appears.  Keep code cells short (< 30 lines) and commented.
4. **Exercises** (6–12 per lesson, see §4).  Mix the three types in every
   lesson: *conceptual* (choice / free text), *maths* (numeric), *coding*.
5. **Summary** (bullets), **References** (full citations with DOIs, in the
   format of §6), **What's next**.

Rules:

* One seed (`rng`) per notebook; no `np.random.seed`.  Any cell may be re-run
  in any order after the setup cell without breaking (except that exercise
  cells must be run before cells that `use()` them).
* Every cell must run in the *stub* pass (unsolved exercises).  Cells that use
  a student's function go *after* its check and obtain it through
  `fn = use("N.k", fn)` so that the reference implementation is substituted if
  the exercise is unfinished.
* Runtime: the whole notebook must execute in < 3 minutes in the solution pass
  (keep simulations small; 20–60 simulated subjects, 100–300 trials).
* Figures: use the course style (`setup_style()`), label axes with units,
  title every figure, legend when > 1 series.  System colours: MF blue, MB
  orange, SR aqua (`plotting.SYSTEM_COLORS`).
* No internet access is needed.  Optional dependencies (PyMC) are wrapped in
  `try/except ImportError` with a clear message and the lesson still works.
* Tone: friendly, precise, no hand-waving.  Say *why* before *how*.  Name the
  paper that introduced each idea.
* Mathematics: define quantities with equations in display mode, then show the
  same thing in code.  Use the course notation (§5).

## 4. Exercises

### 4.1 In the notebook

A markdown cell introduces the exercise:

```
### ✏️ Exercise 3.1 — The Rescorla–Wagner update  *(coding)*
Implement ... (what, inputs, outputs, and one sentence on why it matters).
```

followed by **one code cell** marked `# %% [exercise 3.1]` in the source that
contains the stub and, as its last line, the check call:

```python
def rw_update(value, reward, alpha):
    """Return the updated value after observing `reward`."""
    prediction_error = ...            # <- fill in
    new_value = ...                   # <- fill in
    return new_value

check("3.1", rw_update)
```

Conventions for stubs:
* early / short exercises: fill-in-the-blank with `...` placeholders;
* longer exercises: a docstring and `raise NotImplementedError("your code here")`;
* numeric: `answer = None   # <- your number`, then `check("3.2", answer)`;
* multiple choice: `check("3.3", None)   # <- replace None with "a", "b", "c" or "d"`;
  the options are listed in the markdown cell as **(a)** ... **(b)** ...;
* free text: `check("3.4", """ your answer """)`.

### 4.2 In `rlcourse/exercises/lessonNN.py`

```python
from ..checker import CodeExercise, NumericExercise, ChoiceExercise, FreeTextExercise
from .. import models

EXERCISES = [
    CodeExercise(
        id="3.1", title="The Rescorla–Wagner update",
        reference=models.rw_update,                       # its source is the revealed solution
        signature="rw_update(value, reward, alpha)",
        cases=[{"value": 0.5, "reward": 1.0, "alpha": 0.1}, ...],   # deterministic!
        bug_variants={                                    # common mistakes -> targeted message
            "raw_reward": (lambda value, reward, alpha: value + alpha * reward,
                           "You added alpha × reward instead of alpha × (reward − value)..."),
        },
        hints=["...", "...", "..."],                      # 3 hints, from gentle to explicit
        explanation="Why the solution works / the insight.",
        common_mistakes="What usually goes wrong (shown with the solution).",
        check_call='check("3.1", rw_update)',             # must match the notebook cell
    ),
    NumericExercise(id="3.2", title="...", answer=0.6, atol=1e-3,
                    distractors=[(0.5, "That is the old value: you forgot to apply the update."), ...],
                    hints=[...], explanation="...", check_call='check("3.2", answer)'),
    ChoiceExercise(id="3.3", title="...", options={"a": "...", "b": "...", "c": "..."}, correct="b",
                   feedback={"a": "why a is wrong", "c": "why c is wrong"},
                   hints=[...], explanation="...", check_call='check("3.3", "b")'),
    FreeTextExercise(id="3.4", title="...", model_answer="...",
                     concepts=[("prediction error", ["prediction error", "delta", "surprise"]), ...],
                     min_concepts=2, hints=[...], explanation="...",
                     check_call='check("3.4", """<model answer>""")'),
]
```

Requirements:
* `cases` must be deterministic (seeded `np.random.default_rng(k)` objects are
  fine — they are deep-copied for the student and the reference).
* Stochastic simulators are compared with a **consistency check**, not by
  matching random streams: write a `compare(out, ref, case)` or `diagnose(out,
  ref, case)` that verifies the student's output obeys the model (see lesson 3,
  exercise 3.8 and `models.simulation_is_consistent`).
* Each exercise has ≥ 2 hints (3 is the norm), an `explanation`, and
  `common_mistakes` for coding exercises.  Hints go from *conceptual nudge* →
  *where to look* → *almost the code*.
* `check_call` must be exactly the last line of the notebook cell.
* **Never put an `import` (or any other code the lesson needs later) inside an
  exercise cell**: the solution pass replaces the whole cell with the reference
  source plus the check call.  Imports that a reference needs (e.g. `softmax`,
  `choose`, `sigmoid`) go in a separate code cell *before* the exercise.
* `solution_cell` (automatic) = reference source + check call; the solution is
  what `scripts/run_notebooks.py --mode solution` executes, so the reference
  function must be importable and self-contained (uses only numpy and other
  `rlcourse` functions already imported in the notebook, e.g. `models.softmax`).
  If a solution needs a helper, make the helper part of `rlcourse.models`.
* Reference functions that are revealed must be written as a student would
  write them: explicit, commented, no cleverness.

### 4.3 How feedback unfolds (the checker's contract)

* attempt 1 fails → diagnosis (bug variant / generic) + hint 1
* attempt 2 fails → diagnosis + hint 2
* attempt 3 fails → diagnosis + hint 3
* attempt 4 fails → the worked solution + explanation + common mistakes
* any pass → "correct" + explanation (+ model answer for free text)
* `hint(id)`, `reveal(id)`, `reset(id)`, `progress()` are always available.

## 5. Notation and code conventions

| symbol | code | meaning |
|---|---|---|
| $V$, $Q(a)$ | `V`, `Q` | value of a state / action |
| $\delta$ | `delta`, `prediction_error` | reward prediction error |
| $\alpha$ | `alpha` | learning rate ∈ (0, 1) |
| $\beta$ | `beta` | inverse temperature > 0 |
| $\gamma$ | `gamma` | discount factor ∈ [0, 1] |
| $\lambda$ | `lam` | eligibility trace decay (lesson 4, 9) / volatility learning rate (lesson 12) |
| $w$ | `w` | model-based weight ∈ [0, 1] |
| $p$ / $\kappa$ | `p`, `kappa` | perseveration / stickiness |
| $\theta$ | `params`, `theta` | parameter vector (constrained space) |
| $x$ | `x` | parameters in unconstrained space (`fitting.to_constrained`) |

* Choices are 0-based ints; rewards floats; trial loops are explicit `for t in range(n_trials)`.
* Likelihoods: `nll_model(params, *data)` returns the negative log-likelihood.
* Parameter kinds for fitting: `"unit"` (sigmoid), `"pos"` (exp), `"real"`.

## 6. References

Cite primary sources (the paper that introduced the model) *and* a tutorial
source when one exists.  Format in the notebooks:

> Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., & Dolan, R. J. (2011).
> Model-based influences on humans' choices and striatal prediction errors.
> *Neuron, 69*(6), 1204–1215. https://doi.org/10.1016/j.neuron.2011.02.027

Only cite works you are sure exist, with the correct authors, year and venue.
When an implementation follows a specific equation of a paper, say so in the
code comment (e.g. `# Piray & Daw 2020, Eq. 19`).  The full reading list is in
`docs/READING_LIST.md`.
