"""Exercises of lesson 11: Bayesian learners I — uncertainty-aware learning and exploration."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise


def _steady_state_gain(process_var, obs_var):
    """Steady-state Kalman gain (used to define the numeric answer of 11.5)."""
    v = (-process_var + np.sqrt(process_var ** 2 + 4 * process_var * obs_var)) / 2   # steady-state posterior variance
    return (v + process_var) / (v + process_var + obs_var)


_CP_OUTCOMES_A = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=float)
_CP_OUTCOMES_B = (np.random.default_rng(110).random(30) < np.r_[np.full(15, 0.8), np.full(15, 0.2)]).astype(float)
_GRID = np.linspace(0.01, 0.99, 50)


def _cp_no_hazard(outcomes, hazard, grid):
    grid = np.asarray(grid, float)
    post = np.full(len(grid), 1.0 / len(grid))
    predictions = np.zeros(len(outcomes))
    posteriors = np.zeros((len(outcomes), len(grid)))
    for t, o in enumerate(outcomes):
        predictions[t] = np.sum(post * grid)
        post = post * (grid if o == 1 else 1 - grid)
        post = post / np.sum(post)
        posteriors[t] = post
    return predictions, posteriors


def _cp_predict_after(outcomes, hazard, grid):
    grid = np.asarray(grid, float)
    post = np.full(len(grid), 1.0 / len(grid))
    predictions = np.zeros(len(outcomes))
    posteriors = np.zeros((len(outcomes), len(grid)))
    for t, o in enumerate(outcomes):
        prior = (1 - hazard) * post + hazard / len(grid)
        post = prior * (grid if o == 1 else 1 - grid)
        post = post / np.sum(post)
        predictions[t] = np.sum(post * grid)
        posteriors[t] = post
    return predictions, posteriors


def _cp_hazard_on_grid(outcomes, hazard, grid):
    grid = np.asarray(grid, float)
    post = np.full(len(grid), 1.0 / len(grid))
    predictions = np.zeros(len(outcomes))
    posteriors = np.zeros((len(outcomes), len(grid)))
    for t, o in enumerate(outcomes):
        prior = (1 - hazard) * post + hazard          # forgot to divide the hazard mass by the grid size
        prior = prior / prior.sum()
        predictions[t] = np.sum(prior * grid)
        post = prior * (grid if o == 1 else 1 - grid)
        post = post / np.sum(post)
        posteriors[t] = post
    return predictions, posteriors


EXERCISES = [
    # ------------------------------------------------------------------ 11.1
    CodeExercise(
        id="11.1", title="The Beta–Bernoulli update",
        reference=models.beta_update,
        signature="beta_update(a, b, reward)",
        cases=[{"a": 1, "b": 1, "reward": 1}, {"a": 3, "b": 2, "reward": 0}, {"a": 2.5, "b": 4.0, "reward": 1.0},
               {"a": 10, "b": 10, "reward": 0}],
        bug_variants={
            "swapped": (lambda a, b, reward: (a + (1 - reward), b + reward),
                        "You added the reward to b and the non-reward to a — it is the other way round: a counts successes (reward = 1), b counts failures."),
            "both": (lambda a, b, reward: (a + reward, b + reward),
                     "You added the reward to both counts. A reward increments a; a non-reward increments b: `b + (1 - reward)`."),
            "only_a": (lambda a, b, reward: (a + reward, b),
                       "You never update b. When the reward is 0 the failure count b must go up by one."),
        },
        hints=["A Beta(a, b) belief is a pair of counts: a pseudo-successes and b pseudo-failures.",
               "A reward of 1 adds one success; a reward of 0 adds one failure.",
               "`return a + reward, b + (1 - reward)`"],
        explanation="Bayesian updating of a Bernoulli probability is just counting: a += reward, b += 1 − reward "
                    "(lesson 2). The posterior mean a/(a + b) then moves by (reward − mean)/(a + b + 1) — a "
                    "Rescorla–Wagner update whose learning rate shrinks as evidence accumulates.",
        common_mistakes="Swapping the roles of a and b, or forgetting to increment b on a non-reward.",
        check_call='check("11.1", beta_update)',
    ),
    # ------------------------------------------------------------------ 11.2
    NumericExercise(
        id="11.2", title="The learning rate of a Bayesian learner",
        answer=1 / 11, atol=1e-4,
        distractors=[(1 / 12, "Off by one: before the 9th outcome the learner has seen 8 outcomes, so a + b = 2 + 8 = 10 and the rate is 1/(10 + 1)."),
                     (1 / 10, "You forgot the +1: the posterior mean moves by (r − m)/(a + b + 1), where a + b is the count *before* the update."),
                     (1 / 9, "You ignored the two pseudo-counts of the Beta(1, 1) prior: a + b starts at 2, not 0.")],
        hints=["Before the 9th outcome the learner has updated 8 times, starting from a + b = 2.",
               "The posterior mean a/(a + b) changes by (reward − mean)/(a + b + 1), so the effective learning rate is 1/(a + b + 1).",
               "1 / (2 + 8 + 1)"],
        explanation="Starting from Beta(1, 1), after 8 outcomes a + b = 10, so the 9th outcome moves the mean by "
                    "1/11 of the prediction error. The learning rate falls as 1/(n + 3) with the number of outcomes "
                    "n: exactly the behaviour that is optimal for a *stationary* probability and disastrous for a "
                    "changing one.",
        check_call='check("11.2", 1 / 11)',
    ),
    # ------------------------------------------------------------------ 11.3
    CodeExercise(
        id="11.3", title="One Kalman-filter update",
        reference=models.kalman_update,
        signature="kalman_update(mu, var, reward, obs_var, process_var=0.0)",
        cases=[{"mu": 0.0, "var": 1.0, "reward": 1.0, "obs_var": 1.0, "process_var": 0.0},
               {"mu": 0.3, "var": 0.2, "reward": 1.0, "obs_var": 0.5, "process_var": 0.1},
               {"mu": 0.5, "var": 0.05, "reward": 0.0, "obs_var": 1.0, "process_var": 0.0},
               {"mu": -1.0, "var": 2.0, "reward": 0.5, "obs_var": 0.25, "process_var": 0.3}],
        bug_variants={
            "no_process_in_gain": (lambda mu, var, reward, obs_var, process_var=0.0: (lambda k: (mu + k * (reward - mu), (1 - k) * (var + process_var), k))(var / (var + obs_var)),
                                   "The gain must use the *prior* variance for this trial, var + process_var: the belief first diffuses, then it is updated."),
            "no_process_in_var": (lambda mu, var, reward, obs_var, process_var=0.0: (lambda k: (mu + k * (reward - mu), (1 - k) * var, k))((var + process_var) / (var + process_var + obs_var)),
                                  "The updated variance is (1 − k) × (var + process_var), not (1 − k) × var: the process noise has to be added before the update."),
            "inverted_gain": (lambda mu, var, reward, obs_var, process_var=0.0: (lambda k: (mu + k * (reward - mu), (1 - k) * (var + process_var), k))(obs_var / (var + process_var + obs_var)),
                              "Your gain is obs_var / (prior var + obs_var), i.e. the share of the *observation* noise. The gain is the share of the prior uncertainty: (var + process_var) / (var + process_var + obs_var)."),
            "var_not_updated": (lambda mu, var, reward, obs_var, process_var=0.0: (lambda k: (mu + k * (reward - mu), var, k))((var + process_var) / (var + process_var + obs_var)),
                                "You returned the old variance. After seeing an outcome the uncertainty shrinks: var_new = (1 − k) × (var + process_var)."),
        },
        hints=["First let the belief diffuse: the variance for this trial is var + process_var.",
               "The Kalman gain is k = prior_var / (prior_var + obs_var); the mean moves by k × (reward − mu) — a prediction error scaled by k.",
               "`prior_var = var + process_var; k = prior_var / (prior_var + obs_var); return mu + k * (reward - mu), (1 - k) * prior_var, k`"],
        explanation="The Kalman filter (Kalman 1960) is the exact Bayesian update for a Gaussian belief about a "
                    "drifting quantity observed with Gaussian noise. Its gain k plays the role of the learning rate: "
                    "it is the fraction of the total uncertainty that is due to the learner's own uncertainty rather "
                    "than to the noise of the observation (Gershman 2015 calls this model the Kalman Rescorla–Wagner).",
        common_mistakes="Forgetting to add the process variance before computing the gain, inverting the gain, or not "
                        "shrinking the variance after the update.",
        check_call='check("11.3", kalman_update)',
    ),
    # ------------------------------------------------------------------ 11.4
    NumericExercise(
        id="11.4", title="The first Kalman gain",
        answer=0.5, atol=1e-4,
        distractors=[(1.0, "A gain of 1 would mean the learner ignores its prior entirely; that happens only when the prior variance is infinite."),
                     (0.25, "You may have multiplied the two variances or squared something; the gain is a ratio of variances."),
                     (1 / 3, "That is obs_var / (prior_var + obs_var) if you added a process variance of 1 — but here there is no process noise.")],
        hints=["k = prior variance / (prior variance + observation variance), with prior variance = var + process_var.",
               "1 / (1 + 1)"],
        explanation="With equal prior and observation variance the learner trusts its prior and the new outcome "
                    "equally, so it moves halfway: k = 0.5. After that update the variance halves, so the next gain "
                    "is 1/3, then 1/4, ... — the 1/n schedule of a running mean.",
        check_call='check("11.4", 0.5)',
    ),
    # ------------------------------------------------------------------ 11.5
    NumericExercise(
        id="11.5", title="Where does the learning rate settle?",
        answer=float(_steady_state_gain(0.1, 1.0)), atol=2e-3,
        distractors=[(0.1, "That is the process variance itself; the steady-state gain depends on both variances."),
                     (0.5, "That is the gain on the very first update, before the uncertainty has shrunk."),
                     (float((-0.1 + np.sqrt(0.01 + 0.4)) / 2 + 0.1), "That is the steady-state *prior* variance (v + q); the gain is that divided by (v + q + obs_var)."),
                     (0.0, "With process noise the gain cannot go to zero: the belief keeps diffusing, so there is always something to learn.")],
        hints=["At steady state the posterior variance v no longer changes: v = (1 − k)(v + q) with k = (v + q)/(v + q + σ²), where q is the process variance and σ² the observation variance.",
               "Either solve v² + q v − q σ² = 0 for v and then compute k, or simply iterate your `kalman_update` for 100 trials with any reward sequence and read off the gain.",
               "v = (−q + √(q² + 4 q σ²)) / 2 ≈ 0.270 and k = (v + q) / (v + q + σ²)"],
        explanation="With process variance 0.1 and observation variance 1 the gain settles at ≈ 0.27: every trial "
                    "the belief forgets a little (the variance grows by 0.1), and that steady leak keeps the learner "
                    "responsive. A Kalman learner in a drifting world therefore behaves like a Rescorla–Wagner "
                    "learner with α ≈ 0.27 once the initial uncertainty has been resolved — the two models differ "
                    "mainly early on and after changes.",
        check_call='check("11.5", 0.2702)',
    ),
    # ------------------------------------------------------------------ 11.6
    CodeExercise(
        id="11.6", title="Upper-confidence-bound values",
        reference=models.ucb_values,
        signature="ucb_values(mu, var, c)",
        cases=[{"mu": np.array([0.5, 0.4]), "var": np.array([0.01, 0.25]), "c": 1.0},
               {"mu": np.array([0.0, 0.0, 0.0]), "var": np.array([1.0, 4.0, 0.25]), "c": 2.0},
               {"mu": [0.7, 0.6, 0.65], "var": [0.04, 0.09, 0.0], "c": 0.5}],
        bug_variants={
            "variance_not_sd": (lambda mu, var, c: np.asarray(mu, float) + c * np.asarray(var, float),
                                "You added c × variance. The bonus is c × *standard deviation*: take the square root of the variance."),
            "minus": (lambda mu, var, c: np.asarray(mu, float) - c * np.sqrt(np.asarray(var, float)),
                      "You subtracted the bonus — that would make the agent *avoid* uncertain options (a lower confidence bound)."),
            "c_inside": (lambda mu, var, c: np.asarray(mu, float) + np.sqrt(c * np.asarray(var, float)),
                         "c should multiply the standard deviation, not sit inside the square root."),
        },
        hints=["Optimism in the face of uncertainty: value each option by its mean *plus* a bonus that grows with how uncertain you are about it.",
               "The bonus is c times the standard deviation, i.e. c × sqrt(var).",
               "`return np.asarray(mu, float) + c * np.sqrt(np.asarray(var, float))`"],
        explanation="UCB (Auer, Cesa-Bianchi & Fischer 2002) adds an *information bonus* to uncertain options, so "
                    "the agent deliberately samples what it knows least about — directed exploration (Wilson et al. "
                    "2014). As the uncertainty of an option shrinks, so does its bonus, and the agent exploits.",
        common_mistakes="Using the variance instead of the standard deviation, or subtracting the bonus.",
        check_call='check("11.6", ucb_values)',
    ),
    # ------------------------------------------------------------------ 11.7
    ChoiceExercise(
        id="11.7", title="Which strategy is *directed* exploration?",
        question="Which of these choice strategies implements *directed* exploration — a deliberate preference for options the agent is uncertain about?",
        options={"a": "A softmax on the posterior means with a small inverse temperature.",
                 "b": "Upper-confidence-bound choice: pick the option with the largest mean + c × sd.",
                 "c": "Epsilon-greedy choice.",
                 "d": "Thompson sampling: draw one sample per option from its posterior and pick the largest."},
        correct="b",
        feedback={"a": "A low inverse temperature adds *random* exploration: it does not know which options are uncertain.",
                  "c": "Epsilon-greedy explores uniformly at random, regardless of uncertainty.",
                  "d": "Thompson sampling does use uncertainty — but through randomness in the samples. Gershman (2018) "
                       "classifies it as uncertainty-dependent *random* exploration; directed exploration adds a "
                       "deterministic bonus for uncertainty."},
        hints=["Directed exploration means the agent is drawn *towards* uncertain options on purpose; random exploration means it just adds noise to its choices.",
               "Which rule contains an explicit term that rewards uncertainty?"],
        explanation="Wilson et al. (2014) showed that humans use both kinds: an information bonus for uncertain "
                    "options (directed) and more choice variability when uncertain (random). In Gershman's (2018) "
                    "hybrid model the UCB bonus captures the former and Thompson-like sampling the latter; fitted "
                    "to human bandit data both components are present.",
        check_call='check("11.7", "b")',
    ),
    # ------------------------------------------------------------------ 11.8
    CodeExercise(
        id="11.8", title="A change-point filter on a grid",
        reference=models.grid_changepoint_filter,
        signature="grid_changepoint_filter(outcomes, hazard, grid)",
        cases=[{"outcomes": _CP_OUTCOMES_A, "hazard": 0.1, "grid": _GRID},
               {"outcomes": _CP_OUTCOMES_B, "hazard": 0.05, "grid": np.linspace(0.02, 0.98, 25)},
               {"outcomes": np.array([1.0, 0.0, 1.0]), "hazard": 0.0, "grid": _GRID}],
        bug_variants={
            "no_hazard": (_cp_no_hazard, "Your filter never allows a change-point: the prediction step `prior = (1 − h) × post + h / n_grid` is missing, so the belief just accumulates evidence forever."),
            "predict_after": (_cp_predict_after, "Your `predictions[t]` are computed *after* seeing outcome t. The prediction for trial t must use the belief before that outcome (right after the change-point step)."),
            "hazard_mass": (_cp_hazard_on_grid, "The hazard mass must be spread over the grid: add `hazard / len(grid)` to every grid point, not `hazard`."),
        },
        hints=["Each trial has two steps. (1) Predict: with probability `hazard` the probability was re-drawn uniformly, so `prior = (1 - hazard) * post + hazard / len(grid)`; the prediction is `np.sum(prior * grid)`.",
               "(2) Update: multiply the prior by the likelihood of the outcome at every grid point (`grid` if the outcome was 1, `1 - grid` otherwise) and normalise.",
               "Store the prediction *before* the update and the normalised posterior *after* it, for every trial."],
        explanation="This is exact Bayesian inference for a probability that occasionally jumps (a simplified "
                    "version of the model of Nassar et al. 2010 and of the ideal observer of Behrens et al. 2007, "
                    "with the hazard rate known). The predict step injects a little uniform belief every trial, so "
                    "surprising outcomes can quickly shift the posterior — which is why the *effective* learning rate "
                    "of this learner spikes after a change and decays during stable periods.",
        common_mistakes="Omitting the change-point (prediction) step, recording the prediction after the update, or "
                        "adding the hazard mass without dividing it by the number of grid points.",
        check_call='check("11.8", grid_changepoint_filter)',
    ),
    # ------------------------------------------------------------------ 11.9
    CodeExercise(
        id="11.9", title="The Pearce–Hall hybrid",
        reference=models.pearce_hall_update,
        signature="pearce_hall_update(v, assoc, reward, eta, kappa)",
        cases=[{"v": 0.5, "assoc": 1.0, "reward": 1.0, "eta": 0.3, "kappa": 0.5},
               {"v": 0.8, "assoc": 0.2, "reward": 0.0, "eta": 0.5, "kappa": 0.8},
               {"v": 0.1, "assoc": 0.6, "reward": 0.0, "eta": 0.1, "kappa": 1.0},
               {"v": 0.4, "assoc": 0.4, "reward": 1.0, "eta": 1.0, "kappa": 0.2}],
        bug_variants={
            "signed": (lambda v, assoc, reward, eta, kappa: (lambda d: (v + kappa * assoc * d, eta * d + (1 - eta) * assoc, d))(reward - v),
                       "The associability tracks the *absolute* prediction error |delta|: a negative surprise is still a surprise."),
            "new_assoc_first": (lambda v, assoc, reward, eta, kappa: (lambda d: (lambda a_new: (v + kappa * a_new * d, a_new, d))(eta * abs(d) + (1 - eta) * assoc))(reward - v),
                                "You used the *updated* associability in the value update. In Pearce–Hall the current trial's surprise changes the learning rate for the *next* trial: update v with the old associability, then update the associability."),
            "swapped": (lambda v, assoc, reward, eta, kappa: (lambda d: (v + eta * assoc * d, kappa * abs(d) + (1 - kappa) * assoc, d))(reward - v),
                        "You swapped eta and kappa: kappa scales the value update (v += kappa × assoc × delta) and eta is the rate at which the associability tracks |delta|."),
        },
        hints=["Compute the prediction error delta = reward − v first.",
               "Value: v_new = v + kappa × assoc × delta (the current associability scales learning). Associability: assoc_new = eta × |delta| + (1 − eta) × assoc.",
               "Return `(v_new, assoc_new, delta)`, updating v with the *old* associability."],
        explanation="Pearce & Hall (1980) proposed that a cue's associability — how much is learned about it — "
                    "rises when its outcomes are surprising and falls when they are well predicted. The resulting "
                    "learning rate kappa × assoc is high after changes and low in stable periods: a cheap heuristic "
                    "version of the Bayesian learners of this lesson, and one that fits conditioning data well "
                    "(Li et al. 2011).",
        common_mistakes="Using the signed instead of the absolute prediction error, updating the associability before the value, or swapping eta and kappa.",
        check_call='check("11.9", pearce_hall_update)',
    ),
    # ------------------------------------------------------------------ 11.10
    FreeTextExercise(
        id="11.10", title="Why the learning rate should change",
        question="In 2–4 sentences and in Bayesian terms: why should the learning rate be high just after a change in the environment and low during a long stable period?",
        model_answer="In a Bayesian learner the learning rate is the share of the total uncertainty that belongs to "
                     "the learner's own estimate rather than to the noise of the outcomes. During a long stable "
                     "period the estimate is based on many outcomes, so its uncertainty is small compared with the "
                     "outcome noise, and each new outcome deserves only a small weight: a low learning rate averages "
                     "the noise away. After a change the old estimate is wrong, so the uncertainty about the current "
                     "value is suddenly large (a change-point learner puts probability mass back on all values), and "
                     "new outcomes deserve a large weight until the estimate has caught up. The learning rate should "
                     "therefore track the learner's uncertainty, which rises with change and falls with evidence.",
        concepts=[("uncertainty of the estimate", ["uncertain", "confiden", "belief", "variance", "precision", "how sure", "not sure", "trust"]),
                  ("outcome noise / averaging", ["noise", "noisy", "variab", "random", "average", "many outcomes", "many trials", "evidence"]),
                  ("change / surprise", ["change", "volatil", "revers", "shift", "surpris", "jump", "out of date", "outdated", "wrong"])],
        min_concepts=2,
        hints=["Think of the Kalman gain: a ratio of the learner's uncertainty to the total uncertainty (learner + outcome noise).",
               "What happens to the learner's uncertainty after many consistent outcomes, and what should happen to it after a change-point?"],
        explanation="The learning rate is not a fixed trait of the learner but a statement about how much it trusts "
                    "its current estimate relative to the noisiness of what it observes. Lesson 12 turns this into "
                    "a model that *estimates* how changeable the world is — the volatile Kalman filter.",
        check_call='check("11.10", """In a Bayesian learner the learning rate is the share of the total uncertainty that belongs to the learner\'s own estimate rather than to the noise of the outcomes. During a long stable period the estimate rests on many outcomes, its uncertainty is small compared with the outcome noise, and each new outcome deserves a small weight that averages the noise away. After a change the old estimate is wrong and the uncertainty about the current value is suddenly large, so new outcomes deserve a large weight until the estimate has caught up.""")',
    ),
]
