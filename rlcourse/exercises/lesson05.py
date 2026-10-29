"""Exercises of lesson 5: fitting models to behaviour by maximum likelihood."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise
from ..fitting import to_constrained
from ..models import nll_rw, softmax

# --------------------------------------------------------------------------- #
# deterministic data for the test cases
# --------------------------------------------------------------------------- #
_rng = np.random.default_rng(55)
_sim_a = models.simulate_rw_bandit([0.7, 0.3], 0.3, 4.0, 120, _rng)
_sim_b = models.simulate_rw_bandit([0.2, 0.5, 0.8], 0.15, 2.5, 150, _rng, q0=0.0)
_sim_c = models.simulate_rw_bandit([0.6, 0.4], 0.5, 3.0, 200, _rng)


# --------------------------------------------------------------------------- #
# reference solution of 5.6 (lives here because it is specific to this lesson)
# --------------------------------------------------------------------------- #
def fit_rw(choices, rewards, n_starts, rng):
    """Maximum-likelihood fit of the RW model.  Returns (alpha, beta, nll)."""
    from scipy.optimize import minimize

    def objective(x):
        # x lives in unconstrained space: alpha = sigmoid(x[0]), beta = exp(x[1])
        alpha, beta = to_constrained(x, ["unit", "pos"])
        return nll_rw((alpha, beta), choices, rewards)

    best = None
    for _ in range(n_starts):
        x0 = rng.normal(0, 1.5, size=2)                 # a random starting point
        result = minimize(objective, x0, method="L-BFGS-B")
        if best is None or result.fun < best.fun:       # keep the best of the restarts
            best = result
    alpha, beta = to_constrained(best.x, ["unit", "pos"])
    return float(alpha), float(beta), float(best.fun)


def _fit_compare(out, ref, case):
    try:
        alpha, beta, nll = out
    except Exception:
        return False
    if not (0 < alpha < 1 and beta > 0):
        return False
    true_nll = nll_rw((alpha, beta), case["choices"], case["rewards"])
    if abs(true_nll - nll) > 1e-4:
        return False
    return nll <= ref[2] + 1e-3


def _fit_diagnose(out, ref, case):
    if not isinstance(out, (tuple, list)) or len(out) != 3:
        return "Return a tuple `(alpha, beta, nll)` with three numbers."
    alpha, beta, nll = out
    if not (0 < alpha < 1 and beta > 0):
        return ("The returned parameters are outside their ranges (alpha must be in (0, 1), beta > 0). Did you return the "
                "*unconstrained* values x instead of transforming them back with `to_constrained`?")
    true_nll = nll_rw((alpha, beta), case["choices"], case["rewards"])
    if abs(true_nll - nll) > 1e-4:
        return (f"The nll you return ({nll:.3f}) is not the nll of the parameters you return ({true_nll:.3f}). Return the "
                "objective value of the *best* restart together with *its* parameters.")
    if nll > ref[2] + 1e-3:
        return (f"Your fit is valid but worse than the reference (nll {nll:.3f} vs {ref[2]:.3f}): a local minimum or a "
                "stopped-too-early optimiser. Use all `n_starts` restarts, draw each start with `rng.normal(0, 1.5, size=2)`, "
                "and keep the restart with the *lowest* objective.")
    return None


# --------------------------------------------------------------------------- #
# buggy variants
# --------------------------------------------------------------------------- #
def _nll_update_first(params, choices, rewards, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        Q[c] = Q[c] + alpha * (r - Q[c])
        p = softmax(Q, beta)
        nll -= np.log(p[c])
    return nll


def _nll_positive(params, choices, rewards, n_arms=2, q0=0.5):
    return -nll_rw(params, choices, rewards, n_arms, q0)


def _nll_sum_probs(params, choices, rewards, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    total = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, beta)
        total -= p[c]
        Q[c] = Q[c] + alpha * (r - Q[c])
    return total


def _nll_no_learning(params, choices, rewards, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, beta)
        nll -= np.log(p[c])
    return nll


def _nll_favourite(params, choices, rewards, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, beta)
        nll -= np.log(np.max(p))
        Q[c] = Q[c] + alpha * (r - Q[c])
    return nll


def _nll_no_beta(params, choices, rewards, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, 1.0)
        nll -= np.log(p[c])
        Q[c] = Q[c] + alpha * (r - Q[c])
    return nll


def _tc_flipped_sigmoid(x, kinds):
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    for i, kind in enumerate(kinds):
        out[i] = 1 / (1 + np.exp(x[i])) if kind == "unit" else (np.exp(x[i]) if kind == "pos" else x[i])
    return out


def _tc_exp_for_unit(x, kinds):
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    for i, kind in enumerate(kinds):
        out[i] = np.exp(x[i]) if kind in ("unit", "pos") else x[i]
    return out


def _tc_identity(x, kinds):
    return np.atleast_1d(np.asarray(x, float)).copy()


def _tc_log_for_pos(x, kinds):
    x = np.atleast_1d(np.asarray(x, float))
    out = np.empty_like(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, kind in enumerate(kinds):
            out[i] = 1 / (1 + np.exp(-x[i])) if kind == "unit" else (np.log(x[i]) if kind == "pos" else x[i])
    return out


EXERCISES = [
    # ------------------------------------------------------------------ 5.1
    FreeTextExercise(
        id="5.1", title="Why not fit by simulating?",
        question="In 2–4 sentences: why can we not estimate a participant's parameters by simulating the model with candidate "
                 "parameters and counting how many of the simulated choices match the participant's choices?",
        model_answer="A simulated agent samples its own choices at random, so even the true parameters would produce a different "
                     "sequence every run and a different history of values, and the match would be dominated by chance. The "
                     "likelihood avoids sampling altogether: it replays the model on the participant's *observed* choices and "
                     "rewards, so the values on each trial are the ones the participant's own history implies, and it scores each "
                     "observed choice by the *probability* the model assigns to it. Probabilities vary smoothly with the parameters "
                     "and can be multiplied across trials (summed as logs), which makes them something we can maximise.",
        concepts=[("randomness of simulated choices", ["random", "stochastic", "chance", "different every", "different each", "noise",
                                                       "sample", "luck", "vary from run", "varies from run"]),
                  ("replaying the observed history", ["observed", "actual", "participant's choices", "participant's own", "history",
                                                      "replay", "real choices", "the data", "same values"]),
                  ("probabilities instead of samples", ["probabilit", "likelihood", "how likely", "smooth", "continuous"])],
        min_concepts=2,
        hints=["Run the same simulation twice with the same parameters: do you get the same choices?",
               "What does the model assign to *each observed choice* when it is replayed on the participant's own history?"],
        explanation="Fitting works with probabilities, not samples: the model is replayed on the observed sequence and each observed "
                    "choice is scored by its probability under the model.",
        check_call='check("5.1", """A simulated agent samples its own choices at random, so even the true parameters produce a different sequence every run and the match with the data would be dominated by chance. The likelihood instead replays the model on the participant\'s observed choices and rewards, so the values on each trial are those implied by the participant\'s own history, and scores each observed choice by the probability the model assigns to it. Probabilities vary smoothly with the parameters and can be multiplied across trials, which is what we maximise.""")',
    ),
    # ------------------------------------------------------------------ 5.2
    CodeExercise(
        id="5.2", title="The negative log-likelihood of the RW model",
        reference=models.nll_rw,
        signature="nll_rw(params, choices, rewards, n_arms=2, q0=0.5)",
        cases=[{"params": (0.3, 4.0), "choices": _sim_a["choices"], "rewards": _sim_a["rewards"]},
               {"params": (0.5, 1.0), "choices": _sim_a["choices"], "rewards": _sim_a["rewards"]},
               {"params": (0.15, 2.5), "choices": _sim_b["choices"], "rewards": _sim_b["rewards"], "n_arms": 3, "q0": 0.0},
               {"params": (0.05, 20.0), "choices": _sim_c["choices"], "rewards": _sim_c["rewards"]}],
        bug_variants={
            "update_first": (_nll_update_first, "You updated the values *before* computing the probability of the choice. On trial t the "
                                                "participant chose using the values that existed before seeing trial t's reward: compute "
                                                "the probability first, then learn from the outcome."),
            "positive": (_nll_positive, "You returned the log-likelihood; the function must return its *negative* (a quantity to minimise)."),
            "sum_probs": (_nll_sum_probs, "You summed probabilities. The likelihood is a *product* of probabilities, so its log is a sum "
                                          "of *log* probabilities: subtract np.log(p[c]) on every trial."),
            "no_learning": (_nll_no_learning, "The values never change: after scoring each choice, update the chosen arm with the RW rule."),
            "favourite": (_nll_favourite, "You scored the model's *favourite* option (the largest probability) instead of the option the "
                                          "participant actually chose: use p[c], where c is the observed choice."),
            "no_beta": (_nll_no_beta, "The softmax ignores beta: use softmax(Q, beta)."),
        },
        hints=["Walk through the trials exactly as the participant experienced them: on each trial compute the softmax probabilities "
               "from the current values, take the probability of the choice that was *actually* made, then update the chosen "
               "arm's value with the reward that was *actually* received.",
               "Accumulate `nll -= np.log(p[c])` (a sum of negative log-probabilities), and return the total.",
               "`p = softmax(Q, beta); nll -= np.log(p[c]); Q[c] = Q[c] + alpha * (r - Q[c])` inside the loop."],
        explanation="The likelihood replays the model on the participant's own history: the values on every trial are the ones their "
                    "choices and rewards imply, and each observed choice is scored by the probability the model gave it *before* "
                    "that trial's outcome was known. Summing −log p over trials gives the negative log-likelihood, which the fitting "
                    "routine minimises. The order — score first, then learn — matters.",
        common_mistakes="Updating before scoring; returning the log-likelihood with the wrong sign; scoring the model's preferred option "
                        "rather than the observed one; summing probabilities instead of log-probabilities.",
        check_call='check("5.2", nll_rw)',
    ),
    # ------------------------------------------------------------------ 5.3
    NumericExercise(
        id="5.3", title="A two-trial likelihood by hand",
        answer=float(-np.log(0.5) - np.log(1 / (1 + np.exp(-0.5)))), atol=2e-3,
        distractors=[(float(-np.log(0.5)), "That is the first trial only: the second choice also contributes −ln P(choice 2)."),
                     (float(-2 * np.log(0.5)), "You did not let the model learn: after the first reward Q(A) = 0.75, so the second choice of A is more probable than 0.5."),
                     (float(np.log(0.5) + np.log(1 / (1 + np.exp(-0.5)))), "Right magnitude, wrong sign: the *negative* log-likelihood is positive."),
                     (float(0.5 * (1 / (1 + np.exp(-0.5)))), "That is the *product* of the two probabilities (the likelihood). Take −ln of it."),
                     (float(-np.log(1 / (1 + np.exp(-0.5))) - np.log(1 / (1 + np.exp(-0.75)))), "You updated the value *before* scoring each choice. On trial 1 both values are still 0.5.")],
        hints=["Trial 1: both values are 0.5, so P(A) = 0.5 whatever beta is; contribution −ln 0.5.",
               "After the reward Q(A) = 0.5 + 0.5 × (1 − 0.5) = 0.75 and Q(B) = 0.5. Trial 2: P(A) = 1 / (1 + exp(−β (Q(A) − Q(B)))) with β = 2.",
               "−ln 0.5 − ln σ(0.5) = 0.693 + 0.474."],
        explanation="−ln 0.5 = 0.693 for the first choice; after learning, Q(A) = 0.75 so P(A) = σ(2 × 0.25) = 0.622 and the "
                    "second choice contributes −ln 0.622 = 0.474; the total is 1.167. The second choice is *less* surprising to the "
                    "model than the first, which is what learning does to the likelihood.",
        check_call='check("5.3", -np.log(0.5) - np.log(1 / (1 + np.exp(-0.5))))',
    ),
    # ------------------------------------------------------------------ 5.4
    ChoiceExercise(
        id="5.4", title="Reading a likelihood surface",
        question="The negative log-likelihood is nearly flat along one parameter (a long valley). What does this mean?",
        options={"a": "That parameter is estimated very precisely.",
                 "b": "The data barely constrain that parameter: many values explain the choices about equally well, so its estimate is unreliable.",
                 "c": "The model is wrong and should be discarded.",
                 "d": "The optimiser is guaranteed to find the global minimum."},
        correct="b",
        feedback={"a": "It is the opposite: a precise estimate corresponds to a *sharp* minimum, where moving the parameter quickly makes the data less likely.",
                  "c": "A flat likelihood says something about the *data* (too few trials, or a task that does not probe that parameter), not about the truth of the model.",
                  "d": "Flat regions are exactly where optimisers wander and stop early; they make the fit *less* reliable."},
        hints=["Think about what the curvature of the surface tells you: how much does the fit worsen if you move the parameter a little?",
               "If the fit hardly changes along a direction, the data cannot tell those parameter values apart."],
        explanation="The curvature of the log-likelihood around its maximum measures how well a parameter is pinned down (lesson 8 turns "
                    "this into a posterior standard deviation). A flat valley means poor identifiability: collect more trials, change "
                    "the task, or fix/regularise the parameter (lesson 6 and 8).",
        check_call='check("5.4", "b")',
    ),
    # ------------------------------------------------------------------ 5.5
    CodeExercise(
        id="5.5", title="Parameter transforms",
        reference=to_constrained,
        signature="to_constrained(x, kinds)",
        cases=[{"x": [0.0, 0.0, 0.0], "kinds": ["unit", "pos", "real"]},
               {"x": [2.0, -1.0], "kinds": ["unit", "pos"]},
               {"x": np.array([-3.0, 1.5, 2.0, 0.3]), "kinds": ["unit", "pos", "unit", "real"]},
               {"x": [10.0, -10.0], "kinds": ["unit", "unit"]}],
        bug_variants={
            "flipped_sigmoid": (_tc_flipped_sigmoid, "Your sigmoid is mirrored: 1 / (1 + exp(−x)) increases with x; you have exp(+x) in the denominator."),
            "exp_for_unit": (_tc_exp_for_unit, "You used exp for the 'unit' kind; exp is unbounded above. Learning rates need the sigmoid, 1 / (1 + exp(−x))."),
            "identity": (_tc_identity, "No transform was applied. 'unit' → sigmoid(x), 'pos' → exp(x), 'real' → x."),
            "log_for_pos": (_tc_log_for_pos, "You used log for the 'pos' kind. log maps *to* the unconstrained space; to go back use exp(x)."),
        },
        hints=["Loop over the kinds: 'unit' → sigmoid(x) = 1 / (1 + exp(−x)) gives a number in (0, 1); 'pos' → exp(x) gives a positive number; 'real' → x unchanged.",
               "Build an output array of the same length as x and fill it element by element.",
               "`theta[i] = sigmoid(x[i]) if kind == 'unit' else np.exp(x[i]) if kind == 'pos' else x[i]`"],
        explanation="Optimisers work on the whole real line. By optimising x and mapping it through the sigmoid or the exponential we "
                    "guarantee that alpha stays in (0, 1) and beta stays positive without any constraint handling — and the "
                    "likelihood becomes smoother near the boundaries, which helps the optimiser.",
        common_mistakes="Mirroring the sigmoid, using exp for the unit kind, or forgetting to transform at all.",
        check_call='check("5.5", to_constrained)',
    ),
    # ------------------------------------------------------------------ 5.6
    CodeExercise(
        id="5.6", title="Fit the RW model by maximum likelihood",
        reference=fit_rw,
        signature="fit_rw(choices, rewards, n_starts, rng)",
        cases=[{"choices": _sim_a["choices"], "rewards": _sim_a["rewards"], "n_starts": 5, "rng": np.random.default_rng(3)},
               {"choices": _sim_c["choices"], "rewards": _sim_c["rewards"], "n_starts": 4, "rng": np.random.default_rng(4)}],
        compare=_fit_compare,
        diagnose=_fit_diagnose,
        hints=["Write an objective function of the unconstrained vector x: transform it with `to_constrained(x, ['unit', 'pos'])` and "
               "return `nll_rw((alpha, beta), choices, rewards)`.",
               "Call `scipy.optimize.minimize(objective, x0, method='L-BFGS-B')` from `n_starts` random starting points "
               "`x0 = rng.normal(0, 1.5, size=2)` and keep the result with the smallest `.fun`.",
               "Transform the best `result.x` back with `to_constrained` and return `(alpha, beta, result.fun)`."],
        explanation="This is the complete recipe used for every model in the course: an objective in unconstrained space, a general "
                    "optimiser, several random restarts, and the best result mapped back to the parameter space. "
                    "`fitting.fit_mle` does the same for any model and any number of parameters.",
        common_mistakes="Returning the unconstrained x instead of the transformed parameters; returning the objective of the last "
                        "restart instead of the best; using too few restarts.",
        check_call='check("5.6", fit_rw)',
    ),
    # ------------------------------------------------------------------ 5.7
    ChoiceExercise(
        id="5.7", title="Why several starting points?",
        question="Why do we run the optimiser from several random starting points and keep the best result?",
        options={"a": "To average the parameter estimates across runs.",
                 "b": "Because the optimiser can get stuck in a local minimum or stop in a flat region; different starts explore different parts of the surface.",
                 "c": "To estimate the uncertainty of the parameters from the spread of the runs.",
                 "d": "Because scipy requires at least five calls to converge."},
        correct="b",
        feedback={"a": "We keep the single best run, we never average: runs that ended in a worse minimum are simply wrong.",
                  "c": "The spread of restarts reflects the optimiser's failures, not the data's uncertainty. Uncertainty comes from the curvature of the likelihood or the posterior (lesson 8).",
                  "d": "There is no such requirement; one call can converge perfectly well — the issue is *where* it converges."},
        hints=["Gradient-based optimisers only go downhill from where they start.",
               "What happens if the surface has two valleys, or a long flat plateau?"],
        explanation="A local optimiser finds *a* minimum near its start. Restarting from several points and keeping the lowest value is "
                    "the simplest safeguard against local minima and premature stops on flat regions; the restarts should agree "
                    "closely when the fit is trustworthy.",
        check_call='check("5.7", "b")',
    ),
    # ------------------------------------------------------------------ 5.8
    NumericExercise(
        id="5.8", title="Pseudo-R² of a fit",
        answer=float(1 - 100 / (200 * np.log(2))), atol=1e-3,
        distractors=[(0.5, "You compared with n instead of with the chance log-likelihood n × ln 2 (two options)."),
                     (float(100 / (200 * np.log(2))), "That is the ratio nll / nll_chance; the pseudo-R² is 1 minus that ratio."),
                     (float(-(1 - 100 / (200 * np.log(2)))), "Sign error: the pseudo-R² is 1 − nll / nll_chance, positive here."),
                     (float(1 - 100 / (200 * np.log(10))), "The chance model picks between *two* options, so its −log-probability per trial is ln 2, not ln 10.")],
        hints=["A model that chooses at random between two options assigns probability ½ to every choice: nll_chance = n × ln 2.",
               "pseudo-R² = 1 − nll / nll_chance.",
               "1 − 100 / (200 × 0.693)."],
        explanation="nll_chance = 200 ln 2 ≈ 138.6, so pseudo-R² = 1 − 100/138.6 ≈ 0.279: the model explains about 28% of the "
                    "'choice uncertainty' a random chooser would have. Values near 0 mean the model does no better than chance — a "
                    "useful sanity check before interpreting any parameter.",
        check_call='check("5.8", 1 - 100 / (200 * np.log(2)))',
    ),
]
