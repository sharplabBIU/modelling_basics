"""Exercises of lesson 6: parameter recovery, identifiability and model recovery."""
from __future__ import annotations

import numpy as np

from .. import fitting, models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference implementations (their source is revealed as the solution)
# --------------------------------------------------------------------------- #


def simulate_and_fit(alpha, beta, n_trials, rng, n_starts=3, probs=(0.7, 0.3)):
    """Simulate one RW + softmax agent with known parameters on a bandit, then fit
    the RW model to the simulated choices.  Returns (alpha_hat, beta_hat)."""
    sim = models.simulate_rw_bandit(probs, alpha, beta, n_trials, rng)
    fit = fitting.fit_mle(models.nll_rw, (sim["choices"], sim["rewards"]), ["unit", "pos"],
                          n_starts=n_starts, rng=rng)
    alpha_hat, beta_hat = fit["params"]              # the *constrained* estimates
    return float(alpha_hat), float(beta_hat)


def recovery_summary(true, recovered):
    """Per-parameter Pearson correlation and mean bias (recovered - true) between
    true and recovered parameters, both of shape (n_subjects, n_params).
    Returns {"r": array (n_params,), "bias": array (n_params,)}."""
    true = np.asarray(true, dtype=float)
    recovered = np.asarray(recovered, dtype=float)
    n_params = true.shape[1]
    r = np.zeros(n_params)
    bias = np.zeros(n_params)
    for j in range(n_params):
        r[j] = np.corrcoef(true[:, j], recovered[:, j])[0, 1]
        bias[j] = np.mean(recovered[:, j] - true[:, j])
    return {"r": r, "bias": bias}


def best_model_by_bic(nlls, n_trials):
    """Return the name of the model with the lowest BIC.  ``nlls`` maps model names
    (keys of models.MODEL_SPECS) to fitted negative log-likelihoods."""
    best_name, best_bic = None, np.inf
    for name, nll in nlls.items():
        k = len(models.MODEL_SPECS[name][1])          # number of free parameters
        value = fitting.bic(nll, k, n_trials)         # k * log(n) + 2 * nll
        if value < best_bic:
            best_name, best_bic = name, value
    return best_name


def confusion_matrix(sim_labels, fit_labels, n_models):
    """Confusion matrix of a model-recovery study: rows = generating model,
    columns = best-fitting model, entries = proportion of data sets (rows sum to 1)."""
    sim_labels = np.asarray(sim_labels, dtype=int)
    fit_labels = np.asarray(fit_labels, dtype=int)
    counts = np.zeros((n_models, n_models))
    for s, f in zip(sim_labels, fit_labels):
        counts[s, f] += 1
    return counts / counts.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Helpers for the checker (comparisons, diagnoses, buggy variants)
# --------------------------------------------------------------------------- #


def _fit_compare(out, ref, case):
    try:
        a, b = float(out[0]), float(out[1])
    except Exception:
        return False
    if len(out) != 2:
        return False
    if np.allclose([a, b], ref, atol=1e-6):
        return True
    # A correct implementation that consumes the random numbers differently sees a
    # different data set: accept estimates that are plausibly close to the truth.
    return 0 < a < 1 and b > 0 and abs(a - case["alpha"]) < 0.3 and abs(np.log(b / case["beta"])) < 1.0


def _fit_diagnose(out, ref, case):
    if isinstance(out, dict):
        return ("Return the two estimates as a tuple `(alpha_hat, beta_hat)`, not the whole fit "
                "dictionary (they are in `fit['params']`).")
    try:
        values = [float(v) for v in out]
    except Exception:
        return "Return exactly two numbers: `(alpha_hat, beta_hat)`."
    if len(values) != 2:
        return f"Return exactly two numbers, you returned {len(values)}."
    a, b = values
    if not (0 < a < 1) or b <= 0:
        if abs(models.sigmoid(a) - ref[0]) < 0.05:
            return ("You returned the *unconstrained* values (`fit['x']`, where alpha is a logit and beta a "
                    "log). Return `fit['params']`, which holds the transformed estimates.")
        return ("`alpha_hat` must lie in (0, 1) and `beta_hat` must be positive. Did you return the "
                "unconstrained values, or pass the parameter kinds in the wrong order (['unit', 'pos'])?")
    if abs(a - ref[1]) < 0.3 and abs(b - ref[0]) < 0.3:
        return "The two estimates seem to be swapped: return `(alpha_hat, beta_hat)` in that order."
    return ("Your estimates are far from the generating parameters. Check that you simulate with "
            "`models.simulate_rw_bandit(probs, alpha, beta, n_trials, rng)` and fit `models.nll_rw` "
            "on `(sim['choices'], sim['rewards'])` with kinds `['unit', 'pos']`.")


def _bias_flipped(true, recovered):
    s = recovery_summary(true, recovered)
    return {"r": s["r"], "bias": -s["bias"]}


def _r_squared(true, recovered):
    s = recovery_summary(true, recovered)
    return {"r": s["r"] ** 2, "bias": s["bias"]}


def _argmin_nll(nlls, n_trials):
    return min(nlls, key=lambda name: nlls[name])


def _aic_winner(nlls, n_trials):
    return min(nlls, key=lambda name: fitting.aic(nlls[name], len(models.MODEL_SPECS[name][1])))


def _counts_only(sim_labels, fit_labels, n_models):
    counts = np.zeros((n_models, n_models))
    for s, f in zip(np.asarray(sim_labels, int), np.asarray(fit_labels, int)):
        counts[s, f] += 1
    return counts


def _transposed(sim_labels, fit_labels, n_models):
    return confusion_matrix(fit_labels, sim_labels, n_models)


def _column_normalised(sim_labels, fit_labels, n_models):
    counts = _counts_only(sim_labels, fit_labels, n_models)
    col = counts.sum(axis=0, keepdims=True)
    return np.divide(counts, col, out=np.zeros_like(counts), where=col > 0)


# deterministic test data for 6.2
_rng = np.random.default_rng(606)
_TRUE_BIG = np.column_stack([_rng.uniform(0.1, 0.9, 20), np.exp(_rng.uniform(0.0, 2.3, 20))])
_REC_BIG = _TRUE_BIG + np.column_stack([_rng.normal(0.02, 0.08, 20), _rng.normal(-0.3, 1.0, 20)])
_TRUE_SMALL = np.array([[0.1, 1.0], [0.3, 2.0], [0.5, 3.0], [0.7, 4.0]])
_REC_SMALL = np.array([[0.2, 1.5], [0.25, 2.5], [0.55, 2.8], [0.6, 4.5]])

# the small table of exercise 6.3
_TABLE_TRUE = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
_TABLE_REC = np.array([0.20, 0.25, 0.55, 0.60, 0.95])
_PEARSON = float(np.corrcoef(_TABLE_TRUE, _TABLE_REC)[0, 1])


EXERCISES = [
    # ------------------------------------------------------------------ 6.1
    CodeExercise(
        id="6.1", title="Simulate, then fit",
        reference=simulate_and_fit,
        signature="simulate_and_fit(alpha, beta, n_trials, rng, n_starts=3, probs=(0.7, 0.3))",
        cases=[{"alpha": 0.3, "beta": 4.0, "n_trials": 150, "rng": np.random.default_rng(61)},
               {"alpha": 0.7, "beta": 2.0, "n_trials": 200, "rng": np.random.default_rng(62)}],
        compare=_fit_compare,
        diagnose=_fit_diagnose,
        hints=["Two steps: `sim = models.simulate_rw_bandit(probs, alpha, beta, n_trials, rng)` produces the data; "
               "`fitting.fit_mle(models.nll_rw, (sim['choices'], sim['rewards']), ['unit', 'pos'], n_starts=n_starts, rng=rng)` fits them.",
               "`fit_mle` returns a dictionary; the constrained estimates are `fit['params']` (an array with alpha first, then beta).",
               "`alpha_hat, beta_hat = fit['params']` and `return float(alpha_hat), float(beta_hat)`."],
        explanation="This two-line pipeline — simulate with known parameters, fit, compare — is the whole idea of "
                    "parameter recovery. Everything else in the section is bookkeeping: repeating it for many "
                    "parameter values and summarising how close the estimates come to the truth.",
        common_mistakes="Returning the unconstrained values `fit['x']` instead of `fit['params']`; passing the "
                        "parameter kinds in the wrong order; fitting the rewards instead of the choices.",
        check_call='check("6.1", simulate_and_fit)',
    ),
    # ------------------------------------------------------------------ 6.2
    CodeExercise(
        id="6.2", title="Summarising a recovery study",
        reference=recovery_summary,
        signature="recovery_summary(true, recovered)",
        cases=[{"true": _TRUE_SMALL, "recovered": _REC_SMALL},
               {"true": _TRUE_BIG, "recovered": _REC_BIG}],
        bug_variants={
            "bias_flipped": (_bias_flipped, "Your bias has the wrong sign: bias is defined as recovered − true "
                                            "(positive = over-estimation)."),
            "r_squared": (_r_squared, "You returned r² rather than the correlation coefficient r itself."),
        },
        hints=["Loop over the parameters (the columns). For column j, `np.corrcoef(true[:, j], recovered[:, j])[0, 1]` is "
               "Pearson's r and `np.mean(recovered[:, j] - true[:, j])` is the bias.",
               "`np.corrcoef` returns a 2×2 matrix; the correlation you want is the off-diagonal entry `[0, 1]`.",
               "Collect the values in two arrays of length n_params and return `{'r': r, 'bias': bias}`."],
        explanation="The correlation tells you whether the *ordering* of subjects is preserved (essential for "
                    "individual-differences research); the bias tells you whether estimates are systematically too "
                    "high or too low. Both can be fine while the estimates are still noisy, so always look at the "
                    "scatter plot as well.",
        common_mistakes="Computing true − recovered for the bias; returning the whole correlation matrix; pooling the "
                        "parameters into one correlation.",
        check_call='check("6.2", recovery_summary)',
    ),
    # ------------------------------------------------------------------ 6.3
    NumericExercise(
        id="6.3", title="Recovery correlation by hand",
        answer=_PEARSON, atol=2e-3,
        distractors=[(1.0, "That is the rank (Spearman) correlation: the two orderings agree perfectly, but the "
                           "question asks for Pearson's r, which also depends on the sizes of the deviations."),
                     (_PEARSON ** 2, "That is r², the squared correlation; the question asks for r."),
                     (0.01, "That is the mean bias (recovered − true), not the correlation.")],
        hints=["Pearson's r = cov(x, y) / (sd(x) sd(y)). Centre both columns (subtract their means) first.",
               "`np.corrcoef([0.1, 0.3, 0.5, 0.7, 0.9], [0.2, 0.25, 0.55, 0.6, 0.95])[0, 1]` does it for you — but "
               "try the formula by hand once.",
               "cov = mean of the products of the centred values = 0.074; sd(true) = 0.283, sd(rec) ≈ 0.271 "
               "(population sds), so r ≈ 0.074 / (0.283 × 0.271)."],
        explanation="r ≈ 0.966: the recovered values track the true ones closely even though every single estimate "
                    "is off by 0.05–0.10. A high r is compatible with substantial estimation error — which is why "
                    "r alone does not tell you whether an individual estimate can be trusted.",
        check_call=f'check("6.3", {_PEARSON:.6f})',
    ),
    # ------------------------------------------------------------------ 6.4
    ChoiceExercise(
        id="6.4", title="When choices are deterministic",
        question="A simulated participant's choices are perfectly consistent with always picking the option that currently has the higher value. What happens to the maximum-likelihood estimate of beta, and why?",
        options={"a": "beta_hat = 0: consistent choices mean there is no exploration to explain.",
                 "b": "beta_hat grows without bound: every increase in beta makes the observed choices more probable, so the likelihood has no maximum at any finite value.",
                 "c": "beta_hat = 1, because the data are perfectly consistent with the model.",
                 "d": "The optimiser returns its starting value, because beta drops out of the likelihood when choices are deterministic."},
        correct="b",
        feedback={"a": "beta = 0 makes every choice a coin flip — the opposite of perfectly consistent choices.",
                  "c": "Nothing is special about 1: for any finite beta the observed choices still have probability below 1, "
                       "and a larger beta raises it.",
                  "d": "beta never drops out of the likelihood: P(choice) = 1 / (1 + exp(−beta ΔQ)) depends on beta on every "
                       "trial. The problem is that the likelihood keeps increasing with beta."},
        hints=["Write the probability of an observed choice as 1 / (1 + exp(−beta × ΔQ)) with ΔQ > 0 on every trial. What "
               "happens to it as beta → ∞?",
               "Each observed choice's probability approaches 1 as beta grows, so the product (the likelihood) approaches "
               "its largest possible value only in the limit: the maximum is at infinity."],
        explanation="The data say 'very deterministic' but not *how* deterministic, so the maximum-likelihood estimate of "
                    "beta is at infinity — a boundary fit. The same happens to alpha → 1 for a participant who always "
                    "repeats after a reward and switches after a loss. Remedies: bounds on the parameters, a prior "
                    "(MAP estimation, lesson 8), or more trials in which some inconsistent choices occur.",
        check_call='check("6.4", "b")',
    ),
    # ------------------------------------------------------------------ 6.5
    CodeExercise(
        id="6.5", title="Picking the winning model with BIC",
        reference=best_model_by_bic,
        signature="best_model_by_bic(nlls, n_trials)",
        cases=[{"nlls": {"bias": 103.0, "wsls": 95.0, "rw": 90.0, "rw_stick": 88.5}, "n_trials": 150},
               {"nlls": {"bias": 120.0, "wsls": 70.0, "rw": 85.0, "rw_asym": 84.0}, "n_trials": 200},
               {"nlls": {"rw": 100.0, "rw_asym": 99.0}, "n_trials": 100}],
        bug_variants={
            "argmin_nll": (_argmin_nll, "You picked the model with the lowest negative log-likelihood. A model with "
                                        "more parameters always fits at least as well, so you must add the "
                                        "penalty: BIC = k ln n + 2 nll."),
            "aic": (_aic_winner, "You used AIC's penalty (2 per parameter). The exercise asks for BIC, whose penalty "
                                 "is ln n per parameter — the two criteria can disagree, as they do here."),
        },
        hints=["BIC = k × ln(n) + 2 × nll, where k is the number of free parameters and n the number of trials; "
               "the *lowest* BIC wins.",
               "`len(models.MODEL_SPECS[name][1])` gives k (the list of parameter names), and `fitting.bic(nll, k, n)` "
               "computes the criterion.",
               "Loop over `nlls.items()`, compute each model's BIC, keep the name with the smallest value."],
        explanation="The penalty k ln n is what stops the most flexible model from winning automatically. Lesson 7 "
                    "explains where the formula comes from and when AIC or cross-validation are better choices; "
                    "for model recovery any sensible criterion can be used, as long as it is the same one you will "
                    "apply to the real data.",
        common_mistakes="Forgetting the penalty (choosing the lowest nll), or using k instead of k ln n.",
        check_call='check("6.5", best_model_by_bic)',
    ),
    # ------------------------------------------------------------------ 6.6
    CodeExercise(
        id="6.6", title="The confusion matrix",
        reference=confusion_matrix,
        signature="confusion_matrix(sim_labels, fit_labels, n_models)",
        cases=[{"sim_labels": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], "fit_labels": [0, 0, 1, 0, 1, 1, 1, 2, 2, 2, 1, 2], "n_models": 3},
               {"sim_labels": np.array([0, 0, 1, 1, 1, 1]), "fit_labels": np.array([1, 1, 1, 0, 1, 1]), "n_models": 2},
               {"sim_labels": [0, 0, 1, 1, 2, 2, 3, 3], "fit_labels": [0, 3, 1, 1, 2, 1, 3, 3], "n_models": 4}],
        bug_variants={
            "counts": (_counts_only, "You returned counts. The confusion matrix holds *proportions*: divide each row by "
                                     "its sum so that every row adds up to 1."),
            "transposed": (_transposed, "Your matrix is transposed: rows must index the *generating* model and "
                                        "columns the *best-fitting* model."),
            "column_normalised": (_column_normalised, "You normalised the columns: that is the *inversion* matrix "
                                                      "P(generating | best fit). The confusion matrix normalises "
                                                      "each row: P(best fit | generating)."),
        },
        hints=["Start from a matrix of zeros of shape (n_models, n_models); for every data set add 1 at "
               "[generating model, winning model].",
               "`for s, f in zip(sim_labels, fit_labels): counts[s, f] += 1`",
               "Divide by the row sums: `counts / counts.sum(axis=1, keepdims=True)`."],
        explanation="Each row is a conditional distribution: given that model i generated the data, how often did "
                    "each candidate win? A diagonal matrix means perfect recovery; mass off the diagonal tells you "
                    "which models are confusable in this task with this many trials.",
        common_mistakes="Returning counts instead of proportions; swapping rows and columns; normalising columns.",
        check_call='check("6.6", confusion_matrix)',
    ),
    # ------------------------------------------------------------------ 6.7
    NumericExercise(
        id="6.7", title="From confusion to inversion",
        answer=14 / 24, atol=2e-3,
        distractors=[(14 / 20, "That is the row-normalised entry P(best fit = B | generating = B) — the confusion "
                               "matrix. The question asks the reverse: given that B won, which model generated the data?"),
                     (14 / 60, "That is the joint proportion of all data sets that were generated by B *and* won by B. "
                               "Condition on the column (B won)."),
                     (24 / 60, "That is the overall proportion of data sets in which B won.")],
        hints=["You are told that model B won, so restrict attention to the *column* of B.",
               "In that column, 4 + 14 + 6 = 24 data sets were won by B; 14 of them were generated by B.",
               "14 / 24"],
        explanation="P(generating = B | best fit = B) = 14/24 ≈ 0.58. This 'inversion matrix' answers the question you "
                    "actually face with real data — the model that won — and it can differ a lot from the "
                    "confusion matrix when some models win too often (here C is often mistaken for B).",
        check_call='check("6.7", 14 / 24)',
    ),
    # ------------------------------------------------------------------ 6.8
    ChoiceExercise(
        id="6.8", title="Reading a confusion matrix",
        question="In a confusion matrix, the entry in row RW and column RW+stickiness is 0.30. This means that ...",
        options={"a": "30% of the data sets simulated from RW were best fit by RW+stickiness.",
                 "b": "30% of the data sets best fit by RW+stickiness were generated by RW.",
                 "c": "RW+stickiness is the true model with probability 0.30.",
                 "d": "RW+stickiness fits 30% worse than RW."},
        correct="a",
        feedback={"b": "That is what the *inversion* matrix (columns normalised) tells you.",
                  "c": "A statement about the true model would require the real data and a prior over models; the "
                       "confusion matrix only describes simulations.",
                  "d": "The matrix contains proportions of data sets, not differences in fit."},
        hints=["Rows of the confusion matrix condition on the generating model; each row sums to 1.",
               "Row RW: of all the data sets generated by RW, which model won how often?"],
        explanation="Rows condition on the truth (confusion matrix), columns on the winner (inversion matrix). When "
                    "the entry is large the extra stickiness parameter is being 'found' in data that do not contain "
                    "it — a sign that the task does not let you tell the two models apart.",
        check_call='check("6.8", "a")',
    ),
    # ------------------------------------------------------------------ 6.9
    FreeTextExercise(
        id="6.9", title="When recovery fails",
        question="Parameter recovery for your model is poor (r ≈ 0.3 for the learning rate). Name at least two things you could do about it, and say why each helps.",
        model_answer="(1) Collect more trials per participant: estimation noise shrinks with the number of informative "
                     "choices. (2) Redesign the task so that the parameter has a clear behavioural signature — for "
                     "example add reversals so that the learning rate visibly affects how fast people adapt, or use "
                     "reward probabilities that make choices depend on the values. (3) Simplify the model: fix or "
                     "remove parameters that trade off with the one you care about, or constrain them to a "
                     "plausible range. (4) Use hierarchical (group-level) priors, which regularise individual "
                     "estimates and reduce noise, at the cost of some shrinkage towards the group mean. Whatever you "
                     "change, rerun the recovery study to confirm that it worked.",
        concepts=[("more data / trials", ["more trials", "more data", "longer", "more sessions", "number of trials",
                                          "additional trials", "more choices", "more observations"]),
                  ("task design", ["design", "task", "revers", "manipulat", "condition", "probabilit", "contrast",
                                   "informative"]),
                  ("simplify / constrain the model", ["simpl", "fewer parameter", "fix", "constrain", "reduce",
                                                      "drop", "remove", "bound", "plausible range"]),
                  ("hierarchical priors", ["hierarch", "prior", "regulari", "shrink", "group-level", "group level",
                                           "empirical bayes", "pool"])],
        min_concepts=2,
        hints=["Think about the two ingredients of recovery: how much information each trial carries, and how many trials there are.",
               "Also think about the model side: could another parameter be absorbing the variance, and could prior knowledge help?"],
        explanation="Poor recovery is a property of the model–task–sample-size combination, so each of those three "
                    "can be changed. The one thing you cannot do is interpret the parameter anyway.",
        check_call='check("6.9", """Collect more trials per participant so that estimation noise shrinks. Redesign the task, for example add reversals so that the learning rate has a clear behavioural signature. Simplify the model by fixing parameters that trade off with the one of interest, or use hierarchical priors that regularise the estimates.""")',
    ),
]
