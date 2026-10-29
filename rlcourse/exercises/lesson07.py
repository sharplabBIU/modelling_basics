"""Exercises of lesson 7: model comparison."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference implementations (their source is revealed as the solution)
# --------------------------------------------------------------------------- #


def information_criteria(nll, k, n):
    """Return (AIC, BIC) for a fitted model with negative log-likelihood ``nll``,
    ``k`` free parameters and ``n`` data points (choices)."""
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll
    return aic, bic


def nll_rw_on_trials(params, choices, rewards, test_mask, n_arms=2, q0=0.5):
    """Negative log-likelihood of the RW + softmax model restricted to the trials
    where ``test_mask`` is True.  The model is replayed over the *whole* sequence
    (so the values on a test trial reflect the full learning history), but only
    the test trials are scored.  params = (alpha, beta)."""
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for t, (c, r) in enumerate(zip(choices, rewards)):
        p = models.softmax(Q, beta)
        if test_mask[t]:
            nll -= np.log(p[c])                  # score only the held-out trials ...
        Q[c] = Q[c] + alpha * (r - Q[c])         # ... but learn from every trial
    return nll


def bms_alpha(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8):
    """Random-effects Bayesian model selection (Stephan et al. 2009): the variational
    iteration that returns the Dirichlet parameters ``alpha`` of the posterior over
    model frequencies.  ``log_evidence`` has shape (n_subjects, n_models)."""
    from scipy.special import digamma, logsumexp

    L = np.asarray(log_evidence, dtype=float)
    n_subjects, n_models = L.shape
    alpha = np.full(n_models, float(alpha0))
    for _ in range(max_iter):
        # E-step: posterior probability of each model for each subject
        log_u = L + digamma(alpha) - digamma(alpha.sum())
        g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
        # M-step: prior counts plus the expected number of subjects using each model
        alpha_new = alpha0 + g.sum(axis=0)
        if np.max(np.abs(alpha_new - alpha)) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return alpha


def exceedance_probabilities(alpha, n_samples, rng):
    """Exceedance probabilities by Monte Carlo: draw the model frequencies r from
    Dirichlet(alpha) many times and count how often each model has the largest
    frequency."""
    alpha = np.asarray(alpha, dtype=float)
    samples = rng.dirichlet(alpha, size=n_samples)          # one row = one draw of r
    winners = np.argmax(samples, axis=1)                     # the most frequent model in each draw
    return np.bincount(winners, minlength=len(alpha)) / n_samples


# --------------------------------------------------------------------------- #
# Buggy variants and test data
# --------------------------------------------------------------------------- #


def _ic_no_factor(nll, k, n):
    return 2 * k + nll, k * np.log(n) + nll


def _ic_log10(nll, k, n):
    return 2 * k + 2 * nll, k * np.log10(n) + 2 * nll


def _ic_swapped(nll, k, n):
    return k * np.log(n) + 2 * nll, 2 * k + 2 * nll


def _ic_n_not_log(nll, k, n):
    return 2 * k + 2 * nll, k * n + 2 * nll


def _subset_only(params, choices, rewards, test_mask, n_arms=2, q0=0.5):
    mask = np.asarray(test_mask, bool)
    return models.nll_rw(params, np.asarray(choices)[mask], np.asarray(rewards)[mask], n_arms, q0)


def _ignore_mask(params, choices, rewards, test_mask, n_arms=2, q0=0.5):
    return models.nll_rw(params, choices, rewards, n_arms, q0)


def _learn_only_on_test(params, choices, rewards, test_mask, n_arms=2, q0=0.5):
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for t, (c, r) in enumerate(zip(choices, rewards)):
        if test_mask[t]:
            p = models.softmax(Q, beta)
            nll -= np.log(p[c])
            Q[c] = Q[c] + alpha * (r - Q[c])
    return nll


def _bms_no_digamma(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8):
    from scipy.special import logsumexp
    L = np.asarray(log_evidence, float)
    g = np.exp(L - logsumexp(L, axis=1, keepdims=True))
    return alpha0 + g.sum(axis=0)


def _bms_no_alpha0(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8):
    from scipy.special import digamma, logsumexp
    L = np.asarray(log_evidence, float)
    alpha = np.full(L.shape[1], float(alpha0))
    for _ in range(max_iter):
        log_u = L + digamma(alpha) - digamma(alpha.sum())
        g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
        alpha_new = g.sum(axis=0)
        if np.max(np.abs(alpha_new - alpha)) < tol:
            return alpha_new
        alpha = alpha_new
    return alpha


def _bms_one_iteration(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8):
    return bms_alpha(log_evidence, alpha0, max_iter=1, tol=tol)


def _bms_wrong_axis(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8):
    from scipy.special import digamma, logsumexp
    L = np.asarray(log_evidence, float)
    alpha = np.full(L.shape[1], float(alpha0))
    for _ in range(max_iter):
        log_u = L + digamma(alpha) - digamma(alpha.sum())
        g = np.exp(log_u - logsumexp(log_u, axis=0, keepdims=True))
        alpha_new = alpha0 + g.sum(axis=0)
        if np.max(np.abs(alpha_new - alpha)) < tol:
            return alpha_new
        alpha = alpha_new
    return alpha


def _expected_frequencies(alpha, n_samples, rng):
    alpha = np.asarray(alpha, float)
    return alpha / alpha.sum()


def _xp_compare(out, ref):
    out = np.asarray(out, float)
    return out.shape == np.shape(ref) and np.allclose(out, ref, atol=0.03)


# deterministic data for 7.5
_sim = models.simulate_rw_bandit([0.7, 0.3], 0.3, 4.0, 100, np.random.default_rng(707))
_C, _R = _sim["choices"], _sim["rewards"]
_MASK_HALF = np.arange(100) >= 50
_MASK_ODD = (np.arange(100) % 2) == 1
_MASK_ALL = np.ones(100, dtype=bool)

# deterministic log-evidence matrices for 7.6
_rng7 = np.random.default_rng(77)
_L1 = _rng7.normal(0.0, 2.0, (12, 3))
_L1[:, 0] += 2.0
_L2 = _rng7.normal(0.0, 1.0, (8, 2))
_L2[:4, 1] += 3.0


EXERCISES = [
    # ------------------------------------------------------------------ 7.1
    CodeExercise(
        id="7.1", title="AIC and BIC",
        reference=information_criteria,
        signature="information_criteria(nll, k, n)",
        cases=[{"nll": 100.0, "k": 3, "n": 200}, {"nll": 50.5, "k": 1, "n": 80}, {"nll": 1234.5, "k": 7, "n": 2000}],
        bug_variants={
            "no_factor": (_ic_no_factor, "The deviance term is 2 × nll in both criteria, not nll."),
            "log10": (_ic_log10, "BIC uses the natural logarithm ln(n), not log10."),
            "swapped": (_ic_swapped, "You returned (BIC, AIC); the order must be (AIC, BIC)."),
            "n_not_log": (_ic_n_not_log, "The BIC penalty is k × ln(n), not k × n."),
        },
        hints=["AIC = 2k + 2·nll and BIC = k·ln(n) + 2·nll (lower is better for both).",
               "`np.log` is the natural logarithm.",
               "`return 2 * k + 2 * nll, k * np.log(n) + 2 * nll`"],
        explanation="Both criteria are '2 × nll plus a penalty for complexity'. AIC's penalty (2 per parameter) "
                    "estimates how much the in-sample fit overstates out-of-sample fit; BIC's (ln n per parameter) "
                    "approximates the log marginal likelihood (model evidence) for large n. Because ln n > 2 as soon "
                    "as n > 7, BIC is almost always the more conservative of the two.",
        common_mistakes="Forgetting the factor 2 on the nll; using log10; mixing up the order of the outputs.",
        check_call='check("7.1", information_criteria)',
    ),
    # ------------------------------------------------------------------ 7.2
    NumericExercise(
        id="7.2", title="BIC by hand",
        answer=3 * np.log(200) + 200, atol=0.01,
        distractors=[(206.0, "That is the AIC (penalty 2 per parameter). BIC's penalty is ln(n) per parameter."),
                     (3 * np.log(200) + 100, "You forgot to double the negative log-likelihood: BIC = k ln(n) + 2 nll."),
                     (3 * np.log10(200) + 200, "You used log10; the BIC uses the natural logarithm."),
                     (3 * np.log(200), "That is only the penalty; add the deviance term 2 × nll."),
                     (3 * 200 + 200, "The penalty is k × ln(n), not k × n.")],
        hints=["BIC = k ln(n) + 2 nll with k = 3, n = 200, nll = 100.",
               "ln(200) ≈ 5.298.",
               "3 × 5.298 + 2 × 100 ≈ 215.9"],
        explanation="BIC ≈ 215.9. On its own the number means nothing; only *differences* between models fitted to "
                    "the same data are interpretable. A difference of 10 or more is conventionally 'very strong' "
                    "evidence (Kass & Raftery 1995).",
        check_call='check("7.2", 3 * np.log(200) + 200)',
    ),
    # ------------------------------------------------------------------ 7.3
    NumericExercise(
        id="7.3", title="From a BIC difference to a Bayes factor",
        answer=float(np.exp(3.0)), atol=0.05,
        distractors=[(float(np.exp(6.0)), "You forgot the factor 1/2: the log evidence is approximately −BIC/2, so "
                                          "ln BF ≈ ΔBIC/2 = 3."),
                     (3.0, "3 is the *log* Bayes factor (ΔBIC/2); the question asks for the Bayes factor itself, exp(3)."),
                     (float(np.exp(-3.0)), "You inverted the ratio: the Bayes factor is in favour of the model with "
                                           "the *lower* BIC, so it is larger than 1.")],
        hints=["The BIC approximates −2 × log evidence, so the log Bayes factor between two models is ΔBIC / 2.",
               "ΔBIC = 6 gives ln BF = 3.",
               "BF = exp(3) ≈ 20.1"],
        explanation="A BIC difference of 6 corresponds to a Bayes factor of about 20 in favour of the model with the "
                    "lower BIC: the data are about 20 times more probable under it. Kass & Raftery (1995) call "
                    "2 ln BF between 6 and 10 'strong' evidence.",
        check_call='check("7.3", float(np.exp(3.0)))',
    ),
    # ------------------------------------------------------------------ 7.4
    ChoiceExercise(
        id="7.4", title="AIC versus BIC",
        question="For a data set of n = 150 choices, why does BIC penalise an extra parameter more heavily than AIC?",
        options={"a": "Because BIC multiplies the negative log-likelihood by n.",
                 "b": "Because BIC's penalty per parameter is ln(n) ≈ 5.0, larger than AIC's 2; this is the case whenever n > e² ≈ 7.4.",
                 "c": "Because BIC uses base-10 logarithms.",
                 "d": "It does not: AIC always penalises more."},
        correct="b",
        feedback={"a": "Both criteria use the same deviance term 2 × nll; they differ only in the penalty.",
                  "c": "BIC uses the natural logarithm; the base is not what makes it stricter.",
                  "d": "Compare the penalties: 2 per parameter (AIC) versus ln(n) per parameter (BIC)."},
        hints=["Write the two penalties side by side: 2k versus k ln(n).",
               "For which n is ln(n) larger than 2?"],
        explanation="BIC's penalty grows with the sample size, AIC's does not. The two answer different questions: "
                    "AIC asks which model will predict new data best, BIC asks which model is most probably the "
                    "data-generating one. With plenty of data they often agree; when they disagree, say so.",
        check_call='check("7.4", "b")',
    ),
    # ------------------------------------------------------------------ 7.5
    CodeExercise(
        id="7.5", title="Scoring held-out trials",
        reference=nll_rw_on_trials,
        signature="nll_rw_on_trials(params, choices, rewards, test_mask, n_arms=2, q0=0.5)",
        cases=[{"params": (0.3, 4.0), "choices": _C, "rewards": _R, "test_mask": _MASK_HALF},
               {"params": (0.6, 2.0), "choices": _C, "rewards": _R, "test_mask": _MASK_ODD},
               {"params": (0.2, 6.0), "choices": _C, "rewards": _R, "test_mask": _MASK_ALL}],
        bug_variants={
            "subset_only": (_subset_only, "You dropped the training trials and replayed the model on the test trials "
                                          "only. The learning history matters: replay the *whole* sequence and only "
                                          "*score* the trials where `test_mask` is True."),
            "ignore_mask": (_ignore_mask, "You scored every trial. Only the trials where `test_mask` is True should "
                                          "contribute to the sum."),
            "learn_only_on_test": (_learn_only_on_test, "You only update the values on test trials. The agent learned "
                                                        "from every trial — update on all of them, score only the test ones."),
        },
        hints=["Copy the loop of `models.nll_rw` and add one condition: subtract log p[c] only when `test_mask[t]` is True.",
               "The update `Q[c] = Q[c] + alpha * (r - Q[c])` must happen on *every* trial, inside the loop but outside the condition.",
               "`for t, (c, r) in enumerate(zip(choices, rewards)):` gives you the trial index t to look up `test_mask[t]`."],
        explanation="Cross-validating a learning model is subtle: the held-out trials cannot be evaluated in isolation "
                    "because the model's predictions on them depend on everything that happened before. The "
                    "solution is to replay the full sequence and sum the log-likelihood only over the held-out trials.",
        common_mistakes="Evaluating the model on the test trials alone (losing the history); scoring all trials; "
                        "updating only on test trials.",
        check_call='check("7.5", nll_rw_on_trials)',
    ),
    # ------------------------------------------------------------------ 7.6
    CodeExercise(
        id="7.6", title="Random-effects Bayesian model selection",
        reference=bms_alpha,
        signature="bms_alpha(log_evidence, alpha0=1.0, max_iter=500, tol=1e-8)",
        cases=[{"log_evidence": _L1}, {"log_evidence": _L2, "alpha0": 1.0}],
        rtol=1e-3, atol=1e-3,
        bug_variants={
            "no_digamma": (_bms_no_digamma, "You weighted the models by the raw evidence only. The E-step must add the "
                                            "expected log frequency ψ(α_k) − ψ(Σα) of each model, and the two steps "
                                            "must be iterated until α stops changing."),
            "no_alpha0": (_bms_no_alpha0, "The M-step adds the prior counts: α = α0 + Σ_n g_nk."),
            "one_iteration": (_bms_one_iteration, "You stopped after one iteration. Repeat the E-step and M-step until α converges."),
            "wrong_axis": (_bms_wrong_axis, "The posterior over models must be normalised *within each subject* "
                                            "(over the model axis, axis=1), not across subjects."),
        },
        hints=["Keep two quantities: α (one entry per model, start at α0) and g (one row per subject, one column per "
               "model). Alternate: E-step computes g from α and the evidence; M-step computes α from g.",
               "E-step: `log_u = L + digamma(alpha) - digamma(alpha.sum())`, then normalise within each row: "
               "`g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))`.",
               "M-step: `alpha = alpha0 + g.sum(axis=0)`; stop when the largest change in α is below `tol`."],
        explanation="The algorithm is an expectation–maximisation-like scheme: given the current belief about how "
                    "common each model is (α), assign each subject to the models (g); given the assignments, update "
                    "the counts (α). The digamma terms are the Bayesian correction that makes rarely-used models "
                    "less attractive, and the log-sum-exp keeps the numbers finite.",
        common_mistakes="Leaving out the digamma terms, forgetting α0, normalising over the wrong axis, or not iterating.",
        check_call='check("7.6", bms_alpha)',
    ),
    # ------------------------------------------------------------------ 7.7
    CodeExercise(
        id="7.7", title="Exceedance probabilities",
        reference=exceedance_probabilities,
        signature="exceedance_probabilities(alpha, n_samples, rng)",
        cases=[{"alpha": np.array([8.0, 3.0, 1.0]), "n_samples": 20000, "rng": np.random.default_rng(71)},
               {"alpha": np.array([2.0, 2.0]), "n_samples": 20000, "rng": np.random.default_rng(72)},
               {"alpha": np.array([1.5, 6.0, 4.0, 1.5]), "n_samples": 20000, "rng": np.random.default_rng(73)}],
        compare=_xp_compare,
        bug_variants={
            "expected_frequencies": (_expected_frequencies, "You returned the expected frequencies α / Σα. The "
                                                            "exceedance probability of model k is the probability "
                                                            "that r_k is the *largest* frequency — estimate it by "
                                                            "sampling r from the Dirichlet."),
        },
        hints=["`rng.dirichlet(alpha, size=n_samples)` returns an array (n_samples, n_models): each row is one draw of the frequencies r.",
               "For each draw find the model with the largest frequency (`np.argmax(..., axis=1)`).",
               "Count how often each model wins and divide by `n_samples` (`np.bincount(winners, minlength=len(alpha)) / n_samples`)."],
        explanation="The exceedance probability turns the Dirichlet posterior into a single number per model: how "
                    "sure are we that this model is the most common one in the population? Because it is a "
                    "Monte Carlo estimate, your numbers can differ from the reference by a percent or two.",
        common_mistakes="Returning the expected frequencies; returning counts instead of proportions.",
        check_call='check("7.7", exceedance_probabilities)',
    ),
    # ------------------------------------------------------------------ 7.8
    NumericExercise(
        id="7.8", title="Expected model frequency",
        answer=12.3 / 18.0, atol=2e-3,
        distractors=[(11.3 / 15.0, "You subtracted the prior counts (α0 = 1) before dividing; the expected frequency "
                                   "under a Dirichlet(α) is simply α_k / Σ_j α_j."),
                     (1 / 3, "That is the prior expected frequency; use the posterior α."),
                     (12.3, "Divide by the sum of all α to turn the count into a frequency.")],
        hints=["The expected value of r_k under Dirichlet(α) is α_k divided by the sum of all α_j.",
               "Σα = 12.3 + 4.2 + 1.5 = 18.",
               "12.3 / 18"],
        explanation="r_1 = 12.3/18 ≈ 0.68: the model is estimated to be used by about two thirds of the population. "
                    "The expected frequency is a *proportion of people*, not a probability that the model is correct.",
        check_call='check("7.8", 12.3 / 18.0)',
    ),
    # ------------------------------------------------------------------ 7.9
    ChoiceExercise(
        id="7.9", title="What an exceedance probability means",
        question="An exceedance probability of 0.90 for model B means ...",
        options={"a": "90% of the participants are best described by model B.",
                 "b": "The posterior probability that model B is more frequent in the population than any other candidate model is 0.90.",
                 "c": "Model B is the true model with probability 0.90.",
                 "d": "Model B's parameters are recovered with 90% accuracy."},
        correct="b",
        feedback={"a": "That would be an expected frequency r_B = 0.9. The exceedance probability is about which "
                       "model is the *most* frequent, not about how frequent it is.",
                  "c": "Bayesian model selection compares the candidates you supplied; none of them has to be true "
                       "(see the posterior predictive checks of section 7.5).",
                  "d": "Exceedance probabilities say nothing about parameter recovery (lesson 6)."},
        hints=["Exceedance probabilities are statements about the population frequencies r, which are uncertain.",
               "φ_k = P(r_k > r_j for all j ≠ k | data)."],
        explanation="An exceedance probability answers 'is this the most common model?', not 'how common is it?' "
                    "(that is r) and not 'is it right?' (that needs absolute fit). Report it together with the "
                    "expected frequencies and, ideally, its protected version.",
        check_call='check("7.9", "b")',
    ),
    # ------------------------------------------------------------------ 7.10
    FreeTextExercise(
        id="7.10", title="Winning is not being right",
        question="Model B has the lowest BIC for every participant and a protected exceedance probability of 0.99. Explain, in 2–4 sentences, why this does not establish that model B describes how participants actually learn, and what you would do about it.",
        model_answer="Model comparison is relative: it only says that B is the best of the candidates you happened to "
                     "include, and all of them may be mis-specified — the true process may simply be absent from the "
                     "set. To check absolute fit you simulate data from model B with the fitted parameters (a "
                     "posterior predictive check) and compare qualitative summary statistics — learning curves, "
                     "stay probabilities, the reward × transition pattern — with those of the real data. Patterns "
                     "the model cannot reproduce falsify it and tell you which mechanism is missing, so you add a "
                     "candidate that has it and compare again.",
        concepts=[("comparison is relative to the candidate set",
                   ["relative", "candidate", "among the models", "compared to", "of the models", "best of",
                    "considered", "included", "set of models", "only tells", "only says"]),
                  ("simulate / posterior predictive check",
                   ["simulat", "posterior predictive", "generate", "reproduce", "predict", "qualitative", "pattern",
                    "summary statistic", "learning curve", "stay prob"]),
                  ("mis-specification / missing mechanism",
                   ["mis-specif", "misspecif", "missing", "not in the set", "wrong model", "true model", "absent",
                    "mechanism", "another model", "new model", "add a model", "falsif"])],
        min_concepts=2,
        hints=["What does 'best' mean when every candidate could be wrong?",
               "Which analysis from lesson 6 compares a model's *simulated* behaviour with the data?"],
        explanation="Relative comparison and absolute validation answer different questions; a complete analysis "
                    "reports both (Palminteri, Wyart & Koechlin 2017).",
        check_call='check("7.10", """Model comparison is relative: it only says that B is the best among the candidate models, and all of them may be mis-specified. To check absolute fit, simulate data from model B with the fitted parameters (a posterior predictive check) and compare qualitative patterns such as learning curves and stay probabilities with the real data; patterns the model cannot reproduce falsify it and point to a missing mechanism.""")',
    ),
]
