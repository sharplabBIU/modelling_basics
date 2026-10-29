"""Exercises of lesson 2: probability and Bayes for modellers."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

_OUTCOMES_7_OF_10 = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]


# --------------------------------------------------------------------------- #
# Buggy variants (used only for diagnosis)
# --------------------------------------------------------------------------- #


def _ll_not_logged(p, outcomes):
    outcomes = np.asarray(outcomes, float)
    return float(np.prod(p ** outcomes * (1 - p) ** (1 - outcomes)))


def _ll_swapped(p, outcomes):
    outcomes = np.asarray(outcomes, float)
    return float(np.sum(outcomes * np.log(1 - p) + (1 - outcomes) * np.log(p)))


def _ll_successes_only(p, outcomes):
    outcomes = np.asarray(outcomes, float)
    return float(np.sum(outcomes * np.log(p)))


def _ll_negative(p, outcomes):
    return -models.bernoulli_log_likelihood(p, outcomes)


def _post_unnormalised(prior, likelihood):
    return np.asarray(prior, float) * np.asarray(likelihood, float)


def _post_added(prior, likelihood):
    s = np.asarray(prior, float) + np.asarray(likelihood, float)
    return s / s.sum()


def _post_prior_normalised_only(prior, likelihood):
    prior = np.asarray(prior, float)
    return prior / prior.sum() * np.asarray(likelihood, float)


def _beta_both_successes(a0, b0, outcomes):
    k = float(np.sum(outcomes))
    return a0 + k, b0 + k


def _beta_swapped(a0, b0, outcomes):
    outcomes = np.asarray(outcomes, float)
    k = float(np.sum(outcomes))
    return a0 + len(outcomes) - k, b0 + k


def _beta_b_all_trials(a0, b0, outcomes):
    k = float(np.sum(outcomes))
    return a0 + k, b0 + len(outcomes)


_P_GREATER_HALF = float(stats.beta.sf(0.5, 8, 4))   # P(p > 0.5 | Beta(8, 4)) = 0.8867...

_GRID = np.linspace(0.01, 0.99, 99)
_LIK_GRID = np.array([models.bernoulli_likelihood(p, _OUTCOMES_7_OF_10) for p in _GRID])

EXERCISES = [
    # ------------------------------------------------------------------ 2.1
    CodeExercise(
        id="2.1", title="The Bernoulli log-likelihood",
        reference=models.bernoulli_log_likelihood,
        signature="bernoulli_log_likelihood(p, outcomes)",
        cases=[{"p": 0.7, "outcomes": _OUTCOMES_7_OF_10},
               {"p": 0.5, "outcomes": [1, 0]},
               {"p": 0.9, "outcomes": np.array([1, 1, 1])},
               {"p": 0.25, "outcomes": [0, 0, 0, 1]}],
        bug_variants={
            "not_logged": (_ll_not_logged, "You returned the likelihood itself (a product of probabilities). The "
                                           "*log*-likelihood is the sum of the logs of those probabilities."),
            "swapped": (_ll_swapped, "You attached log(1 − p) to the successes and log(p) to the failures — the other "
                                     "way round: a success (outcome 1) has probability p."),
            "successes_only": (_ll_successes_only, "You only counted the successes. Each failure (outcome 0) "
                                                   "contributes log(1 − p) as well."),
            "negative": (_ll_negative, "You returned the *negative* log-likelihood. Here we want the log-likelihood "
                                       "itself (the negative version is what we minimise in lesson 5)."),
        },
        hints=["The probability of one outcome x is p if x = 1 and 1 − p if x = 0, which can be written in one "
               "line as p^x (1 − p)^(1 − x).",
               "The log of a product is the sum of the logs: each outcome contributes x log p + (1 − x) log(1 − p).",
               "`np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p))`"],
        explanation="Each outcome contributes log p (success) or log(1 − p) (failure); summing over outcomes "
                    "gives the log of the probability of the whole sequence. With k successes in n trials this "
                    "is k log p + (n − k) log(1 − p): the order of the outcomes does not matter, only the counts.",
        common_mistakes="Returning the product instead of the sum of logs; swapping p and 1 − p; forgetting the "
                        "failure term; returning the negative log-likelihood.",
        check_call='check("2.1", bernoulli_log_likelihood)',
    ),
    # ------------------------------------------------------------------ 2.2
    NumericExercise(
        id="2.2", title="A log-likelihood by hand",
        answer=3 * np.log(0.75) + np.log(0.25), atol=2e-3,
        distractors=[(float(np.exp(3 * np.log(0.75) + np.log(0.25))), "That is the likelihood (the probability of the "
                                                                        "sequence); the question asks for its logarithm."),
                     (3 * np.log(0.75), "You forgot the unrewarded trial, which contributes log(1 − 0.75) = log 0.25."),
                     (float(np.log(0.25)), "You only counted the failure; each of the three successes contributes log 0.75."),
                     (-(3 * np.log(0.75) + np.log(0.25)), "Right magnitude, wrong sign: a log-likelihood of probabilities "
                                                           "below 1 is negative."),
                     (float(np.log(0.75) + np.log(0.25)), "You counted only one success; there are three.")],
        hints=["Three successes each contribute log 0.75, one failure contributes log(1 − 0.75) = log 0.25.",
               "log 0.75 ≈ −0.2877 and log 0.25 ≈ −1.3863 (natural logarithms).",
               "3 × (−0.2877) + (−1.3863)"],
        explanation="3 log 0.75 + log 0.25 ≈ −0.863 − 1.386 = −2.249. Note how a single failure at p = 0.75 "
                    "'costs' more than three successes together: unlikely outcomes dominate the log-likelihood.",
        check_call='check("2.2", 3 * np.log(0.75) + np.log(0.25))',
    ),
    # ------------------------------------------------------------------ 2.3
    NumericExercise(
        id="2.3", title="The maximum-likelihood estimate",
        answer=0.7, atol=1e-3,
        distractors=[(0.3, "That is the proportion of *failures*; the parameter p is the probability of a success."),
                     (7, "That is the number of successes; the estimate is a probability between 0 and 1."),
                     (float(7 / 11), "7/11 is not the maximum of the likelihood; with a *uniform prior* the posterior "
                                     "mean would be 8/12, but the MLE is simply the proportion of successes."),
                     (0.5, "The likelihood of 7 successes in 10 trials is not maximal at 0.5 — plot it or take the derivative.")],
        hints=["Write the log-likelihood as k log p + (n − k) log(1 − p) with k = 7 and n = 10, and find the p at "
               "which its derivative is zero.",
               "The derivative is k/p − (n − k)/(1 − p); set it to 0 and solve for p.",
               "p = k / n"],
        explanation="Setting k/p = (n − k)/(1 − p) gives p = k/n = 0.7: the maximum-likelihood estimate of a "
                    "Bernoulli probability is the observed proportion of successes. (You could also read it off the "
                    "likelihood plot — the maximum is at 0.7.)",
        check_call='check("2.3", 7 / 10)',
    ),
    # ------------------------------------------------------------------ 2.4
    ChoiceExercise(
        id="2.4", title="Why work with log-likelihoods?",
        question="Why do we work with the log-likelihood rather than the likelihood itself?",
        options={"a": "Taking the log turns the likelihood into a probability distribution over the parameter.",
                 "b": "Products of many small probabilities underflow to zero in floating point, whereas sums of logs "
                      "do not; sums are also easier to differentiate and optimise, and the maximum is at the same place.",
                 "c": "Bayes' rule can only be applied to logarithms.",
                 "d": "The log-likelihood is always positive, which optimisers require."},
        correct="b",
        feedback={"a": "No transformation of the likelihood makes it a distribution over the parameter; that needs a "
                       "prior and normalisation (section 2.3).",
                  "c": "Bayes' rule works with probabilities directly; logs are a numerical convenience.",
                  "d": "Log-probabilities are negative (probabilities are below 1); optimisers do not care about sign."},
        hints=["Try `np.prod(np.full(500, 0.3))` in a cell.",
               "The log is a monotonic transformation: the largest likelihood has the largest log-likelihood."],
        explanation="The log is monotonic, so maximising the log-likelihood finds the same parameter as maximising "
                    "the likelihood; but the log turns a product of hundreds of probabilities (which underflows to "
                    "0.0) into a sum of ordinary-sized numbers, and sums are easy to differentiate.",
        check_call='check("2.4", "b")',
    ),
    # ------------------------------------------------------------------ 2.5
    CodeExercise(
        id="2.5", title="Bayes' rule on a grid",
        reference=models.grid_posterior,
        signature="grid_posterior(prior, likelihood)",
        cases=[{"prior": np.full(99, 1 / 99), "likelihood": _LIK_GRID},
               {"prior": stats.beta.pdf(_GRID, 10, 10), "likelihood": _LIK_GRID},
               {"prior": np.array([1.0, 2.0, 3.0, 4.0]), "likelihood": np.array([0.5, 0.25, 0.125, 0.0625])}],
        bug_variants={
            "unnormalised": (_post_unnormalised, "You multiplied prior and likelihood but did not normalise: the "
                                                 "posterior must sum to 1 (divide by the sum of the products)."),
            "added": (_post_added, "You *added* prior and likelihood. Bayes' rule multiplies them."),
            "prior_only": (_post_prior_normalised_only, "You normalised the prior but not the product. Normalise "
                                                        "*after* multiplying by the likelihood."),
        },
        hints=["The posterior is proportional to prior × likelihood, evaluated at each grid point.",
               "Multiply the two arrays element-wise, then divide by the sum so that the result sums to 1.",
               "`unnormalised = prior * likelihood; return unnormalised / np.sum(unnormalised)`"],
        explanation="On a grid, Bayes' rule is two lines: multiply the prior by the likelihood at every grid point, "
                    "then divide by the sum. The division is the 'evidence' P(data) in disguise — it is simply "
                    "whatever constant makes the posterior sum to 1. Note that the prior need not be normalised "
                    "either: any constant cancels.",
        common_mistakes="Forgetting to normalise; adding instead of multiplying; normalising before multiplying.",
        check_call='check("2.5", grid_posterior)',
    ),
    # ------------------------------------------------------------------ 2.6
    CodeExercise(
        id="2.6", title="The Beta–Bernoulli update",
        reference=models.beta_posterior,
        signature="beta_posterior(a0, b0, outcomes)",
        cases=[{"a0": 1, "b0": 1, "outcomes": _OUTCOMES_7_OF_10},
               {"a0": 2, "b0": 5, "outcomes": [0, 0, 0]},
               {"a0": 10, "b0": 10, "outcomes": np.array([1])},
               {"a0": 1, "b0": 1, "outcomes": []}],
        bug_variants={
            "both_successes": (_beta_both_successes, "You added the number of successes to *both* a and b. "
                                                     "b counts the failures: b0 + (number of trials − number of successes)."),
            "swapped": (_beta_swapped, "You added the failures to a and the successes to b — the other way round: "
                                       "a counts successes."),
            "b_all_trials": (_beta_b_all_trials, "You added the total number of trials to b; b should only grow by the "
                                                 "number of *failures*."),
        },
        hints=["a counts successes and b counts failures: each outcome adds 1 to one of them.",
               "Number of successes = `np.sum(outcomes)`; number of failures = `len(outcomes) - np.sum(outcomes)`.",
               "`return a0 + n_success, b0 + n_failure`"],
        explanation="Because Beta(a, b) ∝ p^(a−1) (1 − p)^(b−1) has the same form as the Bernoulli likelihood "
                    "p^k (1 − p)^(n−k), the posterior is again a Beta distribution with the counts added: this is "
                    "what *conjugacy* means. a and b behave like pseudo-counts of successes and failures.",
        common_mistakes="Adding the successes to both parameters; adding the total number of trials instead of the "
                        "number of failures; returning the posterior mean instead of (a, b).",
        check_call='check("2.6", beta_posterior)',
    ),
    # ------------------------------------------------------------------ 2.7
    NumericExercise(
        id="2.7", title="The posterior mean",
        answer=8 / 12, atol=1e-3,
        distractors=[(0.7, "That is the maximum-likelihood estimate (and the posterior *mode*). The posterior "
                           "*mean* is pulled slightly towards the prior mean 0.5: a/(a + b) with a = 8, b = 4."),
                     (float(7 / 11), "You forgot that the Beta(1, 1) prior adds one pseudo-count to *each* of a and b: "
                                     "the posterior is Beta(8, 4), not Beta(7, 4)."),
                     (0.5, "That is the prior mean; the 10 observations move it a long way."),
                     (float(8 / 11), "Check b: with 3 failures and b0 = 1, b = 4 (so a + b = 12).")],
        hints=["With a Beta(1, 1) prior and 7 successes / 3 failures the posterior is Beta(1 + 7, 1 + 3).",
               "The mean of Beta(a, b) is a / (a + b).",
               "8 / (8 + 4)"],
        explanation="Beta(8, 4) has mean 8/12 ≈ 0.667, slightly below the MLE of 0.7: the single pseudo-success "
                    "and pseudo-failure of the uniform prior pull the estimate towards 0.5. With more data the "
                    "pull vanishes (the pseudo-counts become negligible).",
        check_call='check("2.7", 8 / 12)',
    ),
    # ------------------------------------------------------------------ 2.8
    NumericExercise(
        id="2.8", title="A posterior probability",
        answer=_P_GREATER_HALF, atol=2e-3,
        distractors=[(1 - _P_GREATER_HALF, "That is P(p < 0.5), the complement of what was asked."),
                     (float(8 / 12), "That is the posterior *mean*, not the probability that p exceeds 0.5."),
                     (0.5, "The posterior is far from symmetric around 0.5 — look at its plot."),
                     (0.7, "That is the MLE; the question asks for an area under the posterior density.")],
        hints=["You need the area under the Beta(8, 4) density to the right of 0.5.",
               "`scipy.stats.beta.cdf(x, a, b)` gives the area to the *left* of x; the survival function "
               "`scipy.stats.beta.sf(x, a, b)` gives the area to the right.",
               "`stats.beta.sf(0.5, 8, 4)`"],
        explanation="P(p > 0.5 | Beta(8, 4)) ≈ 0.887. Statements like this are the pay-off of having a posterior: "
                    "a likelihood alone cannot tell you how probable it is that the option is better than chance.",
        check_call='check("2.8", stats.beta.sf(0.5, 8, 4))',
    ),
    # ------------------------------------------------------------------ 2.9
    ChoiceExercise(
        id="2.9", title="Which one is a distribution over the parameter?",
        question="Which of the following is a probability distribution over the parameter p?",
        options={"a": "the likelihood P(data | p), viewed as a function of p",
                 "b": "the posterior P(p | data)",
                 "c": "the Bernoulli probability P(x | p) of a single outcome",
                 "d": "the evidence P(data)"},
        correct="b",
        feedback={"a": "The likelihood is a function of p, but it is not a distribution over p: it does not integrate "
                       "to 1 (and nothing forces it to).",
                  "c": "P(x | p) is a distribution over the *outcome* x for a fixed p.",
                  "d": "The evidence is a single number (the normalising constant of Bayes' rule), not a distribution."},
        hints=["Ask of each object: what is random, and what is fixed? Which one assigns probabilities to values of p?",
               "Only one of the four integrates to 1 over p."],
        explanation="Only the posterior assigns probabilities to values of p. The likelihood also varies with p, "
                    "which is why it is so easily confused with a distribution over p — but it is the probability "
                    "of the *data*, and the area under it as a function of p is arbitrary.",
        check_call='check("2.9", "b")',
    ),
    # ------------------------------------------------------------------ 2.10
    FreeTextExercise(
        id="2.10", title="Likelihood versus posterior",
        question="In 2–4 sentences: what is the difference between P(data | θ) and P(θ | data)?",
        model_answer="P(data | θ) is the likelihood: with the data held fixed it is a function of the parameter θ "
                     "that says how probable the observed data would be for each candidate value of θ. It is not a "
                     "probability distribution over θ (it need not integrate to 1), so it cannot by itself tell us "
                     "how probable any value of θ is. P(θ | data) is the posterior: a genuine probability "
                     "distribution over θ expressing our belief after seeing the data. Bayes' rule connects the "
                     "two: the posterior is proportional to the likelihood times the prior, normalised by the "
                     "evidence P(data) — so turning a likelihood into a posterior always requires a prior.",
        concepts=[("likelihood: data fixed, function of θ, not a distribution over θ",
                   ["function of", "not a distribution", "not a probability distribution", "does not integrate",
                    "doesn't integrate", "not sum", "doesn't sum", "does not sum", "data fixed", "data are fixed",
                    "data is fixed", "fixed data", "given the parameter", "given theta", "given θ",
                    "how probable the data", "probability of the data", "how likely the data", "probability of observing"]),
                  ("posterior: a distribution over θ expressing belief, requires a prior",
                   ["prior", "distribution over", "belief", "degree of", "credib", "plausib", "how probable the parameter",
                    "probability of the parameter", "probability of theta", "probability of θ"]),
                  ("Bayes' rule / normalisation connects them",
                   ["bayes", "normalis", "normaliz", "evidence", "marginal", "proportional", "∝", "multiply",
                    "times the prior", "product", "invert"])],
        min_concepts=2,
        hints=["For each object, ask: what is held fixed and what varies? And does it integrate to 1 over θ?",
               "What extra ingredient does Bayes' rule need to turn one into the other?"],
        explanation="The likelihood answers 'how probable are these data for a given θ?'; the posterior answers "
                    "'how probable is each θ given these data?'. The second requires a prior and normalisation; "
                    "the first does not. Keeping them apart avoids the most common misreading of a fitted model.",
        check_call='check("2.10", """P(data | θ) is the likelihood: with the data held fixed it is a function of the parameter that says how probable the observed data are for each value of θ; it is not a probability distribution over θ and does not integrate to 1. P(θ | data) is the posterior, a genuine distribution over θ expressing our belief after seeing the data. Bayes\' rule connects them: the posterior is proportional to the likelihood times the prior, normalised by the evidence, so turning a likelihood into a posterior always requires a prior.""")',
    ),
]
