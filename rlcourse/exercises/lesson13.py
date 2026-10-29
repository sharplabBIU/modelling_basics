"""Exercises of lesson 13: distributional reinforcement learning."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# deterministic test data
# --------------------------------------------------------------------------- #
_rng = np.random.default_rng(13)
_normal = _rng.normal(2.0, 1.0, 400)
_skewed = _rng.exponential(1.5, 400)
_bimodal = np.where(_rng.random(500) < 0.4, _rng.normal(5.0, 0.5, 500), _rng.normal(1.0, 0.5, 500))
_taus20 = np.linspace(0.05, 0.95, 20)
_alpha_pos20 = 0.06 * _taus20
_alpha_neg20 = 0.06 * (1 - _taus20)
_rewards_bernoulli = (_rng.random(60) < 0.5).astype(float)
_rewards_cont = _rng.normal(1.0, 1.0, 40)


# --------------------------------------------------------------------------- #
# lesson-specific reference function (exercise 13.9)
# --------------------------------------------------------------------------- #
def asymmetric_value_trajectory(rewards, alpha_pos, alpha_neg, v0=0.5):
    """Values of a single learner with asymmetric learning rates.

    Returns an array of length n + 1: the value before each of the n trials,
    followed by the value after the last update."""
    rewards = np.asarray(rewards, dtype=float)
    V = np.zeros(len(rewards) + 1)
    V[0] = v0
    v = v0
    for t, r in enumerate(rewards):
        delta = r - v                                     # prediction error
        alpha = alpha_pos if delta > 0 else alpha_neg     # which learning rate applies
        v = v + alpha * delta
        V[t + 1] = v
    return V


# --------------------------------------------------------------------------- #
# buggy variants used for targeted diagnoses
# --------------------------------------------------------------------------- #
def _expectile_as_quantile(samples, tau, n_iter=200):
    return float(np.quantile(samples, tau))


def _expectile_mirrored(samples, tau, n_iter=200):
    return models.expectile(samples, 1 - tau, n_iter)


def _expectile_mean(samples, tau, n_iter=200):
    return float(np.mean(samples))


def _etd_swapped(values, reward, alpha_pos, alpha_neg):
    return models.expectile_td_update(values, reward, alpha_neg, alpha_pos)


def _etd_symmetric(values, reward, alpha_pos, alpha_neg):
    values = np.asarray(values, float)
    return values + np.asarray(alpha_pos, float) * (reward - values)


def _etd_positive_only(values, reward, alpha_pos, alpha_neg):
    values = np.asarray(values, float)
    delta = reward - values
    return values + np.where(delta > 0, alpha_pos, 0.0) * delta


def _etd_change_only(values, reward, alpha_pos, alpha_neg):
    values = np.asarray(values, float)
    delta = reward - values
    return np.where(delta > 0, alpha_pos, alpha_neg) * delta


def _loss_mirrored(e, samples, tau):
    return models.expectile_loss(e, samples, 1 - tau)


def _loss_absolute(e, samples, tau):
    x = np.asarray(samples, float)
    return float(np.mean(np.where(x > e, tau, 1 - tau) * np.abs(x - e)))


def _qtd_reversed(values, reward, alpha, taus):
    values = np.asarray(values, float)
    return values + alpha * (np.asarray(taus, float) - (reward > values).astype(float))


def _qtd_scaled(values, reward, alpha, taus):
    values = np.asarray(values, float)
    return values + alpha * (np.asarray(taus, float) - (reward < values).astype(float)) * np.abs(reward - values)


def _qtd_flipped(values, reward, alpha, taus):
    values = np.asarray(values, float)
    return values - alpha * (np.asarray(taus, float) - (reward < values).astype(float))


def _traj_swapped(rewards, alpha_pos, alpha_neg, v0=0.5):
    return asymmetric_value_trajectory(rewards, alpha_neg, alpha_pos, v0)


def _traj_symmetric(rewards, alpha_pos, alpha_neg, v0=0.5):
    return asymmetric_value_trajectory(rewards, alpha_pos, alpha_pos, v0)


def _traj_after_only(rewards, alpha_pos, alpha_neg, v0=0.5):
    return asymmetric_value_trajectory(rewards, alpha_pos, alpha_neg, v0)[1:]


EXERCISES = [
    # ------------------------------------------------------------------ 13.1
    ChoiceExercise(
        id="13.1", title="Why go beyond the mean?",
        question="Option A always pays 0.5. Option B pays 1 with probability 0.5 and 0 otherwise. "
                 "A learner that represents each option by a single expected value (lesson 3) ...",
        options={"a": "will, after enough trials, assign B a higher value because it sometimes pays 1",
                 "b": "will assign both options the same value and therefore cannot represent that B is riskier",
                 "c": "will assign A a higher value because its outcomes are less variable",
                 "d": "cannot learn the value of B at all because its outcomes vary"},
        correct="b",
        feedback={"a": "The expected value of B is 0.5 × 1 + 0.5 × 0 = 0.5, the same as A. A mean-tracking learner converges to the same number for both.",
                  "c": "A mean-only learner has no term for variability, so it has no reason to prefer the safer option.",
                  "d": "It can learn the mean perfectly well (the delta rule averages the outcomes); what it loses is everything *about* the spread."},
        hints=["Compute the expected value of each option.",
               "If two options have the same mean, a learner that stores only the mean ends up with the same number for both — whatever their spread."],
        explanation="Both options have expected value 0.5, so any learner that summarises an option by its mean "
                    "treats them as identical. Preferences for or against risk, and the possibility that the brain "
                    "tracks the spread of outcomes, are exactly what distributional RL adds.",
        check_call='check("13.1", "b")',
    ),
    # ------------------------------------------------------------------ 13.2
    CodeExercise(
        id="13.2", title="Computing an expectile",
        reference=models.expectile,
        signature="expectile(samples, tau, n_iter=200)",
        cases=[{"samples": _normal, "tau": 0.5},
               {"samples": _normal, "tau": 0.1},
               {"samples": _skewed, "tau": 0.9},
               {"samples": _bimodal, "tau": 0.7},
               {"samples": np.array([0.0, 1.0] * 50), "tau": 0.8}],
        rtol=1e-4, atol=1e-6,
        bug_variants={
            "quantile": (_expectile_as_quantile, "You computed the τ-*quantile* (the value with a fraction τ of the "
                                                 "samples below it). The expectile is a *weighted mean*: samples above it "
                                                 "weigh τ, samples below it weigh 1 − τ."),
            "mirrored": (_expectile_mirrored, "Your weights are the wrong way round: samples *above* the current "
                                              "estimate get weight τ, samples *below* it get weight 1 − τ."),
            "mean": (_expectile_mean, "You returned the plain mean — that is the τ = 0.5 expectile only. "
                                      "The weights must depend on τ."),
        },
        hints=["The τ-expectile e is a weighted mean of the samples in which samples above e have weight τ and samples "
               "below e have weight 1 − τ. The catch: the weights depend on e itself.",
               "Solve it by iteration: start from the ordinary mean, compute the weights given the current e, take the "
               "weighted mean, and repeat until e stops changing.",
               "`e = np.mean(x)`; loop: `weights = np.where(x > e, tau, 1 - tau)`; `e = np.sum(weights * x) / np.sum(weights)`."],
        explanation="The expectile is a fixed point: e = Σ w_i x_i / Σ w_i with w_i = τ for x_i > e and 1 − τ otherwise. "
                    "At τ = 0.5 all weights are equal and e is the mean. Iterating the weighted mean converges "
                    "quickly because the loss is strictly convex. Expectiles were introduced by Newey & Powell (1987) "
                    "as 'asymmetric least squares'.",
        common_mistakes="Confusing expectiles with quantiles, putting the weights the wrong way round, or stopping "
                        "after one weighted mean instead of iterating to convergence.",
        check_call='check("13.2", expectile)',
    ),
    # ------------------------------------------------------------------ 13.3
    NumericExercise(
        id="13.3", title="An expectile by hand",
        answer=0.8, atol=1e-6,
        distractors=[(0.5, "That is the mean (the τ = 0.5 expectile). Which side of the mean does τ = 0.8 pull towards?"),
                     (1.0, "That is the 0.8-*quantile* (80% of the mass is at or below 1). Expectiles weigh the "
                           "distances to the outcomes, not just their ranks, so the answer lies strictly between 0 and 1."),
                     (0.2, "That is the 0.2-expectile — the weights are the wrong way round.")],
        hints=["Write the defining balance: τ × E[(X − e)+] = (1 − τ) × E[(e − X)+], where (·)+ keeps only positive parts.",
               "With outcomes 0 and 1 (each with probability 0.5): E[(X − e)+] = 0.5 (1 − e) and E[(e − X)+] = 0.5 e.",
               "0.8 × 0.5 (1 − e) = 0.2 × 0.5 e  →  solve for e."],
        explanation="0.8 (1 − e) = 0.2 e gives e = 0.8. For a 50/50 outcome of 0 or 1 the τ-expectile is simply τ: an "
                    "'optimistic' predictor (τ > 0.5) sits above the mean, a 'pessimistic' one below it, and the whole "
                    "family of expectiles between 0 and 1 traces out the distribution.",
        check_call='check("13.3", 0.8)',
    ),
    # ------------------------------------------------------------------ 13.4
    CodeExercise(
        id="13.4", title="Distributional TD with asymmetric learning rates",
        reference=models.expectile_td_update,
        signature="expectile_td_update(values, reward, alpha_pos, alpha_neg)",
        cases=[{"values": np.array([0.2, 0.5, 0.8]), "reward": 1.0, "alpha_pos": np.array([0.1, 0.2, 0.3]), "alpha_neg": np.array([0.3, 0.2, 0.1])},
               {"values": np.array([0.2, 0.5, 0.8]), "reward": 0.0, "alpha_pos": np.array([0.1, 0.2, 0.3]), "alpha_neg": np.array([0.3, 0.2, 0.1])},
               {"values": np.linspace(0, 6, 20), "reward": 2.5, "alpha_pos": _alpha_pos20, "alpha_neg": _alpha_neg20},
               {"values": np.array([1.0, 1.0]), "reward": 1.0, "alpha_pos": np.array([0.5, 0.5]), "alpha_neg": np.array([0.5, 0.5])},
               {"values": 0.4, "reward": 1.0, "alpha_pos": 0.3, "alpha_neg": 0.1}],
        bug_variants={
            "swapped": (_etd_swapped, "You applied alpha_pos to *negative* prediction errors and alpha_neg to positive ones — swap them."),
            "symmetric": (_etd_symmetric, "Every predictor is updated with alpha_pos only; the learning rate must depend on the sign of the prediction error."),
            "positive_only": (_etd_positive_only, "Negative prediction errors do not update the values at all — they should, with alpha_neg."),
            "change_only": (_etd_change_only, "You returned the *change* (learning rate × error) instead of the updated values."),
        },
        hints=["Compute the prediction error of every predictor at once: `delta = reward - values` (an array).",
               "Pick the learning rate element-wise with `np.where(delta > 0, alpha_pos, alpha_neg)`.",
               "`return values + lr * delta`."],
        explanation="Each predictor runs its own delta rule, but with a learning rate that depends on the sign of its "
                    "error. A predictor with alpha_pos > alpha_neg is pulled up by good surprises more than it is pulled "
                    "down by bad ones, so it settles above the mean — at the expectile with τ = alpha_pos / (alpha_pos + alpha_neg).",
        common_mistakes="Swapping the two learning rates, forgetting that the selection must be element-wise, or "
                        "returning the change instead of the new values.",
        check_call='check("13.4", expectile_td_update)',
    ),
    # ------------------------------------------------------------------ 13.5
    NumericExercise(
        id="13.5", title="From learning rates to asymmetry",
        answer=0.75, atol=1e-6,
        distractors=[(3.0, "That is the ratio alpha_pos / alpha_neg. The asymmetry τ is alpha_pos divided by the *sum* of the two rates."),
                     (0.25, "That is alpha_neg / (alpha_pos + alpha_neg) = 1 − τ: the pessimistic complement."),
                     (0.2, "That is the difference of the two rates; τ is a ratio."),
                     (0.3, "That is alpha_pos itself; divide it by alpha_pos + alpha_neg.")],
        hints=["τ = alpha_pos / (alpha_pos + alpha_neg).",
               "0.3 / (0.3 + 0.1)."],
        explanation="τ = 0.3 / 0.4 = 0.75: this learner converges to the 0.75-expectile of its rewards, an optimistic value. "
                    "Only the *ratio* of the two learning rates matters for where the value settles; their sum sets how fast it gets there.",
        check_call='check("13.5", 0.75)',
    ),
    # ------------------------------------------------------------------ 13.6
    CodeExercise(
        id="13.6", title="The expectile loss",
        reference=models.expectile_loss,
        signature="expectile_loss(e, samples, tau)",
        cases=[{"e": 2.0, "samples": _normal, "tau": 0.5},
               {"e": 1.0, "samples": _normal, "tau": 0.2},
               {"e": 3.0, "samples": _bimodal, "tau": 0.8},
               {"e": 0.5, "samples": np.array([0.0, 1.0, 1.0, 0.0]), "tau": 0.3}],
        bug_variants={
            "mirrored": (_loss_mirrored, "Your weights are the wrong way round: errors where the sample lies *above* e are weighted by τ, those below by 1 − τ."),
            "absolute": (_loss_absolute, "You used the absolute error — that is the *quantile* loss. The expectile loss uses the *squared* error."),
        },
        hints=["The expectile loss is a weighted mean of squared errors (x − e)².",
               "The weight of each squared error is τ if x > e and 1 − τ otherwise — the same weights as in Exercise 13.2.",
               "`weights = np.where(x > e, tau, 1 - tau)`; `return np.mean(weights * (x - e) ** 2)`."],
        explanation="The τ-expectile is the value of e that minimises this asymmetric squared loss (just as the mean "
                    "minimises the ordinary squared error). Dabney et al. (2020) decode a reward distribution by "
                    "searching for a set of outcomes whose expectiles match the values read out from dopamine neurons — "
                    "i.e. by minimising a loss of exactly this kind.",
        common_mistakes="Mirroring the weights, using absolute instead of squared errors, or summing instead of averaging "
                        "(the minimiser is the same, but the checker compares values).",
        check_call='check("13.6", expectile_loss)',
    ),
    # ------------------------------------------------------------------ 13.7
    ChoiceExercise(
        id="13.7", title="The dopamine evidence",
        question="Which finding of Dabney et al. (2020) supports a distributional code in dopamine neurons?",
        options={"a": "All dopamine neurons respond identically to reward prediction errors, as the classic theory predicts.",
                 "b": "Different neurons scale positive and negative prediction errors differently, and the reward size at which each "
                      "neuron's response flips sign (its reversal point) is ordered by that asymmetry — as expected if each neuron "
                      "encodes a different expectile.",
                 "c": "Dopamine neurons fire in proportion to the variance of the reward distribution.",
                 "d": "Dopamine neurons respond to rewards but never to reward omissions."},
        correct="b",
        feedback={"a": "The opposite: the paper's point is the *diversity* across neurons, which the classic single-value theory does not predict.",
                  "c": "The variance is not encoded by any single neuron; the distribution is encoded *collectively* by many neurons with different asymmetries.",
                  "d": "Dopamine neurons do respond to omissions (with a dip, lesson 4); the distributional evidence concerns how differently they weigh gains and losses."},
        hints=["Think about what a *population* of expectile learners looks like: each has its own τ and its own reversal point.",
               "The key prediction is a *correlation across neurons* between the asymmetry of their responses and their reversal points."],
        explanation="In Dabney et al. (2020) each neuron's positive and negative responses had different slopes (an asymmetry τ), the "
                    "reversal points were spread out and ordered by τ, and the reward distribution of the task could be decoded from "
                    "the population — three signatures of a distributional, expectile-like code.",
        check_call='check("13.7", "b")',
    ),
    # ------------------------------------------------------------------ 13.8
    CodeExercise(
        id="13.8", title="Quantile-regression TD",
        reference=models.quantile_td_update,
        signature="quantile_td_update(values, reward, alpha, taus)",
        cases=[{"values": np.array([0.2, 0.5, 0.8]), "reward": 1.0, "alpha": 0.1, "taus": np.array([0.25, 0.5, 0.75])},
               {"values": np.array([0.2, 0.5, 0.8]), "reward": 0.0, "alpha": 0.1, "taus": np.array([0.25, 0.5, 0.75])},
               {"values": np.linspace(0, 6, 20), "reward": 2.5, "alpha": 0.05, "taus": _taus20},
               {"values": np.array([1.0, 1.0]), "reward": 1.0, "alpha": 0.2, "taus": np.array([0.1, 0.9])}],
        bug_variants={
            "reversed": (_qtd_reversed, "Your indicator is reversed: it should be 1 when the reward is *below* the value (r < v)."),
            "scaled": (_qtd_scaled, "You multiplied the step by the size of the error. The quantile update moves each value by a "
                                    "fixed step alpha × (τ − 1[r < v]) that does *not* scale with the error — that is what makes it "
                                    "converge to a quantile rather than an expectile."),
            "flipped": (_qtd_flipped, "The sign of the update is flipped: values must move *up* when τ − 1[r < v] is positive."),
        },
        hints=["The update is v_i ← v_i + alpha × (τ_i − 1[r < v_i]), where 1[·] is 1 if the condition holds and 0 otherwise.",
               "Build the indicator for all predictors at once: `(reward < values).astype(float)`.",
               "`return values + alpha * (taus - indicator)`."],
        explanation="Each predictor steps up by alpha τ when the reward is above it and down by alpha (1 − τ) when below. "
                    "The expected step is zero exactly when P(r < v) = τ, i.e. at the τ-quantile. Unlike the expectile update, the "
                    "step size ignores how far the reward was from the value.",
        common_mistakes="Reversing the indicator, scaling the step by the prediction error (which turns it into an expectile-style update), or flipping the sign.",
        check_call='check("13.8", quantile_td_update)',
    ),
    # ------------------------------------------------------------------ 13.9
    CodeExercise(
        id="13.9", title="A single asymmetric learner over trials",
        reference=asymmetric_value_trajectory,
        signature="asymmetric_value_trajectory(rewards, alpha_pos, alpha_neg, v0=0.5)",
        cases=[{"rewards": _rewards_bernoulli, "alpha_pos": 0.3, "alpha_neg": 0.1},
               {"rewards": _rewards_bernoulli, "alpha_pos": 0.1, "alpha_neg": 0.3, "v0": 0.0},
               {"rewards": _rewards_cont, "alpha_pos": 0.2, "alpha_neg": 0.2, "v0": 1.0},
               {"rewards": [1.0, 0.0, 0.0, 1.0], "alpha_pos": 0.5, "alpha_neg": 0.25}],
        bug_variants={
            "swapped": (_traj_swapped, "alpha_pos must be used when the prediction error is positive (reward > value) and alpha_neg otherwise — you have them swapped."),
            "symmetric": (_traj_symmetric, "You used alpha_pos for every update; the learning rate must depend on the sign of the prediction error."),
            "after_only": (_traj_after_only, "Your array starts *after* the first update. Index 0 must hold the initial value v0 (length n + 1)."),
        },
        hints=["Allocate `V = np.zeros(len(rewards) + 1)` and put the initial value in `V[0]`.",
               "Loop over trials: compute `delta = r - v`, choose `alpha_pos` if `delta > 0` else `alpha_neg`, update `v`, store it in `V[t + 1]`.",
               "The structure is exactly the RW loop of lesson 3 with one extra `if` on the sign of the error."],
        explanation="With noisy rewards the value never settles exactly, but it fluctuates around the τ-expectile of the reward "
                    "distribution with τ = alpha_pos / (alpha_pos + alpha_neg): a single asymmetric learner is a *biased* "
                    "estimator of the mean — optimistic if alpha_pos > alpha_neg, pessimistic otherwise.",
        common_mistakes="Swapping the learning rates, forgetting the initial value at index 0, or returning only the final value.",
        check_call='check("13.9", asymmetric_value_trajectory)',
    ),
    # ------------------------------------------------------------------ 13.10
    FreeTextExercise(
        id="13.10", title="Asymmetric learning, expectiles and mood",
        question="In 3–5 sentences: how does the distributional (expectile) view explain what a person with alpha_pos > alpha_neg "
                 "ends up believing about an option, and why might this matter for mood or psychopathology?",
        model_answer="A learner whose positive learning rate exceeds its negative learning rate converges to an expectile above "
                     "the mean, with asymmetry τ = alpha_pos / (alpha_pos + alpha_neg): it is systematically optimistic about the "
                     "option's value, whereas alpha_pos < alpha_neg produces a pessimistic estimate below the mean. The ratio of "
                     "the two learning rates therefore determines *where* in the reward distribution the value sits, not just "
                     "how quickly it is learned. This gives a precise computational reading of affective biases: an optimistic "
                     "bias in healthy learning (Lefebvre et al., 2017), and reduced positive or enhanced negative learning as a "
                     "candidate mechanism for pessimistic expectations in low mood, anhedonia or anxiety — while Pulcu & Browning "
                     "(2017) warn that such asymmetries can also be rational adaptations to environments in which rewards and "
                     "punishments have different statistics.",
        concepts=[("optimistic / pessimistic value (an expectile above or below the mean)",
                   ["optimis", "pessimis", "expectile", "above the mean", "below the mean", "overestimat", "underestimat",
                    "higher than the mean", "lower than the mean", "biased"]),
                  ("the ratio of the learning rates sets the asymmetry τ",
                   ["ratio", "relative", "tau", "τ", "alpha_pos / (", "proportion", "dominat", "larger than", "greater than",
                    "bigger than", "exceeds"]),
                  ("relevance to mood / psychopathology",
                   ["mood", "depress", "anhedon", "anxi", "affect", "symptom", "clinical", "psychiatr", "disorder", "patient"])],
        min_concepts=2,
        hints=["Where does a predictor with alpha_pos > alpha_neg settle relative to the mean, and which quantity determines that?",
               "Then connect: what would a systematically pessimistic value estimate feel like, and which parameters would you expect to differ in depression or anxiety?"],
        explanation="Asymmetric learning rates turn a value estimate into an expectile: the learner's optimism or pessimism is "
                    "a parameter that can be measured from choices, which is why asymmetric-learning models are a staple of "
                    "computational psychiatry (lesson 14).",
        check_call='check("13.10", """A learner whose positive learning rate exceeds its negative learning rate converges to an expectile above the mean, with asymmetry tau = alpha_pos / (alpha_pos + alpha_neg): it is systematically optimistic about the option, whereas the opposite asymmetry produces a pessimistic estimate below the mean. The ratio of the learning rates sets where in the reward distribution the value sits. This matters for mood and psychopathology because reduced positive or enhanced negative learning is a candidate mechanism for pessimistic expectations in depression, anhedonia or anxiety, although such asymmetries can also be rational adaptations to the statistics of rewards and punishments.""")',
    ),
]
