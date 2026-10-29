"""Exercises of lesson 8: Bayesian parameter estimation and hierarchical models."""
from __future__ import annotations

import numpy as np

from .. import fitting, models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference functions specific to this lesson (their source is the revealed solution)
# --------------------------------------------------------------------------- #


def log_posterior_rw(x, choices, rewards, mu, sd):
    """Log posterior (up to a constant) of the RW model's *unconstrained*
    parameters x = (logit alpha, log beta), with independent Normal priors
    N(mu[i], sd[i]) on each unconstrained parameter."""
    from scipy import stats

    theta = fitting.to_constrained(x, ["unit", "pos"])        # x -> (alpha, beta)
    log_likelihood = -models.nll_rw(theta, choices, rewards)  # the likelihood of the choices
    log_prior = np.sum(stats.norm.logpdf(x, mu, sd))          # the prior, in the unconstrained space
    return log_likelihood + log_prior


def em_m_step(x_hat, var_hat):
    """M-step of the empirical-Bayes EM algorithm (Huys et al. 2011; Gershman 2016).

    ``x_hat``   : array (n_subjects, n_params) of per-subject MAP estimates
    ``var_hat`` : array (n_subjects, n_params) of their Laplace variances
    Returns the group mean ``mu`` and standard deviation ``sd`` (unconstrained space)."""
    x_hat = np.asarray(x_hat, float)
    var_hat = np.asarray(var_hat, float)
    mu = np.mean(x_hat, axis=0)
    second_moment = np.mean(x_hat ** 2 + var_hat, axis=0)     # E[x^2] includes each subject's uncertainty
    sd = np.sqrt(second_moment - mu ** 2)
    return mu, sd


# --------------------------------------------------------------------------- #
# helpers for the test cases
# --------------------------------------------------------------------------- #

_rng_data = np.random.default_rng(80)
_SIM_A = models.simulate_rw_bandit([0.7, 0.3], 0.3, 5.0, 40, _rng_data)
_SIM_B = models.simulate_rw_bandit([0.2, 0.8], 0.6, 2.0, 25, _rng_data)

_MH_MU = np.array([1.0, -1.0])
_MH_SD = np.array([1.0, 0.5])


def _gaussian_log_target(x):
    """log density (up to a constant) of a 2-D Gaussian with mean (1, -1) and sd (1, 0.5)."""
    x = np.asarray(x, float)
    return float(-0.5 * np.sum(((x - _MH_MU) / _MH_SD) ** 2))


def _mh_unpack(out):
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        return None, None
    samples, acc = out
    try:
        samples = np.asarray(samples, float)
        acc = float(acc)
    except Exception:
        return None, None
    return samples, acc


def _mh_compare(out, ref, case):
    samples, acc = _mh_unpack(out)
    if samples is None or samples.shape != (case["n_samples"], 2):
        return False
    if not (0.1 <= acc <= 0.9):
        return False
    if not np.all(np.isfinite(samples)):
        return False
    mean_ok = np.all(np.abs(samples.mean(axis=0) - _MH_MU) < 0.2)
    sd_ok = np.all(np.abs(samples.std(axis=0) / _MH_SD - 1) < 0.3)
    return bool(mean_ok and sd_ok)


def _mh_diagnose(out, ref, case):
    samples, acc = _mh_unpack(out)
    if samples is None:
        return "Return a tuple `(samples, acceptance_rate)` where `samples` is an array of shape (n_samples, dim)."
    if samples.shape != (case["n_samples"], 2):
        return (f"`samples` should have shape (n_samples, dim) = {(case['n_samples'], 2)}, but has shape "
                f"{samples.shape}. Store a *copy* of the current position after every iteration (accepted or not).")
    if not np.all(np.isfinite(samples)):
        return "The samples contain NaN or inf — check that `log_target` is called on the proposal, not on something else."
    if acc >= 0.99:
        return ("Your acceptance rate is ~1: every proposal is accepted, so the chain is a pure random walk. "
                "Accept only when log(u) < log_target(proposal) − log_target(current), with u ~ Uniform(0, 1).")
    if acc <= 0.01:
        return ("Your acceptance rate is ~0: proposals are (almost) never accepted. Check the sign of the "
                "comparison (accept when the *proposal* is more probable) and that you compare log(u), not u, "
                "with the log ratio.")
    sds = samples.std(axis=0)
    if np.any(sds / _MH_SD > 1.3):
        return ("The samples are more spread out than the target: you are keeping proposals that should have "
                "been rejected. When a proposal is rejected the chain must *stay* at the current point (and "
                "that repeated point is still recorded as a sample).")
    if np.any(sds / _MH_SD < 0.7):
        return ("The samples are less spread out than the target: perhaps you overwrite the stored samples, "
                "record the proposal instead of the current state, or update `x` without updating its stored log density.")
    if np.any(np.abs(samples.mean(axis=0) - _MH_MU) >= 0.2):
        return ("The sample mean is off. After accepting a proposal, update *both* the position and its stored "
                "log density, and make sure the proposal is centred on the *current* position.")
    return None


def _mh_always_accept(log_target, x0, n_samples, proposal_sd, rng):
    x = np.atleast_1d(np.asarray(x0, float))
    samples = []
    for _ in range(n_samples):
        x = x + rng.normal(0, proposal_sd, size=x.shape)
        samples.append(x.copy())
    return np.array(samples), 1.0


# --------------------------------------------------------------------------- #
EXERCISES = [
    # ------------------------------------------------------------------ 8.1
    ChoiceExercise(
        id="8.1", title="A suspicious maximum-likelihood estimate",
        question="A participant completed only 30 trials and chose the same option on 28 of them. Maximum "
                 "likelihood gives beta ≈ 400. What is the most likely explanation?",
        options={"a": "The participant really is extraordinarily consistent, so beta is genuinely huge.",
                 "b": "With so few, near-deterministic choices the likelihood keeps increasing as beta grows, so "
                      "the estimate drifts to the boundary: the data cannot pin beta down.",
                 "c": "The optimiser got stuck; a few random restarts would give a sensible value.",
                 "d": "alpha must have been estimated as exactly 0."},
        correct="b",
        feedback={"a": "A very large beta is *consistent* with the data, but so is beta = 20 or 50: the likelihood "
                       "is almost flat beyond some value, so the exact number means little.",
                  "c": "Restarts fix local optima; here the problem is that there is no finite optimum (or a very "
                       "shallow one). Restarts would find the same boundary.",
                  "d": "alpha and beta are different parameters; nothing about alpha follows from this symptom."},
        hints=["Think about what the likelihood looks like as a function of beta when almost every choice went to the "
               "option with the higher value.",
               "If choices are (nearly) deterministic, a larger beta makes every observed choice *more* probable, "
               "without end: the likelihood has no interior maximum."],
        explanation="Near-deterministic choice data make the likelihood monotone in beta, so the MLE runs off to "
                    "the boundary (or to wherever the optimiser stops). This is not a bug but a property of the "
                    "data: they are compatible with any large beta. A prior expresses that beta = 400 is a priori "
                    "implausible and brings the estimate back to a sensible range — the first motivation for "
                    "Bayesian estimation.",
        check_call='check("8.1", "b")',
    ),
    # ------------------------------------------------------------------ 8.2
    CodeExercise(
        id="8.2", title="The log posterior of the RW model",
        reference=log_posterior_rw,
        signature="log_posterior_rw(x, choices, rewards, mu, sd)",
        cases=[{"x": np.array([0.0, 1.0]), "choices": _SIM_A["choices"], "rewards": _SIM_A["rewards"],
                "mu": np.array([0.0, 1.0]), "sd": np.array([1.5, 1.0])},
               {"x": np.array([-1.2, 2.0]), "choices": _SIM_A["choices"], "rewards": _SIM_A["rewards"],
                "mu": np.array([0.0, 1.0]), "sd": np.array([1.5, 1.0])},
               {"x": np.array([0.8, -0.5]), "choices": _SIM_B["choices"], "rewards": _SIM_B["rewards"],
                "mu": np.array([-0.5, 0.5]), "sd": np.array([1.0, 2.0])}],
        bug_variants={
            "no_prior": (lambda x, choices, rewards, mu, sd: -models.nll_rw(fitting.to_constrained(x, ["unit", "pos"]), choices, rewards),
                         "You returned the log-likelihood only: the prior term `sum(norm.logpdf(x, mu, sd))` is missing."),
            "sign": (lambda x, choices, rewards, mu, sd: models.nll_rw(fitting.to_constrained(x, ["unit", "pos"]), choices, rewards)
                     + np.sum(__import__("scipy").stats.norm.logpdf(x, mu, sd)),
                     "You added the *negative* log-likelihood. `nll_rw` returns −log p(data | θ); the log posterior "
                     "needs +log p(data | θ), i.e. `-models.nll_rw(...)`."),
            "minus_prior": (lambda x, choices, rewards, mu, sd: -models.nll_rw(fitting.to_constrained(x, ["unit", "pos"]), choices, rewards)
                            - np.sum(__import__("scipy").stats.norm.logpdf(x, mu, sd)),
                            "You *subtracted* the log prior. Bayes' rule multiplies likelihood and prior, so their logs are *added*."),
            "prior_on_theta": (lambda x, choices, rewards, mu, sd: -models.nll_rw(fitting.to_constrained(x, ["unit", "pos"]), choices, rewards)
                               + np.sum(__import__("scipy").stats.norm.logpdf(fitting.to_constrained(x, ["unit", "pos"]), mu, sd)),
                               "You evaluated the prior on (alpha, beta) instead of on the unconstrained x. In this lesson the "
                               "Normal priors live in the unconstrained space (logit alpha, log beta)."),
            "no_transform": (lambda x, choices, rewards, mu, sd: -models.nll_rw(x, choices, rewards)
                             + np.sum(__import__("scipy").stats.norm.logpdf(x, mu, sd)),
                             "You passed x straight to `nll_rw`. The likelihood needs (alpha, beta) = "
                             "`fitting.to_constrained(x, ['unit', 'pos'])`."),
        },
        hints=["Three steps: map x to (alpha, beta) with `fitting.to_constrained(x, ['unit', 'pos'])`; get the "
               "log-likelihood as `-models.nll_rw(theta, choices, rewards)`; add the log prior.",
               "The log prior of independent Normals is the *sum* over parameters of `scipy.stats.norm.logpdf(x, mu, sd)` "
               "(evaluated at the unconstrained x, not at alpha and beta).",
               "`theta = fitting.to_constrained(x, ['unit', 'pos']); return -models.nll_rw(theta, choices, rewards) + np.sum(stats.norm.logpdf(x, mu, sd))`"],
        explanation="log p(x | data) = log p(data | x) + log p(x) + const. The likelihood is the same function you "
                    "maximised in lesson 5 (with the sign flipped), and the prior is a sum of Normal log densities "
                    "in the unconstrained space, where the parameters are free to take any real value.",
        common_mistakes="Forgetting the prior, adding the *negative* log-likelihood, subtracting the prior, or "
                        "evaluating the prior on (alpha, beta) rather than on x.",
        check_call='check("8.2", log_posterior_rw)',
    ),
    # ------------------------------------------------------------------ 8.3
    NumericExercise(
        id="8.3", title="A posterior standard deviation from the Hessian",
        answer=0.2, atol=1e-4,
        distractors=[(0.04, "1/25 is the posterior *variance*; the question asks for the standard deviation."),
                     (5.0, "√25 = 5 is the inverse of the answer: the variance is 1/H, not H."),
                     (25.0, "H itself is the curvature; the variance is its inverse and the sd the square root of that.")],
        hints=["Under the Laplace approximation the posterior variance is the inverse of the second derivative "
               "(curvature) of the negative log posterior at its minimum.",
               "Variance = 1 / 25; the standard deviation is the square root of the variance."],
        explanation="A sharply curved negative log posterior (large second derivative) means a narrow posterior: "
                    "variance = 1/H = 0.04, sd = 0.2. With several parameters the same idea uses the inverse "
                    "Hessian *matrix*, whose diagonal holds the variances.",
        check_call='check("8.3", 0.2)',
    ),
    # ------------------------------------------------------------------ 8.4
    NumericExercise(
        id="8.4", title="Transforming a credible interval",
        answer=float(models.sigmoid(0.5 + 1.96 * 0.4)), atol=1e-3,
        distractors=[(float(models.sigmoid(0.5 + 0.4)), "You forgot the factor 1.96: a 95% interval extends 1.96 sd on each side."),
                     (float(models.sigmoid(0.5)) + 1.96 * 0.4, "You added 1.96 × sd *after* transforming to the alpha scale. "
                                                               "Build the interval in logit space first, then transform its end points."),
                     (float(models.sigmoid(0.5 - 1.96 * 0.4)), "That is the *lower* bound of the interval."),
                     (0.5 + 1.96 * 0.4, "That is the upper bound in logit space; transform it with the sigmoid.")],
        hints=["Compute the interval in the unconstrained (logit) space: x̂ ± 1.96 × sd.",
               "Then map the end points to the alpha scale with the sigmoid 1 / (1 + exp(−x)).",
               "sigmoid(0.5 + 1.96 × 0.4) = sigmoid(1.284)"],
        explanation="Intervals are built where the Gaussian approximation lives (the unconstrained space) and "
                    "their end points are then transformed: sigmoid(1.284) ≈ 0.783. The transformed interval "
                    "(0.43, 0.78) is asymmetric around sigmoid(0.5) = 0.62 — which is exactly right, because "
                    "the posterior on the alpha scale is skewed.",
        check_call='check("8.4", 1 / (1 + np.exp(-(0.5 + 1.96 * 0.4))))',
    ),
    # ------------------------------------------------------------------ 8.5
    CodeExercise(
        id="8.5", title="A Metropolis–Hastings sampler",
        reference=fitting.metropolis_hastings,
        signature="metropolis_hastings(log_target, x0, n_samples, proposal_sd, rng)",
        cases=[{"log_target": _gaussian_log_target, "x0": np.zeros(2), "n_samples": 4000, "proposal_sd": 1.0,
                "rng": np.random.default_rng(7)}],
        compare=_mh_compare,
        diagnose=_mh_diagnose,
        bug_variants={
            "always_accept": (_mh_always_accept,
                              "You accept every proposal, so the chain is a random walk that never settles on the "
                              "target. Compare log(u) with the difference of log densities and keep the old point "
                              "when the proposal is rejected."),
        },
        hints=["Loop n_samples times: propose `x_new = x + rng.normal(0, proposal_sd, size=x.shape)`, compute "
               "`log_target(x_new)`, accept if `np.log(rng.random()) < log_target(x_new) - log_target(x)`.",
               "When a proposal is rejected the chain stays where it is — and that repeated position is *still* "
               "appended to the samples. Count accepted proposals to compute the acceptance rate.",
               "Store the current log density so that you evaluate `log_target` only once per iteration; append "
               "`x.copy()` each iteration; return `(np.array(samples), n_accepted / n_samples)`."],
        explanation="Random-walk Metropolis–Hastings (Metropolis et al. 1953; Hastings 1970) needs only the "
                    "*unnormalised* log density: proposals that increase the density are always accepted and those "
                    "that decrease it are accepted with probability equal to the density ratio. In the long run "
                    "the chain visits each region in proportion to its posterior probability. The checker verifies "
                    "that your samples reproduce the mean and spread of a known Gaussian target and that the "
                    "acceptance rate is in a plausible range.",
        common_mistakes="Accepting every proposal; comparing u instead of log(u) with the log ratio; not recording "
                        "the (repeated) current point after a rejection; centring the proposal on x0 rather than on "
                        "the current position.",
        check_call='check("8.5", metropolis_hastings)',
    ),
    # ------------------------------------------------------------------ 8.6
    ChoiceExercise(
        id="8.6", title="Reading the acceptance rate",
        question="Your Metropolis–Hastings chain accepts 98% of its proposals. What does this tell you?",
        options={"a": "The sampler has converged perfectly; the posterior is well characterised.",
                 "b": "The proposals are too small: the chain creeps along, consecutive samples are highly "
                      "correlated and the posterior is explored very slowly.",
                 "c": "The proposals are too large: most moves jump into the tails and are wasted.",
                 "d": "The posterior must be flat (uniform)."},
        correct="b",
        feedback={"a": "A high acceptance rate is not a sign of convergence: tiny steps are almost always accepted "
                       "whether or not the chain has found the posterior.",
                  "c": "Too-large proposals produce the *opposite* symptom: a very low acceptance rate, with the chain "
                       "stuck at the same point for long stretches.",
                  "d": "A flat posterior would also accept most proposals, but it is far less likely than a too-small "
                       "proposal scale, and it would not explain the slow drift you see in the trace plot."},
        hints=["Look at the trace plots for three proposal scales: which one moves like a slow drift, which one like a staircase?",
               "A proposal that barely moves is almost always accepted (the density ratio is close to 1), but it "
               "takes many steps to cross the posterior — the samples are strongly autocorrelated."],
        explanation="For random-walk Metropolis the sweet spot is an acceptance rate of roughly 20–50%: large "
                    "enough to move, small enough not to be rejected constantly. Tune the proposal scale to land "
                    "there, discard the burn-in, and check autocorrelation (or the effective sample size) before "
                    "trusting the samples.",
        check_call='check("8.6", "b")',
    ),
    # ------------------------------------------------------------------ 8.7
    CodeExercise(
        id="8.7", title="The M-step of hierarchical EM",
        reference=em_m_step,
        signature="em_m_step(x_hat, var_hat)",
        cases=[{"x_hat": np.array([[0.1, 1.2], [-0.4, 0.8], [0.6, 1.5], [-1.0, 1.1], [0.3, 0.9]]),
                "var_hat": np.array([[0.2, 0.1], [0.5, 0.2], [0.1, 0.05], [0.8, 0.3], [0.3, 0.1]])},
               {"x_hat": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
                "var_hat": np.array([[0.25, 1.0], [0.25, 1.0], [0.25, 1.0]])},
               {"x_hat": np.random.default_rng(3).normal(0, 1, (12, 3)), "var_hat": np.random.default_rng(4).uniform(0.05, 0.5, (12, 3))}],
        bug_variants={
            "ignore_var": (lambda x_hat, var_hat: (np.mean(x_hat, axis=0), np.std(x_hat, axis=0)),
                           "You ignored `var_hat`: the group variance must include each subject's own posterior "
                           "uncertainty, E[x²] = mean(x_hat² + var_hat)."),
            "ddof1": (lambda x_hat, var_hat: (np.mean(x_hat, axis=0), np.sqrt(np.mean(np.asarray(x_hat) ** 2 + np.asarray(var_hat), axis=0) * len(x_hat) / (len(x_hat) - 1) - np.mean(x_hat, axis=0) ** 2 * len(x_hat) / (len(x_hat) - 1))),
                      "You applied a sample-size correction (n − 1). The EM M-step uses plain means (maximum likelihood "
                      "of the group parameters)."),
            "variance": (lambda x_hat, var_hat: (np.mean(x_hat, axis=0), np.mean(np.asarray(x_hat) ** 2 + np.asarray(var_hat), axis=0) - np.mean(x_hat, axis=0) ** 2),
                         "You returned the *variance*; the function should return the standard deviation (take the square root)."),
        },
        hints=["The group mean is the average of the subjects' estimates (column by column).",
               "The group variance is E[x²] − mu², where E[x²] = mean(x_hat² + var_hat) — the Laplace variance of each "
               "subject adds to the spread, because a subject's true parameter is only known up to that uncertainty.",
               "`mu = x_hat.mean(0); sd = np.sqrt(np.mean(x_hat**2 + var_hat, 0) - mu**2)`"],
        explanation="In the M-step the group prior is re-estimated from the subjects' posteriors. Including "
                    "`var_hat` is what makes this *empirical Bayes* rather than a mere average of point estimates: "
                    "subjects whose parameters are poorly constrained contribute uncertainty, not false precision, "
                    "to the group variance (Huys et al. 2011, Eq. in Methods; Gershman 2016).",
        common_mistakes="Ignoring the Laplace variances, using an n − 1 correction, or returning the variance instead of the sd.",
        check_call='check("8.7", em_m_step)',
    ),
    # ------------------------------------------------------------------ 8.8
    NumericExercise(
        id="8.8", title="How much shrinkage?",
        answer=0.25, atol=1e-4,
        distractors=[(0.75, "That is the weight on the *group mean*; the question asks for the weight on the subject's own estimate."),
                     (1 / 3, "τ²/σ² is the ratio of the two variances, not a weight; the weight is τ²/(τ² + σ²)."),
                     (0.5, "Equal weights would need equal variances; here the likelihood is three times more uncertain than the prior.")],
        hints=["Combining a Normal prior N(mu, tau²) with a Normal likelihood N(x̂, sigma²) gives a posterior mean that is a "
               "precision-weighted average of mu and x̂.",
               "The weight on x̂ is tau² / (tau² + sigma²): the less uncertain source gets more weight.",
               "1 / (1 + 3)"],
        explanation="Posterior mean = w x̂ + (1 − w) mu with w = τ²/(τ² + σ²) = 1/(1 + 3) = 0.25: a subject whose data "
                    "are three times more uncertain than the group spread is pulled three quarters of the way "
                    "towards the group mean. Subjects with many clean trials (small σ²) are barely moved.",
        check_call='check("8.8", 0.25)',
    ),
    # ------------------------------------------------------------------ 8.9
    ChoiceExercise(
        id="8.9", title="Reading an MCMC diagnostic",
        question="After sampling a hierarchical model, ArviZ reports R-hat = 1.5 for the group mean of beta. What does this mean?",
        options={"a": "The posterior standard deviation of that parameter is 1.5.",
                 "b": "The chains disagree with each other: they have not converged (mixed), so the samples cannot be trusted yet.",
                 "c": "Sampling was 1.5 times faster than expected.",
                 "d": "The prior on beta was too wide."},
        correct="b",
        feedback={"a": "R-hat is a ratio of between-chain to within-chain variability (Gelman & Rubin 1992), not a standard deviation.",
                  "c": "R-hat says nothing about speed.",
                  "d": "A wide prior does not by itself produce a high R-hat; poor mixing, multimodality or a badly "
                       "parameterised model do."},
        hints=["R-hat compares the variance *between* chains with the variance *within* chains.",
               "If every chain had explored the same posterior, the two would agree and R-hat would be ≈ 1.00 "
               "(values above about 1.01 are a warning)."],
        explanation="R-hat ≈ 1 is a necessary (not sufficient) sign of convergence. R-hat = 1.5 means at least one "
                    "chain is sampling a different region than the others: run longer, reparameterise (e.g. the "
                    "non-centred form), or inspect trace plots for multimodality.",
        check_call='check("8.9", "b")',
    ),
    # ------------------------------------------------------------------ 8.10
    FreeTextExercise(
        id="8.10", title="When does hierarchical fitting help, and when can it mislead?",
        question="In 3–5 sentences: when does a hierarchical (group-level) model improve parameter estimates, and in "
                 "what situation can it produce a misleading result?",
        model_answer="Hierarchical fitting helps when each participant provides few or noisy trials: the group "
                     "distribution acts as an empirically estimated prior that pulls (shrinks) implausible "
                     "individual estimates towards the group, reducing extreme values and improving parameter "
                     "recovery. It can mislead when the group structure in the model does not match the design: "
                     "fitting patients and controls under one common prior shrinks both towards the same mean and "
                     "can hide a real group difference (and estimating the prior from one group only biases the "
                     "other), and strong shrinkage can make individual differences look smaller than they are. The "
                     "remedy is to put the group or covariate inside the hierarchical model (lesson 14), and to check "
                     "with simulations that the effect of interest survives the shrinkage.",
        concepts=[("shrinkage / pooling towards the group", ["shrink", "pool", "regulari", "borrow", "towards the group",
                                                              "toward the group", "group mean", "group distribution",
                                                              "group prior", "pulled"]),
                  ("few or noisy trials", ["few trials", "few data", "noisy", "little data", "small sample", "uncertain",
                                           "unreliable", "extreme", "boundary", "poorly constrained"]),
                  ("group differences / covariates", ["group difference", "groups", "patients", "controls", "clinical",
                                                      "covariate", "separate", "mask", "hide", "individual differences",
                                                      "between-group", "condition"])],
        min_concepts=2,
        hints=["Think about what shrinkage does to a participant with 30 noisy trials, and what it does to a real "
               "difference between two groups fitted under one common prior.",
               "Key words: shrinkage/pooling, few noisy trials, group differences or covariates inside the model."],
        explanation="Hierarchical models trade a little bias for a lot of variance reduction. That trade is "
                    "excellent for noisy individual estimates and dangerous when the quantity of interest is a "
                    "difference between groups that the model does not know about.",
        check_call='check("8.10", """Hierarchical fitting helps when each participant provides few or noisy trials: the group distribution acts as a prior that shrinks implausible individual estimates towards the group mean, reducing extreme values and improving parameter recovery. It can mislead when the group structure in the model does not match the design: fitting patients and controls under one common prior shrinks both towards the same mean and can hide a real group difference, so the group or covariate should be put inside the hierarchical model.""")',
    ),
]
