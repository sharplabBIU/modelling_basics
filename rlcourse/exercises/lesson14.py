"""Exercises of lesson 14: linking model parameters to psychopathology."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import models, tasks
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference implementations (their source is shown when a solution is revealed)
# --------------------------------------------------------------------------- #


def stay_regression_coefficients(data):
    """Logistic regression of 'staying' on the previous trial's reward and transition.

    Coding (Daw et al. 2011; Gillan et al. 2016): stay = 1 if the stage-1 choice
    repeats the previous one; reward = +1 (rewarded) / -1 (not); transition = +1
    (common) / -1 (rare).  Returns (intercept, b_reward, b_transition, b_interaction):
    b_reward is the model-free index, b_interaction the model-based index.
    """
    import statsmodels.api as sm

    a1 = np.asarray(data["a1"])
    r = np.asarray(data["r"], dtype=float)
    common = np.asarray(data["common"], dtype=float)
    stay = (a1[1:] == a1[:-1]).astype(float)        # did trial t repeat the choice of trial t-1?
    reward = 2 * r[:-1] - 1                           # outcome of the *previous* trial, coded +1/-1
    transition = 2 * common[:-1] - 1                  # transition of the previous trial, +1/-1
    X = np.column_stack([np.ones(len(stay)), reward, transition, reward * transition])
    result = sm.Logit(stay, X).fit(disp=0)
    return tuple(float(b) for b in result.params)


def symptom_regression(df, param, symptom, covariates=()):
    """Standardised regression of a fitted parameter on a symptom score, controlling for covariates.

    Every variable is z-scored first, so the returned coefficient is a standardised
    beta (comparable across parameters).  Returns (beta_symptom, p_value).
    """
    import statsmodels.api as sm

    cols = [param, symptom] + list(covariates)
    z = (df[cols] - df[cols].mean()) / df[cols].std()           # z-score every variable
    X = sm.add_constant(z[[symptom] + list(covariates)].to_numpy())
    result = sm.OLS(z[param].to_numpy(), X).fit()
    return float(result.params[1]), float(result.pvalues[1])   # index 1 = the symptom


def split_half_reliability(x_half1, x_half2):
    """Split-half reliability of a parameter: the correlation between the estimates from
    two halves of the data, corrected to full length with the Spearman–Brown formula."""
    r_half = np.corrcoef(x_half1, x_half2)[0, 1]
    return 2 * r_half / (1 + r_half)


def power_by_simulation(r_true, n, n_sims, rng, alpha=0.05):
    """Statistical power of a Pearson correlation test, estimated by simulation.

    Draws ``n_sims`` samples of ``n`` bivariate-normal pairs with correlation
    ``r_true``, tests each with ``scipy.stats.pearsonr`` and returns the proportion
    of samples in which p < alpha."""
    from scipy import stats

    cov = np.array([[1.0, r_true], [r_true, 1.0]])
    n_significant = 0
    for _ in range(n_sims):
        xy = rng.multivariate_normal([0.0, 0.0], cov, size=n)
        p_value = stats.pearsonr(xy[:, 0], xy[:, 1])[1]
        n_significant += p_value < alpha
    return n_significant / n_sims


# --------------------------------------------------------------------------- #
# Deterministic test data
# --------------------------------------------------------------------------- #


def _two_step_subject(w, seed, n_trials=200):
    rng = np.random.default_rng(seed)
    task = tasks.TwoStepTask(n_trials=n_trials, rng=rng)
    params = (0.6, 0.6, 0.6, 4.0, 4.0, w, 0.15)
    return models.two_step_simulate(params, task, rng)


_SUBJ_MB = _two_step_subject(0.9, 141)
_SUBJ_MF = _two_step_subject(0.1, 142)


def _symptom_df(seed=143, n=60):
    rng = np.random.default_rng(seed)
    compulsivity = rng.normal(25, 12, n)
    age = rng.normal(28, 6, n)
    anxiety = rng.normal(40, 10, n)
    w = 0.6 - 0.012 * (compulsivity - 25) + 0.004 * (age - 28) + rng.normal(0, 0.12, n)
    return pd.DataFrame({"w": np.clip(w, 0.01, 0.99), "compulsivity": compulsivity, "age": age, "anxiety": anxiety})


_DF = _symptom_df()
_rng_rel = np.random.default_rng(144)
_true_w = _rng_rel.normal(0.5, 0.15, 50)
_HALF1 = _true_w + _rng_rel.normal(0, 0.1, 50)
_HALF2 = _true_w + _rng_rel.normal(0, 0.1, 50)


# --------------------------------------------------------------------------- #
# Buggy variants and diagnostics
# --------------------------------------------------------------------------- #


def _stay_reg_01_coding(data):
    import statsmodels.api as sm
    a1 = np.asarray(data["a1"]); r = np.asarray(data["r"], float); common = np.asarray(data["common"], float)
    stay = (a1[1:] == a1[:-1]).astype(float)
    X = np.column_stack([np.ones(len(stay)), r[:-1], common[:-1], r[:-1] * common[:-1]])
    return tuple(float(b) for b in sm.Logit(stay, X).fit(disp=0).params)


def _stay_reg_no_lag(data):
    import statsmodels.api as sm
    a1 = np.asarray(data["a1"]); r = np.asarray(data["r"], float); common = np.asarray(data["common"], float)
    stay = (a1[1:] == a1[:-1]).astype(float)
    reward, transition = 2 * r[1:] - 1, 2 * common[1:] - 1
    X = np.column_stack([np.ones(len(stay)), reward, transition, reward * transition])
    return tuple(float(b) for b in sm.Logit(stay, X).fit(disp=0).params)


def _symptom_reg_raw(df, param, symptom, covariates=()):
    import statsmodels.api as sm
    X = sm.add_constant(df[[symptom] + list(covariates)].to_numpy())
    result = sm.OLS(df[param].to_numpy(), X).fit()
    return float(result.params[1]), float(result.pvalues[1])


def _symptom_reg_reversed(df, param, symptom, covariates=()):
    import statsmodels.api as sm
    cols = [param, symptom] + list(covariates)
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    X = sm.add_constant(z[[param] + list(covariates)].to_numpy())
    result = sm.OLS(z[symptom].to_numpy(), X).fit()
    return float(result.params[1]), float(result.pvalues[1])


def _symptom_reg_no_constant(df, param, symptom, covariates=()):
    import statsmodels.api as sm
    cols = [param, symptom] + list(covariates)
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    X = z[[symptom] + list(covariates)].to_numpy()
    result = sm.OLS(z[param].to_numpy(), X).fit()
    return float(result.params[0]), float(result.pvalues[0])


def _power_compare(out, ref, case):
    try:
        return abs(float(out) - float(ref)) <= 0.1
    except Exception:
        return False


def _power_diagnose(out, ref, case):
    try:
        o = float(out)
    except Exception:
        return "Return a single number: the proportion of simulations with p < alpha."
    if o > 1:
        return "You returned a *count* of significant results; divide by the number of simulations."
    if abs(o - (1 - float(ref))) <= 0.1 and abs(float(ref) - 0.5) > 0.15:
        return "Your value is about 1 − power: you counted the *non*-significant simulations (p ≥ alpha)."
    if abs(o - float(ref)) > 0.1 and case["r_true"] == 0 and o > 0.2:
        return ("With r_true = 0 the power should equal the false-positive rate alpha (≈ 0.05). Check that each "
                "simulation draws *fresh* data with the requested correlation and that you compare p with alpha.")
    return None


def _split_half_uncorrected(x_half1, x_half2):
    return np.corrcoef(x_half1, x_half2)[0, 1]


def _split_half_wrong_formula(x_half1, x_half2):
    r = np.corrcoef(x_half1, x_half2)[0, 1]
    return r / (1 + r)


# --------------------------------------------------------------------------- #
EXERCISES = [
    # ------------------------------------------------------------------ 14.1
    CodeExercise(
        id="14.1", title="The one-trial-back regression (model-free and model-based indices)",
        reference=stay_regression_coefficients,
        signature="stay_regression_coefficients(data)",
        cases=[{"data": _SUBJ_MB}, {"data": _SUBJ_MF}],
        atol=2e-3, rtol=1e-3,
        bug_variants={
            "coding_01": (_stay_reg_01_coding,
                          "You coded reward and transition as 0/1. Use +1/−1 (`2 * x - 1`): with 0/1 coding the "
                          "'main effects' change meaning and the interaction is no longer the model-based index "
                          "that the literature reports."),
            "no_lag": (_stay_reg_no_lag,
                       "You aligned staying on trial t with the outcome of trial t itself. Staying on trial t depends "
                       "on what happened on trial t − 1: use `r[:-1]` and `common[:-1]` against `stay = a1[1:] == a1[:-1]`."),
        },
        hints=["Build three arrays of length n_trials − 1: `stay` (choice t equals choice t−1), the previous trial's "
               "reward coded ±1 and the previous trial's transition coded ±1.",
               "The design matrix has four columns: a column of ones, reward, transition and reward × transition. "
               "Fit `statsmodels.api.Logit(stay, X).fit(disp=0)` and return `result.params` as a tuple.",
               "`X = np.column_stack([np.ones(len(stay)), reward, transition, reward * transition])`"],
        explanation="With ±1 coding the reward coefficient measures how much a reward (regardless of transition) "
                    "increases staying — the model-free signature — and the reward × transition coefficient measures "
                    "how much that effect depends on whether the transition was common — the model-based signature. "
                    "Gillan et al. (2016) report that this interaction correlates at r ≈ 0.87 with the fitted w.",
        common_mistakes="0/1 instead of ±1 coding; misaligning trials; forgetting the intercept column.",
        check_call='check("14.1", stay_regression_coefficients)',
    ),
    # ------------------------------------------------------------------ 14.2
    NumericExercise(
        id="14.2", title="A model-based index from a stay-probability table",
        answer=0.33, atol=1e-3,
        distractors=[(0.085, "That is the model-*free* index (mean stay after reward minus mean stay after no reward)."),
                     (-0.01, "You subtracted in the wrong order in one of the two brackets: compute (common − rare) "
                             "for rewarded trials and (common − rare) for unrewarded trials, then subtract the second from the first."),
                     (0.16, "That is only the rewarded half of the contrast; the unrewarded trials contribute too."),
                     (0.17, "That is only the unrewarded half of the contrast; the rewarded trials contribute too."),
                     (0.165, "You averaged the two halves of the contrast instead of adding them (a matter of convention, "
                             "but the definition in the text adds them).")],
        hints=["MB index = [P(stay | rewarded, common) − P(stay | rewarded, rare)] − [P(stay | unrewarded, common) − P(stay | unrewarded, rare)].",
               "(0.80 − 0.64) − (0.55 − 0.72)",
               "0.16 − (−0.17) = 0.33"],
        explanation="A purely model-free agent shows the same common − rare difference whether or not it was rewarded "
                    "(index 0); a purely model-based agent stays more after rewarded-common and unrewarded-rare trials, "
                    "producing a positive interaction. 0.33 is a strongly model-based pattern.",
        check_call='check("14.2", 0.33)',
    ),
    # ------------------------------------------------------------------ 14.3
    CodeExercise(
        id="14.3", title="Relating a parameter to a symptom score with covariates",
        reference=symptom_regression,
        signature="symptom_regression(df, param, symptom, covariates=())",
        cases=[{"df": _DF, "param": "w", "symptom": "compulsivity", "covariates": ("age", "anxiety")},
               {"df": _DF, "param": "w", "symptom": "compulsivity", "covariates": ()},
               {"df": _DF, "param": "w", "symptom": "anxiety", "covariates": ("age",)}],
        atol=1e-4, rtol=1e-4,
        bug_variants={
            "raw": (_symptom_reg_raw, "You did not z-score the variables: the coefficient is then in raw units "
                                      "(change in w per questionnaire point) rather than a standardised beta."),
            "reversed": (_symptom_reg_reversed, "You regressed the symptom on the parameter. Here the fitted parameter "
                                                "is the outcome and the symptom (plus covariates) the predictor."),
            "no_constant": (_symptom_reg_no_constant, "You forgot the intercept (`sm.add_constant`). Even with z-scored "
                                                      "variables, always include it; and index the symptom's coefficient correctly."),
        },
        hints=["z-score every column involved: `(df[cols] - df[cols].mean()) / df[cols].std()`.",
               "Design matrix: `sm.add_constant(z[[symptom] + list(covariates)].to_numpy())`; outcome: `z[param]`.",
               "`result = sm.OLS(y, X).fit()`; the symptom's coefficient and p-value are `result.params[1]` and `result.pvalues[1]`."],
        explanation="Standardising makes the coefficient a partial correlation-like quantity: the expected change in w "
                    "(in SD units) per SD of compulsivity, holding age and anxiety fixed. Controlling for covariates "
                    "guards against the most obvious confounds, but not against unmeasured ones.",
        common_mistakes="Raw instead of standardised coefficients; swapping outcome and predictor; forgetting the intercept.",
        check_call='check("14.3", symptom_regression)',
    ),
    # ------------------------------------------------------------------ 14.4
    ChoiceExercise(
        id="14.4", title="What a parameter–symptom correlation shows",
        question="Fitted w correlates negatively with compulsivity (r = −0.3, p < .01) in a large online sample. Which conclusion is justified?",
        options={"a": "People with compulsive symptoms cannot plan.",
                 "b": "In this task, higher compulsivity goes with a lower estimated weight on model-based values — provided w is recoverable and reliable and the relation survives obvious confounds.",
                 "c": "Compulsivity causes a reduction in model-based control.",
                 "d": "The two-step task can be used to diagnose obsessive–compulsive disorder."},
        correct="b",
        feedback={"a": "A lower *weight* on model-based values in one task is not an inability to plan; the effect is "
                       "dimensional and modest in size, and the task measures a specific trade-off.",
                  "c": "Correlational data cannot establish causal direction (nor rule out a third variable).",
                  "d": "An r of −0.3 explains ~9% of the variance: useful for theory, useless for diagnosis."},
        hints=["Think about what a correlation in cross-sectional data can and cannot tell you, and about the conditions "
               "under which a fitted parameter is interpretable at all (lessons 6 and this one).",
               "The most defensible statement is also the most carefully hedged one."],
        explanation="Gillan et al. (2016) found exactly this kind of relation (compulsivity ↔ reduced model-based control) "
                    "and interpreted it dimensionally. Parameter–symptom correlations are informative about mechanisms "
                    "only to the extent that the parameter is recoverable, reliable and not confounded — and they "
                    "never establish causation on their own.",
        check_call='check("14.4", "b")',
    ),
    # ------------------------------------------------------------------ 14.5
    CodeExercise(
        id="14.5", title="Split-half reliability with the Spearman–Brown correction",
        reference=split_half_reliability,
        signature="split_half_reliability(x_half1, x_half2)",
        cases=[{"x_half1": _HALF1, "x_half2": _HALF2},
               {"x_half1": np.array([0.1, 0.4, 0.5, 0.9, 0.3]), "x_half2": np.array([0.2, 0.3, 0.6, 0.8, 0.35])}],
        bug_variants={
            "uncorrected": (_split_half_uncorrected, "You returned the raw correlation between the halves. Each half "
                                                     "uses only half the data, so apply the Spearman–Brown correction 2r / (1 + r)."),
            "wrong_formula": (_split_half_wrong_formula, "The Spearman–Brown formula is 2r / (1 + r), not r / (1 + r)."),
        },
        hints=["First compute the Pearson correlation between the two half-session estimates (`np.corrcoef`).",
               "Then correct it for the fact that each half contains only half of the trials: r_full = 2 r / (1 + r).",
               "`r = np.corrcoef(x_half1, x_half2)[0, 1]; return 2 * r / (1 + r)`"],
        explanation="Reliability is the proportion of variance in a measure that is stable rather than noise. The "
                    "correlation between two halves underestimates the reliability of the whole session because each "
                    "half is noisier; Spearman–Brown corrects for that (assuming the halves are parallel).",
        common_mistakes="Forgetting the correction; mixing up the formula; using the covariance instead of the correlation.",
        check_call='check("14.5", split_half_reliability)',
    ),
    # ------------------------------------------------------------------ 14.6
    NumericExercise(
        id="14.6", title="Spearman–Brown by hand",
        answer=2 * 0.5 / 1.5, atol=1e-3,
        distractors=[(0.5, "That is the uncorrected half-session correlation."),
                     (0.25, "You squared the correlation; the formula is 2r / (1 + r)."),
                     (1 / 3, "You computed r / (1 + r); the numerator is 2r."),
                     (0.75, "That is (1 + r) / 2, not 2r / (1 + r).")],
        hints=["Spearman–Brown: r_full = 2 r_half / (1 + r_half).",
               "2 × 0.5 / (1 + 0.5)"],
        explanation="2 × 0.5 / 1.5 ≈ 0.667: doubling the number of trials raises the reliability from 0.5 to 0.67, "
                    "which is one reason longer tasks (or several sessions) are so valuable for individual differences.",
        check_call='check("14.6", 2 * 0.5 / 1.5)',
    ),
    # ------------------------------------------------------------------ 14.7
    NumericExercise(
        id="14.7", title="Attenuation: the ceiling on an observable correlation",
        answer=float(np.sqrt(0.5 * 0.8)), atol=1e-3,
        distractors=[(0.4, "That is the product of the reliabilities; the ceiling is its square root."),
                     (0.65, "You averaged the reliabilities. The ceiling is sqrt(rel_x × rel_y)."),
                     (0.5, "The ceiling is not simply the smaller reliability."),
                     (float(np.sqrt(0.8)), "You used only one of the two reliabilities.")],
        hints=["Classical test theory: r_observed = r_true × sqrt(rel_x × rel_y).",
               "The maximum is reached when r_true = 1: sqrt(0.5 × 0.8)."],
        explanation="Even a perfect true relation would show up as r ≈ 0.63 when one measure has reliability 0.5 and "
                    "the other 0.8. Observed parameter–symptom correlations of 0.1–0.3 must be read against this ceiling "
                    "(Brown et al. 2020; Zorowitz & Niv 2023).",
        check_call='check("14.7", np.sqrt(0.5 * 0.8))',
    ),
    # ------------------------------------------------------------------ 14.9
    CodeExercise(
        id="14.9", title="Power by simulation",
        reference=power_by_simulation,
        signature="power_by_simulation(r_true, n, n_sims, rng, alpha=0.05)",
        cases=[{"r_true": 0.3, "n": 60, "n_sims": 300, "rng": np.random.default_rng(145)},
               {"r_true": 0.0, "n": 40, "n_sims": 300, "rng": np.random.default_rng(146)},
               {"r_true": 0.5, "n": 30, "n_sims": 300, "rng": np.random.default_rng(147), "alpha": 0.01}],
        compare=_power_compare,
        diagnose=_power_diagnose,
        hints=["Inside a loop of n_sims iterations: draw n pairs with correlation r_true, e.g. with "
               "`rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=n)`, and test them with `scipy.stats.pearsonr`.",
               "Count how often the p-value is below alpha and divide by n_sims.",
               "`p = stats.pearsonr(xy[:, 0], xy[:, 1])[1]; n_significant += p < alpha` ... `return n_significant / n_sims`"],
        explanation="Simulation-based power analysis generalises to any design: replace the bivariate normal with "
                    "'simulate participants from the model with the hypothesised parameter–symptom relation, fit them, "
                    "run the planned test' and you have the power of the *whole* pipeline, including the noise added "
                    "by parameter estimation.",
        common_mistakes="Returning a count instead of a proportion; comparing r instead of p with the threshold; "
                        "re-using the same sample in every iteration.",
        check_call='check("14.9", power_by_simulation)',
    ),
    # ------------------------------------------------------------------ 14.8
    ChoiceExercise(
        id="14.8", title="Reading a reliability estimate",
        question="The split-half reliability of fitted w in a design is 0.4 (and the symptom questionnaire's reliability is about 0.9). Which statement follows?",
        options={"a": "Even a perfect true relation between w and the symptom would be expected to appear as r ≈ 0.6 at most; small observed correlations are not surprising.",
                 "b": "w is useless as an individual-differences measure and should be abandoned.",
                 "c": "The true correlation between w and the symptom must be smaller than 0.4.",
                 "d": "Nothing can raise the reliability: it is a fixed property of the parameter."},
        correct="a",
        feedback={"b": "Low reliability limits what can be detected; it does not make the measure useless, and it can be improved.",
                  "c": "Reliability bounds the *observable* correlation (sqrt(0.4 × 0.9) ≈ 0.6); it says nothing about the size of the true relation.",
                  "d": "Reliability depends on the task length, the design and the estimation method: hierarchical fitting "
                       "raised the reliability of model-based indices in Brown et al. (2020)."},
        hints=["Use the attenuation formula from the previous exercise.",
               "sqrt(0.4 × 0.9) ≈ 0.6 is the ceiling for the *observed* correlation."],
        explanation="Reliability caps the observable correlation at sqrt(rel_x × rel_y) ≈ 0.6 here. The remedies are "
                    "design-side (more trials, repeated sessions, tasks tuned for individual differences; Zorowitz & "
                    "Niv 2023) and estimation-side (hierarchical models; Brown et al. 2020).",
        check_call='check("14.8", "a")',
    ),
    # ------------------------------------------------------------------ 14.10
    FreeTextExercise(
        id="14.10", title="Why a significant correlation is not enough",
        question="In 3–5 sentences: a fitted parameter correlates significantly with a symptom score. List the checks you would run before interpreting this as evidence about a mechanism, and say why each matters.",
        model_answer="(1) Parameter recovery in this exact design: if the parameter cannot be recovered from simulated "
                     "data, or trades off with another parameter (e.g. learning rate with inverse temperature), the "
                     "correlation may reflect a different quantity than the one named. (2) Reliability: a parameter that "
                     "is mostly estimation noise caps the observable correlation and makes the result fragile; split-half "
                     "or test–retest reliability (ideally with hierarchical estimation) tells you how much signal there is. "
                     "(3) Confounds and covariates: age, IQ, other symptom dimensions, overall performance or inverse "
                     "temperature may drive both the parameter and the symptom; include them and check specificity. "
                     "(4) Model comparison and posterior predictive checks: if a different model describes the behaviour "
                     "better, the parameter may not mean what the model says; check that the model captures the "
                     "behavioural signature the parameter is supposed to index. (5) Causality and replication: the data "
                     "are correlational, so no causal direction follows; pre-register and replicate, and estimate the "
                     "effect size rather than only its p-value.",
        concepts=[("parameter recovery / identifiability", ["recover", "identif", "trade-off", "tradeoff", "trade off",
                                                            "simulat", "confus"]),
                  ("reliability", ["reliab", "test-retest", "test retest", "split-half", "split half", "noise", "noisy"]),
                  ("confounds / covariates", ["confound", "covariate", "control for", "controlling", "age", "iq", "third variable",
                                              "specific", "other symptom"]),
                  ("model comparison / alternative models", ["model comparison", "alternative model", "other model",
                                                              "compare models", "better model", "posterior predictive",
                                                              "mis-specif", "misspecif", "falsif"]),
                  ("causality / replication", ["caus", "direction", "replicat", "pre-regist", "preregist", "correlation is not",
                                               "cross-sectional", "effect size"])],
        min_concepts=3,
        hints=["Think of the checks from lessons 6 and 7 (recovery, model comparison) and of this lesson's reliability and confound sections.",
               "At least three distinct checks, each with a one-clause reason."],
        explanation="The checks are the difference between 'a number correlates with a questionnaire' and 'a mechanism "
                    "differs with symptoms'. Each check removes one way the correlation could be an artefact.",
        check_call='check("14.10", """Before interpreting it I would check parameter recovery and identifiability in this exact design, because a parameter that trades off with another cannot be interpreted. I would estimate the reliability of the parameter (split-half or test-retest), because noise caps the observable correlation. I would control for confounds and covariates such as age and other symptom dimensions, and test specificity. I would run model comparison and posterior predictive checks to make sure the model captures the behaviour. Finally the data are correlational, so no causal direction follows, and the effect should be replicated.""")',
    ),
]
