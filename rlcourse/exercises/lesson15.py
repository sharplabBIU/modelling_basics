"""Exercises of lesson 15: the capstone project."""
from __future__ import annotations

import numpy as np

from .. import fitting, models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference implementations
# --------------------------------------------------------------------------- #


def summarise_subject(choices, rewards, p_arm0):
    """Model-agnostic summary of one participant of the capstone task.

    ``p_arm0[t]`` is the reward probability of arm 0 on trial t (arm 1 pays with
    probability 1 - p_arm0[t]).  Returns a dict with the reward rate, the
    proportion of choices of the currently better arm, and the probability of
    repeating the previous choice after a reward and after no reward."""
    choices = np.asarray(choices)
    rewards = np.asarray(rewards, dtype=float)
    p_arm0 = np.asarray(p_arm0, dtype=float)
    better = np.where(p_arm0 > 0.5, 0, 1)                       # which arm is better on each trial
    stay = (choices[1:] == choices[:-1]).astype(float)          # did trial t repeat trial t-1?
    previous_reward = rewards[:-1]
    return {"reward_rate": float(np.mean(rewards)),
            "p_better": float(np.mean(choices == better)),
            "p_stay_win": float(np.mean(stay[previous_reward == 1])),
            "p_stay_lose": float(np.mean(stay[previous_reward == 0]))}


def fit_subject_all_models(choices, rewards, rng, n_starts=3):
    """Fit every candidate model in ``models.MODEL_SPECS`` to one participant by
    maximum likelihood.  Returns {model: {"nll", "k", "params"}} — everything
    needed for BIC and for simulating from the fitted model."""
    out = {}
    for name, (nll_fn, param_names, kinds) in models.MODEL_SPECS.items():
        fit = fitting.fit_mle(nll_fn, (choices, rewards), kinds, n_starts=n_starts, rng=rng)
        out[name] = {"nll": fit["nll"], "k": len(kinds), "params": dict(zip(param_names, fit["params"]))}
    return out


def simulate_asym_subject(alpha_pos, alpha_neg, beta, p_arm0, rng, q0=0.5):
    """Simulate a Rescorla–Wagner learner with asymmetric learning rates on the
    capstone task (reward probabilities ``p_arm0`` for arm 0, ``1 - p_arm0`` for arm 1).

    Returns dict(choices, rewards, Q) with Q the values *before* each choice."""
    p_arm0 = np.asarray(p_arm0, dtype=float)
    n_trials = len(p_arm0)
    Q = np.full(2, q0, dtype=float)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    Q_history = np.zeros((n_trials, 2))
    for t in range(n_trials):
        Q_history[t] = Q
        p = models.softmax(Q, beta)
        c = models.choose(p, rng)
        p_reward = p_arm0[t] if c == 0 else 1 - p_arm0[t]
        r = float(rng.random() < p_reward)
        delta = r - Q[c]
        alpha = alpha_pos if delta > 0 else alpha_neg         # asymmetric learning
        Q[c] = Q[c] + alpha * delta
        choices[t], rewards[t] = c, r
    return {"choices": choices, "rewards": rewards, "Q": Q_history}


# --------------------------------------------------------------------------- #
# Deterministic test data (the capstone task: 80 trials at 0.7/0.3, then 80 at 0.3/0.7)
# --------------------------------------------------------------------------- #

_P_ARM0 = np.r_[np.full(80, 0.7), np.full(80, 0.3)]


def _subject(alpha_pos, alpha_neg, beta, seed):
    return simulate_asym_subject(alpha_pos, alpha_neg, beta, _P_ARM0, np.random.default_rng(seed))


_S1 = _subject(0.5, 0.2, 4.0, 151)
_S2 = _subject(0.2, 0.5, 3.0, 152)
_S3 = _subject(0.3, 0.3, 6.0, 153)


# --------------------------------------------------------------------------- #
# Checks and diagnostics
# --------------------------------------------------------------------------- #


def _summary_better_flipped(choices, rewards, p_arm0):
    out = summarise_subject(choices, rewards, p_arm0)
    out["p_better"] = 1 - out["p_better"]
    return out


def _summary_no_lag(choices, rewards, p_arm0):
    choices = np.asarray(choices); rewards = np.asarray(rewards, float); p_arm0 = np.asarray(p_arm0, float)
    better = np.where(p_arm0 > 0.5, 0, 1)
    stay = (choices[1:] == choices[:-1]).astype(float)
    return {"reward_rate": float(rewards.mean()), "p_better": float(np.mean(choices == better)),
            "p_stay_win": float(np.mean(stay[rewards[1:] == 1])), "p_stay_lose": float(np.mean(stay[rewards[1:] == 0]))}


def _fits_compare(out, ref, case):
    if not isinstance(out, dict) or set(out) != set(ref):
        return False
    for name in ref:
        try:
            o, r = float(out[name]["nll"]), float(ref[name]["nll"])
            if int(out[name]["k"]) != int(ref[name]["k"]):
                return False
        except Exception:
            return False
        if not (r - 2.0 <= o <= r + 0.5):
            return False
    return True


def _fits_diagnose(out, ref, case):
    if not isinstance(out, dict):
        return "Return a dictionary with one entry per model name in `models.MODEL_SPECS`."
    missing = [m for m in ref if m not in out]
    if missing:
        return f"Missing model(s) {missing}: loop over *all* of `models.MODEL_SPECS.items()`."
    for name in ref:
        entry = out[name]
        if not isinstance(entry, dict) or "nll" not in entry or "k" not in entry:
            return f"Each entry should be a dict with keys 'nll', 'k' and 'params' (problem with {name!r})."
        if int(entry["k"]) != int(ref[name]["k"]):
            return f"`k` for {name!r} should be the number of parameters of that model, i.e. `len(kinds)`."
        o, r = float(entry["nll"]), float(ref[name]["nll"])
        if o > r + 0.5:
            return (f"The fit of {name!r} is worse than the reference (nll {o:.1f} vs {r:.1f}): use `fitting.fit_mle` with "
                    "the model's `kinds`, the data as `(choices, rewards)` and at least 3 random starts.")
        if o < r - 2.0:
            return (f"The fit of {name!r} is implausibly good (nll {o:.1f} vs {r:.1f}): are you fitting the right "
                    "data, and is `nll` the value returned by `fitting.fit_mle` (`fit['nll']`)?")
    return None


def _asym_consistent(out, alpha_pos, alpha_neg, beta, p_arm0, q0=0.5):
    try:
        choices = np.asarray(out["choices"]).astype(int)
        rewards = np.asarray(out["rewards"], float)
        Q = np.asarray(out["Q"], float)
    except Exception:
        return False
    n = len(p_arm0)
    if Q.shape != (n, 2) or len(choices) != n or len(rewards) != n or not np.allclose(Q[0], q0):
        return False
    if not set(np.unique(rewards)).issubset({0.0, 1.0}):
        return False
    for t in range(n - 1):
        expected = Q[t].copy()
        delta = rewards[t] - Q[t, choices[t]]
        expected[choices[t]] = Q[t, choices[t]] + (alpha_pos if delta > 0 else alpha_neg) * delta
        if not np.allclose(Q[t + 1], expected):
            return False
    return True


def _asym_compare(out, ref, case):
    if not _asym_consistent(out, case["alpha_pos"], case["alpha_neg"], case["beta"], case["p_arm0"], case.get("q0", 0.5)):
        return False
    choices = np.asarray(out["choices"])
    p_arm0 = np.asarray(case["p_arm0"], float)
    better = np.where(p_arm0 > 0.5, 0, 1)
    return np.mean(choices == better) > 0.58


def _asym_diagnose(out, ref, case):
    if not isinstance(out, dict):
        return f"Return a dictionary with keys 'choices', 'rewards' and 'Q' (you returned `{type(out).__name__}`)."
    for key in ("choices", "rewards", "Q"):
        if key not in out:
            return f"The returned dictionary has no key {key!r}."
    p_arm0 = np.asarray(case["p_arm0"], float)
    n = len(p_arm0)
    choices, rewards, Q = np.asarray(out["choices"]), np.asarray(out["rewards"], float), np.asarray(out["Q"], float)
    if Q.shape != (n, 2):
        return f"`Q` should have shape (n_trials, 2) = {(n, 2)}, but has shape {Q.shape}."
    if not np.allclose(Q[0], case.get("q0", 0.5)):
        return "`Q[0]` should hold the initial values: record the values *before* the choice on every trial."
    if not set(np.unique(rewards)).issubset({0.0, 1.0}):
        return "Rewards should be 0 or 1."
    for t in range(n - 1):
        delta = rewards[t] - Q[t, choices[t]]
        expected = Q[t].copy()
        expected[choices[t]] = Q[t, choices[t]] + (case["alpha_pos"] if delta > 0 else case["alpha_neg"]) * delta
        if not np.allclose(Q[t + 1], expected):
            sym = Q[t].copy()
            sym[choices[t]] = Q[t, choices[t]] + case["alpha_pos"] * delta
            if np.allclose(Q[t + 1], sym):
                return "You always used alpha_pos: use alpha_neg when the prediction error is negative."
            sym[choices[t]] = Q[t, choices[t]] + case["alpha_neg"] * delta
            if np.allclose(Q[t + 1], sym):
                return "You always used alpha_neg: use alpha_pos when the prediction error is positive."
            if any(not np.isclose(Q[t + 1, a], Q[t, a]) for a in range(2) if a != choices[t]):
                return "The unchosen arm's value changed: only the chosen arm is updated."
            return (f"On trial {t} the chosen arm's value moves from {Q[t, choices[t]]:.3f} to {Q[t + 1, choices[t]]:.3f}, "
                    f"but the asymmetric rule gives {expected[choices[t]]:.3f} (delta = {delta:+.3f}).")
    better = np.where(p_arm0 > 0.5, 0, 1)
    if np.mean(choices == better) <= 0.58:
        return ("Learning is consistent but the agent does not prefer the better arm: check the reward sampling "
                "(`p_arm0[t]` for arm 0, `1 - p_arm0[t]` for arm 1) and the softmax.")
    return None


# --------------------------------------------------------------------------- #
EXERCISES = [
    # ------------------------------------------------------------------ 15.1
    CodeExercise(
        id="15.1", title="A model-agnostic summary of one participant",
        reference=summarise_subject,
        signature="summarise_subject(choices, rewards, p_arm0)",
        cases=[{"choices": _S1["choices"], "rewards": _S1["rewards"], "p_arm0": _P_ARM0},
               {"choices": _S2["choices"], "rewards": _S2["rewards"], "p_arm0": _P_ARM0}],
        bug_variants={
            "flipped": (_summary_better_flipped, "`p_better` is inverted: arm 0 is the better arm when `p_arm0 > 0.5`, "
                                                 "otherwise arm 1 is."),
            "no_lag": (_summary_no_lag, "The stay probabilities are misaligned: staying on trial t (choice t equals "
                                        "choice t−1) must be conditioned on the reward of trial t−1, i.e. `rewards[:-1]`."),
        },
        hints=["The better arm on trial t is arm 0 if `p_arm0[t] > 0.5`, else arm 1 (`np.where`).",
               "`stay = (choices[1:] == choices[:-1])` has one entry per trial from the second one; condition it on `rewards[:-1]`.",
               "`p_stay_win = stay[rewards[:-1] == 1].mean()`, `p_stay_lose = stay[rewards[:-1] == 0].mean()`."],
        explanation="These four numbers describe behaviour without committing to a model: the reward rate and "
                    "P(better arm) measure performance, and the two stay probabilities are the simplest signature of "
                    "learning from positive vs negative feedback — the very thing the asymmetric model formalises.",
        common_mistakes="Inverting the better arm; conditioning on the current rather than the previous reward.",
        check_call='check("15.1", summarise_subject)',
    ),
    # ------------------------------------------------------------------ 15.2
    CodeExercise(
        id="15.2", title="Fit all candidate models to one participant",
        reference=fit_subject_all_models,
        signature="fit_subject_all_models(choices, rewards, rng, n_starts=3)",
        cases=[{"choices": _S1["choices"], "rewards": _S1["rewards"], "rng": np.random.default_rng(154)},
               {"choices": _S3["choices"], "rewards": _S3["rewards"], "rng": np.random.default_rng(155)}],
        compare=_fits_compare,
        diagnose=_fits_diagnose,
        hints=["`models.MODEL_SPECS` maps each model name to `(nll_function, parameter_names, kinds)`; loop over its `.items()`.",
               "For each model call `fitting.fit_mle(nll_fn, (choices, rewards), kinds, n_starts=n_starts, rng=rng)`.",
               "Store `fit['nll']`, `len(kinds)` and `dict(zip(param_names, fit['params']))` under the model's name."],
        explanation="Fitting every candidate with the same procedure and data is what makes the comparison fair. "
                    "The negative log-likelihood and the number of parameters are all BIC needs; the fitted parameters "
                    "are what you will simulate from for the posterior predictive check.",
        common_mistakes="Forgetting a model; passing the data in the wrong order; too few random starts.",
        check_call='check("15.2", fit_subject_all_models)',
    ),
    # ------------------------------------------------------------------ 15.3
    ChoiceExercise(
        id="15.3", title="Reading the model-recovery matrix",
        question="In your model recovery, data simulated from the asymmetric model (rw_asym) are best fit by the plain RW model in about 40% of the simulations, while data simulated from RW are almost never best fit by rw_asym. What does this mean?",
        options={"a": "The asymmetry is hard to detect with 160 trials: when the simulated alpha+ and alpha− are similar the simpler model wins through the BIC penalty, so evidence *for* rw_asym in real data is conservative and a win for RW is ambiguous.",
                 "b": "The asymmetric model is wrong and should be dropped from the comparison.",
                 "c": "BIC is biased and should be replaced by the raw log-likelihood.",
                 "d": "The simulations used too many trials."},
        correct="a",
        feedback={"b": "Recovery says nothing about which model is *true*; it tells you how the comparison behaves when "
                       "each model is true.",
                  "c": "Raw log-likelihood would always favour the model with more parameters; the penalty is doing its job.",
                  "d": "More trials would make the asymmetry *easier* to detect."},
        hints=["Think about what happens to rw_asym when alpha+ ≈ alpha−: it becomes the RW model with one extra parameter.",
               "An asymmetric confusion is still informative: which direction of error is possible and which is not?"],
        explanation="This asymmetric confusion is typical: a nested model can mimic the richer one whenever the extra "
                    "parameter is near its 'null' value. A win for the richer model is then trustworthy; a win for the "
                    "simpler one is not strong evidence against asymmetry.",
        check_call='check("15.3", "a")',
    ),
    # ------------------------------------------------------------------ 15.4
    NumericExercise(
        id="15.4", title="A BIC difference by hand",
        answer=float((2 * np.log(160) + 2 * 95.2) - (3 * np.log(160) + 2 * 92.1)), atol=5e-3,
        distractors=[(6.2, "That is 2 × Δnll: you forgot the penalty terms k ln(n)."),
                     (3.1, "That is the difference in negative log-likelihoods, not in BIC (BIC = k ln n + 2 nll)."),
                     (float(-((2 * np.log(160) + 2 * 95.2) - (3 * np.log(160) + 2 * 92.1))), "Sign: the question asks for BIC(rw) − BIC(rw_asym)."),
                     (float((2 * np.log10(160) + 2 * 95.2) - (3 * np.log10(160) + 2 * 92.1)), "You used log10; BIC uses the natural logarithm."),
                     (float((2 * np.log(160) + 95.2) - (3 * np.log(160) + 92.1)), "You forgot the factor 2 in front of the negative log-likelihood.")],
        hints=["BIC = k ln(n) + 2 nll for each model, then subtract.",
               "BIC(rw) = 2 ln 160 + 2 × 95.2;  BIC(rw_asym) = 3 ln 160 + 2 × 92.1.",
               "ln 160 ≈ 5.075, so the penalty difference is 5.075 and the fit difference is 6.2."],
        explanation="BIC(rw) − BIC(rw_asym) ≈ 1.12 > 0: the asymmetric model wins, but only just — a difference "
                    "below ~2 is 'barely worth mentioning' on the Kass & Raftery scale. The extra parameter bought "
                    "3.1 nats of fit but cost 5.075 / 2 ≈ 2.5 nats of penalty.",
        check_call='check("15.4", (2 * np.log(160) + 2 * 95.2) - (3 * np.log(160) + 2 * 92.1))',
    ),
    # ------------------------------------------------------------------ 15.5
    ChoiceExercise(
        id="15.5", title="Interpreting parameter recovery before the symptom analysis",
        question="Suppose that in this design alpha+ recovers with r = 0.9 but alpha− only with r = 0.4 (and recovered alpha− correlates with recovered beta). What follows for the symptom analysis?",
        options={"a": "A null result for alpha− would be uninformative and a positive one hard to interpret; conclusions about negative-feedback learning need a better design (more trials or more negative prediction errors) or hierarchical estimation.",
                 "b": "alpha− can be dropped from the model because it does not exist.",
                 "c": "The fitting code must be wrong, since both learning rates are estimated by the same procedure.",
                 "d": "r = 0.4 is adequate for correlating alpha− with symptoms as long as the sample is large."},
        correct="a",
        feedback={"b": "Poor recovery is a property of the design and the data, not evidence that the process is absent.",
                  "c": "Two parameters estimated by the same procedure can differ enormously in recoverability: it depends "
                       "on how much the data constrain each of them.",
                  "d": "A large sample does not fix a parameter that trades off with another: the estimates are "
                       "systematically, not just randomly, wrong."},
        hints=["Recall lesson 6: recovery tells you whether the data in *this* design constrain the parameter.",
               "A parameter that correlates with beta in recovery is partly measuring beta."],
        explanation="Recovery is design-specific. Before relating a parameter to symptoms, show that this exact task, "
                    "trial count and fitting method can recover it; otherwise any correlation (or its absence) is "
                    "uninterpretable (Wilson & Collins 2019; Eckstein et al. 2021).",
        check_call='check("15.5", "a")',
    ),
    # ------------------------------------------------------------------ 15.6
    NumericExercise(
        id="15.6", title="Correcting a correlation for attenuation",
        answer=float(-0.30 / np.sqrt(0.6 * 0.8)), atol=1e-3,
        distractors=[(float(-0.30 / (0.6 * 0.8)), "You divided by the product of the reliabilities; divide by its square root."),
                     (float(-0.30 * np.sqrt(0.6 * 0.8)), "You multiplied by sqrt(rel_x × rel_y); the observed correlation is *divided* by it."),
                     (float(0.30 / np.sqrt(0.6 * 0.8)), "Sign: a negative observed correlation stays negative after correction."),
                     (-0.3, "That is the uncorrected observed correlation.")],
        hints=["r_true = r_observed / sqrt(rel_x × rel_y).",
               "−0.30 / sqrt(0.6 × 0.8)"],
        explanation="−0.30 / sqrt(0.48) ≈ −0.43: the disattenuated estimate of the true relation. Report it alongside "
                    "the raw value — it is an estimate with its own (larger) uncertainty, not a licence to claim a big effect.",
        check_call='check("15.6", -0.30 / np.sqrt(0.6 * 0.8))',
    ),
    # ------------------------------------------------------------------ 15.7
    CodeExercise(
        id="15.7", title="Simulate a participant from the fitted asymmetric model",
        reference=simulate_asym_subject,
        signature="simulate_asym_subject(alpha_pos, alpha_neg, beta, p_arm0, rng, q0=0.5)",
        cases=[{"alpha_pos": 0.5, "alpha_neg": 0.2, "beta": 5.0, "p_arm0": _P_ARM0, "rng": np.random.default_rng(156)},
               {"alpha_pos": 0.2, "alpha_neg": 0.5, "beta": 4.0, "p_arm0": _P_ARM0, "rng": np.random.default_rng(157), "q0": 0.0}],
        compare=_asym_compare,
        diagnose=_asym_diagnose,
        hints=["Start from the simulation loop of lesson 3 (exercise 3.8): record Q, softmax, choose, reward, update.",
               "The reward probability of the chosen arm on trial t is `p_arm0[t]` for arm 0 and `1 - p_arm0[t]` for arm 1.",
               "Pick the learning rate by the sign of the prediction error: `alpha = alpha_pos if delta > 0 else alpha_neg`."],
        explanation="Simulating from the fitted parameters is the posterior predictive check: if the simulated "
                    "participants reproduce the summary statistics of the real ones (learning curves, stay "
                    "probabilities), the model captures the behaviour it is meant to explain; if not, the fitted "
                    "parameters describe a model of something else.",
        common_mistakes="Using one learning rate for both signs of the prediction error; sampling the reward with the "
                        "wrong arm's probability; recording Q after the update.",
        check_call='check("15.7", simulate_asym_subject)',
    ),
    # ------------------------------------------------------------------ 15.8
    FreeTextExercise(
        id="15.8", title="Limitations of your analysis",
        question="Write the Limitations paragraph of your report (3–6 sentences).",
        model_answer="The data are simulated, so the ground truth is known and the symptom relation was built in; real "
                     "data add measurement error in the questionnaire, heterogeneity and model mis-specification. The "
                     "design is cross-sectional and correlational, so no causal direction can be inferred between "
                     "anhedonia and the positive learning rate. Parameter estimates are noisy: the split-half "
                     "reliability of alpha+ and the recovery correlations bound how large an observed relation can be, "
                     "and alpha− and beta are less well constrained by 160 trials. The model set was small: an "
                     "unmodelled process (e.g. working memory, changing attention, or a different choice rule) could "
                     "produce the same signature. The sample size gives limited power for small effects, and the "
                     "relation should be replicated in an independent, pre-registered sample with a clinical range of "
                     "symptoms before any clinical interpretation.",
        concepts=[("simulated data / ground truth", ["simulat", "synthetic", "ground truth", "not real", "artificial", "built in"]),
                  ("correlational / causality", ["caus", "correlation", "cross-sectional", "direction"]),
                  ("reliability / recovery / identifiability", ["reliab", "recover", "identif", "noise", "noisy", "trade-off", "tradeoff"]),
                  ("alternative models / mis-specification", ["other model", "alternative", "mis-specif", "misspecif",
                                                             "model set", "unmodel", "working memory", "attention", "different model"]),
                  ("sample size / replication / generalisation", ["sample size", "power", "replicat", "generalis", "generaliz",
                                                                   "clinical", "pre-regist", "preregist"])],
        min_concepts=3,
        hints=["Think about the data (what they are), the design (what it can show), the measurement (how noisy), the "
               "models (what else could explain it) and the inference (sample, replication).",
               "Each sentence should name one limitation and why it matters."],
        explanation="A good limitations section is the mirror image of lesson 14's checklist: it tells the reader "
                    "exactly how far the conclusions reach.",
        check_call='check("15.8", """The data are simulated, so the ground truth is known and the symptom relation was built in; real data add measurement error and heterogeneity. The design is correlational and cross-sectional, so no causal direction can be inferred. The parameter estimates are noisy: the split-half reliability and the recovery correlations bound how large an observed relation can be. The model set was small, so an alternative model could produce the same signature. The sample size limits power and the relation should be replicated in an independent clinical sample.""")',
    ),
]
