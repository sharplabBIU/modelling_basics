"""Exercises of lesson 12: volatility and the volatile Kalman filter (Piray & Daw 2020)."""
from __future__ import annotations

import numpy as np

from .. import fitting, models, tasks
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# deterministic test data
# --------------------------------------------------------------------------- #
_rng = np.random.default_rng(12)
_v_true = np.r_[np.full(50, 0.01), np.full(50, 4.0), np.full(50, 0.01)]
_walk = np.cumsum(_rng.normal(0, np.sqrt(_v_true)))
_cont_outcomes = _walk + _rng.normal(0, 1.0, len(_walk))
_task = tasks.ReversalTask.stable_volatile(rng=np.random.default_rng(1201))
_bin_outcomes = np.array([_task.step(0)[1] for _ in range(_task.n_trials)])
_block = _task.block.copy()


def _simulate_vkf_choices(outcomes, lam, v0, omega, beta, rng):
    """Choices of a VKF agent on a full-information task (outcome = 1 if option 0 paid)."""
    sig = models.vkf_bin(outcomes, lam, v0, omega)
    p0 = models.sigmoid(beta * (2 * sig["pred"] - 1))
    return (rng.random(len(outcomes)) >= p0).astype(int)      # 0 with probability p0


_choices_a = _simulate_vkf_choices(_bin_outcomes, 0.1, 0.1, 0.1, 4.0, np.random.default_rng(1))
_choices_b = _simulate_vkf_choices(_bin_outcomes, 0.3, 0.5, 0.5, 2.0, np.random.default_rng(2))


def _simulate_block_rw_choices(outcomes, block, alpha_stable, alpha_volatile, beta, rng):
    """Choices of a Rescorla–Wagner learner whose learning rate depends on the block."""
    p_hat, choices = 0.5, np.zeros(len(outcomes), int)
    for t, o in enumerate(outcomes):
        p0 = models.sigmoid(beta * (2 * p_hat - 1))
        choices[t] = 0 if rng.random() < p0 else 1
        alpha = alpha_volatile if block[t] == "volatile" else alpha_stable
        p_hat = p_hat + alpha * (o - p_hat)
    return choices


_choices_rw = _simulate_block_rw_choices(_bin_outcomes, _block, 0.15, 0.5, 5.0, np.random.default_rng(3))


# --------------------------------------------------------------------------- #
# lesson-specific reference function (exercise 12.8)
# --------------------------------------------------------------------------- #
def fit_block_learning_rates(outcomes, choices, block, rng):
    """Fit the Rescorla–Wagner reversal model separately to the stable and the
    volatile block.  Returns (alpha_stable, alpha_volatile)."""
    outcomes, choices, block = np.asarray(outcomes), np.asarray(choices), np.asarray(block)
    alphas = []
    for name in ["stable", "volatile"]:
        mask = block == name
        fit = fitting.fit_mle(models.rw_reversal_nll, (outcomes[mask], choices[mask]),
                              ["unit", "pos"], n_starts=3, rng=rng)
        alphas.append(fit["params"][0])                   # params = (alpha, beta)
    return alphas[0], alphas[1]


# --------------------------------------------------------------------------- #
# buggy variants used for targeted diagnoses
# --------------------------------------------------------------------------- #
def _vkf_c_variant(outcomes, lam, v0, sigma2, bug):
    outcomes = np.asarray(outcomes, float)
    n = len(outcomes)
    m, w, v = 0.0, sigma2, v0
    out = {key: np.zeros(n) for key in ["m", "v", "k", "pe", "w"]}
    for t, o in enumerate(outcomes):
        if bug != "record_after":
            out["m"][t], out["v"][t], out["w"][t] = m, v, w
        m_prev, w_prev = m, w
        k = (w / (w + sigma2)) if bug == "gain_without_v" else (w + v) / (w + v + sigma2)
        pe = (m - o) if bug == "pe_sign" else (o - m)
        m = m + k * pe
        w = (1 - k) * w_prev if bug == "var_without_v" else (1 - k) * (w_prev + v)
        w_cov = (1 - k) * w_prev
        if bug == "no_vol_update":
            pass
        elif bug == "no_cov_term":
            v = v + lam * ((m - m_prev) ** 2 + w + w_prev - v)
        else:
            v = v + lam * ((m - m_prev) ** 2 + w + w_prev - 2 * w_cov - v)
        if bug == "record_after":
            out["m"][t], out["v"][t], out["w"][t] = m, v, w
        out["k"][t], out["pe"][t] = k, pe
    return out


def _vkf_b_variant(outcomes, lam, v0, omega, bug):
    outcomes = np.asarray(outcomes, float)
    n = len(outcomes)
    m, w, v = 0.0, omega, v0
    out = {key: np.zeros(n) for key in ["m", "pred", "v", "alpha", "k", "pe", "w"]}
    for t, o in enumerate(outcomes):
        out["m"][t], out["v"][t], out["w"][t] = m, v, w
        out["pred"][t] = m if bug == "pred_no_sigmoid" else models.sigmoid(m)
        m_prev, w_prev = m, w
        pe = (o - m) if bug == "pe_no_sigmoid" else (o - models.sigmoid(m))
        k = (w + v) / (w + v + omega)
        alpha = k if bug == "gain_as_rate" else np.sqrt(w + v)
        m = m + alpha * pe
        w = (1 - k) * (w_prev + v)
        w_cov = (1 - k) * w_prev
        if bug != "no_vol_update":
            v = v + lam * ((m - m_prev) ** 2 + w + w_prev - 2 * w_cov - v)
        out["alpha"][t], out["k"][t], out["pe"][t] = alpha, k, pe
    return out


def _make_c(bug):
    def variant(outcomes, lam, v0, sigma2):
        with np.errstate(all="ignore"):          # buggy filters may overflow; keep the student's output clean
            return _vkf_c_variant(outcomes, lam, v0, sigma2, bug)
    return variant


def _make_b(bug):
    def variant(outcomes, lam, v0, omega):
        with np.errstate(all="ignore"):
            return _vkf_b_variant(outcomes, lam, v0, omega, bug)
    return variant


def _nll_variant(params, outcomes, choices, bug):
    lam, v0, omega, beta = params
    pred = models.vkf_bin(outcomes, lam, v0, omega)["pred"]
    if bug == "no_logistic":
        p0 = pred
    elif bug == "no_centering":
        p0 = models.sigmoid(beta * pred)
    else:
        p0 = models.sigmoid(beta * (2 * pred - 1))
    choices = np.asarray(choices)
    if bug == "flipped_coding":
        p_choice = np.where(choices == 1, p0, 1 - p0)
    else:
        p_choice = np.where(choices == 0, p0, 1 - p0)
    ll = float(np.sum(np.log(p_choice)))
    return ll if bug == "positive_ll" else -ll


def _make_nll(bug):
    def variant(params, outcomes, choices):
        with np.errstate(all="ignore"):
            return _nll_variant(params, outcomes, choices, bug)
    return variant


def _blocks_compare(out, ref):
    out = np.asarray(out, float).ravel()
    ref = np.asarray(ref, float).ravel()
    return out.shape == ref.shape and bool(np.allclose(out, ref, atol=0.03))


def _blocks_diagnose(out, ref, case):
    try:
        o = np.asarray(out, float).ravel()
    except Exception:
        return "Return a tuple of two numbers: (alpha_stable, alpha_volatile)."
    r = np.asarray(ref, float).ravel()
    if o.shape != r.shape:
        return f"Return exactly two numbers (alpha_stable, alpha_volatile); you returned {o.size}."
    if np.allclose(o[::-1], r, atol=0.03):
        return "The two learning rates are in the wrong order: return (alpha_stable, alpha_volatile)."
    if np.all((o > 1) | (o < 0)):
        return "The values are outside (0, 1): did you return the unconstrained `x` or the inverse temperature instead of `params[0]`?"
    if np.allclose(o[0], o[1], atol=0.01) and not np.allclose(r[0], r[1], atol=0.01):
        return ("Both blocks gave the same learning rate — did you fit the whole sequence (or the same block) twice "
                "instead of selecting the trials of each block with a boolean mask?")
    return ("The learning rates differ from the reference by more than 0.03. Use `fitting.fit_mle(models.rw_reversal_nll, "
            "(outcomes[mask], choices[mask]), ['unit', 'pos'], n_starts=3, rng=rng)` and read `fit['params'][0]`.")


EXERCISES = [
    # ------------------------------------------------------------------ 12.1
    ChoiceExercise(
        id="12.1", title="Noise or volatility?",
        question="A reversal task has two blocks. In block 1 the better option pays on 75% of trials and never changes; "
                 "in block 2 the better option still pays on 75% of trials but the identity of the better option "
                 "switches every 25 trials. Which statement is correct?",
        options={"a": "The 25% unrewarded trials of block 1 are volatility: the environment is changing on those trials.",
                 "b": "Block 1 has outcome noise but no volatility; block 2 has the same noise and, in addition, volatility.",
                 "c": "Block 2 has less noise than block 1 because the reversals make outcomes more predictable.",
                 "d": "Noise and volatility are the same quantity measured on different time scales."},
        correct="b",
        feedback={"a": "An unrewarded trial from a 75% option is *noise* (an unlucky draw from an unchanged process), not a change of the process.",
                  "c": "The outcome noise (a 25% chance of the unexpected outcome) is identical in both blocks; the reversals add volatility on top of it.",
                  "d": "They are different things: noise is variability of outcomes around a fixed process, volatility is change of the process itself. "
                       "They call for opposite responses (Exercise 12.10)."},
        hints=["Ask of each surprising outcome: is the *process* that generates outcomes still the same, or has it changed?",
               "Noise = unexpected outcomes from an unchanged process; volatility = the process itself changes over time."],
        explanation="Both blocks have the same outcome noise (75/25). Only the second block has volatility, because the "
                    "hidden quantity being learned — which option is better — changes. Behrens et al. (2007) built exactly "
                    "this design to show that people learn faster when the environment is volatile.",
        check_call='check("12.1", "b")',
    ),
    # ------------------------------------------------------------------ 12.2
    NumericExercise(
        id="12.2", title="The Kalman gain by hand",
        answer=0.25, atol=1e-6,
        distractors=[(0.6 / 3.6, "You left the volatility v out of the numerator and denominator: the gain uses the *predictive* variance w + v."),
                     (1.0 / 3.0, "You divided by the observation noise only; the denominator is w + v + σ²."),
                     (0.75, "That is 1 − k, the weight kept on the old prediction."),
                     (0.2, "You used w / (w + v + σ²) — the volatility must also appear in the numerator.")],
        hints=["Piray & Daw (2020), Eq. 9: k = (w + v) / (w + v + σ²).",
               "Here w + v = 1.0 and σ² = 3."],
        explanation="k = 1 / (1 + 3) = 0.25. The gain compares how uncertain the prediction is (w + v) with how noisy an "
                    "observation is (σ²): a quarter of the prediction error is incorporated. More volatility (larger v) "
                    "raises the gain; more noise (larger σ²) lowers it.",
        check_call='check("12.2", 0.25)',
    ),
    # ------------------------------------------------------------------ 12.3
    CodeExercise(
        id="12.3", title="The continuous volatile Kalman filter",
        reference=models.vkf_continuous,
        signature="vkf_continuous(outcomes, lam, v0, sigma2)",
        cases=[{"outcomes": _cont_outcomes, "lam": 0.1, "v0": 0.5, "sigma2": 1.0},
               {"outcomes": _cont_outcomes, "lam": 0.3, "v0": 0.1, "sigma2": 0.25},
               {"outcomes": [1.0, 0.0, 2.0, 1.0, 3.0], "lam": 0.2, "v0": 0.5, "sigma2": 1.0}],
        bug_variants={
            "no_vol_update": (_make_c("no_vol_update"), "Your volatility never changes — that is the ordinary Kalman filter. Add the "
                                                        "volatility update (Eq. 13) at the end of each trial."),
            "gain_without_v": (_make_c("gain_without_v"), "Your gain uses w alone; it must use the predictive variance w + v (Eq. 9)."),
            "var_without_v": (_make_c("var_without_v"), "The variance update must start from the *predictive* variance w_prev + v: "
                                                        "w = (1 − k)(w_prev + v) (Eq. 11)."),
            "no_cov_term": (_make_c("no_cov_term"), "The volatility error is missing the covariance term −2 w_cov with w_cov = (1 − k) w_prev (Eqs. 12–13)."),
            "pe_sign": (_make_c("pe_sign"), "Your prediction error has the wrong sign: it is outcome − prediction."),
            "record_after": (_make_c("record_after"), "You record m, v and w *after* the update; the arrays must hold the values before each "
                                                      "trial's outcome (the prediction the agent makes)."),
        },
        hints=["Keep three state variables (m, w, v) and, on each trial, first store them, then compute k = (w + v)/(w + v + σ²) "
               "and the prediction error o − m.",
               "Update in this order: m ← m + k·pe; w ← (1 − k)(w_prev + v); w_cov ← (1 − k) w_prev; "
               "v ← v + λ[(m − m_prev)² + w + w_prev − 2 w_cov − v]. Save m_prev and w_prev before overwriting.",
               "Look at the structure of `models.vkf_bin` if you have already solved 12.4 — the continuous version is the same loop with k as the learning rate and no sigmoid."],
        explanation="Equations 9–13 of Piray & Daw (2020): a Kalman filter whose process noise v is itself learned by a "
                    "delta rule. The volatility error compares the posterior expectation of the squared change in the "
                    "hidden state, (m_t − m_{t−1})² + w_t + w_{t−1} − 2 w_cov, with the current estimate v. Setting λ = 0 "
                    "recovers the standard Kalman filter with process noise v0.",
        common_mistakes="Forgetting v in the gain or in the variance update, dropping the covariance term, flipping the "
                        "prediction error, or storing the post-update values.",
        check_call='check("12.3", vkf_continuous)',
    ),
    # ------------------------------------------------------------------ 12.4
    CodeExercise(
        id="12.4", title="The binary volatile Kalman filter",
        reference=models.vkf_bin,
        signature="vkf_bin(outcomes, lam, v0, omega)",
        cases=[{"outcomes": _bin_outcomes, "lam": 0.1, "v0": 0.1, "omega": 0.1},
               {"outcomes": _bin_outcomes, "lam": 0.2, "v0": 0.5, "omega": 0.5},
               {"outcomes": [1, 1, 0, 1, 0, 0, 0, 1], "lam": 0.1, "v0": 0.2, "omega": 0.3}],
        bug_variants={
            "gain_as_rate": (_make_b("gain_as_rate"), "You updated m with the gain k. In the binary VKF the learning rate is "
                                                      "α = √(w + v) (Eq. 15); k is only used for the variance update."),
            "pe_no_sigmoid": (_make_b("pe_no_sigmoid"), "The prediction error must be computed in probability space: "
                                                        "o − sigmoid(m), not o − m (Eq. 14)."),
            "pred_no_sigmoid": (_make_b("pred_no_sigmoid"), "`pred` must be the predicted *probability* sigmoid(m), not the logit m."),
            "no_vol_update": (_make_b("no_vol_update"), "Your volatility never changes — add the volatility update (Eq. 19)."),
        },
        hints=["The loop is the continuous VKF's loop with three changes: the prediction error is o − sigmoid(m), the "
               "learning rate is α = √(w + v) instead of k, and the gain k = (w + v)/(w + v + ω) is used only for w.",
               "Store m, sigmoid(m), v and w before the update; after computing pe, k and α: m ← m + α·pe; "
               "w ← (1 − k)(w_prev + v); w_cov ← (1 − k) w_prev; v ← v + λ[(m − m_prev)² + w + w_prev − 2 w_cov − v].",
               "Initial values: m = 0, w = ω, v = v0 (as in the authors' code)."],
        explanation="Equations 14–19 of Piray & Daw (2020). The hidden state is a logit, so the prediction is sigmoid(m) and "
                    "the error lives in probability space. The learning rate √(w + v) grows with both the posterior "
                    "uncertainty w and the estimated volatility v, and shrinks as observations accumulate in a stable stretch.",
        common_mistakes="Using k as the learning rate, forgetting the sigmoid in the prediction error, or returning the "
                        "logit instead of the probability as `pred`.",
        check_call='check("12.4", vkf_bin)',
    ),
    # ------------------------------------------------------------------ 12.5
    NumericExercise(
        id="12.5", title="The first learning rate",
        answer=float(np.sqrt(0.3 + 0.1)), atol=1e-4,
        distractors=[(0.4, "That is w + v; the learning rate is its square root (Eq. 15)."),
                     (0.4 / 0.7, "That is the gain k = (w + v)/(w + v + ω); in the binary VKF the learning rate is √(w + v)."),
                     (float(np.sqrt(0.3)), "You forgot the initial volatility v0: before the first trial w = ω and v = v0."),
                     (float(np.sqrt(0.1)), "You forgot ω: the initial posterior variance w is set to ω.")],
        hints=["Before the first outcome, w = ω and v = v0.",
               "α_1 = √(w + v) = √(ω + v0)."],
        explanation="α_1 = √(0.3 + 0.1) ≈ 0.632. The very first update is large because both uncertainty terms are at their "
                    "initial values; as evidence accumulates w shrinks and, in a stable world, v shrinks too, so later "
                    "updates are smaller — unless surprises push v back up.",
        check_call='check("12.5", np.sqrt(0.3 + 0.1))',
    ),
    # ------------------------------------------------------------------ 12.6
    ChoiceExercise(
        id="12.6", title="Which parameter learns volatility?",
        question="Which VKF parameter determines how quickly the volatility estimate v responds to a run of surprising outcomes?",
        options={"a": "ω (or σ²), the noise parameter", "b": "v0, the initial volatility", "c": "λ, the volatility update rate", "d": "β, the inverse temperature"},
        correct="c",
        feedback={"a": "ω scales how informative each outcome is (through the gain k); it does not set the speed of the volatility update.",
                  "b": "v0 is only the starting value of v; the speed at which v moves is λ.",
                  "d": "β belongs to the choice rule, not to learning."},
        hints=["Look at the volatility update v ← v + λ(... − v): which symbol plays the role of a learning rate for v?",
               "It is the rate in front of the volatility prediction error."],
        explanation="λ is the learning rate *of the learning rate*: it sets how fast the estimate of volatility — and with it "
                    "α = √(w + v) — tracks recent evidence of change. Piray & Daw (2020) fit λ to individuals; lesson 14 "
                    "discusses how such parameters have been related to anxiety.",
        check_call='check("12.6", "c")',
    ),
    # ------------------------------------------------------------------ 12.7
    CodeExercise(
        id="12.7", title="Likelihood of choices under the VKF",
        reference=models.vkf_bin_nll,
        signature="vkf_bin_nll(params, outcomes, choices)",
        cases=[{"params": (0.1, 0.1, 0.1, 4.0), "outcomes": _bin_outcomes, "choices": _choices_a},
               {"params": (0.3, 0.5, 0.5, 2.0), "outcomes": _bin_outcomes, "choices": _choices_b},
               {"params": (0.2, 0.2, 0.2, 1.0), "outcomes": [1, 1, 0, 1, 0, 0], "choices": [0, 0, 0, 1, 1, 1]}],
        rtol=1e-6, atol=1e-6,
        bug_variants={
            "no_logistic": (_make_nll("no_logistic"), "You used the predicted probability directly as the choice probability. The choice rule is "
                                                      "P(choose 0) = sigmoid(β (2·pred − 1)): the inverse temperature must enter."),
            "no_centering": (_make_nll("no_centering"), "The logistic must be applied to β (2·pred − 1), i.e. the *difference* between the two "
                                                        "options' predicted probabilities (pred − (1 − pred)), not to β·pred."),
            "flipped_coding": (_make_nll("flipped_coding"), "The choice coding is flipped: p0 is the probability of choosing option 0 "
                                                            "(choices == 0), and 1 − p0 that of option 1."),
            "positive_ll": (_make_nll("positive_ll"), "You returned the log-likelihood; the function must return its *negative*."),
        },
        hints=["Run `models.vkf_bin(outcomes, lam, v0, omega)` (or your own) and take the `pred` array: the predicted probability that option 0 pays, before each trial.",
               "Turn it into choice probabilities: p0 = sigmoid(β (2·pred − 1)); the probability of the observed choice is p0 when the choice is 0 and 1 − p0 otherwise.",
               "Return −Σ log(probability of the observed choice)."],
        explanation="This is the replay logic of lesson 5 applied to the VKF: the learner's predictions are computed from the "
                    "observed outcomes (which, in a full-information task, do not depend on the choices), then each observed "
                    "choice is scored under the logistic choice rule. The prediction 2·pred − 1 is the difference between the "
                    "two options' predicted reward probabilities, so β plays its usual role.",
        common_mistakes="Omitting β, applying the logistic to pred instead of the difference 2·pred − 1, flipping the choice coding, or forgetting the minus sign.",
        check_call='check("12.7", vkf_bin_nll)',
    ),
    # ------------------------------------------------------------------ 12.8
    CodeExercise(
        id="12.8", title="Learning rates per block",
        reference=fit_block_learning_rates,
        signature="fit_block_learning_rates(outcomes, choices, block, rng)",
        cases=[{"outcomes": _bin_outcomes, "choices": _choices_rw, "block": _block, "rng": np.random.default_rng(8)}],
        compare=_blocks_compare,
        diagnose=_blocks_diagnose,
        hints=["Build a boolean mask for each block (`block == 'stable'`, `block == 'volatile'`) and select the outcomes and choices of that block.",
               "Fit the two-parameter RW reversal model to each block with `fitting.fit_mle(models.rw_reversal_nll, (outcomes[mask], choices[mask]), ['unit', 'pos'], n_starts=3, rng=rng)`.",
               "The learning rate is `fit['params'][0]`; return the stable block's first, then the volatile block's."],
        explanation="Fitting the same model separately to each block turns 'does the learning rate adapt?' into a measurable "
                    "quantity per participant — the analysis used by Behrens et al. (2007) and Browning et al. (2015). The "
                    "checker accepts your answer within 0.03 of the reference because different random restarts can land on "
                    "slightly different optima.",
        common_mistakes="Fitting the whole sequence instead of each block, returning the inverse temperature, or swapping the order of the two blocks.",
        check_call='check("12.8", fit_block_learning_rates)',
    ),
    # ------------------------------------------------------------------ 12.9
    ChoiceExercise(
        id="12.9", title="How the VKF's learning rate behaves",
        question="Which statement about the binary VKF's learning rate α_t = √(w_t + v_t) is correct?",
        options={"a": "It is constant across trials once the filter has been initialised.",
                 "b": "It rises when the volatility estimate v rises after a run of surprising outcomes, and falls as the posterior variance w shrinks during a stable run.",
                 "c": "It depends only on the most recent prediction error.",
                 "d": "It is equal to the volatility update rate λ."},
        correct="b",
        feedback={"a": "Both w and v change on every trial, so α does too — that is the whole point of the model.",
                  "c": "The most recent error enters only indirectly, through the volatility update; α summarises *accumulated* uncertainty and volatility.",
                  "d": "λ sets how fast v changes; it is not the learning rate itself."},
        hints=["α_t is built from two uncertainty terms: which events make w shrink, and which make v grow?",
               "w shrinks as outcomes accumulate; v grows when the estimated hidden state has been moving a lot."],
        explanation="α = √(w + v) couples the learning rate to two kinds of uncertainty: the posterior variance w (which falls "
                    "as evidence accumulates) and the estimated volatility v (which rises when the hidden state appears to "
                    "move). This is the mechanism by which a Bayesian learner reproduces the higher learning rates of volatile "
                    "environments that Behrens et al. (2007) measured in people.",
        check_call='check("12.9", "b")',
    ),
    # ------------------------------------------------------------------ 12.10
    FreeTextExercise(
        id="12.10", title="Noise versus volatility",
        question="In 3–5 sentences: why do outcome noise and volatility have *opposite* effects on the learning rate an ideal learner should use?",
        model_answer="Outcome noise makes each individual observation less informative about the hidden quantity: a surprising "
                     "outcome is probably just an unlucky draw, so the learner should average over many outcomes and use a *low* "
                     "learning rate. Volatility means the hidden quantity itself changes, so old observations become outdated and "
                     "the learner should weight recent outcomes more heavily — a *high* learning rate. In the Kalman gain "
                     "k = (w + v)/(w + v + σ²) this shows up directly: volatility v enters the numerator (raising the gain) while "
                     "noise σ² enters only the denominator (lowering it). The right learning rate is therefore a ratio of how "
                     "much the world changes to how noisy it is, which is why a learner must estimate both and why confusing the "
                     "two (treating noise as change, or change as noise) produces systematically wrong learning rates.",
        concepts=[("noise makes single outcomes less informative → lower learning rate",
                   ["noise", "noisy", "unlucky", "less informative", "uninformative", "average", "averag", "lower learning rate",
                    "smaller learning rate", "slow", "discount", "unreliable"]),
                  ("volatility makes old evidence outdated → higher learning rate",
                   ["volatil", "change", "outdated", "obsolete", "stale", "recent", "higher learning rate", "larger learning rate",
                    "faster", "track", "adapt"]),
                  ("the learning rate reflects the ratio / balance of the two",
                   ["ratio", "relative", "balance", "trade", "numerator", "denominator", "gain", "k =", "proportion", "compare"])],
        min_concepts=2,
        hints=["Think about what a surprising outcome *means* in each case: an unlucky draw, or a sign that the world has changed?",
               "Then look at where v and σ² sit in the Kalman gain k = (w + v)/(w + v + σ²)."],
        explanation="Noise argues for averaging (low learning rate); volatility argues for forgetting (high learning rate). An "
                    "ideal learner sets its learning rate by the balance of the two — which is exactly what the gain of the "
                    "(volatile) Kalman filter computes, and what Pulcu & Browning (2019) argue affective disorders get wrong.",
        check_call='check("12.10", """Outcome noise makes each individual observation less informative about the hidden quantity: a surprising outcome is probably just an unlucky draw, so the learner should average over many outcomes and use a lower learning rate. Volatility means the hidden quantity itself changes, so old observations become outdated and the learner should weight recent outcomes more heavily, i.e. use a higher learning rate. In the Kalman gain the volatility enters the numerator while the noise enters only the denominator, so the right learning rate is the ratio of how much the world changes to how noisy it is.""")',
    ),
]
