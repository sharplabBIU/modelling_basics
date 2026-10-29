"""Exercises of lesson 3: Rescorla–Wagner learning and softmax choice."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# helpers for the simulation exercise (3.8): compare by *consistency*, not by
# matching random streams
# --------------------------------------------------------------------------- #


def _sim_compare(out, ref, case):
    if not isinstance(out, dict):
        return False
    if not models.simulation_is_consistent(out, case["probs"], case["alpha"], case["beta"], case.get("q0", 0.5)):
        return False
    choices = np.asarray(out["choices"])
    best = int(np.argmax(case["probs"]))
    # with a decent beta and alpha the best arm must be chosen clearly above chance
    return np.mean(choices[len(choices) // 2:] == best) > 1.0 / len(case["probs"]) + 0.15


def _sim_diagnose(out, ref, case):
    if not isinstance(out, dict):
        return f"Return a dictionary with keys 'choices', 'rewards' and 'Q' (you returned `{type(out).__name__}`)."
    for key in ("choices", "rewards", "Q"):
        if key not in out:
            return f"The returned dictionary has no key {key!r}."
    probs, alpha, beta, q0 = case["probs"], case["alpha"], case["beta"], case.get("q0", 0.5)
    n_trials, n_arms = case["n_trials"], len(probs)
    choices, rewards, Q = np.asarray(out["choices"]), np.asarray(out["rewards"], float), np.asarray(out["Q"], float)
    if Q.shape != (n_trials, n_arms):
        return f"`Q` should have shape (n_trials, n_arms) = {(n_trials, n_arms)}, but has shape {Q.shape}."
    if len(choices) != n_trials or len(rewards) != n_trials:
        return "`choices` and `rewards` should both have one entry per trial."
    if not np.allclose(Q[0], q0):
        return "`Q[0]` should contain the *initial* values (q0) — record the values *before* the choice and update on each trial."
    if not set(np.unique(rewards)).issubset({0.0, 1.0}):
        return "Rewards should be 0 or 1 (`float(rng.random() < probs[c])`)."
    if choices.min() < 0 or choices.max() >= n_arms:
        return "Choices should be integers between 0 and n_arms - 1."
    for t in range(n_trials - 1):
        expected = Q[t].copy()
        expected[choices[t]] = models.rw_update(Q[t, choices[t]], rewards[t], alpha)
        if not np.allclose(Q[t + 1], expected):
            unchosen_changed = any(not np.isclose(Q[t + 1, a], Q[t, a]) for a in range(n_arms) if a != choices[t])
            if unchosen_changed:
                return "The values of *unchosen* arms changed between trials: only the chosen arm should be updated."
            if t + 2 < n_trials and np.allclose(Q[t + 1], Q[t]) and not np.allclose(Q[t + 2], Q[t + 1]):
                return "The recorded `Q` lags behind the updates — are you recording the values before the update on the *previous* trial twice?"
            shifted = Q[t].copy()
            shifted[choices[t]] = models.rw_update(Q[t, choices[t]], rewards[t], alpha)
            if t + 1 < n_trials and np.allclose(Q[t], shifted) is False and np.allclose(Q[t + 1], Q[t]):
                return "`Q[t+1]` equals `Q[t]`: the update is not being stored (did you update a copy instead of `Q[c]`?)"
            return (f"On trial {t} the chosen arm's value moves from {Q[t, choices[t]]:.3f} to {Q[t + 1, choices[t]]:.3f}, "
                    f"but the RW rule with alpha={alpha} and reward={rewards[t]:.0f} gives {expected[choices[t]]:.3f}. "
                    "Check the update and that `Q[t]` stores the values *before* the choice.")
    best = int(np.argmax(probs))
    if np.mean(choices[n_trials // 2:] == best) <= 1.0 / n_arms + 0.15:
        return ("Learning is consistent, but the agent does not prefer the best arm — check that the softmax uses the "
                "current values (`softmax(Q, beta)`) and that `choose` receives the probabilities.")
    return None


def _eps_greedy_others_only(values, epsilon):
    values = np.asarray(values, float)
    best = values == values.max()
    k = len(values)
    probs = np.where(best, (1 - epsilon) / best.sum(), epsilon / max(k - best.sum(), 1))
    return probs


def _eps_greedy_no_ties(values, epsilon):
    values = np.asarray(values, float)
    probs = np.full(len(values), epsilon / len(values))
    probs[int(np.argmax(values))] += 1 - epsilon
    return probs


EXERCISES = [
    # ------------------------------------------------------------------ 3.1
    CodeExercise(
        id="3.1", title="The Rescorla–Wagner update",
        reference=models.rw_update,
        signature="rw_update(value, reward, alpha)",
        cases=[{"value": 0.5, "reward": 1.0, "alpha": 0.1},
               {"value": 0.2, "reward": 0.0, "alpha": 0.5},
               {"value": 0.0, "reward": 1.0, "alpha": 1.0},
               {"value": 0.8, "reward": 0.0, "alpha": 0.0},
               {"value": np.array([0.2, 0.6]), "reward": 1.0, "alpha": 0.3}],
        bug_variants={
            "raw_reward": (lambda value, reward, alpha: value + alpha * reward,
                           "You added alpha × reward to the value. The update must use the *prediction error* "
                           "(reward − value), otherwise the value grows without bound."),
            "flipped_pe": (lambda value, reward, alpha: value + alpha * (value - reward),
                           "Your prediction error has the wrong sign (value − reward). It should be reward − value: "
                           "a reward *larger* than expected must *increase* the value."),
            "no_old_value": (lambda value, reward, alpha: alpha * (reward - value),
                             "You returned only the change (alpha × error). The new value is the *old value plus* the change."),
            "swapped": (lambda value, reward, alpha: (1 - alpha) * reward + alpha * value,
                        "You swapped the roles of alpha and 1 − alpha: the new value is (1 − alpha) × old value + alpha × reward."),
        },
        hints=["The prediction error is the difference between what happened and what was expected: reward − value.",
               "Move the value a fraction `alpha` of the way from where it is towards the reward: new = old + alpha × error.",
               "`prediction_error = reward - value` and `new_value = value + alpha * prediction_error`."],
        explanation="The value moves a fraction alpha of the way towards the outcome. Equivalently, the new value is a "
                    "weighted average of the old value (weight 1 − alpha) and the new reward (weight alpha). The "
                    "prediction error reward − value is the quantity that dopamine neurons appear to signal (lesson 4).",
        common_mistakes="Using the raw reward instead of the prediction error; flipping the sign of the error; "
                        "returning the change rather than the updated value.",
        check_call='check("3.1", rw_update)',
    ),
    # ------------------------------------------------------------------ 3.2
    NumericExercise(
        id="3.2", title="Two updates by hand",
        answer=0.48, atol=1e-3,
        distractors=[(0.6, "That is the value after the *first* trial only. Apply the rule a second time with reward 0."),
                     (0.4, "You applied the second update to the initial value 0.5 instead of to the updated value 0.6."),
                     (0.7, "It looks like you added alpha × reward instead of alpha × (reward − value)."),
                     (0.5, "The value did not change — remember to apply both updates.")],
        hints=["Apply the rule once with reward 1 (the value goes up), then again with reward 0 (it goes down).",
               "Trial 1: 0.5 + 0.2 × (1 − 0.5) = 0.6. Now use 0.6 as the starting value for trial 2.",
               "Trial 2: 0.6 + 0.2 × (0 − 0.6) = ?"],
        explanation="0.5 → 0.6 after the reward, then 0.6 + 0.2 × (0 − 0.6) = 0.48 after the non-reward. Notice that "
                    "the drop (0.12) is larger than the rise (0.10) because the second surprise was larger "
                    "(|0 − 0.6| = 0.6 versus |1 − 0.5| = 0.5): the size of an update scales with the size of the surprise.",
        check_call='check("3.2", 0.48)',
    ),
    # ------------------------------------------------------------------ 3.3
    ChoiceExercise(
        id="3.3", title="Extreme learning rates",
        question="With a learning rate of alpha = 1, the value after each trial equals ...",
        options={"a": "the average of all rewards received so far",
                 "b": "the most recent reward",
                 "c": "the initial value, forever",
                 "d": "the sum of all rewards received so far"},
        correct="b",
        feedback={"a": "An average of *all* past rewards would need a learning rate that shrinks over trials (1/t), "
                       "not a constant 1 — see section 3.3.",
                  "c": "That is what happens with alpha = 0: nothing is ever learned.",
                  "d": "The RW rule never sums rewards; it moves the value *towards* the reward."},
        hints=["Plug alpha = 1 into V + alpha × (R − V) and simplify.",
               "V + 1 × (R − V) = R: the old value cancels out completely."],
        explanation="With alpha = 1 the update V + (R − V) = R throws away all history: the value is just the last "
                    "reward. With alpha = 0 nothing changes. Intermediate values trade off memory and flexibility.",
        check_call='check("3.3", "b")',
    ),
    # ------------------------------------------------------------------ 3.4
    NumericExercise(
        id="3.4", title="The weight of an old outcome",
        answer=0.3 * 0.7 ** 3, atol=1e-4,
        distractors=[(0.3 * 0.7 ** 2, "Off by one: the most recent outcome has weight alpha × (1 − alpha)^0, the one "
                                      "before it alpha × (1 − alpha)^1, so the outcome 3 trials earlier has exponent 3."),
                     (0.7 ** 3, "You forgot the factor alpha in front: every outcome's weight is alpha × (1 − alpha)^k."),
                     (0.3 ** 3, "The decay factor is (1 − alpha), not alpha."),
                     (0.3 * 0.7 ** 4, "Off by one in the other direction: 3 trials before the most recent means exponent 3.")],
        hints=["Unrolling the RW recursion gives V_t = sum_k alpha (1 − alpha)^k R_(t−k) (plus a vanishing term for the initial value).",
               "The outcome k trials before the most recent one has weight alpha × (1 − alpha)^k; here k = 3 and alpha = 0.3.",
               "0.3 × 0.7^3"],
        explanation="RW is an exponentially weighted moving average: 0.3 × 0.7³ ≈ 0.103. The weights decay "
                    "geometrically, so roughly the last 1/alpha trials matter — an agent with alpha = 0.3 effectively "
                    "remembers the last ~3 outcomes.",
        check_call='check("3.4", 0.3 * 0.7**3)',
    ),
    # ------------------------------------------------------------------ 3.5
    CodeExercise(
        id="3.5", title="A numerically stable softmax",
        reference=models.softmax,
        signature="softmax(values, beta=1.0)",
        cases=[{"values": [0.6, 0.4], "beta": 5.0},
               {"values": [1.0, 2.0, 3.0], "beta": 1.0},
               {"values": [0.0, 0.0], "beta": 2.0},
               {"values": np.array([0.5, 0.5, 0.5, 0.5]), "beta": 10.0},
               {"values": [1000.0, 1001.0], "beta": 1.0}],
        bug_variants={
            "no_beta": (lambda values, beta=1.0: np.exp(np.asarray(values, float)) / np.sum(np.exp(np.asarray(values, float))),
                        "Your function ignores beta: the values must be multiplied by beta *before* exponentiating."),
            "unstable": (lambda values, beta=1.0: np.exp(beta * np.asarray(values, float)) / np.sum(np.exp(beta * np.asarray(values, float))),
                         "Your softmax is mathematically right but overflows for large inputs (exp(1000) is inf, and inf/inf "
                         "is nan). Subtract the maximum of beta × values before exponentiating — it cancels in the ratio."),
            "normalise_values": (lambda values, beta=1.0: np.asarray(values, float) / np.sum(values),
                                 "You divided the values by their sum. The softmax exponentiates beta × values first."),
            "unnormalised": (lambda values, beta=1.0: np.exp(beta * (np.asarray(values, float) - np.max(values))),
                             "You forgot to divide by the sum: probabilities must add up to 1."),
            "negative_beta": (lambda values, beta=1.0: models.softmax(values, -beta),
                              "Your probabilities favour the *lowest* value — check the sign of beta."),
        },
        hints=["p_i = exp(beta × v_i) / sum_j exp(beta × v_j). Compute the numerator for every option as an array, then divide by its sum.",
               "To avoid overflow, subtract max(beta × values) from every element before `np.exp`: the ratio is unchanged.",
               "`x = beta * np.asarray(values, float); x = x - np.max(x); e = np.exp(x); return e / e.sum()`"],
        explanation="Subtracting the maximum before exponentiating does not change the ratio (it multiplies numerator "
                    "and denominator by the same constant) but keeps every exponent ≤ 0, so nothing overflows. Always "
                    "write softmax this way; with large beta or large values the naive version returns NaN.",
        common_mistakes="Forgetting beta, forgetting to normalise, or the naive exp without subtracting the maximum.",
        check_call='check("3.5", softmax)',
    ),
    # ------------------------------------------------------------------ 3.6
    NumericExercise(
        id="3.6", title="Softmax by hand",
        answer=float(models.sigmoid(1.0)), atol=1e-3,
        distractors=[(0.6, "That is the *value* of option A, not its choice probability."),
                     (float(models.sigmoid(-1.0)), "That is the probability of choosing B (or you flipped a sign)."),
                     (float(models.sigmoid(0.2)), "It looks like beta was left out: the exponent is beta × (Q_A − Q_B) = 5 × 0.2 = 1."),
                     (0.5, "With beta = 5 and different values the choice is not 50/50 — check your exponent.")],
        hints=["For two options the softmax reduces to a logistic function of the value difference: P(A) = 1 / (1 + exp(−beta (Q_A − Q_B))).",
               "Here beta (Q_A − Q_B) = 5 × (0.6 − 0.4) = 1.",
               "1 / (1 + exp(−1)) ≈ 0.731"],
        explanation="P(A) = 1 / (1 + e^−1) ≈ 0.731. A value difference of 0.2 at beta = 5 corresponds to one 'unit' "
                    "of decision evidence, which maps to ~73%: the inverse temperature sets the exchange rate between "
                    "value differences and choice probabilities.",
        check_call='check("3.6", 1 / (1 + np.exp(-1.0)))',
    ),
    # ------------------------------------------------------------------ 3.7
    ChoiceExercise(
        id="3.7", title="What the inverse temperature does",
        question="Increasing beta (keeping the values the same) makes the agent ...",
        options={"a": "choose more randomly, exploring more",
                 "b": "choose the higher-valued option more consistently (exploit more)",
                 "c": "learn faster from each reward",
                 "d": "assign higher values to every option"},
        correct="b",
        feedback={"a": "It is the other way round: a *small* beta makes choices random (beta = 0 gives uniform probabilities).",
                  "c": "beta affects only the choice rule, not the learning rule — learning speed is set by alpha.",
                  "d": "beta rescales the values inside the softmax but does not change the values themselves."},
        hints=["Look at the two-option formula P(A) = 1 / (1 + exp(−beta × ΔQ)): what happens as beta → ∞ and as beta → 0?",
               "Large beta: even a tiny value difference produces a near-certain choice. Small beta: 50/50 regardless."],
        explanation="beta is often called the exploitation parameter: larger beta means more deterministic (greedy) "
                    "choices, smaller beta more random ones. Note that beta multiplies the values, so a change in "
                    "the scale of rewards is indistinguishable from a change in beta — keep that in mind when "
                    "interpreting fitted values (lesson 14).",
        check_call='check("3.7", "b")',
    ),
    # ------------------------------------------------------------------ 3.8
    CodeExercise(
        id="3.8", title="Simulate a complete RW + softmax agent",
        reference=models.simulate_rw_bandit,
        signature="simulate_rw_bandit(probs, alpha, beta, n_trials, rng, q0=0.5)",
        cases=[{"probs": [0.8, 0.2], "alpha": 0.3, "beta": 5.0, "n_trials": 300, "rng": np.random.default_rng(1)},
               {"probs": [0.3, 0.5, 0.7], "alpha": 0.1, "beta": 4.0, "n_trials": 400, "rng": np.random.default_rng(2), "q0": 0.0}],
        compare=_sim_compare,
        diagnose=_sim_diagnose,
        hints=["Inside the trial loop: (1) store the current values in `Q_history[t]`, (2) compute choice probabilities "
               "with `softmax(Q, beta)`, (3) sample a choice with `choose(p, rng)`, (4) sample a reward with "
               "`float(rng.random() < probs[c])`, (5) update only `Q[c]` with `rw_update`.",
               "Record `Q_history[t] = Q` *before* the update so that `Q[0]` holds the initial values; use `Q_history[t] = Q` "
               "(NumPy copies the row on assignment) or `Q.copy()`.",
               "The checker does not require the same random numbers as the reference: it verifies that your `Q` "
               "trajectory is consistent with your own choices and rewards, and that the agent prefers the best arm."],
        explanation="This loop is the template for every simulation in the course: observe → choose → outcome → learn. "
                    "Recording the values *before* each choice is the convention we use everywhere, because those are "
                    "the values that determined the choice (and the values a likelihood function needs, lesson 5).",
        common_mistakes="Recording Q after the update (so Q[0] is not q0); updating all arms instead of the chosen one; "
                        "forgetting to convert the reward to a float; returning the final Q instead of its history.",
        check_call='check("3.8", simulate_rw_bandit)',
    ),
    # ------------------------------------------------------------------ 3.9
    CodeExercise(
        id="3.9", title="Epsilon-greedy choice probabilities",
        reference=models.epsilon_greedy_probs,
        signature="epsilon_greedy_probs(values, epsilon)",
        cases=[{"values": [0.6, 0.4], "epsilon": 0.1},
               {"values": [0.5, 0.5], "epsilon": 0.2},
               {"values": [1.0, 2.0, 3.0], "epsilon": 0.3},
               {"values": np.array([0.2, 0.9, 0.9, 0.1]), "epsilon": 0.4}],
        bug_variants={
            "others_only": (_eps_greedy_others_only,
                            "You gave the greedy option exactly 1 − epsilon. In the definition used here the random choice is made "
                            "among *all* options, so the greedy option also receives its share epsilon / k."),
            "no_ties": (_eps_greedy_no_ties,
                        "Ties: when several options share the highest value they must *split* the greedy probability 1 − epsilon equally."),
        },
        hints=["Start with the random part: every option gets epsilon / k. Then add the greedy part 1 − epsilon to the best option(s).",
               "Find the best options with `values == values.max()` (a boolean array); if there are several, divide (1 − epsilon) among them.",
               "`probs = np.full(k, epsilon / k); best = values == values.max(); probs[best] += (1 - epsilon) / best.sum()`"],
        explanation="Epsilon-greedy explores *uniformly* (every non-best option is equally likely) whereas the softmax "
                    "explores *in proportion to value* (a nearly-as-good option is chosen more often than a bad one). "
                    "Both give the agent a chance to discover that its current favourite is not the best.",
        common_mistakes="Giving the greedy option 1 − epsilon instead of 1 − epsilon + epsilon/k; not handling ties.",
        check_call='check("3.9", epsilon_greedy_probs)',
    ),
    # ------------------------------------------------------------------ 3.10
    FreeTextExercise(
        id="3.10", title="Stability versus flexibility",
        question="In 2–4 sentences: why does a high learning rate hurt in a stable but noisy environment, and why does it help right after a reversal?",
        model_answer="A high learning rate makes the value track the most recent outcomes closely. In a stable but noisy "
                     "environment the recent outcomes are mostly noise (a single unrewarded trial from a 70% option is "
                     "not evidence that it has become bad), so a high-alpha learner's values jump around and it "
                     "sometimes switches to the worse option; a low alpha averages over many trials and smooths the "
                     "noise away. After a reversal the old average is wrong, and the same quick tracking lets a "
                     "high-alpha learner adapt within a few trials, whereas a low-alpha learner keeps choosing the "
                     "previously good option for a long time. The best learning rate therefore depends on how often "
                     "the world changes relative to how noisy it is — which is exactly what the Bayesian learners of "
                     "lessons 11 and 12 estimate.",
        concepts=[("noise / single outcomes", ["noise", "noisy", "fluctuat", "random", "single", "one trial", "last reward",
                                               "recent", "jump", "overreact", "over-react", "chance"]),
                  ("averaging / stability", ["average", "averag", "stable", "smooth", "many trials", "history", "integrat",
                                             "accumulat", "memory", "slow"]),
                  ("adaptation after change", ["adapt", "flexib", "quick", "fast", "revers", "change", "track", "switch",
                                               "catch up", "update"])],
        min_concepts=2,
        hints=["Think about what a high alpha does to the value after one unlucky outcome from a good option.",
               "Then think about how many trials a low-alpha learner needs after a reversal to change its mind."],
        explanation="The learning rate sets a speed–accuracy trade-off between averaging out noise and tracking "
                    "change. There is no universally good value: it depends on the statistics of the environment.",
        check_call='check("3.10", """A high learning rate makes the value track the most recent outcomes closely. In a stable but noisy environment the recent outcomes are mostly noise, so the values jump around and the agent sometimes switches to the worse option; a low learning rate averages over many trials and smooths the noise away. After a reversal the old average is wrong, and quick tracking lets the agent adapt within a few trials, whereas a low learning rate keeps choosing the previously good option for a long time.""")',
    ),
]
