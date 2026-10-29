"""Exercises of lesson 1: tasks, agents and simulation."""
from __future__ import annotations

import numpy as np

from .. import models
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# Reference implementations specific to this lesson (their source is revealed
# as the solution, so they are written the way a student would write them).
# --------------------------------------------------------------------------- #


def simulate_random_agent(probs, n_trials, rng):
    """An agent that chooses every option with equal probability.

    Returns dict(choices=int array, rewards=float array)."""
    probs = np.asarray(probs, dtype=float)
    n_arms = len(probs)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    for t in range(n_trials):
        c = int(rng.integers(n_arms))               # every option is equally likely
        r = float(rng.random() < probs[c])          # Bernoulli reward of the chosen option
        choices[t], rewards[t] = c, r
    return {"choices": choices, "rewards": rewards}


def simulate_wsls(probs, n_trials, rng):
    """Win-stay / lose-shift: repeat the last choice after a reward, switch to
    another option after no reward (the first choice is random).

    Returns dict(choices=int array, rewards=float array)."""
    probs = np.asarray(probs, dtype=float)
    n_arms = len(probs)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    for t in range(n_trials):
        if t == 0:
            c = int(rng.integers(n_arms))                      # no history yet: pick at random
        elif rewards[t - 1] == 1:
            c = choices[t - 1]                                 # win  -> stay
        else:
            others = [a for a in range(n_arms) if a != choices[t - 1]]
            c = int(rng.choice(others))                        # lose -> shift to another option
        r = float(rng.random() < probs[c])
        choices[t], rewards[t] = c, r
    return {"choices": choices, "rewards": rewards}


def stay_probabilities_bandit(choices, rewards):
    """P(repeat the previous choice | previous trial rewarded) and
    P(repeat | previous trial not rewarded).  Returns a tuple (p_stay_win, p_stay_lose)."""
    choices = np.asarray(choices)
    rewards = np.asarray(rewards, dtype=float)
    stay = choices[1:] == choices[:-1]          # did trial t repeat the choice of trial t-1?
    previous_rewarded = rewards[:-1] == 1       # was trial t-1 rewarded?
    p_stay_win = np.mean(stay[previous_rewarded])
    p_stay_lose = np.mean(stay[~previous_rewarded])
    return float(p_stay_win), float(p_stay_lose)


# --------------------------------------------------------------------------- #
# Buggy variants (used only for diagnosis)
# --------------------------------------------------------------------------- #


def _stay_current_reward(choices, rewards):
    choices = np.asarray(choices); rewards = np.asarray(rewards, float)
    stay = choices[1:] == choices[:-1]
    rewarded = rewards[1:] == 1
    return float(np.mean(stay[rewarded])), float(np.mean(stay[~rewarded]))


def _stay_swapped(choices, rewards):
    win, lose = stay_probabilities_bandit(choices, rewards)
    return lose, win


def _stay_counts(choices, rewards):
    choices = np.asarray(choices); rewards = np.asarray(rewards, float)
    stay = choices[1:] == choices[:-1]
    previous_rewarded = rewards[:-1] == 1
    return float(np.sum(stay[previous_rewarded])), float(np.sum(stay[~previous_rewarded]))


def _stay_unconditional(choices, rewards):
    choices = np.asarray(choices)
    p = float(np.mean(choices[1:] == choices[:-1]))
    return p, p


# --------------------------------------------------------------------------- #
# Comparison helpers for the stochastic simulators: check the *structure* and
# the *statistics* of the output, never the exact random numbers.
# --------------------------------------------------------------------------- #


def _structure_problem(out, case):
    if not isinstance(out, dict):
        return f"Return a dictionary with keys 'choices' and 'rewards' (you returned `{type(out).__name__}`)."
    for key in ("choices", "rewards"):
        if key not in out:
            return f"The returned dictionary has no key {key!r}."
    n_trials, n_arms = case["n_trials"], len(case["probs"])
    choices = np.asarray(out["choices"])
    rewards = np.asarray(out["rewards"], dtype=float)
    if len(choices) != n_trials or len(rewards) != n_trials:
        return "`choices` and `rewards` should both have exactly one entry per trial."
    if not np.all(np.mod(choices, 1) == 0):
        return "`choices` should be integers (the 0-based index of the chosen option)."
    if choices.min() < 0 or choices.max() >= n_arms:
        return f"Choices should be integers between 0 and n_arms - 1 = {n_arms - 1}."
    if not set(np.unique(rewards)).issubset({0.0, 1.0}):
        return "Rewards should be 0 or 1 (use `float(rng.random() < probs[c])`)."
    return None


def _per_arm_problem(out, case, min_count=40, tol=0.1):
    probs = np.asarray(case["probs"], float)
    choices = np.asarray(out["choices"]).astype(int)
    rewards = np.asarray(out["rewards"], float)
    for a in range(len(probs)):
        mask = choices == a
        if mask.sum() >= min_count and abs(rewards[mask].mean() - probs[a]) > tol:
            return (f"The rewards do not follow the task: option {a} was rewarded on "
                    f"{rewards[mask].mean():.0%} of the trials on which it was chosen, but its reward "
                    f"probability is {probs[a]}. Draw the reward of the *chosen* option: "
                    "`float(rng.random() < probs[c])`.")
    return None


def _random_compare(out, ref, case):
    if _structure_problem(out, case) or _per_arm_problem(out, case):
        return False
    choices = np.asarray(out["choices"]).astype(int)
    n_arms = len(case["probs"])
    freqs = np.bincount(choices, minlength=n_arms) / len(choices)
    return bool(np.all(np.abs(freqs - 1.0 / n_arms) < 0.07))


def _random_diagnose(out, ref, case):
    msg = _structure_problem(out, case)
    if msg:
        return msg
    msg = _per_arm_problem(out, case)
    if msg:
        return msg
    choices = np.asarray(out["choices"]).astype(int)
    n_arms = len(case["probs"])
    freqs = np.bincount(choices, minlength=n_arms) / len(choices)
    return (f"A random agent should choose every option equally often, but your choice frequencies are "
            f"{np.round(freqs, 2)}. Use `rng.integers(n_arms)` (or `models.choose` with equal probabilities).")


def _wsls_violation(out):
    choices = np.asarray(out["choices"]).astype(int)
    rewards = np.asarray(out["rewards"], float)
    for t in range(1, len(choices)):
        if rewards[t - 1] == 1 and choices[t] != choices[t - 1]:
            return (f"On trial {t} the previous choice was rewarded, but your agent switched "
                    f"(from {choices[t - 1]} to {choices[t]}). After a reward the agent must *stay*.")
        if rewards[t - 1] == 0 and choices[t] == choices[t - 1]:
            return (f"On trial {t} the previous choice was not rewarded, but your agent repeated it. "
                    "After a non-reward the agent must *shift* to a different option.")
    return None


def _wsls_compare(out, ref, case):
    if _structure_problem(out, case) or _per_arm_problem(out, case):
        return False
    return _wsls_violation(out) is None


def _wsls_diagnose(out, ref, case):
    msg = _structure_problem(out, case)
    if msg:
        return msg
    msg = _wsls_violation(out)
    if msg:
        return msg + " (Hint: decide the choice from `rewards[t - 1]` and `choices[t - 1]`, which are already stored.)"
    return _per_arm_problem(out, case)


# --------------------------------------------------------------------------- #
EXERCISES = [
    # ------------------------------------------------------------------ 1.1
    ChoiceExercise(
        id="1.1", title="Task variable, parameter or data?",
        question="Which of the following is a *parameter* of a model of a participant?",
        options={"a": "the reward probability of option A",
                 "b": "the number of trials in the experiment",
                 "c": "the participant's learning rate",
                 "d": "the sequence of rewards the participant received"},
        correct="c",
        feedback={"a": "The reward probability belongs to the *task*: the experimenter sets it and knows it. "
                       "A parameter is a property of the agent that we do not know and must estimate.",
                  "b": "The number of trials is a design choice of the experimenter, not a property of the agent.",
                  "d": "The rewards are *data* — observations. Parameters are the unknown numbers inside the "
                       "agent that we infer from the data."},
        hints=["A parameter is a number inside the *agent* that the hypothesis leaves free and that may differ "
               "between people.",
               "Which of the four options is unknown to the experimenter and has to be estimated from behaviour?"],
        explanation="Task variables (reward probabilities, trial numbers) are set by the experimenter; data "
                    "(choices, rewards) are observed; parameters (learning rate, inverse temperature) are "
                    "properties of the agent that the model leaves free and that we estimate from the data. "
                    "Keeping these three categories apart is the first step in any modelling project.",
        check_call='check("1.1", "c")',
    ),
    # ------------------------------------------------------------------ 1.2
    CodeExercise(
        id="1.2", title="The running mean",
        reference=models.running_mean,
        signature="running_mean(x)",
        cases=[{"x": [1, 0, 1, 1]}, {"x": [0, 0, 0]}, {"x": np.arange(5.0)},
               {"x": (np.random.default_rng(3).random(50) < 0.7).astype(float)}],
        bug_variants={
            "cumsum": (lambda x: np.cumsum(np.asarray(x, float)),
                       "You returned the cumulative *sum*; divide each entry by the number of outcomes seen so far."),
            "total_n": (lambda x: np.cumsum(np.asarray(x, float)) / len(x),
                        "You divided every cumulative sum by the *total* number of outcomes; entry t must be "
                        "divided by t + 1 (the number of outcomes seen up to and including trial t)."),
            "overall_mean": (lambda x: float(np.mean(x)),
                             "You returned a single number (the overall mean); the running mean is an array with "
                             "one estimate per trial."),
        },
        hints=["The estimate after t outcomes is the sum of the first t outcomes divided by t.",
               "`np.cumsum(x)` gives all the partial sums at once; the number of outcomes seen so far is "
               "`np.arange(1, len(x) + 1)`.",
               "`return np.cumsum(x) / np.arange(1, len(x) + 1)`"],
        explanation="Dividing the partial sums by the number of outcomes seen gives the estimate an observer "
                    "would hold after each trial. Because the outcomes are 0/1, this is also the observer's "
                    "best guess of the reward probability p — the law of large numbers says it converges to p, "
                    "but the plots show how slowly.",
        common_mistakes="Dividing by the total number of trials instead of the number seen so far; returning the "
                        "cumulative sum; returning a scalar.",
        check_call='check("1.2", running_mean)',
    ),
    # ------------------------------------------------------------------ 1.3
    NumericExercise(
        id="1.3", title="Expected reward of a random chooser",
        answer=0.5, atol=1e-6,
        distractors=[(0.7, "That is the reward rate of an agent that *always* chooses the best option (the oracle)."),
                     (1.0, "You added the two probabilities. An expectation *weights* each option's probability by "
                           "how often it is chosen (0.5 each) — the weights sum to 1."),
                     (0.21, "You multiplied the two probabilities; that would be the probability that *both* "
                            "options pay off on the same trial, not the expected reward."),
                     (0.35, "That is 0.5 × 0.7: you forgot to add the contribution of option B (0.5 × 0.3)."),
                     (0.4, "That is the difference between the two probabilities, not the expected reward.")],
        hints=["On each trial the agent picks A with probability 0.5 (expected reward 0.7) or B with probability "
               "0.5 (expected reward 0.3).",
               "Expected reward = P(choose A) × P(reward | A) + P(choose B) × P(reward | B).",
               "0.5 × 0.7 + 0.5 × 0.3"],
        explanation="0.5 × 0.7 + 0.5 × 0.3 = 0.5. A random agent earns the *average* of the reward "
                    "probabilities; learning is what lets an agent earn more than that, so the random agent "
                    "is the natural baseline for every learning model in the course.",
        check_call='check("1.3", 0.5)',
    ),
    # ------------------------------------------------------------------ 1.4
    NumericExercise(
        id="1.4", title="Three failures in a row",
        answer=0.027, atol=1e-4,
        distractors=[(0.343, "That is 0.7³, the probability of three *successes* in a row."),
                     (0.9, "You added 0.3 three times. Independent events combine by *multiplying* probabilities."),
                     (0.09, "That is 0.3², the probability of two failures in a row — you need three."),
                     (0.3, "That is the probability of a single failure; three independent failures are much rarer."),
                     (0.973, "That is 1 − 0.027, the probability of *at least one* success in three trials.")],
        hints=["Each trial fails with probability 1 − 0.7 = 0.3, independently of the others.",
               "The probability that several independent events all happen is the product of their probabilities.",
               "0.3 × 0.3 × 0.3"],
        explanation="0.3³ = 0.027: a 70% option fails three times in a row on about 1 run in 37. That is rare, "
                    "but over a 200-trial experiment such runs *will* happen — and an agent that reacts strongly "
                    "to them (a high learning rate, lesson 3) will abandon a good option for no reason.",
        check_call='check("1.4", 0.3**3)',
    ),
    # ------------------------------------------------------------------ 1.5
    CodeExercise(
        id="1.5", title="Simulate a random agent",
        reference=simulate_random_agent,
        signature="simulate_random_agent(probs, n_trials, rng)",
        cases=[{"probs": [0.7, 0.3], "n_trials": 2000, "rng": np.random.default_rng(1)},
               {"probs": [0.2, 0.5, 0.8], "n_trials": 3000, "rng": np.random.default_rng(2)}],
        compare=_random_compare,
        diagnose=_random_diagnose,
        hints=["Inside the loop: pick an option index at random (`rng.integers(n_arms)`), draw its reward "
               "(`float(rng.random() < probs[c])`), and store both.",
               "`choices` must be integer indices between 0 and `n_arms - 1`; `rewards` must be 0.0 or 1.0.",
               "The checker does not need the same random numbers as the reference: it checks that every option "
               "is chosen about equally often and that rewards follow `probs`."],
        explanation="This is the simplest possible agent: no learning, no memory. Its reward rate is the mean of "
                    "the reward probabilities (exercise 1.3), which makes it the baseline every learning model "
                    "must beat. The loop structure (choose → outcome → store) is the skeleton of every "
                    "simulation in the course.",
        common_mistakes="Drawing the reward from the wrong option (e.g. always option 0), returning rewards as "
                        "booleans or as the probabilities themselves, or forgetting to store the choice.",
        check_call='check("1.5", simulate_random_agent)',
    ),
    # ------------------------------------------------------------------ 1.6
    CodeExercise(
        id="1.6", title="Simulate a win-stay / lose-shift agent",
        reference=simulate_wsls,
        signature="simulate_wsls(probs, n_trials, rng)",
        cases=[{"probs": [0.7, 0.3], "n_trials": 1000, "rng": np.random.default_rng(4)},
               {"probs": [0.2, 0.5, 0.8], "n_trials": 1500, "rng": np.random.default_rng(5)}],
        compare=_wsls_compare,
        diagnose=_wsls_diagnose,
        hints=["The choice on trial t depends only on the previous choice and the previous reward: look at "
               "`choices[t - 1]` and `rewards[t - 1]` (the first trial has no history, so choose at random).",
               "After a reward: `c = choices[t - 1]`. After no reward: pick any option *other than* `choices[t - 1]` "
               "(with two options that is `1 - choices[t - 1]`; in general, `rng.choice` among the others).",
               "The checker verifies the rule trial by trial: every rewarded trial must be followed by a repeat, "
               "every unrewarded trial by a switch."],
        explanation="Win-stay / lose-shift is a *heuristic*: it keeps no values and reacts only to the last outcome. "
                    "It is a serious candidate model of behaviour (Wilson & Collins 2019 use it as one of their "
                    "example models), and a useful foil for the learning models to come: it makes very specific "
                    "predictions about stay probabilities (section 1.4).",
        common_mistakes="Deciding from the reward of the *current* trial (which has not happened yet), forgetting "
                        "the random first trial, or 'shifting' to the same option.",
        check_call='check("1.6", simulate_wsls)',
    ),
    # ------------------------------------------------------------------ 1.7
    CodeExercise(
        id="1.7", title="Stay probabilities",
        reference=stay_probabilities_bandit,
        signature="stay_probabilities_bandit(choices, rewards)",
        cases=[{"choices": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1], "rewards": [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]},
               {"choices": np.array([1, 1, 1, 0, 0, 1, 0, 0]), "rewards": np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])},
               {"choices": [0, 1, 0, 1, 0, 1], "rewards": [1, 1, 1, 0, 0, 0]}],
        bug_variants={
            "current_reward": (_stay_current_reward,
                               "You conditioned on the reward of the *current* trial. 'Stay' on trial t must be "
                               "split by the reward of trial t − 1 (use `rewards[:-1]`, aligned with `choices[1:] == choices[:-1]`)."),
            "swapped": (_stay_swapped, "The two numbers are in the wrong order: return (P(stay | rewarded), P(stay | not rewarded))."),
            "counts": (_stay_counts, "You returned *counts* of stays; divide by the number of trials in each condition (use `np.mean`)."),
            "unconditional": (_stay_unconditional, "Both numbers are the overall stay probability: you forgot to split the "
                                                    "trials by whether the previous one was rewarded."),
        },
        hints=["Build a boolean array `stay` for trials 1..T−1: `choices[1:] == choices[:-1]`.",
               "Build a second boolean array for the same trials saying whether the *previous* trial was rewarded: "
               "`rewards[:-1] == 1`. Then average `stay` separately within the two groups.",
               "`np.mean(stay[previous_rewarded])` and `np.mean(stay[~previous_rewarded])`"],
        explanation="Aligning two shifted copies of an array is the standard way to ask 'what happened after ...?'. "
                    "Stay probabilities are a *model-agnostic* summary: they need no model to compute, yet every "
                    "model makes a prediction about them (random: 0.5/0.5; WSLS: 1/0; a learner: somewhere "
                    "in between), which is why they are used to validate fitted models (lessons 6–7 and 9).",
        common_mistakes="Misaligning the reward with the stay indicator (off by one trial), returning counts, "
                        "or swapping the two conditions.",
        check_call='check("1.7", stay_probabilities_bandit)',
    ),
    # ------------------------------------------------------------------ 1.8
    NumericExercise(
        id="1.8", title="The long-run reward rate of win-stay / lose-shift",
        answer=0.58, atol=0.012,
        distractors=[(0.7, "That is how often the agent chooses option A in the long run, not its reward rate: "
                           "multiply each choice frequency by the option's reward probability and add."),
                     (0.5, "That is the random agent's reward rate; WSLS does better because it drifts towards "
                           "the option that pays more often."),
                     (0.49, "That is 0.7 × 0.7: you forgot the trials on which the agent chooses B and is rewarded "
                            "(0.3 × 0.3)."),
                     (0.6, "Close, but not exact. Work out P(choose A) first (hint: it is the probability that a "
                           "trial is rewarded given... ask yourself what makes the agent arrive at A).")],
        hints=["Either simulate many trials with your `simulate_wsls` and take the mean reward, or reason: the "
               "agent is at A on the next trial if it was at A and was rewarded (0.7) *or* at B and was not "
               "rewarded (0.7).",
               "So P(at A next) = 0.7 × P(at A) + 0.7 × P(at B) = 0.7 regardless of where it was. In the long "
               "run the agent chooses A on 70% of trials.",
               "Reward rate = 0.7 × 0.7 + 0.3 × 0.3"],
        explanation="P(choose A) = 0.7 in the long run (the agent lands on A whenever the previous trial 'pointed' "
                    "to it: a reward at A or a non-reward at B, both with probability 0.7), so the reward rate "
                    "is 0.7 × 0.7 + 0.3 × 0.3 = 0.58: better than random (0.50) but well short of the oracle "
                    "(0.70). The heuristic over-reacts to single outcomes — the same weakness as a learner "
                    "with a very high learning rate (lesson 3).",
        check_call='check("1.8", 0.7 * 0.7 + 0.3 * 0.3)',
    ),
    # ------------------------------------------------------------------ 1.9
    ChoiceExercise(
        id="1.9", title="Why simulate before fitting?",
        question="Why does the modelling cycle insist on simulating a model *before* fitting it to data?",
        options={"a": "Because the simulated choices are what we fit the model to, instead of collecting data.",
                 "b": "To see what each parameter does to behaviour, to catch bugs, and to find out whether the "
                      "parameters can be recovered from data at all.",
                 "c": "Because fitting a model requires it to generate the same choices as the participant.",
                 "d": "To remove the randomness from the model so that it makes deterministic predictions."},
        correct="b",
        feedback={"a": "We fit models to *real* data; simulations are a tool to understand and test the model, "
                       "not a replacement for the experiment.",
                  "c": "Fitting never requires reproducing the participant's choices (they are random); it "
                       "evaluates how *probable* the observed choices are (lesson 5).",
                  "d": "Models keep their randomness; simulation shows the *range* of behaviour they produce."},
        hints=["Think of simulation as a test bench: what can you learn about a model by running it with known "
               "parameters?",
               "If two parameter settings produced indistinguishable behaviour, what would that mean for fitting?"],
        explanation="Simulating with known parameters tells you what behavioural signature each parameter leaves, "
                    "exposes bugs, and — by fitting the simulated data — reveals whether the parameters are "
                    "recoverable (lesson 6). Without that, a fitted parameter value is uninterpretable.",
        check_call='check("1.9", "b")',
    ),
    # ------------------------------------------------------------------ 1.10
    FreeTextExercise(
        id="1.10", title="Fitting without simulating",
        question="In 2–4 sentences: what could go wrong if you fitted a model to participants' data without ever simulating it?",
        model_answer="Several things. (1) You would not know whether the parameters are identifiable: two "
                     "parameters might trade off so that many combinations produce the same behaviour, and the "
                     "fitted values would be meaningless — only simulating and re-fitting (parameter recovery) "
                     "reveals this. (2) You would not know what behavioural signature the model actually "
                     "produces, so you could not tell whether it captures the qualitative patterns in the data "
                     "or is mis-specified. (3) Bugs in the model code would go unnoticed, because a fitting "
                     "routine will happily return 'best' parameters for a broken likelihood. (4) You might "
                     "interpret a parameter as measuring one thing when, in this task, it mostly reflects another.",
        concepts=[("identifiability / parameter recovery", ["recover", "identif", "trade-off", "tradeoff", "trade off",
                                                            "confound", "distinguish", "disentangl", "same behaviour",
                                                            "same behavior", "many combinations"]),
                  ("behavioural signature / mis-specification", ["signature", "specif", "wrong model", "capture",
                                                                 "qualitative", "pattern", "reproduce", "produce the",
                                                                 "generate the", "doesn't actually", "does not actually"]),
                  ("bugs", ["bug", "error in the code", "coding error", "mistake in the code", "implementation",
                            "wrong code", "typo", "broken"]),
                  ("interpretation", ["interpret", "meaning", "what the parameter", "what a parameter", "reflect"])],
        min_concepts=2,
        hints=["Think about what the fitting routine would return if two parameters had exactly the same effect "
               "on behaviour — and whether you could tell.",
               "Think about a model whose code contains a mistake: would fitting it produce an error message?"],
        explanation="Fitting returns numbers no matter what. Simulation is the only way to know whether those "
                    "numbers could mean anything: whether the parameters are recoverable, whether the model "
                    "produces the right kind of behaviour, and whether the code is correct.",
        check_call='check("1.10", """Without simulating you would not know whether the parameters are identifiable: two parameters might trade off so that many combinations produce the same behaviour and the fitted values would be meaningless, which only parameter recovery reveals. You would also not know what behavioural signature the model produces, so you could not tell whether it captures the qualitative patterns in the data or is mis-specified. And bugs in the code would go unnoticed, because fitting returns parameters even for a broken implementation.""")',
    ),
]
