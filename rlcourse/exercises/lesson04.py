"""Exercises of lesson 4: temporal-difference learning, control and dopamine."""
from __future__ import annotations

import numpy as np

from .. import models, tasks
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# --------------------------------------------------------------------------- #
# buggy variants used for targeted diagnoses
# --------------------------------------------------------------------------- #


def _td0_no_bootstrap(V, s, r, s_next, alpha, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    V[s] = V[s] + alpha * (r - V[s])
    return V


def _td0_ignore_done(V, s, r, s_next, alpha, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    V[s] = V[s] + alpha * (r + gamma * V[s_next] - V[s])
    return V


def _td0_update_next(V, s, r, s_next, alpha, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    target = r if done else r + gamma * V[s_next]
    V[s_next] = V[s_next] + alpha * (target - V[s])
    return V


def _td0_no_gamma(V, s, r, s_next, alpha, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    target = r if done else r + V[s_next]
    V[s] = V[s] + alpha * (target - V[s])
    return V


def _td0_flipped(V, s, r, s_next, alpha, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    target = r if done else r + gamma * V[s_next]
    V[s] = V[s] + alpha * (V[s] - target)
    return V


def _ql_same_action(Q, s, a, r, s_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * Q[s_next, a]
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def _ql_mean(Q, s, a, r, s_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * np.mean(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def _ql_ignore_done(Q, s, a, r, s_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])
    return Q


def _ql_all_actions(Q, s, a, r, s_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * np.max(Q[s_next])
    Q[s] = Q[s] + alpha * (target - Q[s])
    return Q


def _sarsa_max(Q, s, a, r, s_next, a_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def _sarsa_ignore_done(Q, s, a, r, s_next, a_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    Q[s, a] = Q[s, a] + alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
    return Q


def _sarsa_update_next(Q, s, a, r, s_next, a_next, alpha, gamma=1.0, done=False):
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * Q[s_next, a_next]
    Q[s_next, a_next] = Q[s_next, a_next] + alpha * (target - Q[s, a])
    return Q


def _tdl_replacing(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    V = np.array(V, dtype=float)
    e = np.zeros_like(V)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        done = t == len(rewards) - 1
        delta = (r if done else r + gamma * V[s_next]) - V[s]
        e[s] = 1.0
        V = V + alpha * delta * e
        e = gamma * lam * e
    return V


def _tdl_no_trace(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    V = np.array(V, dtype=float)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        done = t == len(rewards) - 1
        delta = (r if done else r + gamma * V[s_next]) - V[s]
        V[s] = V[s] + alpha * delta
    return V


def _tdl_no_decay(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    V = np.array(V, dtype=float)
    e = np.zeros_like(V)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        done = t == len(rewards) - 1
        delta = (r if done else r + gamma * V[s_next]) - V[s]
        e[s] += 1.0
        V = V + alpha * delta * e
    return V


def _tdl_ignore_terminal(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    V = np.array(V, dtype=float)
    e = np.zeros_like(V)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        delta = r + gamma * V[s_next] - V[s]
        e[s] += 1.0
        V = V + alpha * delta * e
        e = gamma * lam * e
    return V


def _tdl_decay_before_use(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    V = np.array(V, dtype=float)
    e = np.zeros_like(V)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        done = t == len(rewards) - 1
        delta = (r if done else r + gamma * V[s_next]) - V[s]
        e[s] += 1.0
        e = gamma * lam * e
        V = V + alpha * delta * e
    return V


def _ac_no_baseline(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    theta = np.array(theta, dtype=float)
    target = r if done else r + gamma * V[s_next]
    delta = target - V[s]
    V[s] = V[s] + alpha_critic * delta
    theta[s, a] = theta[s, a] + alpha_actor * delta
    return V, theta, delta


def _ac_reward_for_actor(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    theta = np.array(theta, dtype=float)
    target = r if done else r + gamma * V[s_next]
    delta = target - V[s]
    V[s] = V[s] + alpha_critic * delta
    probs = models.softmax(theta[s])
    grad = -probs
    grad[a] += 1.0
    theta[s] = theta[s] + alpha_actor * r * grad
    return V, theta, delta


def _ac_critic_first(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma=1.0, done=False):
    V = np.array(V, dtype=float)
    theta = np.array(theta, dtype=float)
    target = r if done else r + gamma * V[s_next]
    delta = target - V[s]
    V[s] = V[s] + alpha_critic * delta
    delta_after = target - V[s]            # BUG: recomputed with the updated critic
    probs = models.softmax(theta[s])
    grad = -probs
    grad[a] += 1.0
    theta[s] = theta[s] + alpha_actor * delta_after * grad
    return V, theta, delta_after


def _pav_update_current(reward_sequences, alpha, gamma=1.0):
    n_steps = len(reward_sequences[0])
    V = np.zeros(n_steps)
    V_history, delta_history = [V.copy()], []
    for rewards in reward_sequences:
        deltas = np.zeros(n_steps + 1)
        v_prev = 0.0
        for t in range(n_steps):
            deltas[t] = rewards[t] + gamma * V[t] - v_prev
            V[t] = V[t] + alpha * deltas[t]           # BUG: trains the current step
            v_prev = V[t]
        deltas[n_steps] = 0.0 - v_prev
        V_history.append(V.copy())
        delta_history.append(deltas)
    return np.array(V_history), np.array(delta_history)


def _pav_diagnose(out, ref, case):
    seqs = case["reward_sequences"]
    n_trials, n_steps = len(seqs), len(seqs[0])
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        return "Return a tuple `(V_history, delta_history)` of two arrays."
    V_hist, d_hist = np.asarray(out[0], float), np.asarray(out[1], float)
    if V_hist.shape != (n_trials + 1, n_steps):
        return (f"`V_history` should have shape (n_trials + 1, n_steps) = {(n_trials + 1, n_steps)} "
                f"(the values *before* each trial, starting with the initial zeros), but has shape {V_hist.shape}.")
    if d_hist.shape != (n_trials, n_steps + 1):
        return (f"`delta_history` should have shape (n_trials, n_steps + 1) = {(n_trials, n_steps + 1)}: one error per "
                f"time step plus the error at the end of the trial (0 − V[n−1]). Yours has shape {d_hist.shape}.")
    if not np.allclose(V_hist[0], 0):
        return "`V_history[0]` should be the initial values (all zeros) — record the values before each trial."
    t_us = int(np.argmax(seqs[0]))
    if t_us > 0:
        if not np.isclose(d_hist[0, t_us], seqs[0][t_us]):
            return ("On the very first trial every value is 0, so the error at the reward step should equal the reward "
                    "itself: delta_t = r_t + gamma * V[t] - V[t-1].")
        if np.isclose(V_hist[1, t_us - 1], 0) and not np.isclose(V_hist[1, t_us], 0):
            return ("After the first trial the value of the step *before* the reward should have grown, but yours "
                    "updated the step at which the reward arrived. The error at step t trains V[t-1].")
    return None


def _ac_reference_cases():
    V = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.0])
    theta0 = np.zeros((7, 2))
    theta1 = np.zeros((7, 2))
    theta1[3] = [0.5, -0.5]
    theta1[5] = [-1.0, 1.0]
    return [
        {"V": V, "theta": theta0, "s": 3, "a": 1, "r": 0.0, "s_next": 4, "alpha_critic": 0.1, "alpha_actor": 0.2,
         "gamma": 0.9, "done": False},
        {"V": V, "theta": theta1, "s": 3, "a": 0, "r": 0.0, "s_next": 2, "alpha_critic": 0.5, "alpha_actor": 0.1,
         "gamma": 1.0, "done": False},
        {"V": V, "theta": theta1, "s": 5, "a": 1, "r": 1.0, "s_next": 6, "alpha_critic": 0.1, "alpha_actor": 0.3,
         "gamma": 0.9, "done": True},
    ]


_Q = np.array([[0.1, 0.2], [0.3, 0.5], [0.0, 0.9], [0.4, 0.4]])
_Q3 = np.array([[0.0, 0.1, 0.2], [0.7, 0.2, 0.1], [0.3, 0.3, 0.3]])
_pav8 = tasks.PavlovianTrace(n_steps=8, t_us=5)
_pav6 = tasks.PavlovianTrace(n_steps=6, t_us=3)

EXERCISES = [
    # ------------------------------------------------------------------ 4.1
    NumericExercise(
        id="4.1", title="A discounted return",
        answer=0.81, atol=1e-6,
        distractors=[(1.0, "You forgot to discount: the third reward is two steps after the first, so it is multiplied by gamma²."),
                     (0.9, "Off by one: the first reward in the sequence is *not* discounted (gamma⁰), the second is multiplied by gamma, the third by gamma²."),
                     (0.729, "You discounted three times; the third reward comes two steps after the first, so the factor is gamma², not gamma³."),
                     (2.71, "The return is a weighted *sum* of rewards, and two of the three rewards are 0.")],
        hints=["G = r₁ + γ r₂ + γ² r₃ + ...: the first reward is not discounted at all.",
               "Only the third reward is non-zero, and it carries the factor γ² = 0.9².",
               "0 + 0.9 × 0 + 0.9² × 1 = ?"],
        explanation="G = 0 + 0.9·0 + 0.9²·1 = 0.81. A reward two steps away is worth 81% of an immediate one when γ = 0.9; "
                    "with γ = 0.5 it would be worth only 25%. The discount factor sets the agent's effective horizon (about 1/(1 − γ) steps).",
        check_call='check("4.1", 0.81)',
    ),
    # ------------------------------------------------------------------ 4.2
    NumericExercise(
        id="4.2", title="One Bellman step by hand",
        answer=0.75, atol=1e-6,
        distractors=[(0.5, "That is V(B) itself. Apply the Bellman equation to state C: average over the two moves it can make."),
                     (1.0, "That is the reward at the exit. Only half of the moves from C reach it; the other half go back to B."),
                     (0.25, "That is V(A): you looked at the state next to the *unrewarded* exit."),
                     (0.625, "Did you use V(C) ≈ V(B) on the right-hand side? From C the agent moves to B (value 0.5) or to the exit (reward 1, value 0).")],
        hints=["V(C) = ½ × [reward + value after moving left] + ½ × [reward + value after moving right].",
               "Moving left reaches B: reward 0 and value V(B) = 0.5. Moving right reaches the exit: reward 1 and value 0 (terminal).",
               "V(C) = 0.5 × (0 + 0.5) + 0.5 × (1 + 0) = ?"],
        explanation="V(C) = ½(0 + V(B)) + ½(1 + 0) = 0.25 + 0.5 = 0.75. The Bellman equation expresses a value as the expected "
                    "immediate reward plus the (discounted) value of wherever you land next — exactly the quantity TD learning "
                    "will use as its target.",
        check_call='check("4.2", 0.75)',
    ),
    # ------------------------------------------------------------------ 4.3
    CodeExercise(
        id="4.3", title="The TD(0) update",
        reference=models.td0_update,
        signature="td0_update(V, s, r, s_next, alpha, gamma=1.0, done=False)",
        cases=[{"V": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.0], "s": 3, "r": 0.0, "s_next": 4, "alpha": 0.1, "gamma": 0.9, "done": False},
               {"V": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.3], "s": 5, "r": 1.0, "s_next": 6, "alpha": 0.5, "gamma": 0.9, "done": True},
               {"V": [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0], "s": 1, "r": 0.0, "s_next": 0, "alpha": 0.1, "gamma": 1.0, "done": True},
               {"V": np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]), "s": 4, "r": 0.0, "s_next": 5, "alpha": 0.2, "gamma": 1.0, "done": False}],
        bug_variants={
            "no_bootstrap": (_td0_no_bootstrap, "Your target is just the reward. TD(0) bootstraps: the target is r + gamma × V[s_next] "
                                                "(unless the episode ended)."),
            "ignore_done": (_td0_ignore_done, "When `done` is True the episode has ended and nothing follows, so the target must be just r "
                                              "— you still added gamma × V[s_next]."),
            "update_next": (_td0_update_next, "You updated V[s_next]. The TD error corrects the prediction made *in* state s, so V[s] is updated."),
            "no_gamma": (_td0_no_gamma, "You forgot the discount factor: the target is r + gamma × V[s_next]."),
            "flipped": (_td0_flipped, "Your TD error has the wrong sign: delta = target − V[s], so the value moves *towards* the target."),
        },
        hints=["The target is what the Bellman equation says the value should be: r + gamma × V[s_next], or just r if the episode ended.",
               "Compute delta = target − V[s] and move V[s] a fraction alpha towards the target — exactly the Rescorla–Wagner rule with a new target.",
               "`V = np.array(V, float); target = r if done else r + gamma * V[s_next]; V[s] += alpha * (target - V[s]); return V`"],
        explanation="TD(0) is the Rescorla–Wagner rule with the reward replaced by r + gamma × V[s_next]. The value of the next state "
                    "stands in for all the rewards that will follow it, so learning can proceed after every single step instead "
                    "of waiting for the end of the episode. Returning a new array (np.array copies) keeps the caller's values untouched.",
        common_mistakes="Forgetting the `done` case (bootstrapping from a terminal state), updating the wrong state, or dropping gamma.",
        check_call='check("4.3", td0_update)',
    ),
    # ------------------------------------------------------------------ 4.4
    CodeExercise(
        id="4.4", title="TD learning across a conditioning trial",
        reference=models.td_pavlovian,
        signature="td_pavlovian(reward_sequences, alpha, gamma=1.0)",
        cases=[{"reward_sequences": [_pav8.rewards() for _ in range(20)], "alpha": 0.3, "gamma": 0.9},
               {"reward_sequences": [_pav6.rewards() for _ in range(15)] + [_pav6.rewards(omit=True)], "alpha": 0.2, "gamma": 1.0},
               {"reward_sequences": [_pav8.rewards(size=2.0) for _ in range(5)] + [_pav8.rewards(size=0.5)], "alpha": 0.5, "gamma": 0.95}],
        bug_variants={
            "update_current": (_pav_update_current, "You trained the value of the *current* time step with its own error. In this "
                                                    "convention the error computed on arriving at step t compares the new prediction "
                                                    "V[t] with the previous one V[t−1], and it trains V[t−1]."),
        },
        diagnose=_pav_diagnose,
        hints=["Keep one array V of length n_steps. For every trial, loop over the time steps t: compute delta_t = r_t + gamma × V[t] − v_prev, "
               "where v_prev is the value of the previous step (0 before the cue), then — if t > 0 — update V[t−1] by alpha × delta_t.",
               "After the last step the trial ends with nothing to come: delta_end = 0 − V[n−1], and V[n−1] is updated with it. Store "
               "the n_steps + 1 errors of each trial and the values *before* each trial (start with the initial zeros).",
               "Inside the trial loop: `deltas[t] = rewards[t] + gamma * V[t] - v_prev`; `if t > 0: V[t-1] += alpha * deltas[t]`; "
               "`v_prev = V[t]`. Then `deltas[n] = -v_prev; V[n-1] += alpha * deltas[n]`."],
        explanation="Each time step after the cue is a state, so the whole trial is a chain and ordinary TD(0) applies. The error on "
                    "arriving at step t is the reward found there plus the new prediction, minus what the previous step predicted. "
                    "On the first trial only the reward step is surprising; across trials the surprise moves backwards, step by step, "
                    "until it sits at the cue — where nothing earlier could have predicted it. That migration is the whole dopamine story.",
        common_mistakes="Training V[t] instead of V[t−1]; forgetting that nothing is predicted before the cue (v_prev must restart at 0 "
                        "on every trial); omitting the error at the end of the trial; recording the values after the trial as if they "
                        "were the values before it.",
        check_call='check("4.4", td_pavlovian)',
    ),
    # ------------------------------------------------------------------ 4.5
    ChoiceExercise(
        id="4.5", title="What the cue response means",
        question="After learning, the TD error (and the dopamine burst) at the cue represents ...",
        options={"a": "the reward itself, delivered earlier than before",
                 "b": "the discounted reward that the cue now predicts and that was not predicted before the cue appeared",
                 "c": "the motor command to approach the reward",
                 "d": "the agent's uncertainty about whether the reward will come"},
        correct="b",
        feedback={"a": "No reward is delivered at the cue. What changes at the cue is the *prediction*: before it nothing was expected, after it a reward is.",
                  "c": "TD errors are about predictions of reward, not about actions (the actor–critic section shows how actions are trained *by* the error).",
                  "d": "The TD error signals a change in the *expected* (mean) reward, not uncertainty. Learners that track uncertainty are the topic of lessons 11–13."},
        hints=["Write down delta at the cue: delta_0 = r_0 + gamma × V[0] − (value before the cue). What is each term after learning?",
               "r_0 = 0 and nothing predicts anything before the cue, so delta_0 = gamma × V[0]: the value the cue carries."],
        explanation="At the cue delta = 0 + gamma × V[cue] − 0: the cue brings a prediction of future reward where there was none. "
                    "That is why a fully predicted reward produces no response (the reward was already 'cashed in' at the cue) and "
                    "an omitted reward produces a dip (the prediction is not met).",
        check_call='check("4.5", "b")',
    ),
    # ------------------------------------------------------------------ 4.6
    CodeExercise(
        id="4.6", title="The Q-learning update",
        reference=models.q_learning_update,
        signature="q_learning_update(Q, s, a, r, s_next, alpha, gamma=1.0, done=False)",
        cases=[{"Q": _Q, "s": 0, "a": 0, "r": 0.0, "s_next": 1, "alpha": 0.5, "gamma": 0.9, "done": False},
               {"Q": _Q, "s": 2, "a": 0, "r": 1.0, "s_next": 3, "alpha": 0.2, "gamma": 0.9, "done": True},
               {"Q": _Q3, "s": 2, "a": 2, "r": 0.5, "s_next": 1, "alpha": 0.3, "gamma": 0.8, "done": False},
               {"Q": _Q, "s": 1, "a": 1, "r": -1.0, "s_next": 2, "alpha": 1.0, "gamma": 1.0, "done": False}],
        bug_variants={
            "same_action": (_ql_same_action, "You bootstrapped from Q[s_next, a] — the *same* action. Q-learning uses the best action "
                                             "available in the next state: max over a' of Q[s_next, a']."),
            "mean": (_ql_mean, "You averaged the next state's action values. Q-learning takes the *maximum* (the value of the greedy policy)."),
            "ignore_done": (_ql_ignore_done, "When `done` is True nothing follows, so the target is just r — do not add gamma × max Q[s_next]."),
            "all_actions": (_ql_all_actions, "You updated every action of state s. Only the action that was taken, Q[s, a], is updated."),
        },
        hints=["The target is r + gamma × max_a' Q[s_next, a'] (just r if the episode ended): the best the agent could do next.",
               "Then it is the same delta rule as always: Q[s, a] += alpha × (target − Q[s, a]). Only the pair (s, a) changes.",
               "`Q = np.array(Q, float); target = r if done else r + gamma * np.max(Q[s_next]); Q[s, a] += alpha * (target - Q[s, a]); return Q`"],
        explanation="Q-learning's target uses the *best* next action, whatever the agent actually does next. It therefore learns the "
                    "values of the greedy policy while behaving exploratorily — it is *off-policy*.",
        common_mistakes="Using the action actually taken next (that is SARSA), averaging instead of maximising, or forgetting the terminal case.",
        check_call='check("4.6", q_learning_update)',
    ),
    # ------------------------------------------------------------------ 4.7
    CodeExercise(
        id="4.7", title="The SARSA update",
        reference=models.sarsa_update,
        signature="sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma=1.0, done=False)",
        cases=[{"Q": _Q, "s": 0, "a": 0, "r": 0.0, "s_next": 1, "a_next": 0, "alpha": 0.5, "gamma": 0.9, "done": False},
               {"Q": _Q, "s": 2, "a": 0, "r": 1.0, "s_next": 3, "a_next": 1, "alpha": 0.2, "gamma": 0.9, "done": True},
               {"Q": _Q3, "s": 2, "a": 2, "r": 0.5, "s_next": 1, "a_next": 2, "alpha": 0.3, "gamma": 0.8, "done": False},
               {"Q": _Q, "s": 1, "a": 1, "r": -1.0, "s_next": 2, "a_next": 0, "alpha": 1.0, "gamma": 1.0, "done": False}],
        bug_variants={
            "max": (_sarsa_max, "You used the maximum over next actions — that is Q-learning. SARSA bootstraps from the action the agent "
                                "*actually* takes next: Q[s_next, a_next]."),
            "ignore_done": (_sarsa_ignore_done, "When `done` is True the target is just r."),
            "update_next": (_sarsa_update_next, "You updated Q[s_next, a_next]; the pair that is corrected is the one whose prediction was made: Q[s, a]."),
        },
        hints=["The target is r + gamma × Q[s_next, a_next] (just r if the episode ended): the value of the state–action pair the agent moves to.",
               "Then the usual delta rule on Q[s, a].",
               "`Q = np.array(Q, float); target = r if done else r + gamma * Q[s_next, a_next]; Q[s, a] += alpha * (target - Q[s, a]); return Q`"],
        explanation="SARSA takes its name from the quintuple it uses: (S, A, R, S', A'). Because the target includes the action actually "
                    "chosen next — exploratory or not — it learns the value of the policy the agent really follows: it is *on-policy*.",
        common_mistakes="Maximising over next actions (Q-learning), or forgetting the terminal case.",
        check_call='check("4.7", sarsa_update)',
    ),
    # ------------------------------------------------------------------ 4.8
    ChoiceExercise(
        id="4.8", title="On-policy or off-policy?",
        question="Which algorithm learns the value of the policy it actually follows, including the consequences of its own exploratory actions?",
        options={"a": "Q-learning", "b": "SARSA", "c": "both", "d": "neither"},
        correct="b",
        feedback={"a": "Q-learning's target takes the maximum over next actions, so it evaluates the greedy policy regardless of how the agent explores (off-policy).",
                  "c": "Only one of them bootstraps from the action actually taken next.",
                  "d": "One of them does: look at which next-state value each target uses."},
        hints=["Compare the two targets: max_a' Q[s_next, a'] versus Q[s_next, a_next].",
               "The target that includes the *actually chosen* next action, whatever it was, evaluates the behaving policy."],
        explanation="SARSA is on-policy: its values include the cost of its own exploration, which is why it learns the safer path next "
                    "to the cliff. Q-learning is off-policy: it learns the optimal (greedy) values and walks along the edge — and falls "
                    "off while exploring.",
        check_call='check("4.8", "b")',
    ),
    # ------------------------------------------------------------------ 4.9
    CodeExercise(
        id="4.9", title="TD(λ) with eligibility traces",
        reference=models.td_lambda_episode,
        signature="td_lambda_episode(V, states, rewards, alpha, gamma=1.0, lam=0.0)",
        cases=[{"V": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0], "states": [3, 4, 3, 4, 5, 6], "rewards": [0.0, 0.0, 0.0, 0.0, 1.0],
                "alpha": 0.1, "gamma": 0.9, "lam": 0.5},
               {"V": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3], "states": [3, 2, 1, 0], "rewards": [0.0, 0.0, 0.0],
                "alpha": 0.2, "gamma": 1.0, "lam": 0.0},
               {"V": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3], "states": [3, 4, 5, 6], "rewards": [0.0, 0.0, 1.0],
                "alpha": 0.1, "gamma": 0.9, "lam": 1.0},
               {"V": np.full(7, 0.5), "states": [3, 2, 3, 4, 5, 4, 5, 6], "rewards": [0, 0, 0, 0, 0, 0, 1.0],
                "alpha": 0.3, "gamma": 0.95, "lam": 0.8}],
        bug_variants={
            "replacing": (_tdl_replacing, "You *set* the trace of the visited state to 1. With accumulating traces a revisited state's "
                                          "trace is *increased* by 1 (e[s] += 1)."),
            "no_trace": (_tdl_no_trace, "You only updated the current state (that is TD(0)). With traces *every* state is updated by "
                                        "alpha × delta × e[s], so earlier states receive a share of the error."),
            "no_decay": (_tdl_no_decay, "The traces never decay. After each update multiply them by gamma × lam."),
            "ignore_terminal": (_tdl_ignore_terminal, "On the last transition the episode ends: the target is just the reward, with no "
                                                     "bootstrapping from the terminal state."),
            "decay_before_use": (_tdl_decay_before_use, "You decayed the traces *before* using them for the update, so the current state's "
                                                        "trace is gamma × lam instead of 1. Update first, then decay."),
        },
        hints=["Keep a trace vector e (zeros). At each step: compute the TD error delta as in TD(0) (the last transition is terminal), "
               "add 1 to the trace of the current state, update *all* values by alpha × delta × e, then decay e by gamma × lam.",
               "Because V is updated as a whole vector (`V = V + alpha * delta * e`), states visited several steps ago still receive "
               "a (smaller) share of the current error.",
               "`e[s] += 1.0; V = V + alpha * delta * e; e = gamma * lam * e` inside the loop over t."],
        explanation="Eligibility traces keep a fading memory of which states were visited recently; each TD error is then credited to all "
                    "of them in proportion to their trace. λ = 0 gives TD(0) and λ = 1 (for an episodic task) gives a Monte-Carlo-like "
                    "update. Daw et al. (2011) use exactly this λ in their model of the two-step task (lesson 9).",
        common_mistakes="Replacing instead of accumulating traces, forgetting to decay them, decaying before the update, or bootstrapping from the terminal state.",
        check_call='check("4.9", td_lambda_episode)',
    ),
    # ------------------------------------------------------------------ 4.10
    NumericExercise(
        id="4.10", title="How fast a trace fades",
        answer=0.9 ** 2 * 0.5 ** 2, atol=1e-6,
        distractors=[(0.45, "That is the trace after *one* step (γλ). Two steps later it has been multiplied by γλ twice."),
                     (0.81, "You used γ² only: each step multiplies the trace by γ × λ, not by γ alone."),
                     (0.25, "You used λ² only: the trace decays by γ × λ per step."),
                     (0.091, "That is three decay steps; the question asks about two steps after the visit.")],
        hints=["After a visit the state's trace is 1 (accumulating traces, no revisit). Each later step multiplies it by γ × λ.",
               "Two steps later: (γ λ)² = (0.9 × 0.5)²."],
        explanation="(γλ)² = 0.45² = 0.2025. Traces decay geometrically at rate γλ, so with γ = 0.9, λ = 0.5 a state's "
                    "eligibility is halved roughly every step — the credit for a reward spreads only a few steps back.",
        check_call='check("4.10", 0.9**2 * 0.5**2)',
    ),
    # ------------------------------------------------------------------ 4.11
    CodeExercise(
        id="4.11", title="One actor–critic step",
        reference=models.actor_critic_update,
        signature="actor_critic_update(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma=1.0, done=False)",
        cases=_ac_reference_cases(),
        bug_variants={
            "no_baseline": (_ac_no_baseline, "You increased only the preference of the chosen action by alpha × delta. The softmax "
                                             "policy-gradient update is alpha × delta × (one_hot(a) − π(·|s)): the chosen action goes up "
                                             "*and the others go down* in proportion to their probability."),
            "reward_for_actor": (_ac_reward_for_actor, "The actor should learn from the critic's TD error delta, not from the raw reward "
                                                       "— that is the whole point of the architecture."),
            "critic_first": (_ac_critic_first, "You recomputed the error after updating the critic. Compute delta once, from the values "
                                               "*before* the update, and use that same delta for both critic and actor."),
        },
        hints=["First the critic: target = r + gamma × V[s_next] (or r if done), delta = target − V[s], V[s] += alpha_critic × delta.",
               "Then the actor, using the *same* delta: with a softmax policy the gradient of log π(a|s) with respect to the "
               "preferences theta[s, :] is one_hot(a) − softmax(theta[s]).",
               "`grad = -softmax(theta[s]); grad[a] += 1.0; theta[s] += alpha_actor * delta * grad` — and return (V, theta, delta)."],
        explanation="The critic learns state values by TD; its error delta says whether the last action turned out better or worse than "
                    "expected and is broadcast to the actor, which nudges the policy towards actions followed by positive surprises. "
                    "Subtracting the policy probabilities (the −π term) keeps the preferences from drifting and is the exact gradient "
                    "of log π for a softmax policy. In the basal ganglia the critic is mapped onto ventral striatum, the actor onto "
                    "dorsal striatum, and delta onto dopamine (Joel, Niv & Ruppin 2002).",
        common_mistakes="Training the actor with the reward instead of delta, forgetting the −π baseline term, or recomputing delta after the critic's update.",
        check_call='check("4.11", actor_critic_update)',
    ),
    # ------------------------------------------------------------------ 4.12
    FreeTextExercise(
        id="4.12", title="How TD solves temporal credit assignment",
        question="In 2–4 sentences: a reward arrives ten steps after the cue that predicted it. How does TD learning manage to assign value to the cue, and why is the learned signal strongest at the cue rather than at the reward?",
        model_answer="TD learning does not wait for the reward: at every step it updates the previous state's value towards the reward "
                     "received plus the value of the current state (bootstrapping). On the first trials only the step just before the "
                     "reward is surprised and gains value; on later trials that step's value makes the step before it surprising, and "
                     "so on, so value propagates backwards one step per visit (faster with eligibility traces) until it reaches the "
                     "cue. Once every step predicts the reward correctly there is no error at the reward any more; the only remaining "
                     "surprise is the appearance of the cue itself, because nothing before the cue predicts it — so the prediction "
                     "error, and the dopamine response, ends up at the cue.",
        concepts=[("bootstrapping / next-state value", ["bootstrap", "next state", "value of the next", "next step", "following state",
                                                        "subsequent", "plus the value", "estimate of"]),
                  ("backward propagation of value", ["backward", "propagat", "step by step", "one step", "earlier", "back to the cue",
                                                     "moves back", "travels", "spreads", "trace"]),
                  ("prediction error at the cue / fully predicted reward", ["predict", "surpris", "error", "unexpected", "expected", "nothing before"])],
        min_concepts=2,
        hints=["Think about which state is surprised on the very first trial, and which one on the second.",
               "Once the reward is fully predicted by the preceding step, where is the only remaining surprise?"],
        explanation="TD assigns credit backwards through the chain of states by using each state's value as a stand-in for the future; "
                    "after learning, the reward is no longer a surprise, but the cue is.",
        check_call='check("4.12", """TD learning does not wait for the reward: at every step it updates the previous state towards the reward plus the value of the next state (bootstrapping). At first only the step just before the reward is surprised and gains value; on later trials that value makes the earlier step surprising, so value propagates backward step by step until it reaches the cue. Once every step predicts the reward there is no prediction error at the reward any more; the only remaining surprise is the cue itself, because nothing before the cue predicts it.""")',
    ),
]
