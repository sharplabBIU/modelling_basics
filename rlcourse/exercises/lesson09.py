"""Exercises of lesson 9: model-based reinforcement learning and the two-step task."""
from __future__ import annotations

import numpy as np

from .. import models, tasks
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise
from ..models import choose, softmax  # noqa: F401  (used by the buggy variants below)

# =========================================================================== #
# Reference: value iteration (lesson-specific; rewards are received on entering a state)
# =========================================================================== #


def value_iteration(P, R, terminal, gamma, n_iter=100):
    """Optimal state values of a small MDP by value iteration.

    P[s, a, s'] are the transition probabilities, R[s'] the reward received on
    entering s', and ``terminal`` marks the states where the episode ends (their
    value is 0 because nothing follows them).
    """
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)
    n_states, n_actions = P.shape[0], P.shape[1]
    V = np.zeros(n_states)
    for _ in range(n_iter):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            if terminal[s]:
                continue                                        # nothing follows a terminal state
            q = [np.sum(P[s, a] * (R + gamma * V)) for a in range(n_actions)]   # one-step look-ahead
            V_new[s] = max(q)                                   # plan: assume the best action is taken
        V = V_new
    return V


def _vi_variant(P, R, terminal, gamma, n_iter=100, skip_terminal=True, use_max=True, use_gamma=True, use_reward=True):
    P = np.asarray(P, float)
    R = np.asarray(R, float) if use_reward else np.zeros(P.shape[0])
    g = gamma if use_gamma else 1.0
    V = np.zeros(P.shape[0])
    for _ in range(n_iter):
        V_new = np.zeros(P.shape[0])
        for s in range(P.shape[0]):
            if skip_terminal and terminal[s]:
                continue
            q = [np.sum(P[s, a] * (R + g * V)) for a in range(P.shape[1])]
            V_new[s] = max(q) if use_max else np.mean(q)
        V = V_new
    return V


def _vi_diagnose(out, ref, case):
    try:
        o = np.asarray(out, float)
    except Exception:
        return None
    if o.shape == np.shape(ref) and np.allclose(o, 0):
        return "Every value is 0: did you forget to add the reward R[s'] inside the look-ahead?"
    if o.shape == np.shape(ref) and np.any(o > np.max(ref) * 1.5 + 1):
        return ("Some values are far too large — terminal states must keep the value 0 (skip them in the loop), "
                "otherwise the absorbing self-transition keeps adding reward forever.")
    return None


_chain = tasks.random_walk_chain()
_branch = tasks.three_branch_graph()
_grid = tasks.gridworld(3, 3, goal=(0, 2), start=(2, 0))

# =========================================================================== #
# Helpers for the two-step exercises
# =========================================================================== #


def _two_step_nll_variant(params, data, stage2=True, update_first=False, use_lambda=True, use_persev=True):
    """The hybrid-model likelihood with one deliberate mistake switched on."""
    alpha1, alpha2, lam, beta1, beta2, w, p_stick = params
    if not use_lambda:
        lam = 0.0
    if not use_persev:
        p_stick = 0.0
    transitions = np.array([[0.7, 0.3], [0.3, 0.7]])
    Q_mf = np.zeros(2)
    Q2 = np.zeros((2, 2))
    prev_a1 = None
    nll = 0.0
    for a1, s2, a2, r in zip(data["a1"], data["s2"], data["a2"], data["r"]):
        if update_first:                                      # peeking at the outcome before choosing
            delta1 = Q2[s2, a2] - Q_mf[a1]
            Q_mf[a1] += alpha1 * delta1
            delta2 = r - Q2[s2, a2]
            Q2[s2, a2] += alpha2 * delta2
            Q_mf[a1] += alpha1 * lam * delta2
        Q_mb = models.two_step_mb_values(Q2, transitions)
        Q_net = w * Q_mb + (1 - w) * Q_mf
        rep = np.zeros(2)
        if prev_a1 is not None:
            rep[prev_a1] = 1.0
        p1 = models.softmax(Q_net + p_stick * rep, beta1)
        nll -= np.log(p1[a1])
        if stage2:
            p2 = models.softmax(Q2[s2], beta2)
            nll -= np.log(p2[a2])
        if not update_first:
            delta1 = Q2[s2, a2] - Q_mf[a1]
            Q_mf[a1] += alpha1 * delta1
            delta2 = r - Q2[s2, a2]
            Q2[s2, a2] += alpha2 * delta2
            Q_mf[a1] += alpha1 * lam * delta2
        prev_a1 = a1
    return nll


def _stay_transposed(data):
    return models.stay_probabilities(data).T


def _stay_switch(data):
    return 1 - models.stay_probabilities(data)


def _stay_shifted(data):
    """Uses the *current* trial's reward and transition instead of the previous trial's."""
    a1 = np.asarray(data["a1"]); r = np.asarray(data["r"]); common = np.asarray(data["common"])
    stay = (a1[1:] == a1[:-1]).astype(float)
    cur_r, cur_common = r[1:], common[1:]
    out = np.zeros((2, 2))
    for i, rewarded in enumerate([1, 0]):
        for j, is_common in enumerate([1, 0]):
            mask = (cur_r == rewarded) & (cur_common == is_common)
            out[i, j] = stay[mask].mean() if mask.any() else np.nan
    return out


def _ts_structure_problem(out, case):
    if not isinstance(out, dict):
        return f"Return a dictionary with the keys 'a1', 's2', 'a2', 'r' and 'common' (you returned `{type(out).__name__}`)."
    for key in ("a1", "s2", "a2", "r", "common"):
        if key not in out:
            return f"The returned dictionary has no key {key!r}."
    n = case["task"].n_trials
    try:
        a1 = np.asarray(out["a1"]).astype(int)
        s2 = np.asarray(out["s2"]).astype(int)
        a2 = np.asarray(out["a2"]).astype(int)
        r = np.asarray(out["r"], float)
        common = np.asarray(out["common"]).astype(int)
    except Exception:
        return "The entries of the returned dictionary must be arrays of numbers (one entry per trial)."
    if not (len(a1) == len(s2) == len(a2) == len(r) == len(common) == n):
        return f"Every array must have one entry per trial (n_trials = {n})."
    for name, arr in (("a1", a1), ("s2", s2), ("a2", a2), ("common", common)):
        if arr.min() < 0 or arr.max() > 1:
            return f"`{name}` must contain only 0s and 1s."
    if not set(np.unique(r)).issubset({0.0, 1.0}):
        return "Rewards `r` must be 0 or 1 (take them from `task.step2(a2)`)."
    expected_s2 = np.where(common == 1, a1, 1 - a1)
    if not np.array_equal(s2, expected_s2):
        return ("`s2` is not consistent with `a1` and `common`: after a common transition s2 == a1, after a rare one "
                "s2 == 1 - a1. Take both from `task.step1(a1)`, which returns (state2, is_common).")
    return None


def _ts_signature_problem(out, case):
    w = case["params"][5]
    stay = models.stay_probabilities(out)
    if np.isnan(stay).any():
        return "Some cells of the stay-probability matrix are empty: one of the four trial types never occurs in your data."
    s = np.round(stay, 2).tolist()
    if w >= 0.99:
        if not (stay[0, 0] - stay[0, 1] > 0.08 and stay[1, 1] - stay[1, 0] > 0.08):
            return ("With w = 1 the agent must show the model-based *interaction*: high stay after rewarded-common "
                    "and unrewarded-rare trials, low stay after rewarded-rare and unrewarded-common trials. Your stay "
                    f"matrix [[rew-common, rew-rare], [unrew-common, unrew-rare]] is {s}. Check that Q_MB is computed "
                    "from `task.transitions` and the *maximum* stage-2 value of each state, that Q_net = w·Q_MB + "
                    "(1 − w)·Q_MF, and that the stage-1 softmax uses Q_net (plus the perseveration bonus).")
    elif w <= 0.01:
        if not (np.min(stay[0]) - np.max(stay[1]) > 0.07):
            return ("With w = 0 the agent must show the model-free *main effect of reward*: stay after any rewarded "
                    f"trial must exceed stay after any unrewarded trial. Your stay matrix is {s}. Check the model-free "
                    "updates: Q_MF[a1] += α1·(Q2[s2, a2] − Q_MF[a1]) and then Q_MF[a1] += α1·λ·(r − Q2[s2, a2]) "
                    "(with the *old* Q2 in the second prediction error), and that Q2 itself is updated.")
    return None


def _ts_compare(out, ref, case):
    return _ts_structure_problem(out, case) is None and _ts_signature_problem(out, case) is None


def _ts_diagnose(out, ref, case):
    return _ts_structure_problem(out, case) or _ts_signature_problem(out, case)


_params_a = (0.5, 0.5, 0.6, 5.0, 5.0, 0.7, 0.1)
_params_b = (0.3, 0.6, 0.9, 3.0, 4.0, 0.2, -0.1)
_data_a = models.two_step_simulate(_params_a, tasks.TwoStepTask(n_trials=300, rng=np.random.default_rng(61)),
                                   np.random.default_rng(62))
_data_b = models.two_step_simulate(_params_b, tasks.TwoStepTask(n_trials=250, rng=np.random.default_rng(63)),
                                   np.random.default_rng(64))
_data_tiny = {"a1": np.array([0, 0, 1, 1, 1, 0, 0, 1]), "s2": np.array([0, 1, 1, 1, 0, 0, 1, 1]),
              "a2": np.array([1, 0, 1, 1, 0, 1, 0, 0]), "r": np.array([1.0, 0, 1, 1, 0, 1, 1, 0]),
              "common": np.array([1, 0, 1, 1, 0, 1, 0, 1])}

# =========================================================================== #
EXERCISES = [
    # ------------------------------------------------------------------ 9.1
    CodeExercise(
        id="9.1", title="Planning by value iteration",
        reference=value_iteration,
        signature="value_iteration(P, R, terminal, gamma, n_iter=100)",
        cases=[{"P": _chain.P, "R": _chain.R, "terminal": _chain.terminal, "gamma": 0.9},
               {"P": _chain.P, "R": _chain.R, "terminal": _chain.terminal, "gamma": 1.0},
               {"P": _branch.P, "R": _branch.R, "terminal": _branch.terminal, "gamma": 0.9},
               {"P": _grid.P, "R": _grid.R, "terminal": _grid.terminal, "gamma": 0.8}],
        bug_variants={
            "no_terminal": (lambda P, R, terminal, gamma, n_iter=100: _vi_variant(P, R, terminal, gamma, n_iter, skip_terminal=False),
                            "Terminal states must keep the value 0 (nothing follows them). Because they are absorbing "
                            "(they transition to themselves), updating them keeps adding their reward every sweep."),
            "mean": (lambda P, R, terminal, gamma, n_iter=100: _vi_variant(P, R, terminal, gamma, n_iter, use_max=False),
                     "You averaged over actions. That evaluates a *random* policy; planning takes the *maximum* over "
                     "actions, because the agent will choose the best one."),
            "no_gamma": (lambda P, R, terminal, gamma, n_iter=100: _vi_variant(P, R, terminal, gamma, n_iter, use_gamma=False),
                         "You forgot the discount factor: the look-ahead is R[s'] + gamma × V[s']."),
        },
        diagnose=_vi_diagnose,
        hints=["For each non-terminal state and each action, compute the one-step look-ahead "
               "sum_s' P[s, a, s'] (R[s'] + gamma V[s']); the new value of the state is the maximum over actions.",
               "Repeat the sweep n_iter times, always using the values of the previous sweep; keep terminal states at 0.",
               "`q = [np.sum(P[s, a] * (R + gamma * V)) for a in range(n_actions)]` and `V_new[s] = max(q)` "
               "inside `for s in range(n_states)`, skipping terminal states."],
        explanation="Value iteration repeatedly applies the Bellman optimality equation V(s) = max_a Σ_s' P(s'|s,a) "
                    "[R(s') + γ V(s')]. Each sweep looks one step further ahead; after enough sweeps the values are "
                    "the best achievable discounted return from every state — computed from the model alone, "
                    "without a single real reward being experienced. That is what makes model-based control flexible.",
        common_mistakes="Not keeping terminal states at 0 (the values explode), averaging instead of maximising over "
                        "actions, forgetting gamma, or using the already-updated values within a sweep (which still "
                        "converges but is not the algorithm the checker expects).",
        check_call='check("9.1", value_iteration)',
    ),
    # ------------------------------------------------------------------ 9.2
    NumericExercise(
        id="9.2", title="A model-based value by hand",
        answer=0.74, atol=1e-3,
        distractors=[(0.66, "That is Q_MB for action 1 (0.3 × 0.8 + 0.7 × 0.6). The question asks for action 0."),
                     (0.5, "You averaged the two stage-2 actions. A model-based planner assumes it will take the *best* "
                           "stage-2 action in each state, so use the maximum."),
                     (0.8, "That is the best stage-2 value in state 0 alone: weight each state's maximum by the "
                           "probability of reaching it (0.7 and 0.3)."),
                     (0.56, "You kept only the common transition (0.7 × 0.8). Add the rare transition's term 0.3 × 0.6.")],
        hints=["Q_MB(a) = Σ_s' P(s' | a) max_a' Q2(s', a'): first find the best stage-2 value in each state.",
               "max in state 0 = 0.8, max in state 1 = 0.6; action 0 reaches state 0 with probability 0.7 and state 1 with 0.3.",
               "0.7 × 0.8 + 0.3 × 0.6"],
        explanation="Q_MB(0) = 0.7 × 0.8 + 0.3 × 0.6 = 0.74. The planner evaluates the first-stage action by imagining "
                    "where it leads and assuming the best second-stage choice there.",
        check_call='check("9.2", 0.74)',
    ),
    # ------------------------------------------------------------------ 9.3
    CodeExercise(
        id="9.3", title="Model-based stage-1 values",
        reference=models.two_step_mb_values,
        signature="two_step_mb_values(Q2, transitions)",
        cases=[{"Q2": np.array([[0.8, 0.2], [0.4, 0.6]]), "transitions": np.array([[0.7, 0.3], [0.3, 0.7]])},
               {"Q2": np.array([[0.3, 0.9], [0.5, 0.5]]), "transitions": np.array([[0.9, 0.1], [0.2, 0.8]])},
               {"Q2": np.zeros((2, 2)), "transitions": np.array([[0.7, 0.3], [0.3, 0.7]])},
               {"Q2": np.array([[0.1, 0.7], [0.95, 0.2]]), "transitions": np.array([[0.7, 0.3], [0.3, 0.7]])}],
        bug_variants={
            "mean": (lambda Q2, transitions: transitions @ np.mean(Q2, axis=1),
                     "You averaged the stage-2 actions. The planner assumes it will pick the *best* stage-2 action, "
                     "so take the maximum over actions in each state (`np.max(Q2, axis=1)`)."),
            "transposed": (lambda Q2, transitions: transitions.T @ np.max(Q2, axis=1),
                           "The transition matrix is used the wrong way round: row a of `transitions` holds P(s' | a)."),
            "unweighted": (lambda Q2, transitions: np.max(Q2, axis=1),
                           "You returned the best value of each *state*; weight them by the probability that each "
                           "first-stage action reaches each state."),
            "wrong_axis": (lambda Q2, transitions: transitions @ np.max(Q2, axis=0),
                           "You maximised over states instead of over stage-2 actions: `axis=1` (within each row/state)."),
        },
        hints=["First compute the best value available in each stage-2 state: `np.max(Q2, axis=1)` (one number per state).",
               "Then weight those two numbers by the transition probabilities of each first-stage action.",
               "`transitions @ np.max(Q2, axis=1)` does both steps at once."],
        explanation="This is a one-step look-ahead with a known model: Q_MB(a) = Σ_s' P(s'|a) max_a' Q2(s', a'). "
                    "Daw et al. (2011) assumed participants know the transition probabilities (they are instructed), "
                    "so only the stage-2 values have to be learned.",
        common_mistakes="Averaging instead of maximising over stage-2 actions, or taking the maximum over the wrong axis.",
        check_call='check("9.3", two_step_mb_values)',
    ),
    # ------------------------------------------------------------------ 9.4
    ChoiceExercise(
        id="9.4", title="The behavioural signature of model-based control",
        question="Which pattern of stay probabilities indicates model-based control in the two-step task?",
        options={"a": "Participants repeat their first-stage choice after a reward regardless of the transition type (a main effect of reward).",
                 "b": "Participants repeat after rewarded-common and unrewarded-rare trials, and switch after rewarded-rare and unrewarded-common trials (a reward × transition interaction).",
                 "c": "Participants almost always repeat their previous first-stage choice, whatever happened.",
                 "d": "Participants repeat more after rare transitions than after common ones, regardless of reward (a main effect of transition)."},
        correct="b",
        feedback={"a": "A main effect of reward is the model-*free* signature: the rewarded action is simply reinforced.",
                  "c": "That is perseveration (the parameter p in the model), which is independent of both systems.",
                  "d": "A main effect of transition on its own is predicted by neither system."},
        hints=["Ask what a planner concludes from a reward obtained after a *rare* transition: which first-stage action usually leads to the rewarded state?",
               "A model-based agent credits the *other* first-stage action after a rare transition, so reward and transition type interact."],
        explanation="A planner knows that a reward obtained in a rarely-reached state is better pursued by the *other* "
                    "first-stage action, which usually leads there. So reward increases staying after common transitions "
                    "but decreases it after rare ones — the interaction that Daw et al. (2011) used as the model-based signature.",
        check_call='check("9.4", "b")',
    ),
    # ------------------------------------------------------------------ 9.5
    CodeExercise(
        id="9.5", title="Simulate the hybrid model-based / model-free agent",
        reference=models.two_step_simulate,
        signature="two_step_simulate(params, task, rng)",
        cases=[{"params": (0.5, 0.5, 0.6, 8.0, 8.0, 1.0, 0.0), "task": tasks.TwoStepTask(n_trials=400, rng=np.random.default_rng(91)),
                "rng": np.random.default_rng(92)},
               {"params": (0.5, 0.5, 1.0, 8.0, 8.0, 0.0, 0.0), "task": tasks.TwoStepTask(n_trials=400, rng=np.random.default_rng(93)),
                "rng": np.random.default_rng(94)}],
        compare=_ts_compare,
        diagnose=_ts_diagnose,
        hints=["Per trial: (1) Q_mb = two_step_mb_values(Q2, task.transitions); Q_net = w·Q_mb + (1 − w)·Q_mf; "
               "(2) stage-1 softmax of Q_net + p·rep with beta1, where rep is 1 for the previous a1; (3) `s2, is_common = task.step1(a1)`; "
               "(4) stage-2 softmax of Q2[s2] with beta2; `r = task.step2(a2)`; (5) learn.",
               "Learning: delta1 = Q2[s2, a2] − Q_mf[a1]; Q_mf[a1] += alpha1·delta1; delta2 = r − Q2[s2, a2]; "
               "Q2[s2, a2] += alpha2·delta2; Q_mf[a1] += alpha1·lam·delta2. Compute delta2 *before* updating Q2.",
               "The checker does not require the same random numbers as the reference: it checks that s2 is consistent "
               "with a1 and `common`, and that with w = 1 your agent shows the interaction pattern and with w = 0 the "
               "main effect of reward."],
        explanation="The agent is the sum of two learners sharing stage-2 values. The model-free learner propagates the "
                    "stage-2 value (δ1) and, through the eligibility trace λ, the final reward (δ2) back to the first-stage "
                    "action it actually took; the model-based learner recomputes first-stage values from the transition "
                    "model on every trial. w mixes them, β1 and β2 set choice determinism and p captures perseveration.",
        common_mistakes="Forgetting the model-based look-ahead (so w has no effect); updating Q2 before computing δ2; "
                        "forgetting the λ-weighted update of Q_mf; using the perseveration bonus on the wrong action; "
                        "forgetting `task.reset()` or calling step1/step2 in the wrong order.",
        check_call='check("9.5", two_step_simulate)',
    ),
    # ------------------------------------------------------------------ 9.6
    CodeExercise(
        id="9.6", title="Stay probabilities",
        reference=models.stay_probabilities,
        signature="stay_probabilities(data)",
        cases=[{"data": _data_tiny}, {"data": _data_a}, {"data": _data_b}],
        bug_variants={
            "transposed": (_stay_transposed, "Rows and columns are swapped: rows must be (rewarded, unrewarded) and columns (common, rare)."),
            "switch": (_stay_switch, "You computed the probability of *switching*; the matrix should hold the probability of *repeating* the first-stage choice."),
            "shifted": (_stay_shifted, "You split trials by the *current* trial's reward and transition. A stay on trial t must be "
                                       "classified by what happened on trial t − 1."),
        },
        hints=["A stay on trial t means a1[t] == a1[t − 1]; so compare `a1[1:]` with `a1[:-1]` (one entry per trial from the second on).",
               "Classify each of those stays by the *previous* trial's reward `r[:-1]` and transition `common[:-1]`.",
               "Loop over rewarded in (1, 0) (rows) and common in (1, 0) (columns); for each combination take the mean of `stay[mask]`."],
        explanation="The stay matrix is the standard model-agnostic summary of two-step behaviour: a main effect of reward "
                    "indicates model-free learning, the reward × transition interaction indicates model-based learning "
                    "(Daw et al. 2011). It is what you plot first, before fitting any model.",
        common_mistakes="Off-by-one alignment of stays with outcomes, swapped rows/columns, or counting switches instead of stays.",
        check_call='check("9.6", stay_probabilities)',
    ),
    # ------------------------------------------------------------------ 9.7
    NumericExercise(
        id="9.7", title="Perseveration can flip a choice",
        answer=float(models.sigmoid(-0.3)), atol=1e-3,
        distractors=[(float(models.sigmoid(0.3)), "You ignored the perseveration bonus: B was chosen on the previous trial, so B receives p = 0.2 before the softmax."),
                     (float(models.sigmoid(0.9)), "You added the bonus to A, but B was the previous choice."),
                     (float(models.sigmoid(-0.1)), "You forgot to multiply by β1 = 3."),
                     (0.5, "The two options are not tied: compute β1 × [(Q_A + 0) − (Q_B + p)].")],
        hints=["With two options, P(A) = 1 / (1 + exp(−β1 [(Q_net(A) + p·rep(A)) − (Q_net(B) + p·rep(B))])).",
               "rep(B) = 1 because B was chosen last time, so the effective values are 0.5 for A and 0.4 + 0.2 = 0.6 for B.",
               "P(A) = 1 / (1 + exp(−3 × (0.5 − 0.6))) = 1 / (1 + exp(0.3))"],
        explanation="P(A) = σ(3 × (0.5 − 0.6)) ≈ 0.426: although A has the higher value, the agent is more likely to "
                    "repeat B. Perseveration is a nuisance parameter here, but leaving it out biases the other estimates — "
                    "a sticky participant would otherwise look like a poor learner.",
        check_call='check("9.7", 1 / (1 + np.exp(0.3)))',
    ),
    # ------------------------------------------------------------------ 9.8
    CodeExercise(
        id="9.8", title="The likelihood of two-step data",
        reference=models.two_step_nll,
        signature="two_step_nll(params, data, transitions=None)",
        cases=[{"params": _params_a, "data": _data_a},
               {"params": _params_b, "data": _data_b},
               {"params": (0.3, 0.7, 0.2, 2.0, 3.0, 0.5, -0.2), "data": _data_a},
               {"params": (0.6, 0.4, 0.8, 6.0, 2.0, 0.9, 0.3), "data": _data_tiny}],
        bug_variants={
            "no_stage2": (lambda params, data, transitions=None: _two_step_nll_variant(params, data, stage2=False),
                          "You only scored the stage-1 choices. The stage-2 choices are data too: add −log p2[a2] on every trial."),
            "update_first": (lambda params, data, transitions=None: _two_step_nll_variant(params, data, update_first=True),
                             "You updated the values with the trial's outcome *before* computing the choice probabilities. "
                             "The likelihood of a choice must use the values the participant had when choosing."),
            "no_lambda": (lambda params, data, transitions=None: _two_step_nll_variant(params, data, use_lambda=False),
                          "The eligibility-trace update is missing: after updating Q2, add alpha1 × lam × delta2 to Q_mf[a1]."),
            "no_persev": (lambda params, data, transitions=None: _two_step_nll_variant(params, data, use_persev=False),
                          "The perseveration term is missing: add p to the value of the action chosen on the previous trial before the stage-1 softmax."),
        },
        hints=["Replay the simulation on the *observed* data: same value computations and updates as in Exercise 9.5, "
               "but instead of sampling choices, accumulate −log p1[a1] and −log p2[a2] for the observed choices.",
               "Order within a trial: compute p1 (from Q_net + p·rep) → score a1 → compute p2 (from Q2[s2]) → score a2 → "
               "learn from (s2, a2, r) → remember a1 for the next trial's perseveration.",
               "If `transitions` is None use `np.array([[0.7, 0.3], [0.3, 0.7]])`."],
        explanation="The likelihood is the simulation turned inside out (lesson 5): the model is run forward on the "
                    "participant's own choices and outcomes, and the probability it assigned to each observed choice "
                    "is multiplied up (summed in log space). Both stages carry information: stage 2 mainly about α2 "
                    "and β2, stage 1 about w, λ, β1 and p.",
        common_mistakes="Scoring only stage 1; updating before scoring; forgetting the λ-term or the perseveration bonus; "
                        "returning the log-likelihood instead of its negative.",
        check_call='check("9.8", two_step_nll)',
    ),
    # ------------------------------------------------------------------ 9.9
    ChoiceExercise(
        id="9.9", title="Which parameter indexes model-based control?",
        question="In the hybrid model of Daw et al. (2011), the degree of model-based control is measured by ...",
        options={"a": "α1, the stage-1 learning rate",
                 "b": "β1, the stage-1 inverse temperature",
                 "c": "w, the weight of the model-based values in the stage-1 decision",
                 "d": "λ, the eligibility-trace parameter"},
        correct="c",
        feedback={"a": "α1 sets how fast the model-free stage-1 values change; it says nothing about planning.",
                  "b": "β1 sets how deterministically the *combined* value drives choice; both systems share it. (In the "
                       "reparameterisation of Gillan et al. 2016 the model-based strength is a separate inverse temperature β_MB.)",
                  "d": "λ sets how much of the final reward reaches the stage-1 model-free values; it shapes the model-free "
                       "main effect, not the model-based interaction."},
        hints=["Look at the equation that combines the two systems: Q_net = w·Q_MB + (1 − w)·Q_MF.",
               "Which parameter makes the agent purely model-based at one extreme and purely model-free at the other?"],
        explanation="w ∈ [0, 1] is the mixing weight: w = 1 is a pure planner, w = 0 a pure habit learner. Gillan et al. "
                    "(2016) instead give each system its own inverse temperature (β_MB, β_MF), which avoids the "
                    "awkward interaction between w and β1 when fitting; the two parameterisations are closely related "
                    "(β_MB ≈ w·β1, β_MF ≈ (1 − w)·β1).",
        check_call='check("9.9", "c")',
    ),
    # ------------------------------------------------------------------ 9.10
    FreeTextExercise(
        id="9.10", title="Why rare transitions are diagnostic",
        question="In 2–4 sentences: why are trials with a rare transition the ones that reveal whether a participant plans?",
        model_answer="On a common transition both systems agree: a reward reinforces the first-stage action that was "
                     "taken (model-free), and that action is also the one most likely to lead back to the rewarded state "
                     "(model-based), so both predict a stay. On a rare transition they disagree. The model-free learner "
                     "still credits the action it just took, because it only knows that the action was followed by "
                     "reward. The model-based learner knows the transition structure, so it realises that the rewarded "
                     "state is usually reached by the *other* first-stage action and switches to it. Rare trials are "
                     "therefore the only trials on which the two systems make different predictions, which is why the "
                     "reward × transition interaction isolates model-based control.",
        concepts=[("knowledge of the transition structure", ["transition", "structure", "model of the task", "map", "knows which", "usually lead", "likely to reach", "probab"]),
                  ("crediting the other / unchosen action", ["other action", "other option", "other choice", "unchosen", "alternative", "switch", "would have", "instead"]),
                  ("model-free credits the taken action", ["model-free", "habit", "cached", "just rewarded", "reinforce", "whatever was rewarded", "action it took", "repeat", "stay"])],
        min_concepts=2,
        hints=["Think about what each system concludes from a reward that followed a rare transition.",
               "The model-free learner credits the action it took; the model-based learner asks which action usually leads to where the reward was."],
        explanation="Experimental designs for dissociating learning systems work by finding trials on which the "
                    "systems make *opposite* predictions. In the two-step task those are the rare-transition trials.",
        check_call='check("9.10", """On a common transition both systems agree, because the action that was rewarded is also the action that usually leads to the rewarded state, so both predict a stay. On a rare transition the model-free learner still credits the action it just took and repeats it, while the model-based learner knows the transition structure and realises that the rewarded state is usually reached by the other first-stage action, so it switches. Rare trials are the only trials where the two systems disagree.""")',
    ),
]
