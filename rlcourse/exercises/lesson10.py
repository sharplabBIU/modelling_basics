"""Exercises of lesson 10: the successor representation."""
from __future__ import annotations

import numpy as np

from .. import models, tasks
from ..checker import ChoiceExercise, CodeExercise, FreeTextExercise, NumericExercise

# =========================================================================== #
# Reference: model-based (planning) values from a learned state-to-state model
# =========================================================================== #


def mb_values_from_model(T_hat, R_hat, gamma, n_iter=200):
    """Model-based values from a learned state-to-state transition model and
    reward vector: iterate the Bellman equation  V <- R + gamma * T V.

    Rewards are attached to states (received when the state is occupied); rows of
    ``T_hat`` belonging to terminal states are all zero, so their value is just
    their own reward.
    """
    T_hat = np.asarray(T_hat, dtype=float)
    R_hat = np.asarray(R_hat, dtype=float)
    V = np.zeros(len(R_hat))
    for _ in range(n_iter):
        V = R_hat + gamma * T_hat @ V          # one more step of look-ahead through the model
    return V


def _mb_no_reward(T_hat, R_hat, gamma, n_iter=200):
    V = np.zeros(len(R_hat))
    for _ in range(n_iter):
        V = gamma * np.asarray(T_hat, float) @ V
    return V


def _mb_transition_reward(T_hat, R_hat, gamma, n_iter=200):
    V = np.zeros(len(R_hat))
    for _ in range(n_iter):
        V = np.asarray(T_hat, float) @ (np.asarray(R_hat, float) + gamma * V)
    return V


def _mb_no_gamma(T_hat, R_hat, gamma, n_iter=200):
    V = np.zeros(len(R_hat))
    for _ in range(n_iter):
        V = np.asarray(R_hat, float) + np.asarray(T_hat, float) @ V
    return V


def _mb_transposed(T_hat, R_hat, gamma, n_iter=200):
    V = np.zeros(len(R_hat))
    for _ in range(n_iter):
        V = np.asarray(R_hat, float) + gamma * np.asarray(T_hat, float).T @ V
    return V


def _mb_diagnose(out, ref, case):
    try:
        o = np.asarray(out, float)
    except Exception:
        return None
    if o.shape == np.shape(ref) and np.allclose(o, 0):
        return "Every value is 0: did you forget to add the reward vector R_hat at each iteration?"
    return None


# buggy variants of the SR functions -----------------------------------------
def _sr_sign(T, gamma):
    n = T.shape[0]
    return np.linalg.inv(np.eye(n) + gamma * np.asarray(T, float))


def _sr_no_inverse(T, gamma):
    n = T.shape[0]
    return np.eye(n) - gamma * np.asarray(T, float)


def _sr_transposed(T, gamma):
    n = T.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * np.asarray(T, float).T)


def _sr_gamma_outside(T, gamma):
    n = T.shape[0]
    return gamma * np.linalg.inv(np.eye(n) - np.asarray(T, float))


def _srtd_no_onehot(M, s, s_next, alpha, gamma):
    M = np.array(M, float)
    M[s] = M[s] + alpha * (gamma * M[s_next] - M[s])
    return M


def _srtd_wrong_onehot(M, s, s_next, alpha, gamma):
    M = np.array(M, float)
    onehot = np.zeros(M.shape[0]); onehot[s_next] = 1.0
    M[s] = M[s] + alpha * (onehot + gamma * M[s_next] - M[s])
    return M


def _srtd_no_alpha(M, s, s_next, alpha, gamma):
    M = np.array(M, float)
    onehot = np.zeros(M.shape[0]); onehot[s] = 1.0
    M[s] = onehot + gamma * M[s_next]
    return M


def _srtd_self_target(M, s, s_next, alpha, gamma):
    M = np.array(M, float)
    onehot = np.zeros(M.shape[0]); onehot[s] = 1.0
    M[s] = M[s] + alpha * (onehot + gamma * M[s] - M[s])
    return M


def _srtd_wrong_row(M, s, s_next, alpha, gamma):
    M = np.array(M, float)
    onehot = np.zeros(M.shape[0]); onehot[s] = 1.0
    M[s_next] = M[s_next] + alpha * (onehot + gamma * M[s_next] - M[s_next])
    return M


def _srr_all(R_hat, s, r, alpha):
    R_hat = np.array(R_hat, float)
    return R_hat + alpha * (r - R_hat)


def _srr_raw(R_hat, s, r, alpha):
    R_hat = np.array(R_hat, float)
    R_hat[s] = R_hat[s] + alpha * r
    return R_hat


# test fixtures ---------------------------------------------------------------
_chain = tasks.random_walk_chain()
_branch = tasks.three_branch_graph()
_T_chain = _chain.transition_matrix()
_T_branch = _branch.transition_matrix()
_T_three = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])    # s0 -> s1 -> s2 (terminal)
_M_partial = np.eye(4) + 0.5 * np.round(np.random.default_rng(10).random((4, 4)), 2)
np.fill_diagonal(_M_partial, 1.0)

# =========================================================================== #
EXERCISES = [
    # ------------------------------------------------------------------ 10.1
    CodeExercise(
        id="10.1", title="The successor representation of a known transition matrix",
        reference=models.sr_analytic,
        signature="sr_analytic(T, gamma)",
        cases=[{"T": _T_three, "gamma": 0.9},
               {"T": _T_chain, "gamma": 0.9},
               {"T": _T_branch, "gamma": 0.9},
               {"T": _T_chain, "gamma": 1.0}],
        bug_variants={
            "sign": (_sr_sign, "The sign inside the inverse is wrong: M = (I − γT)^−1, with a *minus*."),
            "no_inverse": (_sr_no_inverse, "You returned I − γT itself; the SR is its matrix *inverse* (`np.linalg.inv`)."),
            "transposed": (_sr_transposed, "You transposed T. Row s of T holds the probabilities of the next state given s, and row s of M the expected occupancies starting from s."),
            "gamma_outside": (_sr_gamma_outside, "γ belongs inside the inverse: M = (I − γT)^−1 = I + γT + γ²T² + …"),
        },
        hints=["Summing the geometric series I + γT + γ²T² + … gives a closed form, exactly like 1 + x + x² + … = 1/(1 − x).",
               "The matrix version of 1/(1 − x) is the inverse of (I − γT).",
               "`np.linalg.inv(np.eye(n) - gamma * T)`"],
        explanation="M = Σ_k γ^k T^k = (I − γT)^−1: entry M[s, s'] is the expected discounted number of future visits "
                    "to s' when starting in s (counting the current state as a visit at k = 0). Every row sums to "
                    "1/(1 − γ) for an endless task; in an episodic task with absorbing terminal states the rows sum "
                    "to less.",
        common_mistakes="A plus instead of a minus, forgetting the inverse, or transposing T.",
        check_call='check("10.1", sr_analytic)',
    ),
    # ------------------------------------------------------------------ 10.2
    NumericExercise(
        id="10.2", title="An SR entry by hand",
        answer=0.81, atol=1e-3,
        distractors=[(0.9, "That is M[0, 1] (one step ahead). s2 is reached after *two* steps."),
                     (1.0, "You forgot the discount: an occupancy k steps in the future counts γ^k."),
                     (0.729, "That is γ³ — s2 is reached after two steps, not three."),
                     (2.71, "That is the whole row sum (1 + γ + γ²); the question asks for the single entry M[0, 2].")],
        hints=["M[s, s'] adds up γ^k over the future time steps k at which s' is occupied.",
               "Starting in s0, the agent is in s1 at k = 1 and in s2 at k = 2 (and stays there, but a terminal state ends the episode — its row of T is zero).",
               "γ² = 0.9²"],
        explanation="M[0, 2] = γ² = 0.81: s2 is occupied exactly once, two steps into the future. Check it against "
                    "`sr_analytic` for T = [[0,1,0],[0,0,1],[0,0,0]].",
        check_call='check("10.2", 0.81)',
    ),
    # ------------------------------------------------------------------ 10.3
    CodeExercise(
        id="10.3", title="Learning the SR by temporal-difference updates",
        reference=models.sr_td_update,
        signature="sr_td_update(M, s, s_next, alpha, gamma)",
        cases=[{"M": np.eye(7), "s": 0, "s_next": 2, "alpha": 0.5, "gamma": 0.9},
               {"M": models.sr_analytic(_T_chain, 0.9), "s": 3, "s_next": 4, "alpha": 0.1, "gamma": 0.9},
               {"M": _M_partial, "s": 1, "s_next": 3, "alpha": 0.3, "gamma": 0.95},
               {"M": _M_partial, "s": 2, "s_next": 2, "alpha": 0.3, "gamma": 0.5}],
        bug_variants={
            "no_onehot": (_srtd_no_onehot, "The target is missing the one-hot vector of the current state: being in s counts as a visit to s now (k = 0)."),
            "wrong_onehot": (_srtd_wrong_onehot, "The one-hot vector should mark the *current* state s, not s_next (the future visits to s_next are already in γ M[s_next])."),
            "no_alpha": (_srtd_no_alpha, "You replaced the row by the target. Move it a fraction alpha of the way: M[s] += alpha × (target − M[s])."),
            "self_target": (_srtd_self_target, "The bootstrap term must use the row of the *next* state: target = onehot(s) + γ M[s_next]."),
            "wrong_row": (_srtd_wrong_row, "You updated the row of s_next. The transition s → s_next teaches us about the future of s, so update M[s]."),
        },
        hints=["This is a Rescorla–Wagner update of a whole row: target − M[s], scaled by alpha, where the target is what the row *should* look like after this transition.",
               "The target is one-hot(s) + γ × M[s_next]: 'I am in s now, and from s_next onwards the future looks like M[s_next], discounted'.",
               "`onehot = np.zeros(n); onehot[s] = 1; M[s] = M[s] + alpha * (onehot + gamma * M[s_next] - M[s])` on a copy of M."],
        explanation="The SR obeys a Bellman equation M[s] = onehot(s) + γ Σ_s' T[s, s'] M[s'], so it can be learned by "
                    "TD exactly like a value function — the 'reward' is a one-hot indicator of the current state "
                    "(Dayan 1993). Only the row of the state just left is updated, and terminal rows stay one-hot "
                    "because no transition ever leaves them.",
        common_mistakes="Forgetting the one-hot term, using the wrong one-hot, replacing instead of nudging the row, "
                        "or updating the row of the next state.",
        check_call='check("10.3", sr_td_update)',
    ),
    # ------------------------------------------------------------------ 10.4
    CodeExercise(
        id="10.4", title="Learning the reward of each state",
        reference=models.sr_reward_update,
        signature="sr_reward_update(R_hat, s, r, alpha)",
        cases=[{"R_hat": np.zeros(7), "s": 5, "r": 10.0, "alpha": 0.3},
               {"R_hat": np.array([0.0, 0.0, 0.0, 0.0, 2.0, 7.0, 0.0]), "s": 5, "r": 0.0, "alpha": 0.5},
               {"R_hat": np.array([1.0, 2.0, 3.0]), "s": 0, "r": 1.0, "alpha": 0.2}],
        bug_variants={
            "all": (_srr_all, "You updated every state. Only the state in which the reward was received tells us about its own reward."),
            "raw": (_srr_raw, "You added alpha × r. Use the prediction error r − R_hat[s], otherwise the estimate grows without bound."),
        },
        hints=["This is the Rescorla–Wagner rule of lesson 3 applied to one entry of a vector.",
               "`R_hat[s] = R_hat[s] + alpha * (r - R_hat[s])` on a copy of R_hat."],
        explanation="The SR factorises value into a *where will I be* part (M) and a *what is each place worth* part "
                    "(R̂). Because the two are learned separately, a change in rewards needs only a change in R̂ — "
                    "the key to the SR's flexibility under reward revaluation.",
        common_mistakes="Updating all entries, or using the raw reward instead of the prediction error.",
        check_call='check("10.4", sr_reward_update)',
    ),
    # ------------------------------------------------------------------ 10.6
    NumericExercise(
        id="10.6", title="A value after reward revaluation",
        answer=8.1, atol=1e-3,
        distractors=[(10.0, "You forgot the discount: C3 is two steps ahead of C1, so its reward counts γ² times."),
                     (9.0, "That uses γ instead of γ²: C3 is two steps after C1."),
                     (0.0, "That is the value *before* revaluation. The SR agent recomputes V = M R̂ with the *new* reward vector."),
                     (0.81, "That is the SR entry M[C1, C3]; multiply it by the new reward of C3.")],
        hints=["V(s) = Σ_s' M[s, s'] R̂(s'): only one state now has a non-zero reward.",
               "V(C1) = M[C1, C3] × R̂(C3) = 0.81 × 10"],
        explanation="V(C1) = 0.81 × 10 = 8.1 immediately after the reward vector is updated, without any new experience "
                    "of the path C1 → C2 → C3. A model-free learner would have to walk the path again to learn this.",
        check_call='check("10.6", 8.1)',
    ),
    # ------------------------------------------------------------------ 10.5
    CodeExercise(
        id="10.5", title="Model-based values from a learned model",
        reference=mb_values_from_model,
        signature="mb_values_from_model(T_hat, R_hat, gamma, n_iter=200)",
        cases=[{"T_hat": _T_branch, "R_hat": _branch.R, "gamma": 0.9},
               {"T_hat": _T_branch, "R_hat": np.array([0, 0, 0, 0, 0, 0, 10.0]), "gamma": 0.9},
               {"T_hat": _T_chain, "R_hat": _chain.R, "gamma": 0.95},
               {"T_hat": _T_three, "R_hat": np.array([0.0, 1.0, 5.0]), "gamma": 0.5}],
        bug_variants={
            "transition_reward": (_mb_transition_reward, "You used the 'reward on entering' convention V = T (R + γ V). In this lesson rewards are attached "
                                                         "to states, so that the SR and the model agree: V = R + γ T V."),
            "no_gamma": (_mb_no_gamma, "The discount factor is missing: V = R + γ T V."),
            "transposed": (_mb_transposed, "You transposed T_hat: row s of T_hat gives the next-state probabilities from s."),
        },
        diagnose=_mb_diagnose,
        hints=["Start from V = 0 and repeat: V ← R_hat + γ T_hat @ V. Each iteration looks one step further into the future.",
               "After enough iterations V equals (I − γ T_hat)^−1 R_hat — the same as the SR solution when T_hat is the true T.",
               "`for _ in range(n_iter): V = R_hat + gamma * T_hat @ V`"],
        explanation="A model-based agent does not store values: it recomputes them from its model whenever it needs "
                    "them. Iterating V ← R̂ + γ T̂ V is policy evaluation through the model; the SR agent instead "
                    "caches the result of the iteration over T (the matrix M) and multiplies it by R̂. The two agree "
                    "as long as T̂ and M describe the same world — they come apart when the transitions change.",
        common_mistakes="Forgetting R̂ (all values stay 0), forgetting γ, or using the transition-reward convention.",
        check_call='check("10.5", mb_values_from_model)',
    ),
    # ------------------------------------------------------------------ 10.7
    ChoiceExercise(
        id="10.7", title="Who adapts to which revaluation?",
        question="After the *transition* revaluation (second-stage states lead to different terminal states than before, experienced only from the second-stage states), which agents choose the correct first-stage state at test?",
        options={"a": "Only the model-based agent.",
                 "b": "The model-based and the successor-representation agents.",
                 "c": "All three agents.",
                 "d": "Only the successor-representation agent."},
        correct="a",
        feedback={"b": "The SR agent updates the rows of the second-stage states it experiences, but the rows of the first-stage "
                       "states — which it needs at test — still predict the old terminal states. Look at M[A1] after the revaluation.",
                  "c": "The model-free agent never experiences a first-stage state during the revaluation, so its cached first-stage values cannot change.",
                  "d": "The SR fails here; it only succeeds under *reward* revaluation, where the change is confined to R̂."},
        hints=["Which quantity does each agent use at test, and was that quantity touched by the revaluation experience?",
               "The SR of a first-stage state is only updated when the agent *leaves* that state; during the revaluation it never does."],
        explanation="Transition revaluation requires re-deriving the consequences of a first-stage choice from updated "
                    "second-stage transitions. Only an agent that recomputes values from a model can do this; the SR "
                    "caches the old consequences in M[A1] and M[B1], and the model-free agent caches old values. This "
                    "is the signature Momennejad et al. (2017) used to show that people are not pure SR learners — "
                    "they are partly sensitive to transition revaluation.",
        check_call='check("10.7", "a")',
    ),
    # ------------------------------------------------------------------ 10.8
    FreeTextExercise(
        id="10.8", title="The SR: a bit of both",
        question="In 2–4 sentences: in what sense is the successor representation model-based-like, and in what sense is it model-free-like?",
        model_answer="The SR is model-based-like because it separates knowledge about *where you will end up* (M) from "
                     "knowledge about *what each place is worth* (R̂), so a change in rewards can be absorbed by updating "
                     "R̂ and values are recomputed as M R̂ without new experience of the paths — like a planner that "
                     "re-evaluates the consequences it already knows. It is model-free-like because M is itself a "
                     "cached quantity, learned by temporal-difference prediction errors under the policy the agent "
                     "followed: it is not re-derived from a one-step model, so when transitions or the policy change the "
                     "cached predictions are stale until the relevant states are revisited, exactly like cached "
                     "model-free values.",
        concepts=[("separates rewards from state predictions (flexible to reward change)", ["reward", "revalu", "separat", "factor", "decoupl", "recompute", "re-comput", "without new experience", "multipl"]),
                  ("cached / learned by TD", ["cache", "cached", "temporal", "td", "prediction error", "learned from experience", "stored", "stale", "not re-derived", "not recomputed"]),
                  ("inflexible to transition or policy change", ["transition", "policy", "inflexib", "fail", "stale", "revisit", "structure"])],
        min_concepts=2,
        hints=["Which part of the value computation does the SR recompute at decision time, and which part does it cache?",
               "Think of reward revaluation (where the SR succeeds) and transition revaluation (where it fails)."],
        explanation="The SR sits between the two classical systems: predictive like a model, cached like a habit "
                    "(Dayan 1993; Russek et al. 2017; Gershman 2018).",
        check_call='check("10.8", """The SR is model-based-like because it separates where you will end up (M) from what each place is worth (R), so a change in rewards is handled by updating R and recomputing V = M R without new experience of the paths. It is model-free-like because M is a cached quantity learned by temporal-difference prediction errors under the policy the agent followed, so when transitions or the policy change the cached predictions are stale until the states are revisited.""")',
    ),
    # ------------------------------------------------------------------ 10.9
    ChoiceExercise(
        id="10.9", title="Policy dependence",
        question="An agent learned its SR while exploring a maze at random. It then starts choosing greedily. What is true of the SR rows it learned?",
        options={"a": "Nothing changes: the SR depends only on the transition structure of the maze.",
                 "b": "They are now wrong for the new behaviour: the SR encodes expected future occupancies *under the policy that generated the experience*.",
                 "c": "They are instantly recomputed, because the SR is a model.",
                 "d": "They collapse to the identity matrix."},
        correct="b",
        feedback={"a": "The SR is the discounted occupancy under a *policy*: T in M = (I − γT)^−1 is the state-to-state matrix induced by that policy, not the raw maze.",
                  "c": "The SR is not a one-step model; it is a cached long-run prediction, updated only by new experience (TD).",
                  "d": "The identity is only the *initial* SR of an agent that has not yet experienced any transition."},
        hints=["Where does the policy enter the definition M = (I − γT)^−1?",
               "T[s, s'] = Σ_a π(a|s) P(s'|s, a): change π and you change T, and therefore M."],
        explanation="The SR is policy-dependent. This is a strength (it is what makes it a cheap evaluation of the "
                    "current policy) and a weakness: after a change of policy, or of the transitions, the cached "
                    "predictions are stale until re-learned. Russek et al. (2017) discuss how off-line replay (SR-Dyna) "
                    "can repair this.",
        check_call='check("10.9", "b")',
    ),
]
