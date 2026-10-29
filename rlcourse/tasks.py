"""Simulated experimental tasks.

All tasks take a ``rng`` (``numpy.random.Generator``) so that simulations are
reproducible: ``rng = np.random.default_rng(0)``.

Tasks
-----
Bandit            k-armed bandit with fixed reward probabilities (Bernoulli) or
                  Gaussian rewards.
RestlessBandit    reward probabilities drift as bounded random walks
                  (as in Daw et al. 2006, Nature; Speekenbrink & Konstantinidis 2015).
ReversalTask      two options whose reward contingencies swap at reversal
                  points; ``ReversalTask.stable_volatile`` builds the stable /
                  volatile block design of Behrens et al. (2007, Nat Neurosci).
TwoStepTask       the two-stage Markov decision task of Daw et al. (2011, Neuron).
MDP               a generic small tabular Markov decision process, with
                  ready-made ``random_walk_chain`` (Sutton & Barto 2018, Ex. 6.2),
                  ``three_branch_graph`` (a revaluation design in the spirit of
                  Momennejad et al. 2017) and ``gridworld``.
PavlovianTrace    a cue followed by a reward after a delay, for the
                  temporal-difference / dopamine demonstrations
                  (Schultz, Dayan & Montague 1997; Sutton & Barto 1990).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _rng(rng):
    return np.random.default_rng() if rng is None else rng


# --------------------------------------------------------------------------- #
class Bandit:
    """A k-armed bandit.

    Parameters
    ----------
    probs : sequence of float
        Reward probability of each arm (Bernoulli rewards, 0 or 1).  If
        ``reward_type="gaussian"`` these are the *means* instead and rewards are
        drawn from N(mean, sd).
    """

    def __init__(self, probs, rng=None, reward_type: str = "bernoulli", sd: float = 1.0):
        self.probs = np.asarray(probs, dtype=float)
        self.rng = _rng(rng)
        self.reward_type = reward_type
        self.sd = sd
        self.t = 0

    @property
    def n_arms(self) -> int:
        return len(self.probs)

    def step(self, action: int) -> float:
        """Pull arm ``action`` (0-based) and return the reward."""
        self.t += 1
        if self.reward_type == "bernoulli":
            return float(self.rng.random() < self.probs[action])
        return float(self.rng.normal(self.probs[action], self.sd))

    def reset(self) -> None:
        self.t = 0


class RestlessBandit:
    """Bandit whose reward probabilities drift as bounded Gaussian random walks.

    ``probs`` has shape (n_trials, n_arms) and is generated at construction so
    that every agent run on the same task sees the same environment.
    """

    def __init__(self, n_arms: int = 2, n_trials: int = 200, sd: float = 0.03,
                 bounds=(0.2, 0.8), init_probs=None, rng=None):
        self.rng = _rng(rng)
        self.n_trials = n_trials
        lo, hi = bounds
        p = np.empty((n_trials, n_arms))
        p[0] = self.rng.uniform(lo, hi, size=n_arms) if init_probs is None else np.asarray(init_probs, float)
        for t in range(1, n_trials):
            step = self.rng.normal(0, sd, size=n_arms)
            p[t] = _reflect(p[t - 1] + step, lo, hi)
        self.probs = p
        self.t = 0

    @property
    def n_arms(self) -> int:
        return self.probs.shape[1]

    def step(self, action: int) -> float:
        r = float(self.rng.random() < self.probs[self.t, action])
        self.t += 1
        return r

    def reset(self) -> None:
        self.t = 0


def _reflect(x, lo, hi):
    """Reflect values back into [lo, hi] (keeps random walks inside the bounds)."""
    x = np.asarray(x, float).copy()
    for _ in range(3):
        x = np.where(x > hi, 2 * hi - x, x)
        x = np.where(x < lo, 2 * lo - x, x)
    return np.clip(x, lo, hi)


# --------------------------------------------------------------------------- #
class ReversalTask:
    """Two options; the probability that option 0 is the *good* one swaps at
    reversal points.

    Parameters
    ----------
    schedule : array of shape (n_trials,)
        ``p_good_is_0[t]`` = probability that option 0 pays off on trial *t*.
        Option 1 pays off with probability ``1 - p`` (Behrens-style
        full-information design: the outcome tells you which option was
        rewarded) when ``full_information=True``; otherwise each option is
        rewarded independently with its own probability (``p`` and ``1 - p``).
    """

    def __init__(self, schedule, rng=None, full_information: bool = True):
        self.schedule = np.asarray(schedule, dtype=float)
        self.rng = _rng(rng)
        self.full_information = full_information
        self.t = 0

    @property
    def n_trials(self) -> int:
        return len(self.schedule)

    @classmethod
    def from_reversals(cls, n_trials: int, reversal_points, p_good: float = 0.8, start_good: int = 0,
                       rng=None, full_information: bool = True) -> "ReversalTask":
        sched = np.full(n_trials, p_good if start_good == 0 else 1 - p_good)
        flip = start_good
        for t in range(n_trials):
            if t in set(reversal_points):
                flip = 1 - flip
            sched[t] = p_good if flip == 0 else 1 - p_good
        return cls(sched, rng=rng, full_information=full_information)

    @classmethod
    def stable_volatile(cls, n_stable: int = 120, n_volatile: int = 120, block_len: int = 25,
                        p_good: float = 0.75, volatile_first: bool = False, rng=None,
                        full_information: bool = True) -> "ReversalTask":
        """Behrens et al. (2007) design: a stable block (no reversals) and a
        volatile block (a reversal every ``block_len`` trials)."""
        stable = np.full(n_stable, p_good)
        vol = np.empty(n_volatile)
        good = 1 if not volatile_first else 0
        for t in range(n_volatile):
            if t % block_len == 0:
                good = 1 - good
            vol[t] = p_good if good == 0 else 1 - p_good
        sched = np.concatenate([vol, stable]) if volatile_first else np.concatenate([stable, vol])
        task = cls(sched, rng=rng, full_information=full_information)
        task.block = np.array(["volatile"] * n_volatile + ["stable"] * n_stable) if volatile_first \
            else np.array(["stable"] * n_stable + ["volatile"] * n_volatile)
        return task

    def step(self, action: int):
        """Returns ``(reward, outcome)`` where ``outcome`` is 1 if option 0 was
        the rewarded option on this trial (only meaningful with full information)."""
        p = self.schedule[self.t]
        if self.full_information:
            outcome = int(self.rng.random() < p)  # 1: option 0 rewarded, 0: option 1 rewarded
            reward = float(outcome == 1) if action == 0 else float(outcome == 0)
        else:
            outcome = int(self.rng.random() < (p if action == 0 else 1 - p))
            reward = float(outcome)
        self.t += 1
        return reward, outcome

    def reset(self) -> None:
        self.t = 0


# --------------------------------------------------------------------------- #
class TwoStepTask:
    """The two-stage decision task of Daw et al. (2011).

    * Stage 1: one state, two actions.  Action 0 leads to stage-2 state 0 with
      probability ``p_common`` (a *common* transition) and to state 1 otherwise
      (*rare*); action 1 is the mirror image.
    * Stage 2: two states, each with two actions, rewarded (1) or not (0) with
      probabilities that drift as independent Gaussian random walks
      (sd ``drift_sd``) reflected at ``bounds`` — the original used sd = 0.025
      within [0.25, 0.75].

    ``reward_probs`` has shape (n_trials, 2 states, 2 actions).
    """

    def __init__(self, n_trials: int = 200, p_common: float = 0.7, drift_sd: float = 0.025,
                 bounds=(0.25, 0.75), init_probs=None, rng=None):
        self.rng = _rng(rng)
        self.n_trials = n_trials
        self.p_common = p_common
        lo, hi = bounds
        p = np.empty((n_trials, 2, 2))
        p[0] = self.rng.uniform(lo, hi, size=(2, 2)) if init_probs is None else np.asarray(init_probs, float)
        for t in range(1, n_trials):
            p[t] = _reflect(p[t - 1] + self.rng.normal(0, drift_sd, size=(2, 2)), lo, hi)
        self.reward_probs = p
        # transition matrix P[a, s2]
        self.transitions = np.array([[p_common, 1 - p_common], [1 - p_common, p_common]])
        self.t = 0

    def step1(self, action: int):
        """Take a stage-1 action; returns ``(state2, is_common)``."""
        common = self.rng.random() < self.p_common
        state2 = action if common else 1 - action
        self._state2 = state2
        return int(state2), bool(common)

    def step2(self, action: int) -> float:
        """Take a stage-2 action in the current stage-2 state; returns the reward
        and advances to the next trial."""
        r = float(self.rng.random() < self.reward_probs[self.t, self._state2, action])
        self.t += 1
        return r

    def reset(self) -> None:
        self.t = 0


# --------------------------------------------------------------------------- #
@dataclass
class MDP:
    """A small tabular Markov decision process.

    Attributes
    ----------
    P : array (n_states, n_actions, n_states)
        Transition probabilities ``P[s, a, s']``.
    R : array (n_states,)
        Reward received on *entering* each state.
    terminal : array of bool (n_states,)
    start : int or sequence of int
        Start state(s); a random one is chosen on ``reset``.
    names : list of str
        Optional state labels (for plots).
    """

    P: np.ndarray
    R: np.ndarray
    terminal: np.ndarray
    start: object = 0
    names: list | None = None

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    def reset(self, rng=None) -> int:
        rng = _rng(rng)
        starts = np.atleast_1d(self.start)
        self.state = int(rng.choice(starts))
        return self.state

    def step(self, action: int, rng=None):
        """Returns ``(next_state, reward, done)``."""
        rng = _rng(rng)
        s2 = int(rng.choice(self.n_states, p=self.P[self.state, action]))
        r = float(self.R[s2])
        done = bool(self.terminal[s2])
        self.state = s2
        return s2, r, done

    def transition_matrix(self, policy=None) -> np.ndarray:
        """State-to-state transition matrix ``T[s, s']`` under a policy
        (array (n_states, n_actions) of action probabilities; uniform if None)."""
        if policy is None:
            policy = np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)
        T = np.einsum("sa,sat->st", policy, self.P)
        # terminal states are absorbing (the SR of an episodic task needs this)
        for s in np.where(self.terminal)[0]:
            T[s] = 0.0
        return T


def random_walk_chain(n_states: int = 5) -> MDP:
    """Sutton & Barto (2018) Example 6.2: a chain A-B-C-D-E, starting in the middle,
    moving left or right at random; the right-hand exit pays 1, the left 0.
    State indices: 0 = left terminal, 1..n = chain, n+1 = right terminal."""
    n = n_states + 2
    P = np.zeros((n, 2, n))
    for s in range(1, n - 1):
        P[s, 0, s - 1] = 1.0  # action 0 = left
        P[s, 1, s + 1] = 1.0  # action 1 = right
    P[0, :, 0] = 1.0
    P[n - 1, :, n - 1] = 1.0
    R = np.zeros(n)
    R[n - 1] = 1.0
    terminal = np.zeros(n, bool)
    terminal[[0, n - 1]] = True
    names = ["L"] + [chr(ord("A") + i) for i in range(n_states)] + ["R"]
    return MDP(P, R, terminal, start=(n_states + 1) // 2, names=names)


def three_branch_graph(rewards=(0.0, 10.0, 0.0), p_intended: float = 1.0) -> MDP:
    """A start state with three actions, each leading (with probability
    ``p_intended``) to its own intermediate state, which leads deterministically
    to its own terminal state carrying ``rewards[i]``.  Used for the reward /
    transition revaluation demonstrations (cf. Momennejad et al. 2017)."""
    n = 7  # 0 start, 1-3 intermediate, 4-6 terminal
    P = np.zeros((n, 3, n))
    others = 0.0 if p_intended >= 1.0 else (1 - p_intended) / 2
    for a in range(3):
        for s_mid in range(1, 4):
            P[0, a, s_mid] = p_intended if s_mid == a + 1 else others
    for i in range(1, 4):
        P[i, :, i + 3] = 1.0
    for s in range(4, 7):
        P[s, :, s] = 1.0
    R = np.zeros(n)
    R[4:7] = rewards
    terminal = np.zeros(n, bool)
    terminal[4:7] = True
    names = ["start", "mid-A", "mid-B", "mid-C", "end-A", "end-B", "end-C"]
    return MDP(P, R, terminal, start=0, names=names)


def gridworld(n_rows: int = 4, n_cols: int = 4, goal=(0, 3), start=(3, 0), goal_reward: float = 1.0,
              walls=()) -> MDP:
    """A deterministic grid world with actions up/right/down/left."""
    n = n_rows * n_cols
    idx = lambda r, c: r * n_cols + c  # noqa: E731
    P = np.zeros((n, 4, n))
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    wall_set = {idx(*w) for w in walls}
    for r in range(n_rows):
        for c in range(n_cols):
            s = idx(r, c)
            for a, (dr, dc) in enumerate(moves):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < n_rows and 0 <= c2 < n_cols and idx(r2, c2) not in wall_set:
                    P[s, a, idx(r2, c2)] = 1.0
                else:
                    P[s, a, s] = 1.0
    R = np.zeros(n)
    R[idx(*goal)] = goal_reward
    terminal = np.zeros(n, bool)
    terminal[idx(*goal)] = True
    for w in wall_set:
        P[w, :, :] = 0
        P[w, :, w] = 1
    names = [f"({r},{c})" for r in range(n_rows) for c in range(n_cols)]
    mdp = MDP(P, R, terminal, start=idx(*start), names=names)
    mdp.shape = (n_rows, n_cols)
    return mdp


# --------------------------------------------------------------------------- #
class PavlovianTrace:
    """A conditioning trial represented from cue (CS) onset: ``n_steps`` time
    steps, with a reward (US) of size ``reward`` at step ``t_us``.  Each time
    step after the cue is a state (a *complete serial compound* / tapped delay
    line representation, as in Montague, Dayan & Sejnowski 1996)."""

    def __init__(self, n_steps: int = 15, t_us: int = 10, reward: float = 1.0):
        self.n_steps, self.t_us, self.reward = n_steps, t_us, reward

    def rewards(self, omit: bool = False, t_us=None, size=None) -> np.ndarray:
        """Reward at every time step of one trial (optionally omitted, shifted or resized)."""
        r = np.zeros(self.n_steps)
        if not omit:
            r[self.t_us if t_us is None else t_us] = self.reward if size is None else size
        return r


__all__ = ["Bandit", "RestlessBandit", "ReversalTask", "TwoStepTask", "MDP",
           "random_walk_chain", "three_branch_graph", "gridworld", "PavlovianTrace"]
