"""Reference implementations of every model taught in the course.

The functions here are deliberately written the way a student would write
them (explicit loops, no tricks): when you ask the checker to reveal a
solution, it shows the source code of the corresponding function below.

Conventions used throughout the course
--------------------------------------
* ``choices`` are integer arrays (0-based), ``rewards`` are float arrays.
* ``alpha`` = learning rate, ``beta`` = inverse temperature, ``gamma`` =
  discount factor, ``lam`` = eligibility-trace parameter, ``w`` = model-based
  weight, ``kappa`` = choice stickiness.
* Simulations take ``rng`` (a ``numpy.random.Generator``) and sample choices
  with :func:`choose`.
* Likelihood functions return the *negative* log-likelihood (``nll``) and take
  the parameters as the first argument, in the order given in their docstring.
"""
from __future__ import annotations

import numpy as np

# =========================================================================== #
# Basics
# =========================================================================== #


def sigmoid(x):
    """Logistic function 1 / (1 + exp(-x)), numerically safe for large |x|."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out if out.ndim else float(out)


def logit(p):
    """Inverse of the sigmoid: log(p / (1 - p))."""
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def softmax(values, beta=1.0):
    """Softmax choice probabilities  p_i = exp(beta*v_i) / sum_j exp(beta*v_j).

    The maximum is subtracted before exponentiating, which does not change the
    result but prevents overflow for large beta.
    """
    x = beta * np.asarray(values, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def choose(probs, rng):
    """Sample one option (0-based index) from a vector of probabilities."""
    return int(rng.choice(len(probs), p=probs))


def running_mean(x):
    """Running (cumulative) mean of a sequence: out[t] = mean(x[:t+1])."""
    x = np.asarray(x, dtype=float)
    return np.cumsum(x) / np.arange(1, len(x) + 1)


# =========================================================================== #
# Lesson 2: probability and Bayes
# =========================================================================== #


def bernoulli_log_likelihood(p, outcomes):
    """log p(outcomes | p) for independent Bernoulli(p) outcomes (0/1)."""
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.sum(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def bernoulli_likelihood(p, outcomes):
    """p(outcomes | p) = prod_t p^o_t (1-p)^(1-o_t)."""
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.prod(p ** outcomes * (1 - p) ** (1 - outcomes)))


def grid_posterior(prior, likelihood):
    """Bayes' rule on a grid: posterior ∝ prior × likelihood, normalised to sum to 1."""
    unnormalised = np.asarray(prior, float) * np.asarray(likelihood, float)
    return unnormalised / np.sum(unnormalised)


def beta_posterior(a0, b0, outcomes):
    """Conjugate update of a Beta(a0, b0) prior with Bernoulli outcomes.

    Returns the posterior parameters (a, b) = (a0 + #successes, b0 + #failures).
    """
    outcomes = np.asarray(outcomes, dtype=float)
    n_success = np.sum(outcomes)
    n_failure = len(outcomes) - n_success
    return a0 + n_success, b0 + n_failure


def beta_mean(a, b):
    """Mean of a Beta(a, b) distribution."""
    return a / (a + b)


def beta_variance(a, b):
    """Variance of a Beta(a, b) distribution."""
    return a * b / ((a + b) ** 2 * (a + b + 1))


# =========================================================================== #
# Lesson 3: Rescorla–Wagner / delta rule on bandits
# =========================================================================== #


def rw_update(value, reward, alpha):
    """One Rescorla–Wagner (delta-rule) update:  V <- V + alpha * (R - V)."""
    prediction_error = reward - value
    return value + alpha * prediction_error


def epsilon_greedy_probs(values, epsilon):
    """Choice probabilities of an epsilon-greedy policy (ties share the greedy mass)."""
    values = np.asarray(values, dtype=float)
    k = len(values)
    best = values == values.max()
    probs = np.full(k, epsilon / k)
    probs[best] += (1 - epsilon) / best.sum()
    return probs


def simulate_rw_bandit(probs, alpha, beta, n_trials, rng, q0=0.5):
    """Simulate a Rescorla–Wagner learner with softmax choices on a bandit.

    Parameters
    ----------
    probs : reward probability of each arm
    alpha, beta : learning rate and inverse temperature
    q0 : initial value of every arm

    Returns a dict with ``choices`` (int array), ``rewards`` (float array) and
    ``Q`` (array (n_trials, n_arms): the values *before* each choice).
    """
    probs = np.asarray(probs, dtype=float)
    n_arms = len(probs)
    Q = np.full(n_arms, q0, dtype=float)
    choices = np.zeros(n_trials, dtype=int)
    rewards = np.zeros(n_trials)
    Q_history = np.zeros((n_trials, n_arms))
    for t in range(n_trials):
        Q_history[t] = Q                      # record values before the choice
        p = softmax(Q, beta)                  # choice probabilities
        c = choose(p, rng)                    # sample a choice
        r = float(rng.random() < probs[c])    # sample a reward
        Q[c] = rw_update(Q[c], r, alpha)      # learn (only the chosen arm)
        choices[t], rewards[t] = c, r
    return {"choices": choices, "rewards": rewards, "Q": Q_history}


def simulation_is_consistent(sim, probs, alpha, beta, q0=0.5, tol=1e-8):
    """Check that a simulated data set obeys the RW + softmax model exactly
    (used by the checker: it does not depend on how random numbers were drawn)."""
    try:
        choices = np.asarray(sim["choices"]).astype(int)
        rewards = np.asarray(sim["rewards"], float)
        Q = np.asarray(sim["Q"], float)
    except Exception:
        return False
    n_trials, n_arms = Q.shape
    if len(choices) != n_trials or len(rewards) != n_trials or n_arms != len(probs):
        return False
    if not np.allclose(Q[0], q0):
        return False
    if not set(np.unique(rewards)).issubset({0.0, 1.0}):
        return False
    for t in range(n_trials - 1):
        expected = Q[t].copy()
        expected[choices[t]] = rw_update(Q[t, choices[t]], rewards[t], alpha)
        if not np.allclose(Q[t + 1], expected, atol=tol):
            return False
    return True


# =========================================================================== #
# Lesson 4: temporal-difference learning
# =========================================================================== #


def td0_update(V, s, r, s_next, alpha, gamma=1.0, done=False):
    """TD(0) prediction update for state values.  Returns a *new* array."""
    V = np.array(V, dtype=float)
    target = r if done else r + gamma * V[s_next]
    delta = target - V[s]
    V[s] = V[s] + alpha * delta
    return V


def sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma=1.0, done=False):
    """SARSA (on-policy TD control) update.  Returns a *new* array."""
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * Q[s_next, a_next]
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def q_learning_update(Q, s, a, r, s_next, alpha, gamma=1.0, done=False):
    """Q-learning (off-policy TD control) update.  Returns a *new* array."""
    Q = np.array(Q, dtype=float)
    target = r if done else r + gamma * np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
    return Q


def td_lambda_episode(V, states, rewards, alpha, gamma=1.0, lam=0.0):
    """TD(lambda) with accumulating eligibility traces over one episode.

    ``states`` = [s_0, s_1, ..., s_T] (s_T terminal), ``rewards`` = [r_1, ..., r_T].
    Returns the updated value array.
    """
    V = np.array(V, dtype=float)
    e = np.zeros_like(V)
    for t in range(len(rewards)):
        s, s_next, r = states[t], states[t + 1], rewards[t]
        done = t == len(rewards) - 1
        delta = (r if done else r + gamma * V[s_next]) - V[s]
        e[s] += 1.0                 # accumulate the trace of the visited state
        V = V + alpha * delta * e   # every eligible state gets a share of the error
        e = gamma * lam * e         # traces decay
    return V


def run_td_prediction(mdp, n_episodes, alpha, gamma=1.0, rng=None, policy=None, V0=0.5,
                      lam=None):
    """Run TD(0) (or TD(lambda) if ``lam`` is given) on an MDP under a policy.

    Returns the value history, shape (n_episodes + 1, n_states)."""
    rng = np.random.default_rng() if rng is None else rng
    V = np.full(mdp.n_states, V0, dtype=float)
    V[mdp.terminal] = 0.0
    history = [V.copy()]
    for _ in range(n_episodes):
        s = mdp.reset(rng)
        states, rewards = [s], []
        done = False
        while not done:
            p = np.full(mdp.n_actions, 1 / mdp.n_actions) if policy is None else policy[s]
            a = choose(p, rng)
            s_next, r, done = mdp.step(a, rng)
            if lam is None:
                V = td0_update(V, s, r, s_next, alpha, gamma, done)
            states.append(s_next)
            rewards.append(r)
            s = s_next
        if lam is not None:
            V = td_lambda_episode(V, states, rewards, alpha, gamma, lam)
        history.append(V.copy())
    return np.array(history)


def run_td_control(mdp, n_episodes, alpha, gamma=1.0, epsilon=0.1, rng=None, algorithm="q_learning",
                   max_steps=500):
    """SARSA or Q-learning with epsilon-greedy exploration on an MDP.

    Returns ``(Q, returns)`` where ``returns[i]`` is the total reward of episode i."""
    rng = np.random.default_rng() if rng is None else rng
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s = mdp.reset(rng)
        a = choose(epsilon_greedy_probs(Q[s], epsilon), rng)
        done, total, steps = False, 0.0, 0
        while not done and steps < max_steps:
            s_next, r, done = mdp.step(a, rng)
            a_next = choose(epsilon_greedy_probs(Q[s_next], epsilon), rng)
            if algorithm == "sarsa":
                Q = sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma, done)
            else:
                Q = q_learning_update(Q, s, a, r, s_next, alpha, gamma, done)
            s, a = s_next, a_next
            total += r
            steps += 1
        returns[ep] = total
    return Q, returns


def td_pavlovian(reward_sequences, alpha, gamma=1.0):
    """TD(0) learning of time-step values across Pavlovian conditioning trials
    (the simulation of Schultz, Dayan & Montague 1997 / Montague et al. 1996).

    ``reward_sequences`` is a list of arrays (one per trial) giving the reward at
    every time step from cue onset onwards (time step = state; see
    ``tasks.PavlovianTrace``).  Before the cue nothing is predicted (value 0),
    and after the last time step the trial ends (value 0).

    The TD error at time step t is  delta_t = r_t + gamma * V[t] - V[t-1]  and it
    updates the value of the *previous* time step, V[t-1].  Returns
    ``(V_history, delta_history)``: values before each trial (n_trials + 1, n_steps)
    and prediction errors on each trial (n_trials, n_steps + 1; the last entry is
    the error at trial end).
    """
    n_steps = len(reward_sequences[0])
    V = np.zeros(n_steps)
    V_history, delta_history = [V.copy()], []
    for rewards in reward_sequences:
        deltas = np.zeros(n_steps + 1)
        v_prev = 0.0                                   # no prediction before the cue
        for t in range(n_steps):
            deltas[t] = rewards[t] + gamma * V[t] - v_prev
            if t > 0:
                V[t - 1] = V[t - 1] + alpha * deltas[t]  # the error trains the preceding step
            v_prev = V[t]
        deltas[n_steps] = 0.0 - v_prev                 # trial ends: nothing follows
        V[n_steps - 1] = V[n_steps - 1] + alpha * deltas[n_steps]
        V_history.append(V.copy())
        delta_history.append(deltas)
    return np.array(V_history), np.array(delta_history)


def actor_critic_update(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma=1.0, done=False):
    """One actor–critic step with a softmax actor (``theta`` = action preferences).

    The critic's TD error trains both the state value (critic) and the action
    preferences (actor).  Returns new ``(V, theta, delta)``.
    """
    V = np.array(V, dtype=float)
    theta = np.array(theta, dtype=float)
    target = r if done else r + gamma * V[s_next]
    delta = target - V[s]
    V[s] = V[s] + alpha_critic * delta
    probs = softmax(theta[s])
    grad = -probs
    grad[a] += 1.0                       # d log pi(a|s) / d theta  for a softmax policy
    theta[s] = theta[s] + alpha_actor * delta * grad
    return V, theta, delta


# =========================================================================== #
# Lesson 5–7: likelihoods of bandit models
# =========================================================================== #


def nll_rw(params, choices, rewards, n_arms=2, q0=0.5):
    """Negative log-likelihood of choices under RW learning + softmax.

    params = (alpha, beta)."""
    alpha, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, beta)
        nll -= np.log(p[c])                    # probability of the observed choice
        Q[c] = Q[c] + alpha * (r - Q[c])       # then learn from the observed outcome
    return nll


def nll_rw_asym(params, choices, rewards, n_arms=2, q0=0.5):
    """RW with separate learning rates for positive and negative prediction errors
    (Lefebvre et al. 2017; Niv et al. 2012).  params = (alpha_pos, alpha_neg, beta)."""
    alpha_pos, alpha_neg, beta = params
    Q = np.full(n_arms, q0, dtype=float)
    nll = 0.0
    for c, r in zip(choices, rewards):
        p = softmax(Q, beta)
        nll -= np.log(p[c])
        delta = r - Q[c]
        alpha = alpha_pos if delta > 0 else alpha_neg
        Q[c] = Q[c] + alpha * delta
    return nll


def nll_rw_stick(params, choices, rewards, n_arms=2, q0=0.5):
    """RW + softmax with choice stickiness (perseveration).  params = (alpha, beta, kappa).

    P(a) ∝ exp(beta * Q(a) + kappa * [a == previous choice])."""
    alpha, beta, kappa = params
    Q = np.full(n_arms, q0, dtype=float)
    prev = None
    nll = 0.0
    for c, r in zip(choices, rewards):
        bonus = np.zeros(n_arms)
        if prev is not None:
            bonus[prev] = kappa
        p = softmax(beta * Q + bonus, 1.0)
        nll -= np.log(p[c])
        Q[c] = Q[c] + alpha * (r - Q[c])
        prev = c
    return nll


def nll_wsls(params, choices, rewards, n_arms=2):
    """Noisy win-stay/lose-shift (Wilson & Collins 2019, model 2).  params = (epsilon,).

    With probability 1 - epsilon the agent follows the rule (stay after a reward,
    shift after no reward); otherwise it chooses at random."""
    (epsilon,) = params
    nll = 0.0
    for t, (c, r) in enumerate(zip(choices, rewards)):
        if t == 0:
            p = np.full(n_arms, 1.0 / n_arms)
        else:
            prev_c, prev_r = choices[t - 1], rewards[t - 1]
            p = np.full(n_arms, epsilon / n_arms)
            if prev_r > 0:                           # win -> stay
                p[prev_c] += 1 - epsilon
            else:                                    # lose -> shift (to any other arm)
                others = [a for a in range(n_arms) if a != prev_c]
                for a in others:
                    p[a] += (1 - epsilon) / len(others)
        nll -= np.log(p[c])
    return nll


def nll_bias(params, choices, rewards=None, n_arms=2):
    """A no-learning model: fixed probability ``b`` of choosing arm 0 (two arms).
    params = (b,)."""
    (b,) = params
    choices = np.asarray(choices)
    p0 = np.clip(b, 1e-12, 1 - 1e-12)
    return float(-np.sum(np.where(choices == 0, np.log(p0), np.log(1 - p0))))


def simulate_model(name, params, probs, n_trials, rng, q0=0.5):
    """Simulate choices/rewards from one of the bandit models above
    (``"rw"``, ``"rw_asym"``, ``"rw_stick"``, ``"wsls"``, ``"bias"``) on a
    stationary bandit.  Returns dict(choices, rewards)."""
    probs = np.asarray(probs, float)
    n_arms = len(probs)
    Q = np.full(n_arms, q0)
    choices = np.zeros(n_trials, int)
    rewards = np.zeros(n_trials)
    prev_c, prev_r = None, None
    for t in range(n_trials):
        if name == "rw":
            alpha, beta = params
            p = softmax(Q, beta)
        elif name == "rw_asym":
            _, _, beta = params
            p = softmax(Q, beta)
        elif name == "rw_stick":
            alpha, beta, kappa = params
            bonus = np.zeros(n_arms)
            if prev_c is not None:
                bonus[prev_c] = kappa
            p = softmax(beta * Q + bonus, 1.0)
        elif name == "wsls":
            (eps,) = params
            if prev_c is None:
                p = np.full(n_arms, 1 / n_arms)
            else:
                p = np.full(n_arms, eps / n_arms)
                if prev_r > 0:
                    p[prev_c] += 1 - eps
                else:
                    others = [a for a in range(n_arms) if a != prev_c]
                    for a in others:
                        p[a] += (1 - eps) / len(others)
        elif name == "bias":
            (b,) = params
            p = np.array([b, 1 - b])
        else:
            raise ValueError(name)
        c = choose(p, rng)
        r = float(rng.random() < probs[c])
        if name == "rw":
            Q[c] += params[0] * (r - Q[c])
        elif name == "rw_asym":
            delta = r - Q[c]
            Q[c] += (params[0] if delta > 0 else params[1]) * delta
        elif name == "rw_stick":
            Q[c] += params[0] * (r - Q[c])
        choices[t], rewards[t] = c, r
        prev_c, prev_r = c, r
    return {"choices": choices, "rewards": rewards}


MODEL_SPECS = {
    # name: (nll function, parameter names, parameter kinds for the transforms)
    "bias": (nll_bias, ["b"], ["unit"]),
    "wsls": (nll_wsls, ["epsilon"], ["unit"]),
    "rw": (nll_rw, ["alpha", "beta"], ["unit", "pos"]),
    "rw_asym": (nll_rw_asym, ["alpha_pos", "alpha_neg", "beta"], ["unit", "unit", "pos"]),
    "rw_stick": (nll_rw_stick, ["alpha", "beta", "kappa"], ["unit", "pos", "real"]),
}


# =========================================================================== #
# Lesson 9: the two-step task, hybrid model-based / model-free (Daw et al. 2011)
# =========================================================================== #

TWO_STEP_PARAM_NAMES = ["alpha1", "alpha2", "lam", "beta1", "beta2", "w", "p"]
TWO_STEP_PARAM_KINDS = ["unit", "unit", "unit", "pos", "pos", "unit", "real"]


def two_step_mb_values(Q2, transitions):
    """Model-based stage-1 values: Q_MB(a) = sum_s' P(s'|a) max_a' Q2(s', a')."""
    return transitions @ np.max(Q2, axis=1)


def two_step_simulate(params, task, rng):
    """Simulate the Daw et al. (2011) hybrid agent on a ``TwoStepTask``.

    params = (alpha1, alpha2, lam, beta1, beta2, w, p).  Returns a dict of arrays:
    ``a1`` (stage-1 choice), ``s2`` (stage-2 state), ``a2``, ``r`` (reward) and
    ``common`` (1 if the transition was common)."""
    alpha1, alpha2, lam, beta1, beta2, w, p_stick = params
    n = task.n_trials
    Q_mf = np.zeros(2)        # model-free stage-1 values
    Q2 = np.zeros((2, 2))     # stage-2 values (shared by both systems)
    a1 = np.zeros(n, int); s2 = np.zeros(n, int); a2 = np.zeros(n, int)
    r = np.zeros(n); common = np.zeros(n, int)
    prev_a1 = None
    task.reset()
    for t in range(n):
        # --- stage 1: combine MB and MF values, add perseveration, choose ---
        Q_mb = two_step_mb_values(Q2, task.transitions)
        Q_net = w * Q_mb + (1 - w) * Q_mf
        rep = np.zeros(2)
        if prev_a1 is not None:
            rep[prev_a1] = 1.0
        p1 = softmax(Q_net + p_stick * rep, beta1)
        a1[t] = choose(p1, rng)
        s2[t], is_common = task.step1(a1[t])
        common[t] = int(is_common)
        # --- stage 2 ---
        p2 = softmax(Q2[s2[t]], beta2)
        a2[t] = choose(p2, rng)
        r[t] = task.step2(a2[t])
        # --- model-free learning: SARSA(lambda) ---
        delta1 = Q2[s2[t], a2[t]] - Q_mf[a1[t]]
        Q_mf[a1[t]] += alpha1 * delta1
        delta2 = r[t] - Q2[s2[t], a2[t]]
        Q2[s2[t], a2[t]] += alpha2 * delta2
        Q_mf[a1[t]] += alpha1 * lam * delta2          # eligibility-trace update
        prev_a1 = a1[t]
    return {"a1": a1, "s2": s2, "a2": a2, "r": r, "common": common}


def two_step_nll(params, data, transitions=None):
    """Negative log-likelihood of two-step data under the hybrid model.

    params = (alpha1, alpha2, lam, beta1, beta2, w, p); ``data`` as returned by
    ``two_step_simulate``.  Both stage-1 and stage-2 choices contribute."""
    alpha1, alpha2, lam, beta1, beta2, w, p_stick = params
    if transitions is None:
        transitions = np.array([[0.7, 0.3], [0.3, 0.7]])
    Q_mf = np.zeros(2)
    Q2 = np.zeros((2, 2))
    prev_a1 = None
    nll = 0.0
    for a1, s2, a2, r in zip(data["a1"], data["s2"], data["a2"], data["r"]):
        Q_mb = two_step_mb_values(Q2, transitions)
        Q_net = w * Q_mb + (1 - w) * Q_mf
        rep = np.zeros(2)
        if prev_a1 is not None:
            rep[prev_a1] = 1.0
        p1 = softmax(Q_net + p_stick * rep, beta1)
        nll -= np.log(p1[a1])
        p2 = softmax(Q2[s2], beta2)
        nll -= np.log(p2[a2])
        delta1 = Q2[s2, a2] - Q_mf[a1]
        Q_mf[a1] += alpha1 * delta1
        delta2 = r - Q2[s2, a2]
        Q2[s2, a2] += alpha2 * delta2
        Q_mf[a1] += alpha1 * lam * delta2
        prev_a1 = a1
    return nll


def stay_probabilities(data):
    """The classic 2x2 analysis: P(repeat the stage-1 choice) split by whether the
    previous trial was rewarded (rows: rewarded, unrewarded) and whether its
    transition was common (cols: common, rare)."""
    a1 = np.asarray(data["a1"]); r = np.asarray(data["r"]); common = np.asarray(data["common"])
    stay = (a1[1:] == a1[:-1]).astype(float)
    prev_r, prev_common = r[:-1], common[:-1]
    out = np.zeros((2, 2))
    for i, rewarded in enumerate([1, 0]):
        for j, is_common in enumerate([1, 0]):
            mask = (prev_r == rewarded) & (prev_common == is_common)
            out[i, j] = stay[mask].mean() if mask.any() else np.nan
    return out


# =========================================================================== #
# Lesson 10: the successor representation
# =========================================================================== #


def sr_analytic(T, gamma):
    """The successor representation of a transition matrix:  M = (I - gamma T)^-1."""
    n = T.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * np.asarray(T, float))


def sr_td_update(M, s, s_next, alpha, gamma):
    """TD update of one row of the successor matrix after a transition s -> s_next.

    Target: one-hot(s) + gamma * M[s_next]; terminal states keep their one-hot
    row (they are never updated because no transition leaves them)."""
    M = np.array(M, dtype=float)
    onehot = np.zeros(M.shape[0])
    onehot[s] = 1.0
    target = onehot + gamma * M[s_next]
    M[s] = M[s] + alpha * (target - M[s])
    return M


def sr_value(M, R):
    """State values from the SR and a reward vector:  V = M @ R."""
    return np.asarray(M, float) @ np.asarray(R, float)


def sr_reward_update(R_hat, s, r, alpha):
    """Delta-rule learning of the reward expected in each state."""
    R_hat = np.array(R_hat, dtype=float)
    R_hat[s] = R_hat[s] + alpha * (r - R_hat[s])
    return R_hat


def run_sr_agent(mdp, n_episodes, alpha, gamma, rng, policy=None, M0=None, R0=None, learn_reward=True):
    """Learn the SR (and the reward vector) by TD while following a policy
    (uniform random if None).  Returns ``(M, R_hat)``."""
    M = np.eye(mdp.n_states) if M0 is None else np.array(M0, float)
    R_hat = np.zeros(mdp.n_states) if R0 is None else np.array(R0, float)
    for _ in range(n_episodes):
        s = mdp.reset(rng)
        done = False
        while not done:
            p = np.full(mdp.n_actions, 1 / mdp.n_actions) if policy is None else policy[s]
            a = choose(p, rng)
            s_next, r, done = mdp.step(a, rng)
            M = sr_td_update(M, s, s_next, alpha, gamma)
            if learn_reward:
                R_hat = sr_reward_update(R_hat, s_next, r, alpha)
            s = s_next
    return M, R_hat


# =========================================================================== #
# Lesson 11: Bayesian learners
# =========================================================================== #


def beta_update(a, b, reward):
    """Bayesian update of a Beta(a, b) belief about a reward probability after one
    Bernoulli outcome (reward 1 or 0)."""
    return a + reward, b + (1 - reward)


def beta_update_forgetful(a, b, reward, decay):
    """Beta update with exponential forgetting: old evidence is discounted by
    ``decay`` (< 1) before adding the new outcome, so the belief can track change."""
    return decay * a + reward, decay * b + (1 - reward)


def kalman_update(mu, var, reward, obs_var, process_var=0.0):
    """One Kalman-filter update of a Gaussian belief N(mu, var) about a reward mean.

    First the belief diffuses (var += process_var), then the observation is
    incorporated with gain k.  Returns ``(mu_new, var_new, k)``."""
    var_prior = var + process_var
    k = var_prior / (var_prior + obs_var)
    mu_new = mu + k * (reward - mu)
    var_new = (1 - k) * var_prior
    return mu_new, var_new, k


def kalman_bandit_step(mu, var, action, reward, obs_var, process_var):
    """Kalman update for a multi-armed bandit: every arm's uncertainty grows by
    ``process_var``; only the chosen arm is updated by the observation.
    Returns new ``(mu, var)`` arrays."""
    mu = np.array(mu, float)
    var = np.array(var, float) + process_var
    k = var[action] / (var[action] + obs_var)
    mu[action] = mu[action] + k * (reward - mu[action])
    var[action] = (1 - k) * var[action]
    return mu, var


def ucb_values(mu, var, c):
    """Upper-confidence-bound values: mean + c * standard deviation."""
    return np.asarray(mu, float) + c * np.sqrt(np.asarray(var, float))


def thompson_choice(mu, var, rng):
    """Thompson sampling: draw one sample per arm from its posterior, pick the largest."""
    samples = rng.normal(np.asarray(mu, float), np.sqrt(np.asarray(var, float)))
    return int(np.argmax(samples))


def grid_changepoint_filter(outcomes, hazard, grid):
    """Exact Bayesian tracking of a Bernoulli probability that may jump.

    Each trial, with probability ``hazard`` the true probability is re-drawn
    uniformly (a change-point); the belief is kept on ``grid``.
    Returns ``(predictions, posteriors)``: the predicted reward probability before
    each outcome (n_trials,) and the posterior after each trial (n_trials, n_grid)."""
    grid = np.asarray(grid, float)
    post = np.full(len(grid), 1.0 / len(grid))
    predictions = np.zeros(len(outcomes))
    posteriors = np.zeros((len(outcomes), len(grid)))
    for t, o in enumerate(outcomes):
        # 1. predict: the probability may have changed
        prior = (1 - hazard) * post + hazard / len(grid)
        predictions[t] = np.sum(prior * grid)
        # 2. update with the outcome
        likelihood = grid if o == 1 else 1 - grid
        post = prior * likelihood
        post = post / np.sum(post)
        posteriors[t] = post
    return predictions, posteriors


def pearce_hall_update(v, assoc, reward, eta, kappa):
    """Pearce–Hall (1980) hybrid: the associability tracks the recent |prediction
    error| and scales the learning rate.  Returns ``(v_new, assoc_new, delta)``."""
    delta = reward - v
    v_new = v + kappa * assoc * delta
    assoc_new = eta * abs(delta) + (1 - eta) * assoc
    return v_new, assoc_new, delta


# =========================================================================== #
# Lesson 12: the volatile Kalman filter (Piray & Daw 2020)
# =========================================================================== #


def vkf_continuous(outcomes, lam, v0, sigma2):
    """Volatile Kalman filter for continuous outcomes (Piray & Daw 2020, Eqs. 9–13).

    Returns a dict of per-trial arrays: ``m`` (prediction before the outcome),
    ``v`` (volatility estimate before the outcome), ``k`` (Kalman gain =
    learning rate), ``pe`` (prediction error), ``w`` (posterior variance).
    """
    outcomes = np.asarray(outcomes, float)
    n = len(outcomes)
    m, w, v = 0.0, sigma2, v0
    out = {key: np.zeros(n) for key in ["m", "v", "k", "pe", "w"]}
    for t, o in enumerate(outcomes):
        out["m"][t], out["v"][t], out["w"][t] = m, v, w
        m_prev, w_prev = m, w
        k = (w + v) / (w + v + sigma2)                     # Kalman gain
        pe = o - m
        m = m + k * pe                                    # mean update
        w = (1 - k) * (w_prev + v)                        # variance update
        w_cov = (1 - k) * w_prev                          # covariance of consecutive states
        v = v + lam * ((m - m_prev) ** 2 + w + w_prev - 2 * w_cov - v)   # volatility update
        out["k"][t], out["pe"][t] = k, pe
    return out


def vkf_bin(outcomes, lam, v0, omega):
    """Binary volatile Kalman filter (Piray & Daw 2020, Eqs. 14–19).

    The hidden state ``m`` is a logit; the prediction is sigmoid(m).  Returns a
    dict with ``m``, ``pred`` (= sigmoid(m)), ``v``, ``alpha`` (learning rate),
    ``k``, ``pe`` and ``w``, each an array over trials (values *before* the
    outcome of that trial is seen)."""
    outcomes = np.asarray(outcomes, float)
    n = len(outcomes)
    m, w, v = 0.0, omega, v0
    out = {key: np.zeros(n) for key in ["m", "pred", "v", "alpha", "k", "pe", "w"]}
    for t, o in enumerate(outcomes):
        out["m"][t], out["v"][t], out["w"][t] = m, v, w
        out["pred"][t] = sigmoid(m)
        m_prev, w_prev = m, w
        pe = o - sigmoid(m)                               # prediction error in probability space
        k = (w + v) / (w + v + omega)                     # gain
        alpha = np.sqrt(w + v)                            # learning rate of the binary VKF
        m = m + alpha * pe
        w = (1 - k) * (w_prev + v)
        w_cov = (1 - k) * w_prev
        v = v + lam * ((m - m_prev) ** 2 + w + w_prev - 2 * w_cov - v)
        out["alpha"][t], out["k"][t], out["pe"][t] = alpha, k, pe
    return out


def vkf_bin_nll(params, outcomes, choices):
    """Negative log-likelihood of binary choices when the VKF's prediction that
    option 0 pays off drives a logistic choice rule.

    params = (lam, v0, omega, beta).  ``outcomes[t]`` = 1 if option 0 was the
    rewarded option on trial t (full-information reversal task); ``choices[t]``
    in {0, 1}.  P(choose 0) = sigmoid(beta * (2 * pred - 1))."""
    lam, v0, omega, beta = params
    sig = vkf_bin(outcomes, lam, v0, omega)
    p0 = sigmoid(beta * (2 * sig["pred"] - 1))
    choices = np.asarray(choices)
    p_choice = np.where(choices == 0, p0, 1 - p0)
    return float(-np.sum(np.log(p_choice)))


def rw_reversal_nll(params, outcomes, choices):
    """Same choice rule as ``vkf_bin_nll`` but with a fixed-learning-rate
    Rescorla–Wagner tracker of P(option 0 pays).  params = (alpha, beta)."""
    alpha, beta = params
    p_hat = 0.5
    nll = 0.0
    for o, c in zip(outcomes, choices):
        p0 = sigmoid(beta * (2 * p_hat - 1))
        nll -= np.log(p0 if c == 0 else 1 - p0)
        p_hat = p_hat + alpha * (o - p_hat)
    return nll


def rw_reversal_predictions(outcomes, alpha):
    """Trial-by-trial RW predictions of P(option 0 pays) before each outcome."""
    p_hat, preds = 0.5, np.zeros(len(outcomes))
    for t, o in enumerate(outcomes):
        preds[t] = p_hat
        p_hat = p_hat + alpha * (o - p_hat)
    return preds


# =========================================================================== #
# Lesson 13: distributional reinforcement learning
# =========================================================================== #


def expectile_td_update(values, reward, alpha_pos, alpha_neg):
    """Distributional TD with asymmetric learning rates (Dabney et al. 2020).

    Each value predictor i is updated with alpha_pos[i] when its prediction error
    is positive and alpha_neg[i] when it is negative.  Returns new values."""
    values = np.asarray(values, float)
    delta = reward - values
    lr = np.where(delta > 0, alpha_pos, alpha_neg)
    return values + lr * delta


def taus_from_alphas(alpha_pos, alpha_neg):
    """Asymmetry tau_i = alpha_pos_i / (alpha_pos_i + alpha_neg_i)."""
    alpha_pos, alpha_neg = np.asarray(alpha_pos, float), np.asarray(alpha_neg, float)
    return alpha_pos / (alpha_pos + alpha_neg)


def expectile(samples, tau, n_iter=200):
    """The tau-expectile of a sample: the e that solves
    tau * E[(x-e)+] = (1-tau) * E[(e-x)+]  (iterated re-weighted mean)."""
    x = np.asarray(samples, float)
    e = np.mean(x)
    for _ in range(n_iter):
        weights = np.where(x > e, tau, 1 - tau)
        e_new = np.sum(weights * x) / np.sum(weights)
        if abs(e_new - e) < 1e-12:
            break
        e = e_new
    return float(e)


def expectile_loss(e, samples, tau):
    """Asymmetric squared loss whose minimiser is the tau-expectile."""
    x = np.asarray(samples, float)
    weights = np.where(x > e, tau, 1 - tau)
    return float(np.mean(weights * (x - e) ** 2))


def quantile_td_update(values, reward, alpha, taus):
    """Quantile-regression TD (Dabney et al. 2018): v_i += alpha * (tau_i - 1[r < v_i])."""
    values = np.asarray(values, float)
    taus = np.asarray(taus, float)
    indicator = (reward < values).astype(float)
    return values + alpha * (taus - indicator)


def decode_distribution(expectiles, taus, n_points=None, x0=None):
    """Recover a set of equally-weighted support points whose expectiles match
    the given (expectile, tau) pairs — the decoding used by Dabney et al. (2020).
    Returns the sorted support points."""
    from scipy.optimize import minimize

    expectiles = np.asarray(expectiles, float)
    taus = np.asarray(taus, float)
    n_points = len(expectiles) if n_points is None else n_points
    x0 = np.sort(expectiles) if x0 is None else np.asarray(x0, float)
    if len(x0) != n_points:
        x0 = np.linspace(expectiles.min(), expectiles.max(), n_points)

    def loss(points):
        return sum((expectile(points, t) - e) ** 2 for e, t in zip(expectiles, taus))

    res = minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-12})
    return np.sort(res.x)


__all__ = [name for name in dir() if not name.startswith("_")]
