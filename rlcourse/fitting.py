"""Model fitting and model comparison helpers.

Everything here is written in plain numpy/scipy so that the mechanics are
visible.  The functions are the reference solutions of the exercises in
lessons 5–8 and are reused by later lessons.

Parameter transforms
--------------------
Optimisers work best on unconstrained numbers, but learning rates live in
[0, 1] and inverse temperatures must be positive.  We therefore optimise
*unconstrained* values ``x`` and map them to parameters ``theta``::

    kind "unit" : theta = sigmoid(x)   (0, 1)
    kind "pos"  : theta = exp(x)       (0, inf)
    kind "real" : theta = x            (-inf, inf)
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

from .models import logit, sigmoid

# =========================================================================== #
# Transforms
# =========================================================================== #


def to_constrained(x, kinds):
    """Map unconstrained values to parameters (see module docstring)."""
    x = np.atleast_1d(np.asarray(x, float))
    theta = np.empty_like(x)
    for i, kind in enumerate(kinds):
        if kind == "unit":
            theta[i] = sigmoid(x[i])
        elif kind == "pos":
            theta[i] = np.exp(x[i])
        elif kind == "real":
            theta[i] = x[i]
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return theta


def to_unconstrained(theta, kinds):
    """Inverse of :func:`to_constrained`."""
    theta = np.atleast_1d(np.asarray(theta, float))
    x = np.empty_like(theta)
    for i, kind in enumerate(kinds):
        if kind == "unit":
            x[i] = logit(np.clip(theta[i], 1e-9, 1 - 1e-9))
        elif kind == "pos":
            x[i] = np.log(max(theta[i], 1e-12))
        elif kind == "real":
            x[i] = theta[i]
        else:
            raise ValueError(f"unknown kind {kind!r}")
    return x


# =========================================================================== #
# Maximum likelihood
# =========================================================================== #


def fit_mle(nll, data_args, kinds, n_starts=10, rng=None, x0_sd=1.5, method="L-BFGS-B", log_prior=None):
    """Maximum-likelihood (or MAP, if ``log_prior`` is given) fit with random restarts.

    Parameters
    ----------
    nll : callable ``nll(theta, *data_args) -> float``
    data_args : tuple passed after the parameters
    kinds : parameter kinds (see module docstring), one per parameter
    n_starts : number of random starting points
    log_prior : optional callable ``log_prior(x_unconstrained) -> float`` added to
        the log-likelihood (turns the fit into MAP estimation)

    Returns a dict with ``params`` (constrained), ``x`` (unconstrained),
    ``nll`` (at the optimum; excludes the prior), ``objective`` (what was
    minimised) and ``success``.
    """
    rng = np.random.default_rng() if rng is None else rng
    kinds = list(kinds)

    def objective(x):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            theta = to_constrained(x, kinds)
            val = nll(theta, *data_args)
            if log_prior is not None:
                val = val - log_prior(x)
        return val if np.isfinite(val) else 1e10

    best = None
    for _ in range(n_starts):
        x0 = rng.normal(0, x0_sd, size=len(kinds))
        res = optimize.minimize(objective, x0, method=method)
        if best is None or res.fun < best.fun:
            best = res
    theta = to_constrained(best.x, kinds)
    return {"params": theta, "x": best.x, "nll": float(nll(theta, *data_args)),
            "objective": float(best.fun), "success": bool(best.success), "result": best}


def grid_nll(nll, data_args, grids):
    """Evaluate ``nll`` on a grid of parameter values (for visualising the likelihood).

    ``grids`` is a list of 1-D arrays, one per parameter; returns an array of
    shape (len(grid_1), len(grid_2), ...)."""
    mesh = np.meshgrid(*grids, indexing="ij")
    out = np.empty(mesh[0].shape)
    for idx in np.ndindex(*mesh[0].shape):
        theta = [m[idx] for m in mesh]
        out[idx] = nll(theta, *data_args)
    return out


# =========================================================================== #
# Information criteria and cross-validation
# =========================================================================== #


def aic(nll, k):
    """Akaike information criterion: 2k + 2 * nll."""
    return 2 * k + 2 * nll


def bic(nll, k, n):
    """Bayesian information criterion: k log(n) + 2 * nll  (n = number of data points)."""
    return k * np.log(n) + 2 * nll


def bic_log_evidence(nll, k, n):
    """Approximate log model evidence from the BIC:  -BIC / 2."""
    return -bic(nll, k, n) / 2


def log_sum_exp(x, axis=None):
    return special.logsumexp(x, axis=axis)


# =========================================================================== #
# Bayesian model selection for group studies (Stephan et al. 2009; Rigoux et al. 2014)
# =========================================================================== #


def bms(log_evidence, alpha0=1.0, n_samples=20000, rng=None, max_iter=500, tol=1e-8):
    """Random-effects Bayesian model selection.

    Parameters
    ----------
    log_evidence : array (n_subjects, n_models)
        Log model evidence (e.g. ``-BIC/2`` or a Laplace approximation) of each
        model for each subject.
    alpha0 : prior Dirichlet count for each model.

    Returns a dict with
      ``alpha``   : posterior Dirichlet parameters
      ``r``       : expected model frequencies  alpha / sum(alpha)
      ``xp``      : exceedance probabilities P(r_k > r_j for all j)
      ``bor``     : Bayes omnibus risk, the posterior probability that all
                    models are equally frequent (Rigoux et al. 2014)
      ``pxp``     : protected exceedance probabilities
      ``g``       : posterior probability of each model for each subject
    """
    rng = np.random.default_rng() if rng is None else rng
    L = np.asarray(log_evidence, float)
    n, K = L.shape
    alpha0 = np.full(K, float(alpha0))
    alpha = alpha0.copy()
    for _ in range(max_iter):
        # E-step: posterior over models for each subject
        log_u = L + special.digamma(alpha) - special.digamma(alpha.sum())
        g = np.exp(log_u - special.logsumexp(log_u, axis=1, keepdims=True))
        # M-step: update the Dirichlet counts
        alpha_new = alpha0 + g.sum(axis=0)
        if np.max(np.abs(alpha_new - alpha)) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    r = alpha / alpha.sum()
    # exceedance probabilities by Monte Carlo sampling of the Dirichlet
    samples = rng.dirichlet(alpha, size=n_samples)
    xp = np.bincount(np.argmax(samples, axis=1), minlength=K) / n_samples
    # free energy of the full model (variational bound) vs. the null model
    e_log_r = special.digamma(alpha) - special.digamma(alpha.sum())
    F1 = (np.sum(g * L) + np.sum((g.sum(0) + alpha0 - alpha) * e_log_r)
          + special.gammaln(alpha0.sum()) - np.sum(special.gammaln(alpha0))
          - special.gammaln(alpha.sum()) + np.sum(special.gammaln(alpha))
          - np.sum(g * np.log(np.clip(g, 1e-300, None))))
    F0 = np.sum(special.logsumexp(L - np.log(K), axis=1))
    bor = 1.0 / (1.0 + np.exp(F1 - F0))
    pxp = (1 - bor) * xp + bor / K
    return {"alpha": alpha, "r": r, "xp": xp, "bor": float(bor), "pxp": pxp, "g": g, "F1": float(F1), "F0": float(F0)}


# =========================================================================== #
# Bayesian estimation: priors, Laplace, MCMC, hierarchical EM
# =========================================================================== #


def normal_log_prior(mu, sd):
    """Return a function x -> log N(x | mu, sd) summed over parameters (unconstrained space)."""
    mu = np.atleast_1d(np.asarray(mu, float))
    sd = np.atleast_1d(np.asarray(sd, float))

    def log_prior(x):
        return float(np.sum(stats.norm.logpdf(np.atleast_1d(x), mu, sd)))

    return log_prior


def numerical_hessian(f, x, eps=1e-4):
    """Finite-difference Hessian of a scalar function at ``x``."""
    x = np.asarray(x, float)
    k = len(x)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = eps
            ej = np.zeros(k); ej[j] = eps
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / eps ** 2
            else:
                H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
                H[j, i] = H[i, j]
    return H


def laplace_covariance(neg_log_post, x_hat, eps=1e-4):
    """Laplace approximation: covariance = inverse Hessian of the negative log posterior
    at its minimum (falls back to a diagonal, positive version if needed)."""
    H = numerical_hessian(neg_log_post, x_hat, eps)
    try:
        cov = np.linalg.inv(H)
        if np.all(np.linalg.eigvalsh(cov) > 0):
            return cov
    except np.linalg.LinAlgError:
        pass
    d = np.clip(np.diag(H), 1e-6, None)
    return np.diag(1.0 / d)


def metropolis_hastings(log_target, x0, n_samples, proposal_sd, rng, n_burn=0, thin=1):
    """Random-walk Metropolis–Hastings sampler.

    ``log_target(x)`` is the (unnormalised) log posterior of a vector ``x``.
    Returns ``(samples, acceptance_rate)`` with samples of shape (n_samples, dim).
    """
    x = np.atleast_1d(np.asarray(x0, float))
    lp = log_target(x)
    samples = []
    n_accept = 0
    total = n_burn + n_samples * thin
    for i in range(total):
        proposal = x + rng.normal(0, proposal_sd, size=x.shape)   # propose a nearby point
        lp_prop = log_target(proposal)
        if np.log(rng.random()) < lp_prop - lp:                    # accept with prob min(1, ratio)
            x, lp = proposal, lp_prop
            n_accept += 1
        if i >= n_burn and (i - n_burn) % thin == 0:
            samples.append(x.copy())
    return np.array(samples), n_accept / total


def em_hierarchical(nll, data_list, kinds, n_iter=30, n_starts=3, rng=None, tol=1e-3, verbose=False,
                    mu0=None, sd0=None):
    """Empirical-Bayes (expectation–maximisation) fitting of a hierarchical model
    with a Gaussian group prior in unconstrained space (Huys et al. 2011;
    Gershman 2016).

    Parameters
    ----------
    nll : ``nll(theta, *data) -> float`` for one subject
    data_list : list of tuples, one per subject (passed as ``*data``)
    kinds : parameter kinds for the transforms

    Returns a dict with the group prior (``mu``, ``sd``), the per-subject MAP
    estimates in unconstrained (``x``) and constrained (``params``) space, the
    per-subject Laplace variances (``var``) and the history of ``mu``.
    """
    rng = np.random.default_rng() if rng is None else rng
    k = len(kinds)
    mu = np.zeros(k) if mu0 is None else np.array(mu0, float)
    sd = np.full(k, 2.0) if sd0 is None else np.array(sd0, float)
    n = len(data_list)
    x_hat = np.zeros((n, k))
    var_hat = np.ones((n, k))
    history = [mu.copy()]
    for it in range(n_iter):
        log_prior = normal_log_prior(mu, sd)
        # E-step: MAP estimate and Laplace variance for every subject
        for i, data in enumerate(data_list):
            fit = fit_mle(nll, data, kinds, n_starts=n_starts, rng=rng, log_prior=log_prior)
            x_hat[i] = fit["x"]

            def neg_log_post(x, data=data):
                val = nll(to_constrained(x, kinds), *data) - log_prior(x)
                return val if np.isfinite(val) else 1e10

            var_hat[i] = np.diag(laplace_covariance(neg_log_post, x_hat[i]))
        # M-step: update the group mean and sd
        mu_new = x_hat.mean(axis=0)
        sd_new = np.sqrt(np.mean(x_hat ** 2 + var_hat, axis=0) - mu_new ** 2)
        sd_new = np.clip(sd_new, 1e-3, None)
        change = np.max(np.abs(mu_new - mu))
        mu, sd = mu_new, sd_new
        history.append(mu.copy())
        if verbose:
            print(f"iter {it + 1:2d}  mu = {np.round(mu, 3)}  sd = {np.round(sd, 3)}")
        if change < tol:
            break
    params = np.array([to_constrained(x, kinds) for x in x_hat])
    return {"mu": mu, "sd": sd, "x": x_hat, "var": var_hat, "params": params, "history": np.array(history)}


# =========================================================================== #
# Reliability helpers (lesson 14)
# =========================================================================== #


def spearman_brown(r_half):
    """Correct a split-half correlation for the full test length."""
    return 2 * r_half / (1 + r_half)


def icc_oneway(x1, x2):
    """One-way random-effects intraclass correlation ICC(1) for two measurements
    per subject (Shrout & Fleiss 1979)."""
    data = np.column_stack([x1, x2]).astype(float)
    n, k = data.shape
    grand = data.mean()
    ms_between = k * np.sum((data.mean(axis=1) - grand) ** 2) / (n - 1)
    ms_within = np.sum((data - data.mean(axis=1, keepdims=True)) ** 2) / (n * (k - 1))
    return (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)


__all__ = [name for name in dir() if not name.startswith("_")]
