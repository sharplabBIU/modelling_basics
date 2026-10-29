"""Fit the course models to the synthetic data sets and cache the results.

Lessons 14 and 15 load these cached maximum-likelihood fits so that the
notebooks stay fast; the lessons also show how to produce them live for a few
subjects.  Everything is deterministic (one seed per subject).

    python scripts/make_fits.py            # both data sets (a few minutes, uses all CPU cores)
    python scripts/make_fits.py two_step   # only the two-step fits
    python scripts/make_fits.py bandit     # only the bandit fits

Outputs
-------
data/two_step_compulsivity_fits.csv : subject, alpha1, alpha2, lam, beta1, beta2, w, p, nll, n
data/bandit_anhedonia_fits.csv      : long format — subject, model, k, n, nll, bic and one
                                      column per parameter name (NaN where a model lacks it)
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from rlcourse import datasets, fitting, models  # noqa: E402

N_STARTS = 5


def _fit_two_step(args):
    subject, data = args
    rng = np.random.default_rng(1000 + subject)
    fit = fitting.fit_mle(models.two_step_nll, (data,), models.TWO_STEP_PARAM_KINDS, n_starts=N_STARTS, rng=rng)
    row = {"subject": subject}
    row.update({name: val for name, val in zip(models.TWO_STEP_PARAM_NAMES, fit["params"])})
    row["nll"] = fit["nll"]
    row["n"] = 2 * len(data["a1"])          # two choices per trial
    return row


def _fit_bandit(args):
    subject, choices, rewards = args
    rows = []
    for m, (name, (nll_fn, param_names, kinds)) in enumerate(models.MODEL_SPECS.items()):
        rng = np.random.default_rng(2000 + 10 * subject + m)
        fit = fitting.fit_mle(nll_fn, (choices, rewards), kinds, n_starts=N_STARTS, rng=rng)
        row = {"subject": subject, "model": name, "k": len(kinds), "n": len(choices), "nll": fit["nll"],
               "bic": fitting.bic(fit["nll"], len(kinds), len(choices))}
        row.update({p: v for p, v in zip(param_names, fit["params"])})
        rows.append(row)
    return rows


def make_two_step(verbose=True):
    trials, _ = datasets.load("two_step_compulsivity")
    per_subject = datasets.split_by_subject(trials)
    jobs = [(s, d) for s, d in per_subject.items()]
    with Pool() as pool:
        rows = pool.map(_fit_two_step, jobs)
    df = pd.DataFrame(rows).sort_values("subject")
    out = os.path.join(datasets.DATA_DIR, "two_step_compulsivity_fits.csv")
    df.to_csv(out, index=False)
    if verbose:
        print(f"two-step fits: {len(df)} subjects -> {os.path.relpath(out, ROOT)}")
    return df


def make_bandit(verbose=True):
    trials, _ = datasets.load("bandit_anhedonia")
    per_subject = datasets.split_by_subject(trials)
    jobs = [(s, d["choice"], d["reward"].astype(float)) for s, d in per_subject.items()]
    with Pool() as pool:
        nested = pool.map(_fit_bandit, jobs)
    rows = [r for rows in nested for r in rows]
    all_params = []
    for _, (_, names, _) in models.MODEL_SPECS.items():
        all_params += [p for p in names if p not in all_params]
    df = pd.DataFrame(rows)
    df = df[["subject", "model", "k", "n", "nll", "bic"] + all_params].sort_values(["subject", "model"])
    out = os.path.join(datasets.DATA_DIR, "bandit_anhedonia_fits.csv")
    df.to_csv(out, index=False)
    if verbose:
        print(f"bandit fits: {df.subject.nunique()} subjects x {df.model.nunique()} models -> {os.path.relpath(out, ROOT)}")
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "two_step"):
        make_two_step()
    if which in ("all", "bandit"):
        make_bandit()
