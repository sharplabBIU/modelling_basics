"""A consistent plotting style for the course, plus a few convenience plots.

The categorical palette is a colour-blind-safe ordering (blue, orange, aqua,
yellow, magenta, green, violet, red); call ``setup_style()`` once at the top of
a notebook.  Colours for the three "systems" that recur throughout the course
are fixed so that figures are comparable across lessons:

    MF (model-free)   -> blue
    MB (model-based)  -> orange
    SR (successor)    -> aqua
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SYSTEM_COLORS = {"MF": PALETTE[0], "MB": PALETTE[1], "SR": PALETTE[2], "hybrid": PALETTE[6],
                 "Bayes": PALETTE[4], "data": "#52514e"}
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"


def setup_style() -> None:
    """Apply the course's matplotlib style (safe to call repeatedly)."""
    mpl.rcParams.update({
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.figsize": (7, 4),
        "figure.dpi": 100,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "font.size": 10.5,
        "image.cmap": SEQUENTIAL,
        "savefig.bbox": "tight",
    })


def plot_parameter_recovery(true, fitted, names, axes=None, identity: bool = True):
    """Scatter true vs fitted parameters, one panel per parameter, with r."""
    true = np.atleast_2d(np.asarray(true, float))
    fitted = np.atleast_2d(np.asarray(fitted, float))
    if true.shape[1] != len(names):
        true, fitted = true.T, fitted.T
    k = len(names)
    if axes is None:
        fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.4))
    axes = np.atleast_1d(axes)
    for j, (ax, name) in enumerate(zip(axes, names)):
        x, y = true[:, j], fitted[:, j]
        ax.scatter(x, y, s=22, alpha=0.75, edgecolor="white", linewidth=0.5)
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        if identity:
            ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1, ls="--", label="identity")
        r = np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan
        ax.set_title(f"{name}:  r = {r:.2f}")
        ax.set_xlabel(f"true {name}")
        ax.set_ylabel(f"recovered {name}")
        if identity:
            ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    return axes


def plot_confusion(matrix, names, ax=None, title="Model recovery (rows: simulated, cols: best fit)",
                   fmt="{:.2f}"):
    """Heat-map of a confusion matrix (rows = simulated model, cols = winning model)."""
    m = np.asarray(matrix, float)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * len(names) + 2.5, 1.0 * len(names) + 2))
    im = ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("model with the best fit")
    ax.set_ylabel("model that generated the data")
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, fmt.format(m[i, j]), ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else "#0b0b0b", fontsize=10)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_stay_probabilities(stay, ax=None, title="Stay probability", ylim=(0.5, 1.0), err=None):
    """The classic Daw-style bar plot.  ``stay`` is a 2x2 array
    [rewarded/unrewarded] x [common/rare] (or a dict with those keys)."""
    if isinstance(stay, dict):
        stay = np.array([[stay["rewarded_common"], stay["rewarded_rare"]],
                         [stay["unrewarded_common"], stay["unrewarded_rare"]]])
    stay = np.asarray(stay, float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.6))
    x = np.array([0, 1])
    w = 0.36
    kw = {}
    ax.bar(x - w / 2, stay[:, 0], w, label="common", color=PALETTE[0],
           yerr=None if err is None else np.asarray(err)[:, 0], **kw)
    ax.bar(x + w / 2, stay[:, 1], w, label="rare", color=PALETTE[1],
           yerr=None if err is None else np.asarray(err)[:, 1], **kw)
    ax.set_xticks(x, ["rewarded", "unrewarded"])
    ax.set_ylabel("P(repeat stage-1 choice)")
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(title="transition")
    ax.grid(axis="x", visible=False)
    return ax


def plot_posterior_1d(samples, true=None, ax=None, name="θ", bins=40, color=None):
    """Histogram of posterior samples with (optional) true value."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.hist(samples, bins=bins, density=True, alpha=0.8, color=color or PALETTE[0], edgecolor="white")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    ax.axvspan(lo, hi, color=PALETTE[0], alpha=0.12, label="95% interval")
    ax.axvline(np.mean(samples), color=PALETTE[0], lw=1.5, label="posterior mean")
    if true is not None:
        ax.axvline(true, color=PALETTE[7], lw=1.5, ls="--", label="true value")
    ax.set_xlabel(name)
    ax.set_ylabel("posterior density")
    ax.legend(fontsize=8)
    return ax


def plot_matrix(M, names=None, ax=None, title="", cmap=SEQUENTIAL, fmt="{:.2f}", vmin=None, vmax=None):
    """Annotated heat-map of a small matrix (e.g. a successor representation)."""
    M = np.asarray(M, float)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.7 * M.shape[1] + 2, 0.6 * M.shape[0] + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    if names is not None:
        ax.set_xticks(range(M.shape[1]), names, rotation=45, ha="right")
        ax.set_yticks(range(M.shape[0]), names)
    ax.grid(False)
    vmax_ = np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.6 * vmax_ else "#0b0b0b")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


__all__ = ["setup_style", "PALETTE", "SYSTEM_COLORS", "SEQUENTIAL", "DIVERGING",
           "plot_parameter_recovery", "plot_confusion", "plot_stay_probabilities",
           "plot_posterior_1d", "plot_matrix"]
