"""Figures for the RNAcompete benchmark.

Pure plotting functions that take the ``{label: {rbp: value}}`` mapping produced by
``report.load_summaries`` and render Matplotlib figures. Kept CLI-free so they are importable and
testable; ``scripts/plot_performance.py`` is the CLI wrapper.

Designed so adding models later is free: models become columns (auto-discovered upstream), missing
RBPs render as blank cells, and ordering/colour are derived from the data.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def pretty_label(name: str) -> str:
    """Human-readable model label. AlphaGenome-FT runs are named by encoder resolution (bp/bin)."""
    ens = re.fullmatch(r"(.+)_ens(\d+)", name)
    if ens:
        return f"{pretty_label(ens.group(1))} ×{ens.group(2)}"  # e.g. "ResidualBind ×10"
    if name == "residualbind":
        return "ResidualBind"
    if name == "agft_trunk":
        return "AGFT 128bp"  # final trunk = 128 bp bins
    m = re.fullmatch(r"agft_bin(\d+)", name)
    if m:
        return f"AGFT {m.group(1)}bp"
    return name


def _matrix(summaries: dict[str, dict[str, float]]) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(M, rbps, labels)`` where ``M[i, j]`` is RBP ``i`` for model ``j`` (NaN if absent)."""
    labels = list(summaries)
    rbps = sorted({rbp for col in summaries.values() for rbp in col})
    M = np.full((len(rbps), len(labels)), np.nan)
    for j, label in enumerate(labels):
        for i, rbp in enumerate(rbps):
            M[i, j] = summaries[label].get(rbp, np.nan)
    return M, rbps, labels


def performance_heatmap(
    summaries: dict[str, dict[str, float]],
    *,
    metric_label: str = "Pearson r",
    split: str = "valid",
    cmap: str = "viridis",
    add_mean_row: bool = True,
    sort_models: bool = True,
):
    """Heatmap of per-RBP performance (rows) across models (columns).

    Columns keep the input (dict) order when ``sort_models`` is False, else are ordered best-mean
    first; RBPs are always ordered best-mean first; a separated ``MEAN`` row (per-model mean across
    RBPs) is appended. Missing entries render grey and annotate ``—``. Returns ``(fig, ax)``.
    """
    M, rbps, labels = _matrix(summaries)
    if M.size == 0:
        raise ValueError("No data to plot (no models / RBPs found).")

    # Order columns by model mean (desc) unless an explicit order was supplied; rows by RBP mean.
    col_mean = np.nanmean(M, axis=0)
    if sort_models:
        col_order = np.argsort(-np.nan_to_num(col_mean, nan=-np.inf))
        M, labels, col_mean = M[:, col_order], [labels[k] for k in col_order], col_mean[col_order]

    row_mean = np.nanmean(M, axis=1)
    row_order = np.argsort(-np.nan_to_num(row_mean, nan=-np.inf))
    M, rbps = M[row_order], [rbps[k] for k in row_order]

    if add_mean_row:
        M = np.vstack([M, col_mean])
        rbps = rbps + ["MEAN"]

    vmin, vmax = np.nanmin(M), np.nanmax(M)
    span = vmax - vmin or 1.0
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.85")  # grey for missing

    n_rows, n_cols = M.shape
    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 2.5, 0.5 * n_rows + 1.8))
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rbps)

    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="0.4", fontsize=8)
            else:
                # white text on dark cells, black on light, for legibility
                color = "white" if (v - vmin) / span < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=8)

    if add_mean_row:  # separate the MEAN row
        ax.axhline(n_rows - 1.5, color="white", linewidth=2)

    fig.colorbar(im, ax=ax, label=metric_label, fraction=0.046, pad=0.04)
    ax.set_title(f"RNAcompete RBP binding — {split} ({metric_label})")
    fig.tight_layout()
    return fig, ax


def single_vs_ensemble_bar(
    per_rbp: dict[str, dict],
    *,
    metric_label: str = "test Pearson r",
    split: str = "test",
    single_label: str = "Single members",
    ensemble_label: str = "Ensemble",
):
    """Grouped bars per RBP: a 'single' bar (height = mean of members, with a dot per member) next
    to an 'ensemble' bar (averaged-prediction, no dots).

    ``per_rbp[rbp] = {"members": [scores...], "ensemble": score}``. RBPs are ordered by ensemble
    score (desc). Returns ``(fig, ax)``.
    """
    rbps = sorted(per_rbp, key=lambda r: per_rbp[r]["ensemble"], reverse=True)
    if not rbps:
        raise ValueError("No RBPs to plot.")
    x = np.arange(len(rbps))
    w = 0.38
    single_means = [float(np.mean(per_rbp[r]["members"])) for r in rbps]
    ensemble = [per_rbp[r]["ensemble"] for r in rbps]

    fig, ax = plt.subplots(figsize=(1.05 * len(rbps) + 2, 5))
    ax.bar(x - w / 2, single_means, w, label=f"{single_label} (mean)",
           color="#9ecae1", edgecolor="black", zorder=2)
    ax.bar(x + w / 2, ensemble, w, label=ensemble_label,
           color="#3182bd", edgecolor="black", zorder=2)
    # one dot per member on the single bar; spread horizontally so near-identical members stay visible
    for i, r in enumerate(rbps):
        members = per_rbp[r]["members"]
        n = len(members)
        spread = np.linspace(-w * 0.32, w * 0.32, n) if n > 1 else np.zeros(1)
        ax.scatter((x[i] - w / 2) + spread, members,
                   color="black", s=16, alpha=0.8, zorder=3, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(rbps, rotation=45, ha="right")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Single members vs ensemble — {split}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    return fig, ax


# name -> (builder, default filename stem). Add new figure kinds here.
KINDS = {
    "heatmap": performance_heatmap,
}
