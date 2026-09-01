"""
Generate the flat ResidualBind VTS1 collection figures that mirror the
Synthetic GT collection, excluding coefficient maps.

Inputs are the freshly generated ResidualBind VTS1 pipeline libraries stored
under libraries_used_for_figures/instance_00.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from mRNA_RBP.scripts.figures.core.provenance import stamp_figure  # noqa: E402

COLLECTION_DIR = (
    REPO
    / "mRNA_RBP"
    / "outputs"
    / "ground_truth_collections"
    / "ResidualBind oracle VTS1"
)
FIG_DIR = COLLECTION_DIR / "figures"
LIB_DIR = COLLECTION_DIR / "libraries_used_for_figures"
BINS = 60
C_PAIR = "#4C72B0"
ALPHA = 0.85
NUCS = ("A", "C", "G", "U")
SCORE_KEY = "scores_vts1_residualbind"
LIBRARY_SOURCE_PATHS = (
    LIB_DIR / "activity_balanced.npz",
    LIB_DIR / "pairwise_lib.npz",
    LIB_DIR / "type3.npz",
)


def make_bins(scores, n=BINS, xlim=None):
    if xlim is None:
        lo, hi = np.percentile(scores, [0.2, 99.8])
        pad = max((hi - lo) * 0.03, 1e-6)
        lo -= pad
        hi += pad
    else:
        lo, hi = xlim
    return np.linspace(lo, hi, n + 1)


def stacked_hist(ax, scores, labels, title, xlabel, cmap_name="plasma_r", xlim=None):
    mut_range = sorted(set(labels.astype(int).tolist()))
    colors = {
        m: cm.get_cmap(cmap_name, len(mut_range) + 2)(i + 1)
        for i, m in enumerate(mut_range)
    }
    bins = make_bins(scores, xlim=xlim)
    bottoms = np.zeros(len(bins) - 1)
    widths = np.diff(bins)
    total = len(scores)
    for m in mut_range:
        mask = labels == m
        if not mask.any():
            continue
        counts, _ = np.histogram(scores[mask], bins=bins)
        density = counts / (total * widths)
        ax.bar(
            bins[:-1],
            density,
            width=widths,
            bottom=bottoms,
            color=colors[m],
            align="edge",
            alpha=0.92,
            label=f"{m}-mut (n={mask.sum():,})",
        )
        bottoms += density
    ax.axvline(np.mean(scores), color="k", ls="--", lw=1, label=f"mean={np.mean(scores):.3f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(xlim)
    handles, labels_text = ax.get_legend_handles_labels()
    mean_h = handles.pop(-1)
    mean_l = labels_text.pop(-1)
    ax.legend(
        handles + [mean_h],
        labels_text + [mean_l],
        fontsize=8,
        framealpha=0.85,
        loc="upper left",
        handlelength=1.2,
    )


def plain_hist(ax, scores, title, xlabel, color, xlim=None):
    bins = make_bins(scores, xlim=xlim)
    ax.hist(scores, bins=bins, density=True, color=color, alpha=ALPHA)
    ax.axvline(np.mean(scores), color="k", ls="--", lw=1, label=f"mean={np.mean(scores):.3f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(fontsize=8, framealpha=0.8)


def load_pipeline_libraries():
    activity = np.load(LIB_DIR / "activity_balanced.npz")
    pairwise = np.load(LIB_DIR / "pairwise_lib.npz")
    type3 = np.load(LIB_DIR / "type3.npz")
    return {
        "activity_scores": activity[SCORE_KEY].astype(float),
        "activity_labels": activity["rate_labels"].astype(int),
        "pairwise_scores": pairwise[SCORE_KEY].astype(float),
        "pairwise_nids": pairwise["nuc_ids"].astype(np.uint8),
        "pairwise_edges": pairwise["edges"].astype(int),
        "type3_scores": type3[SCORE_KEY].astype(float),
        "type3_labels": type3["rate_labels"].astype(int),
    }


def save_distribution_figures(cache):
    lib_dist_dir = FIG_DIR / "library_distributions"
    lib_dist_dir.mkdir(parents=True, exist_ok=True)
    xlabel = "ResidualBind VTS1 score relative to WT"
    activity_scores = cache["activity_scores"]
    activity_labels = cache["activity_labels"]
    pairwise_scores = cache["pairwise_scores"]
    type3_scores = cache["type3_scores"]
    type3_labels = cache["type3_labels"]
    all_lib = np.concatenate([activity_scores, pairwise_scores, type3_scores])
    xlim_lib = (make_bins(all_lib)[0], make_bins(all_lib)[-1])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.35)
    stacked_hist(
        axes[0],
        activity_scores,
        activity_labels,
        f"Activity-balanced library  (n={len(activity_scores):,})",
        xlabel,
        "plasma_r",
        xlim_lib,
    )
    plain_hist(
        axes[1],
        pairwise_scores,
        f"Pairwise library  (n={len(pairwise_scores):,})",
        xlabel,
        C_PAIR,
        xlim_lib,
    )
    stacked_hist(
        axes[2],
        type3_scores,
        type3_labels,
        f"Type3 exhaustive library  (n={len(type3_scores):,})",
        xlabel,
        "viridis",
        xlim_lib,
    )
    stamp_figure(fig, library_status="fresh", source_paths=LIBRARY_SOURCE_PATHS)
    fig.savefig(lib_dist_dir / "library_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    stacked_hist(
        ax,
        activity_scores,
        activity_labels,
        f"Activity-balanced library  (n={len(activity_scores):,})",
        xlabel,
        "plasma_r",
    )
    stamp_figure(fig, library_status="fresh", source_paths=[LIB_DIR / "activity_balanced.npz"])
    fig.savefig(lib_dist_dir / "activity_balanced_lib_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    stacked_hist(
        ax,
        type3_scores,
        type3_labels,
        f"Type3 exhaustive library  (n={len(type3_scores):,})",
        xlabel,
        "viridis",
    )
    stamp_figure(fig, library_status="fresh", source_paths=[LIB_DIR / "type3.npz"])
    fig.savefig(lib_dist_dir / "type3_lib_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_heatmaps(cache):
    nuc_ids = cache["pairwise_nids"]
    scores = cache["pairwise_scores"]
    stem_pairs = [tuple(row.tolist()) for row in cache["pairwise_edges"]]
    n_pairs = len(stem_pairs)
    if len(scores) != 16 * n_pairs:
        raise ValueError(f"Expected {16 * n_pairs} pairwise scores, got {len(scores)}")

    mats = []
    for pidx in range(n_pairs):
        block_ids = nuc_ids[pidx * 16 : (pidx + 1) * 16]
        block_scores = scores[pidx * 16 : (pidx + 1) * 16]
        i, j = stem_pairs[pidx]
        mat = np.empty((4, 4), dtype=float)
        for row, score in zip(block_ids, block_scores):
            mat[int(row[i]), int(row[j])] = score
        mats.append(mat)

    vmax = max(np.max(np.abs(m)) for m in mats)
    fig, axes = plt.subplots(1, n_pairs, figsize=(2.35 * n_pairs, 2.65), sharex=True, sharey=True)
    for ax, mat, pair in zip(axes, mats, stem_pairs):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{pair[0]}-{pair[1]}", fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(NUCS, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(NUCS, fontsize=8)
        ax.set_xlabel("j nucleotide", fontsize=8)
        ax.set_ylabel("i nucleotide", fontsize=8)
        for r in range(4):
            for c in range(4):
                ax.text(c, r, f"{mat[r, c]:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="ResidualBind score relative to WT")
    fig.suptitle("VTS1 targeted pairwise library", fontsize=11, y=1.03)
    stamp_figure(fig, library_status="fresh", source_paths=[LIB_DIR / "pairwise_lib.npz"])
    coef_dir = FIG_DIR / "coefficients"
    coef_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(coef_dir / "pairwise_heatmaps_annotated.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cache = load_pipeline_libraries()
    save_distribution_figures(cache)
    save_pairwise_heatmaps(cache)
    print(f"Saved ResidualBind VTS1 collection figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
