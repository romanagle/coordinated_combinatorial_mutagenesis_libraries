"""
Plot mean oracle score vs. number of mutations from WT for an RBP.

Usage examples:
    # VTS1 using the synthetic mRNA-RBP ground-truth oracle (default)
    python mRNA_RBP/plot_rbp_fitness_vs_nmut.py --rbp vts1 --out_path mRNA_RBP/outputs/notebook_plots/rbp_fitness_vs_nmut_vts1.png

    # MSI1 using existing ground-truth libraries
    python mRNA_RBP/plot_rbp_fitness_vs_nmut.py --rbp msi1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from mRNA_RBP.generate_libraries import STEM_SIGMA, generate_pool
from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth

# MSI1 (default sequence used by the main pipeline)
MSI1_SEQ = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS = [(8, 30), (9, 29), (10, 28), (11, 27), (12, 26), (13, 25), (14, 24), (15, 23)]
MSI1_MOTIF_POSITIONS = [17, 18, 19, 20, 21]


def get_rbp_spec(rbp: str):
    rbp = rbp.lower().strip()
    if rbp in ("msi1", "rncmpt00176"):
        return "MSI1", MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_MOTIF_POSITIONS
    if rbp in ("vts1", "rncmpt00111"):
        return "VTS1", VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS
    raise ValueError(f"Unknown RBP {rbp!r}. Choose 'msi1' or 'vts1'.")


def load_gt_scores_for_counts(
    name: str,
    seq: str,
    stem_pairs: list[tuple[int, int]],
    motif_positions: list[int],
    max_mut: int,
    n_per_count: int,
    seed: int,
    score_key: str = "nonlin_additive_pairwise",
):
    """Generate random sequences at each mutation count and score with MrnaRbpGroundTruth."""
    gt = MrnaRbpGroundTruth(seq, stem_pairs, motif_positions, seed=seed, stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
    rng = np.random.default_rng(seed)

    counts, means, stds = [], [], []
    for mc in range(max_mut + 1):
        if mc == 0:
            ids = wt_idx[None, :]
        else:
            ids = generate_pool(wt_oh, n_per_count, mc, rng)
            ids = np.unique(ids, axis=0)
        x = np.eye(4, dtype=np.float32)[ids]
        scores = gt.score_all(x)[score_key]
        counts.append(mc)
        means.append(float(scores.mean()))
        stds.append(float(scores.std()))
        print(f"  mut={mc:2d}  n={len(ids):5d}  mean={means[-1]:.4f}  std={stds[-1]:.4f}")

    return np.array(counts), np.array(means), np.array(stds)


def plot(
    counts: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    rbp_label: str,
    wt_score: float,
    out_path: str,
):
    """Reproduce the rbp_fitness_vs_nmut style."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6.9, 3.9))

    color = "#c44e52"
    ax.plot(counts, means, "o-", color=color, lw=2.5, ms=7, zorder=3)
    ax.fill_between(counts, means - stds, means + stds, color=color, alpha=0.18, zorder=2)

    ax.axhline(
        wt_score, color="black", linestyle="--", lw=1.3, zorder=1, label=f"WT ({wt_score:.2f})"
    )
    ax.legend(loc="lower left", fontsize=10, frameon=True, edgecolor="gray")

    ax.set_xlabel("Number of mutations from WT", fontsize=13)
    ax.set_ylabel("Mean oracle score", fontsize=13)
    rbp_title = "RNCMPT00111 / VTS1" if rbp_label == "VTS1" else "RNCMPT00176 / MSI1"
    ax.set_title(
        f"RBP binding ({rbp_title})\nmean oracle score vs mutation count",
        fontsize=13,
    )
    ax.set_xticks(counts[::2])
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rbp", default="vts1", choices=["msi1", "vts1"])
    ap.add_argument("--max_mut", type=int, default=30)
    ap.add_argument("--n_per_count", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_path",
        default=None,
        help="Output PNG path. Default: mRNA_RBP/outputs/notebook_plots/rbp_fitness_vs_nmut_<rbp>.png",
    )
    args = ap.parse_args()

    name, seq, stem_pairs, motif_positions = get_rbp_spec(args.rbp)
    out_path = args.out_path or str(
        _HERE / "outputs" / "notebook_plots" / f"rbp_fitness_vs_nmut_{name.lower()}.png"
    )

    print(f"Building {name} fitness-vs-n-mut plot (GT oracle, seed={args.seed})...")
    counts, means, stds = load_gt_scores_for_counts(
        name, seq, stem_pairs, motif_positions, args.max_mut, args.n_per_count, args.seed
    )
    plot(counts, means, stds, name, wt_score=0.0, out_path=out_path)


if __name__ == "__main__":
    main()
