"""
mRNA_RBP/plot_library_distributions.py

3-panel library distribution figure:
  1. Type2 library (uniform, 2/3/4-mutation mix), stacked by mutation count
  2. Pairwise library (exhaustive stem-pair co-mutations)
  3. Random library (10% mut, 20k)

Output: outputs/notebook_plots/library_distributions.png / .svg
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import OUT_BASE

GT_KEY   = "scores_nonlin_additive_pairwise"
OUT_DIR  = os.path.join(_HERE, "outputs", "notebook_plots")

C_PAIR   = "#4C72B0"   # blue  (pairwise)
C_RAND   = "#55A868"   # green (random)

BINS     = 60
ALPHA    = 0.85

# Mutation counts in type2 (2, 3, 4) — colour palette
TYPE2_MUT_RANGE = [2, 3, 4]
_CMAP           = cm.get_cmap("plasma_r", len(TYPE2_MUT_RANGE) + 2)
TYPE2_MUT_COLORS = {m: _CMAP(i + 1) for i, m in enumerate(TYPE2_MUT_RANGE)}

INSTANCE = 0   # single instance to display


def load_score(path):
    return np.load(path)[GT_KEY].astype(np.float64)


def load_instance(filename):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Missing: {filename}")
    return load_score(filename)


def make_bins(scores, n=BINS):
    lo, hi = np.percentile(scores, [0.2, 99.8])
    pad = (hi - lo) * 0.03
    return np.linspace(lo - pad, hi + pad, n + 1)


def main():
    inst_dir = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}")

    # Type2: scores + rate_labels (2/3/4-mut)
    t2_path = os.path.join(inst_dir, "type2.npz")
    if not os.path.isfile(t2_path):
        raise FileNotFoundError(f"Missing: {t2_path}")
    d_t2 = np.load(t2_path)
    scores_t2  = d_t2[GT_KEY].astype(np.float64)
    labels_t2  = d_t2["rate_labels"].astype(np.uint8)

    scores_pw   = load_instance(os.path.join(inst_dir, "pairwise_lib.npz"))
    scores_rand = load_instance(os.path.join(inst_dir, "mut10", "lib_20000.npz"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.35)

    # Panel 1 — Type2, stacked by mutation count
    ax = axes[0]
    all_bins = make_bins(scores_t2)
    bottoms = np.zeros(len(all_bins) - 1)
    widths  = np.diff(all_bins)
    total   = len(scores_t2)
    for m in TYPE2_MUT_RANGE:
        mask = labels_t2 == m
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(scores_t2[mask], bins=all_bins)
        density = counts / (total * widths)
        ax.bar(all_bins[:-1], density, width=widths, bottom=bottoms,
               color=TYPE2_MUT_COLORS[m], align="edge", alpha=0.92,
               label=f"{m}-mut (n={mask.sum():,})")
        bottoms += density
    mean_line = ax.axvline(np.mean(scores_t2), color="k", ls="--", lw=1,
                           label=f"mean={np.mean(scores_t2):.3f}")
    ax.set_title(f"Type2 library  (n={len(scores_t2):,})", fontsize=10)
    ax.set_xlabel("GT score (nonlin additive pairwise)")
    ax.set_ylabel("Density")
    handles, lbls = ax.get_legend_handles_labels()
    mean_h = handles.pop(-1); mean_l = lbls.pop(-1)
    ax.legend(handles + [mean_h], lbls + [mean_l],
              fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)

    # Panel 2 — Pairwise
    ax = axes[1]
    bins = make_bins(scores_pw)
    ax.hist(scores_pw, bins=bins, density=True, color=C_PAIR, alpha=ALPHA)
    ax.axvline(np.mean(scores_pw), color="k", ls="--", lw=1,
               label=f"mean={np.mean(scores_pw):.3f}")
    ax.set_title(f"Pairwise library  (n={len(scores_pw):,})", fontsize=10)
    ax.set_xlabel("GT score (nonlin additive pairwise)")
    ax.legend(fontsize=8, framealpha=0.8)

    # Panel 3 — Random
    ax = axes[2]
    bins = make_bins(scores_rand)
    ax.hist(scores_rand, bins=bins, density=True, color=C_RAND, alpha=ALPHA)
    ax.axvline(np.mean(scores_rand), color="k", ls="--", lw=1,
               label=f"mean={np.mean(scores_rand):.3f}")
    ax.set_title(f"Random lib (10% mut, n={len(scores_rand):,})", fontsize=10)
    ax.set_xlabel("GT score (nonlin additive pairwise)")
    ax.legend(fontsize=8, framealpha=0.8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "library_distributions.png")
    out_svg = os.path.join(OUT_DIR, "library_distributions.svg")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_svg}")

    # Standalone pairwise figure
    fig_pw, ax_pw = plt.subplots(figsize=(4.5, 4.2))
    bins_pw = make_bins(scores_pw)
    ax_pw.hist(scores_pw, bins=bins_pw, density=True, color=C_PAIR, alpha=ALPHA)
    ax_pw.axvline(np.mean(scores_pw), color="k", ls="--", lw=1,
                  label=f"mean={np.mean(scores_pw):.3f}")
    ax_pw.set_title(f"Pairwise library  (n={len(scores_pw):,})", fontsize=10)
    ax_pw.set_xlabel("GT score (nonlin additive pairwise)")
    ax_pw.set_ylabel("Density")
    ax_pw.legend(fontsize=8, framealpha=0.8)
    out_pw_svg = os.path.join(OUT_DIR, "pairwise_library_distribution.svg")
    out_pw_png = os.path.join(OUT_DIR, "pairwise_library_distribution.png")
    fig_pw.savefig(out_pw_svg, bbox_inches="tight")
    fig_pw.savefig(out_pw_png, dpi=150, bbox_inches="tight")
    plt.close(fig_pw)
    print(f"Saved → {out_pw_svg}")
    print(f"Saved → {out_pw_png}")

    # Standalone random library figure
    fig_rand, ax_rand = plt.subplots(figsize=(4.5, 4.2))
    bins_rand = make_bins(scores_rand)
    ax_rand.hist(scores_rand, bins=bins_rand, density=True, color=C_RAND, alpha=ALPHA)
    ax_rand.axvline(np.mean(scores_rand), color="k", ls="--", lw=1,
                    label=f"mean={np.mean(scores_rand):.3f}")
    ax_rand.set_title(f"Random lib (10% mut, n={len(scores_rand):,})", fontsize=10)
    ax_rand.set_xlabel("GT score (nonlin additive pairwise)")
    ax_rand.set_ylabel("Density")
    ax_rand.legend(fontsize=8, framealpha=0.8)
    out_rand_svg = os.path.join(OUT_DIR, "rand_lib_dist.svg")
    out_rand_png = os.path.join(OUT_DIR, "rand_lib_dist.png")
    fig_rand.savefig(out_rand_svg, bbox_inches="tight")
    fig_rand.savefig(out_rand_png, dpi=150, bbox_inches="tight")
    plt.close(fig_rand)
    print(f"Saved → {out_rand_svg}")
    print(f"Saved → {out_rand_png}")

    # Standalone type2 library figure
    fig_t2, ax_t2 = plt.subplots(figsize=(4.5, 4.2))
    all_bins_t2 = make_bins(scores_t2)
    bottoms_t2 = np.zeros(len(all_bins_t2) - 1)
    widths_t2 = np.diff(all_bins_t2)
    total_t2 = len(scores_t2)
    for m in TYPE2_MUT_RANGE:
        mask = labels_t2 == m
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(scores_t2[mask], bins=all_bins_t2)
        density = counts / (total_t2 * widths_t2)
        ax_t2.bar(all_bins_t2[:-1], density, width=widths_t2, bottom=bottoms_t2,
                  color=TYPE2_MUT_COLORS[m], align="edge", alpha=0.92,
                  label=f"{m}-mut (n={mask.sum():,})")
        bottoms_t2 += density
    mean_h_t2 = ax_t2.axvline(np.mean(scores_t2), color="k", ls="--", lw=1,
                               label=f"mean={np.mean(scores_t2):.3f}")
    ax_t2.set_title(f"Type2 library  (n={len(scores_t2):,})", fontsize=10)
    ax_t2.set_xlabel("GT score (nonlin additive pairwise)")
    ax_t2.set_ylabel("Density")
    handles_t2, lbls_t2 = ax_t2.get_legend_handles_labels()
    mean_h2 = handles_t2.pop(-1); mean_l2 = lbls_t2.pop(-1)
    ax_t2.legend(handles_t2 + [mean_h2], lbls_t2 + [mean_l2],
                 fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)
    out_t2_svg = os.path.join(OUT_DIR, "type2_lib_dist.svg")
    out_t2_png = os.path.join(OUT_DIR, "type2_lib_dist.png")
    fig_t2.savefig(out_t2_svg, bbox_inches="tight")
    fig_t2.savefig(out_t2_png, dpi=150, bbox_inches="tight")
    plt.close(fig_t2)
    print(f"Saved → {out_t2_svg}")
    print(f"Saved → {out_t2_png}")


if __name__ == "__main__":
    main()
