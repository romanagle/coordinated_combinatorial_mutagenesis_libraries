"""
mRNA_RBP/plot_library_distributions.py

3-panel library distribution figure:
  1. Activity-balanced library (3/5/7/15-mutation mix), stacked by mutation count
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

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS
from mRNA_RBP.gt_init import MrnaRbpGroundTruth

GT_KEY      = "scores_nonlin_additive_pairwise"   # key in .npz files
GT_KEY_LIVE = "nonlin_additive_pairwise"           # key in gt.score_all() dict
OUT_DIR  = os.path.join(_HERE, "outputs", "notebook_plots")

C_PAIR   = "#4C72B0"   # blue  (pairwise)
C_RAND   = "#55A868"   # green (random)

BINS     = 60
ALPHA    = 0.85


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


def main(save_svg: bool = False):
    inst_dir = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}")
    gt = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=INSTANCE)

    # Activity-balanced: recompute GT scores live (same as scatter plot)
    t2_path = os.path.join(inst_dir, "activity_balanced.npz")
    if not os.path.isfile(t2_path):
        t2_path = os.path.join(inst_dir, "type2.npz")
    if not os.path.isfile(t2_path):
        raise FileNotFoundError(f"Missing: {t2_path}")
    d_t2 = np.load(t2_path)
    type2_X    = np.eye(4, dtype=np.float32)[d_t2["nuc_ids"]]
    scores_t2  = gt.score_all(type2_X)[GT_KEY_LIVE].astype(np.float64)
    labels_t2  = d_t2["rate_labels"].astype(int)
    type2_mut_range = sorted(set(labels_t2.tolist()))
    _cmap_t2 = cm.get_cmap("plasma_r", len(type2_mut_range) + 2)
    type2_mut_colors = {m: _cmap_t2(i + 1) for i, m in enumerate(type2_mut_range)}

    scores_pw   = load_instance(os.path.join(inst_dir, "pairwise_lib.npz"))
    scores_rand = load_instance(os.path.join(inst_dir, "mut10", "lib_20000.npz"))

    # Load type3 for the combined figure
    t3_path_early = os.path.join(inst_dir, "type3.npz")
    has_t3 = os.path.isfile(t3_path_early)
    if has_t3:
        d_t3_early   = np.load(t3_path_early)
        _t3e_X       = np.eye(4, dtype=np.float32)[d_t3_early["nuc_ids"]]
        scores_t3e   = gt.score_all(_t3e_X)[GT_KEY_LIVE].astype(np.float64)
        labels_t3e   = d_t3_early["rate_labels"].astype(np.uint8)
        mut_range_t3e = sorted(set(labels_t3e.tolist()))
        cmap_t3e = cm.get_cmap("viridis", len(mut_range_t3e) + 2)
        colors_t3e = {m: cmap_t3e(i + 1) for i, m in enumerate(mut_range_t3e)}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.35)

    # Panel 1 — activity-balanced, stacked by mutation count
    ax = axes[0]
    all_bins = make_bins(scores_t2)
    bottoms = np.zeros(len(all_bins) - 1)
    widths  = np.diff(all_bins)
    total   = len(scores_t2)
    for m in type2_mut_range:
        mask = labels_t2 == m
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(scores_t2[mask], bins=all_bins)
        density = counts / (total * widths)
        ax.bar(all_bins[:-1], density, width=widths, bottom=bottoms,
               color=type2_mut_colors[m], align="edge", alpha=0.92,
               label=f"{m}-mut (n={mask.sum():,})")
        bottoms += density
    mean_line = ax.axvline(np.mean(scores_t2), color="k", ls="--", lw=1,
                           label=f"mean={np.mean(scores_t2):.3f}")
    ax.set_title(f"Activity-balanced library  (n={len(scores_t2):,})", fontsize=10)
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

    # Panel 3 — Type3 (exhaustive), stacked by mutation count
    ax = axes[2]
    if has_t3:
        all_bins_t3c = make_bins(scores_t3e)
        bottoms_t3c  = np.zeros(len(all_bins_t3c) - 1)
        widths_t3c   = np.diff(all_bins_t3c)
        total_t3c    = len(scores_t3e)
        for m in mut_range_t3e:
            mask = labels_t3e == m
            if mask.sum() == 0:
                continue
            counts, _ = np.histogram(scores_t3e[mask], bins=all_bins_t3c)
            density = counts / (total_t3c * widths_t3c)
            ax.bar(all_bins_t3c[:-1], density, width=widths_t3c, bottom=bottoms_t3c,
                   color=colors_t3e[m], align="edge", alpha=0.92,
                   label=f"{m}-mut (n={mask.sum():,})")
            bottoms_t3c += density
        ax.axvline(np.mean(scores_t3e), color="k", ls="--", lw=1,
                   label=f"mean={np.mean(scores_t3e):.3f}")
        ax.set_title(f"Type3 exhaustive library  (n={len(scores_t3e):,})", fontsize=10)
        ax.set_xlabel("GT score (nonlin additive pairwise)")
        handles_t3c, lbls_t3c = ax.get_legend_handles_labels()
        mean_h3c = handles_t3c.pop(-1); mean_l3c = lbls_t3c.pop(-1)
        ax.legend(handles_t3c + [mean_h3c], lbls_t3c + [mean_l3c],
                  fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)
    else:
        ax.set_visible(False)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "library_distributions.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_png}")
    if save_svg:
        out_svg = os.path.join(OUT_DIR, "library_distributions.svg")
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"Saved → {out_svg}")
    plt.close(fig)

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
    out_pw_png = os.path.join(OUT_DIR, "pairwise_library_distribution.png")
    fig_pw.savefig(out_pw_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_pw_png}")
    if save_svg:
        out_pw_svg = os.path.join(OUT_DIR, "pairwise_library_distribution.svg")
        fig_pw.savefig(out_pw_svg, bbox_inches="tight")
        print(f"Saved → {out_pw_svg}")
    plt.close(fig_pw)

    # Standalone random library figure — 3 panels side by side (5%, 10%, 25%)
    mut_rates = [("mut05", "5%"), ("mut10", "10%"), ("mut25", "25%")]
    scores_rand = []
    for mut_dir, label in mut_rates:
        path_r = os.path.join(inst_dir, mut_dir, "lib_20000.npz")
        scores_rand.append(load_instance(path_r))
    bins_r = make_bins(np.concatenate(scores_rand))
    xlim_r = (bins_r[0], bins_r[-1])

    fig_rand, axes_rand = plt.subplots(1, 3, figsize=(13, 4.2))
    fig_rand.subplots_adjust(wspace=0.35)
    for ax_r, (mut_dir, label), s in zip(axes_rand, mut_rates, scores_rand):
        ax_r.hist(s, bins=bins_r, density=True, color=C_RAND, alpha=ALPHA)
        ax_r.axvline(np.mean(s), color="k", ls="--", lw=1,
                     label=f"mean={np.mean(s):.3f}")
        ax_r.set_title(f"Random lib ({label} mut, n={len(s):,})", fontsize=10)
        ax_r.set_xlabel("GT score (nonlin additive pairwise)")
        ax_r.set_ylabel("Density")
        ax_r.set_xlim(xlim_r)
        ax_r.legend(fontsize=8, framealpha=0.8)
    out_rand_png = os.path.join(OUT_DIR, "rand_lib_dist.png")
    fig_rand.savefig(out_rand_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_rand_png}")
    if save_svg:
        out_rand_svg = os.path.join(OUT_DIR, "rand_lib_dist.svg")
        fig_rand.savefig(out_rand_svg, bbox_inches="tight")
        print(f"Saved → {out_rand_svg}")
    plt.close(fig_rand)

    # Standalone activity-balanced library figure
    fig_t2, ax_t2 = plt.subplots(figsize=(4.5, 4.2))
    all_bins_t2 = make_bins(scores_t2)
    bottoms_t2 = np.zeros(len(all_bins_t2) - 1)
    widths_t2 = np.diff(all_bins_t2)
    total_t2 = len(scores_t2)
    for m in type2_mut_range:
        mask = labels_t2 == m
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(scores_t2[mask], bins=all_bins_t2)
        density = counts / (total_t2 * widths_t2)
        ax_t2.bar(all_bins_t2[:-1], density, width=widths_t2, bottom=bottoms_t2,
                  color=type2_mut_colors[m], align="edge", alpha=0.92,
                  label=f"{m}-mut (n={mask.sum():,})")
        bottoms_t2 += density
    mean_h_t2 = ax_t2.axvline(np.mean(scores_t2), color="k", ls="--", lw=1,
                               label=f"mean={np.mean(scores_t2):.3f}")
    ax_t2.set_title(f"Activity-balanced library  (n={len(scores_t2):,})", fontsize=10)
    ax_t2.set_xlabel("GT score (nonlin additive pairwise)")
    ax_t2.set_ylabel("Density")
    handles_t2, lbls_t2 = ax_t2.get_legend_handles_labels()
    mean_h2 = handles_t2.pop(-1); mean_l2 = lbls_t2.pop(-1)
    ax_t2.legend(handles_t2 + [mean_h2], lbls_t2 + [mean_l2],
                 fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)
    out_t2_png = os.path.join(OUT_DIR, "activity_balanced_lib_dist.png")
    fig_t2.savefig(out_t2_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_t2_png}")
    if save_svg:
        out_t2_svg = os.path.join(OUT_DIR, "activity_balanced_lib_dist.svg")
        fig_t2.savefig(out_t2_svg, bbox_inches="tight")
        print(f"Saved → {out_t2_svg}")
    plt.close(fig_t2)

    # Standalone type3 (exhaustive) library figure — stacked by mutation count (1/2/3-mut)
    t3_path = os.path.join(inst_dir, "type3.npz")
    if not os.path.isfile(t3_path):
        print(f"[skip] type3.npz not found at {t3_path}")
    else:
        d_t3 = np.load(t3_path)
        type3_X   = np.eye(4, dtype=np.float32)[d_t3["nuc_ids"]]
        scores_t3 = gt.score_all(type3_X)[GT_KEY_LIVE].astype(np.float64)
        labels_t3 = d_t3["rate_labels"].astype(np.uint8)
        mut_range_t3 = sorted(set(labels_t3.tolist()))
        cmap_t3 = cm.get_cmap("viridis", len(mut_range_t3) + 2)
        colors_t3 = {m: cmap_t3(i + 1) for i, m in enumerate(mut_range_t3)}

        fig_t3, ax_t3 = plt.subplots(figsize=(4.5, 4.2))
        all_bins_t3 = make_bins(scores_t3)
        bottoms_t3 = np.zeros(len(all_bins_t3) - 1)
        widths_t3 = np.diff(all_bins_t3)
        total_t3 = len(scores_t3)
        for m in mut_range_t3:
            mask = labels_t3 == m
            if mask.sum() == 0:
                continue
            counts, _ = np.histogram(scores_t3[mask], bins=all_bins_t3)
            density = counts / (total_t3 * widths_t3)
            ax_t3.bar(all_bins_t3[:-1], density, width=widths_t3, bottom=bottoms_t3,
                      color=colors_t3[m], align="edge", alpha=0.92,
                      label=f"{m}-mut (n={mask.sum():,})")
            bottoms_t3 += density
        mean_h_t3 = ax_t3.axvline(np.mean(scores_t3), color="k", ls="--", lw=1,
                                   label=f"mean={np.mean(scores_t3):.3f}")
        ax_t3.set_title(f"Type3 exhaustive library  (n={len(scores_t3):,})", fontsize=10)
        ax_t3.set_xlabel("GT score (nonlin additive pairwise)")
        ax_t3.set_ylabel("Density")
        handles_t3, lbls_t3 = ax_t3.get_legend_handles_labels()
        mean_h3 = handles_t3.pop(-1); mean_l3 = lbls_t3.pop(-1)
        ax_t3.legend(handles_t3 + [mean_h3], lbls_t3 + [mean_l3],
                     fontsize=8, framealpha=0.85, loc="upper left", handlelength=1.2)
        out_t3_png = os.path.join(OUT_DIR, "type3_lib_dist.png")
        fig_t3.savefig(out_t3_png, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_t3_png}")
        if save_svg:
            out_t3_svg = os.path.join(OUT_DIR, "type3_lib_dist.svg")
            fig_t3.savefig(out_t3_svg, bbox_inches="tight")
            print(f"Saved → {out_t3_svg}")
        plt.close(fig_t3)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    main(save_svg=args.svg)
