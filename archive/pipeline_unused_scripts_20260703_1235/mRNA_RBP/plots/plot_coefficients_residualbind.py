"""
mRNA_RBP/plot_coefficients_residualbind.py

ResidualBind ensemble doesn't ship additive/pairwise coefficients the way the
synthetic MrnaRbpGroundTruth does (alpha, beta) -- it's just a CNN. This
script builds the equivalent grids directly from oracle readouts, with no
surrogate fitting in between:

  additive grid (4 x L)   = single-site mutant scores (ssm_delta), i.e. the
                             ResidualBind score of every (position, nuc)
                             single mutant, WT-anchored to 0.
  pairwise grid (4x4 x 8) = the 16 stem-pair double-mutant scores per stem
                             pair, taken directly from pairwise_lib.npz
                             ResidualBind scores (no model in between).

Both come from the cache produced by score_residualbind_type3.py:
  mRNA_RBP/outputs/residualbind_type3_scores_instance00.npz        (MSI1, default)
  mRNA_RBP/outputs/residualbind_type3_scores_vts1_instance00.npz   (VTS1)

MSI1 and VTS1 are two distinct, independently-trained ResidualBind ensembles
(see oracles.RESIDUALBIND_MSI1_ORACLE / _VTS1_ORACLE) -- do not conflate them.

Output (--tag "" for MSI1/default, --tag vts1 for VTS1):
  outputs/notebook_plots/coefficients_map/coefficients_residualbind[_<tag>].png / .svg
  outputs/notebook_plots/pairwise_grids_residualbind[_<tag>].png / .svg

Usage:
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/plots/plot_coefficients_residualbind.py
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/plots/plot_coefficients_residualbind.py \
        --tag vts1 \
        --score_cache outputs/residualbind_type3_scores_vts1_instance00.npz \
        --label "VTS1 ensemble"
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS
import mRNA_RBP.viz as viz

OUT_DIR = os.path.join(_HERE, "outputs", "notebook_plots", "coefficients_map")
NUCS_LABEL = ['A', 'C', 'G', 'U']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_cache", default=os.path.join(
        _HERE, "outputs", "residualbind_type3_scores_instance00.npz"))
    parser.add_argument("--tag", default="",
                        help="Filename suffix, e.g. 'vts1' for the VTS1 ensemble "
                             "(default '' keeps the original MSI1 filenames)")
    parser.add_argument("--label", default="ResidualBind ensemble",
                        help="Human-readable oracle label used in plot titles")
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    if not os.path.isfile(args.score_cache):
        raise FileNotFoundError(
            f"Score cache not found: {args.score_cache}\n"
            "Run score_residualbind_type3.py first (RBP torch env)."
        )
    suffix = f"_{args.tag}" if args.tag else ""

    d = np.load(args.score_cache)
    L = len(SEQ)

    # ── Additive grid: (4, L) ssm_delta -> (L, 4) alpha-equivalent ──────────
    delta_W = d["ssm_delta"].astype(np.float64)   # (4, L), [nuc, pos]
    alpha_rb = delta_W.T                          # (L, 4)

    # ── Pairwise grid: 128 pairwise_lib scores -> 8 x (4,4) blocks ──────────
    # generate_pairwise_lib enumerates stem_pairs, then ni in range(4), nj in
    # range(4) -- so consecutive runs of 16 map straight onto (ni, nj).
    scores_pw = d["scores_pairwise"].astype(np.float64)   # (128,)
    blocks = scores_pw.reshape(len(STEM_PAIRS), 4, 4)      # (8, 4, 4) [ni, nj]
    beta_rb = {pair: blocks[idx] for idx, pair in enumerate(STEM_PAIRS)}

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Figure 1: combined additive (4xL) + pairwise summary (LxL) ─────────
    fig = viz.plot_coefficients(
        alpha_rb, beta_rb, L,
        stem_pairs=STEM_PAIRS, motif_positions=MOTIF_POSITIONS,
        title=f"{args.label}: measured single + double mutant effects",
    )
    out1 = os.path.join(OUT_DIR, f"coefficients_residualbind{suffix}.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    msg1 = out1
    if args.svg:
        fig.savefig(out1.replace(".png", ".svg"), bbox_inches="tight")
        msg1 = f"{out1} / coefficients_residualbind{suffix}.svg"
    plt.close(fig)
    print(f"Saved → {msg1}")

    # ── Figure 2: full-detail 4x4 grid per stem pair ────────────────────────
    n = len(STEM_PAIRS)
    vmax = max(float(np.abs(scores_pw).max()), 1e-6)
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.RdBu

    fig2, axes = plt.subplots(1, n, figsize=(n * 2.6, 3.2),
                               gridspec_kw={"wspace": 0.35})
    for col, (i, j) in enumerate(STEM_PAIRS):
        ax = axes[col]
        ax.imshow(beta_rb[(i, j)], cmap=cmap, norm=norm, aspect="equal",
                  origin="upper", interpolation="nearest")
        ax.set_xticks(range(4)); ax.set_xticklabels(NUCS_LABEL, fontsize=7)
        ax.set_yticks(range(4)); ax.set_yticklabels(NUCS_LABEL, fontsize=7)
        ax.set_title(f"({i},{j})", fontsize=9)
        ax.set_xlabel("nuc j", fontsize=7)
        if col == 0:
            ax.set_ylabel("nuc i", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig2.colorbar(sm, ax=axes.tolist(), fraction=0.025, pad=0.02,
                  label=f"{args.label} score")
    fig2.suptitle(f"{args.label}: stem-pair double-mutant grids", fontsize=11, y=1.05)

    out2 = os.path.join(OUT_DIR, f"pairwise_grids_residualbind{suffix}.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    msg2 = out2
    if args.svg:
        fig2.savefig(out2.replace(".png", ".svg"), bbox_inches="tight")
        msg2 = f"{out2} / pairwise_grids_residualbind{suffix}.svg"
    plt.close(fig2)
    print(f"Saved → {msg2}")


if __name__ == "__main__":
    main()
