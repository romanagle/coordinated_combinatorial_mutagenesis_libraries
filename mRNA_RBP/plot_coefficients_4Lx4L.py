"""
mRNA_RBP/plot_coefficients_4Lx4L.py

Generate copies of the coefficient plots with a full 4L×4L pairwise panel:
  outputs/notebook_plots/coefficients_gt_4Lx4L.svg
  outputs/notebook_plots/coefficients_surrogate_4Lx4L.svg

Uses the same instance / mut_rate / lib_size as plot_coefficients.py (defaults:
instance=0, mut_rate=10, lib_size=20000).

Usage:
    python mRNA_RBP/plot_coefficients_4Lx4L.py
    python mRNA_RBP/plot_coefficients_4Lx4L.py --mut_rate 25 --lib_size 20000
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
import mRNA_RBP.viz as viz

COEF_DIR = os.path.join(_HERE, "outputs", "surrogate_coefs")
OUT_DIR  = os.path.join(_HERE, "outputs", "notebook_plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20000)
    args = parser.parse_args()

    gt = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS,
                             seed=args.instance, stem_sigma=STEM_SIGMA)
    L  = len(SEQ)

    coef_path = os.path.join(
        COEF_DIR,
        f"coefs_k{args.instance:02d}_mut{args.mut_rate:02d}_lib{args.lib_size}"
        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
    )
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(f"Coef file not found: {coef_path}")

    coefs     = np.load(coef_path)
    y_std     = float(coefs["y_std"]) if "y_std" in coefs else 1.0
    sur_alpha = coefs["alpha"] * y_std
    sur_beta  = {(int(i), int(j)): coefs["J"][i, :, j, :] * y_std
                 for (i, j) in STEM_PAIRS}

    os.makedirs(OUT_DIR, exist_ok=True)

    fig_gt = viz.plot_coefficients_4Lx4L(
        gt.alpha, gt.beta, L,
        stem_pairs=gt.stem_pairs,
        motif_positions=gt.motif_positions,
        title='Ground truth coefficients',
    )
    out_gt = os.path.join(OUT_DIR, "coefficients_gt_4Lx4L.svg")
    fig_gt.savefig(out_gt, bbox_inches="tight")
    plt.close(fig_gt)
    print(f"Saved → {out_gt}")

    fig_sur = viz.plot_coefficients_4Lx4L(
        sur_alpha, sur_beta, L,
        stem_pairs=gt.stem_pairs,
        motif_positions=gt.motif_positions,
        title='Surrogate (learned) coefficients',
    )
    out_sur = os.path.join(OUT_DIR, "coefficients_surrogate_4Lx4L.svg")
    fig_sur.savefig(out_sur, bbox_inches="tight")
    plt.close(fig_sur)
    print(f"Saved → {out_sur}")

    # ── Compact: i-rows only, full-width x-axis ───────────────────────────
    fig_gt_c = viz.plot_coefficients_stem_compact(
        gt.alpha, gt.beta, L,
        stem_pairs=gt.stem_pairs,
        motif_positions=gt.motif_positions,
        title='Ground truth coefficients (compact)',
    )
    out_gt_c = os.path.join(OUT_DIR, "coefficients_gt_stem_compact.svg")
    fig_gt_c.savefig(out_gt_c, bbox_inches="tight")
    plt.close(fig_gt_c)
    print(f"Saved → {out_gt_c}")

    fig_sur_c = viz.plot_coefficients_stem_compact(
        sur_alpha, sur_beta, L,
        stem_pairs=gt.stem_pairs,
        motif_positions=gt.motif_positions,
        title='Surrogate (learned) coefficients (compact)',
    )
    out_sur_c = os.path.join(OUT_DIR, "coefficients_surrogate_stem_compact.svg")
    fig_sur_c.savefig(out_sur_c, bbox_inches="tight")
    plt.close(fig_sur_c)
    print(f"Saved → {out_sur_c}")


if __name__ == "__main__":
    main()
