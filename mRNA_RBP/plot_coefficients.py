"""
mRNA_RBP/plot_coefficients.py

Generate:
  outputs/notebook_plots/coefficients_gt.png       – GT alpha + beta contact map
  outputs/notebook_plots/coefficients_surrogate.png – surrogate alpha + beta

Default condition: instance 0, mut_rate=10%, lib_size=20000, nl+pair surrogate.

Usage:
    python mRNA_RBP/plot_coefficients.py
    python mRNA_RBP/plot_coefficients.py --mut_rate 25 --lib_size 20000
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

COEF_DIR  = os.path.join(_HERE, "outputs", "surrogate_coefs")
OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20000)
    args = parser.parse_args()

    gt   = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS,
                               seed=args.instance, stem_sigma=STEM_SIGMA)
    L    = len(SEQ)

    coef_path = os.path.join(
        COEF_DIR,
        f"coefs_k{args.instance:02d}_mut{args.mut_rate:02d}_lib{args.lib_size}"
        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
    )
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(f"Coef file not found: {coef_path}")

    coefs  = np.load(coef_path)
    y_std  = float(coefs["y_std"]) if "y_std" in coefs else 1.0
    sur_alpha = coefs["alpha"] * y_std
    sur_beta  = {(int(i), int(j)): coefs["J"][i, :, j, :] * y_std
                 for (i, j) in STEM_PAIRS}

    os.makedirs(OUT_DIR, exist_ok=True)

    fig_gt, fig_sur = viz.plot_gt_vs_surrogate(gt, sur_alpha, sur_beta, L)

    fig_gt.savefig(os.path.join(OUT_DIR, "coefficients_gt.png"),
                   dpi=150, bbox_inches="tight")
    fig_gt.savefig(os.path.join(OUT_DIR, "coefficients_gt.svg"),
                   bbox_inches="tight")
    plt.close(fig_gt)
    print(f"Saved → coefficients_gt.png / .svg")

    fig_sur.savefig(os.path.join(OUT_DIR, "coefficients_surrogate.png"),
                    dpi=150, bbox_inches="tight")
    fig_sur.savefig(os.path.join(OUT_DIR, "coefficients_surrogate.svg"),
                    bbox_inches="tight")
    plt.close(fig_sur)
    print(f"Saved → coefficients_surrogate.png / .svg")


if __name__ == "__main__":
    main()
