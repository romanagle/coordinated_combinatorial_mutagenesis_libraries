"""
mRNA_RBP/plots/plot_coefficients_surrogate_residualbind.py

Companion figure to coefficients_residualbind[_<tag>].png: instead of the raw
ResidualBind ensemble SSM + pairwise-lib readouts, this plots the additive
(alpha) and pairwise (J) coefficients of a MAVE-NN nonlinear-additive+pairwise
surrogate *trained on* ResidualBind ensemble oracle labels (20K training
library, 10% mutation rate, instance 0). Same viz.plot_coefficients layout as
coefficients_residualbind.png, so the two can be compared side by side.

Works for either ensemble -- MSI1 (default) or VTS1 -- these are two
distinct, independently-trained ResidualBind ensembles, never conflate them
(see oracles.RESIDUALBIND_MSI1_ORACLE / _VTS1_ORACLE):

  MSI1 (default): outputs_vts1_residualbind/outputs_residualbind_ensemble/
    surrogate_coefs/coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_residualbind_ensemble.npz
  VTS1: outputs/surrogate_coefs/
    coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz
    (produced by experiments/train_surrogate_from_score_cache.py)

Output:
  outputs/notebook_plots/coefficients_map/coefficients_surrogate_residualbind[_<tag>].png / .svg

Usage:
    python mRNA_RBP/plots/plot_coefficients_surrogate_residualbind.py
    python mRNA_RBP/plots/plot_coefficients_surrogate_residualbind.py \
        --coef_path outputs/surrogate_coefs/coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz \
        --out_name coefficients_surrogate_residualbind_vts1 \
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

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS
import mRNA_RBP.viz as viz

OUT_DIR = os.path.join(_HERE, "outputs", "notebook_plots", "coefficients_map")

DEFAULT_COEF_PATH = os.path.join(
    _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
    "surrogate_coefs",
    "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_residualbind_ensemble.npz",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coef_path", default=DEFAULT_COEF_PATH)
    parser.add_argument("--out_name", default="coefficients_surrogate_residualbind")
    parser.add_argument("--label", default="ResidualBind ensemble")
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    if not os.path.isfile(args.coef_path):
        raise FileNotFoundError(
            f"Coef file not found: {args.coef_path}\n"
            "Run lib_size_spearman.py / train_surrogate.py with "
            "--oracle residualbind_ensemble first."
        )

    coefs  = np.load(args.coef_path)
    L      = len(SEQ)
    y_std  = float(coefs["y_std"]) if "y_std" in coefs else 1.0

    alpha = coefs["alpha"].astype(np.float64) * y_std          # (L, 4)
    J     = coefs["J"].astype(np.float64) * y_std              # (L, 4, L, 4)
    beta  = {(i, j): J[i, :, j, :] for (i, j) in STEM_PAIRS
             if i < J.shape[0] and j < J.shape[2]}

    os.makedirs(OUT_DIR, exist_ok=True)

    fig = viz.plot_coefficients(
        alpha, beta, L,
        stem_pairs=STEM_PAIRS, motif_positions=MOTIF_POSITIONS,
        title=f"Nonlinear additive+pairwise surrogate — trained on {args.label} "
              "(10% mut, 20K lib)",
    )
    out = os.path.join(OUT_DIR, args.out_name + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    msg = out
    if args.svg:
        fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
        msg = f"{out} / {args.out_name}.svg"
    plt.close(fig)
    print(f"Saved → {msg}")


if __name__ == "__main__":
    main()
