"""
mRNA_RBP/plot_pairwise_heatmaps.py

Generate outputs/notebook_plots/pairwise_heatmaps.png:
  3-row pairwise evaluation heatmap (GT vs surrogate vs SSM baseline)
  for the first 3 stem pairs, with a colorbar scale and no title.

Default condition: instance 0, mut_rate=10%, lib_size=20000,
surrogate config = nonlinear additive + pairwise.

Usage:
    python mRNA_RBP/plot_pairwise_heatmaps.py
    python mRNA_RBP/plot_pairwise_heatmaps.py --instance 0 --mut_rate 10 --lib_size 20000
    python mRNA_RBP/plot_pairwise_heatmaps.py --n_pairs 3 --out path/to/out.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS, OUT_BASE, STEM_SIGMA
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_pairwise_dataset, make_ssm_deltas, predict_ssm
import mRNA_RBP.viz as viz

COEF_DIR = os.path.join(_HERE, "outputs", "surrogate_coefs")


def _coef_path(instance: int, mut_rate: int, lib_size: int) -> str:
    return os.path.join(
        COEF_DIR,
        f"coefs_k{instance:02d}_mut{mut_rate:02d}_lib{lib_size}"
        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance",  type=int, default=0)
    parser.add_argument("--mut_rate",  type=int, default=10,
                        help="Mutation rate percent (5, 10, or 25)")
    parser.add_argument("--lib_size",  type=int, default=20000)
    parser.add_argument("--n_pairs",   type=int, default=3,
                        help="Number of stem pairs (rows) to display")
    parser.add_argument("--out", default=os.path.join(
        _HERE, "outputs", "notebook_plots", "pairwise_heatmaps.png"))
    args = parser.parse_args()

    coef_path = _coef_path(args.instance, args.mut_rate, args.lib_size)
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(
            f"Surrogate coef file not found:\n  {coef_path}\n"
            "Run lib_size_spearman.py first to generate surrogate coefficients."
        )

    print(f"Loading coefs: {os.path.basename(coef_path)}")
    coefs  = np.load(coef_path)
    alpha  = coefs["alpha"]   # (L, 4)
    J      = coefs["J"]       # (L, 4, L, 4)
    y_mean = float(coefs["y_mean"]) if "y_mean" in coefs else 0.0
    y_std  = float(coefs["y_std"])  if "y_std"  in coefs else 1.0

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=args.instance,
                               stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()

    x_pw, y_pw_gt, pair_labels, nuc_combos = make_pairwise_dataset(wt_oh, gt)

    # Surrogate: additive term + pairwise stem contributions, un-standardized
    sur_pred = np.einsum("nla,la->n", x_pw.astype(np.float64), alpha.astype(np.float64))
    J_np = J.astype(np.float64)
    for ni in range(len(x_pw)):
        xi = x_pw[ni]
        for (ei, ej) in STEM_PAIRS:
            sur_pred[ni] += float(xi[ei] @ J_np[ei, :, ej, :] @ xi[ej])
    sur_pred = sur_pred * y_std + y_mean

    # SSM baseline
    delta_W  = make_ssm_deltas(wt_oh, gt)
    y_pw_ssm = predict_ssm(x_pw, delta_W)

    # Spearman ρ labels for column headers
    rho_sur = float(spearmanr(y_pw_gt, sur_pred)[0])
    rho_ssm = float(spearmanr(y_pw_gt, y_pw_ssm)[0])
    print(f"Surrogate ρ={rho_sur:.2f}   SSM ρ={rho_ssm:.2f}")

    wt_nuc_ids = np.argmax(wt_oh, axis=1)

    fig = viz.plot_pairwise_eval(
        y_pw_gt,
        sur_pred.astype(np.float32),
        pair_labels,
        nuc_combos,
        STEM_PAIRS,
        y_ssm=y_pw_ssm.astype(np.float32),
        max_pairs=args.n_pairs,
        show_title=False,
        per_col_norm=True,
        wt_nuc_ids=wt_nuc_ids,
        sur_label=f"Surrogate (nl+pair)\nρ={rho_sur:.2f}",
        ssm_label=f"SSM baseline\nρ={rho_ssm:.2f}",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    svg_out = os.path.splitext(args.out)[0] + ".svg"
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out} / {os.path.basename(svg_out)}")


if __name__ == "__main__":
    main()
