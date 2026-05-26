"""
mRNA_RBP/plot_pairwise_heatmaps_r2.py

Copy of plot_pairwise_heatmaps.py with Pearson R² shown in column headers
instead of Spearman ρ.

Outputs:
  outputs/notebook_plots/pairwise_heatmaps_r2.png
  outputs/notebook_plots/pairwise_heatmaps_r2.svg

Usage:
    python mRNA_RBP/plot_pairwise_heatmaps_r2.py
    python mRNA_RBP/plot_pairwise_heatmaps_r2.py --n_pairs 3 --out path/to/out.png
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
from mRNA_RBP.evaluate import make_pairwise_dataset, make_ssm_deltas, predict_ssm
import mRNA_RBP.viz as viz

COEF_DIR = os.path.join(_HERE, "outputs", "surrogate_coefs")


def _r2(y_true, y_pred):
    return float(np.corrcoef(np.asarray(y_true), np.asarray(y_pred))[0, 1] ** 2)


def _coef_path(instance, mut_rate, lib_size):
    return os.path.join(
        COEF_DIR,
        f"coefs_k{instance:02d}_mut{mut_rate:02d}_lib{lib_size}"
        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance",  type=int, default=0)
    parser.add_argument("--mut_rate",  type=int, default=10)
    parser.add_argument("--lib_size",  type=int, default=20000)
    parser.add_argument("--n_pairs",   type=int, default=3)
    parser.add_argument("--out", default=os.path.join(
        _HERE, "outputs", "notebook_plots", "pairwise_heatmaps_r2.png"))
    args = parser.parse_args()

    coef_path = _coef_path(args.instance, args.mut_rate, args.lib_size)
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(f"Coef file not found: {coef_path}")

    print(f"Loading coefs: {os.path.basename(coef_path)}")
    coefs  = np.load(coef_path)
    alpha  = coefs["alpha"]
    J      = coefs["J"]
    y_mean = float(coefs["y_mean"]) if "y_mean" in coefs else 0.0
    y_std  = float(coefs["y_std"])  if "y_std"  in coefs else 1.0

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=args.instance,
                               stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()

    x_pw, y_pw_gt, pair_labels, nuc_combos = make_pairwise_dataset(wt_oh, gt)

    # Surrogate prediction
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

    r2_sur = _r2(y_pw_gt, sur_pred)
    r2_ssm = _r2(y_pw_gt, y_pw_ssm)
    print(f"Surrogate R²={r2_sur:.3f}   SSM R²={r2_ssm:.3f}")

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
        sur_label=f"Surrogate (nl+pair)\nR²={r2_sur:.3f}",
        ssm_label=f"SSM baseline\nR²={r2_ssm:.3f}",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    svg_out = os.path.splitext(args.out)[0] + ".svg"
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out} / {os.path.basename(svg_out)}")


if __name__ == "__main__":
    main()
