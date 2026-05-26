"""
mRNA_RBP/plot_ssm_scatter.py

Scatter plot of SSM predictions vs GT scores on the Type 2 and Pairwise
eval libraries.  One panel per library; points pooled across all available
instances.  Annotates Spearman ρ and Pearson R².

Outputs:
  outputs/notebook_plots/ssm_scatter.png
  outputs/notebook_plots/ssm_scatter.svg

Usage:
    python mRNA_RBP/plot_ssm_scatter.py
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_ssm_deltas, predict_ssm

INST_DIR = os.path.join(_HERE, "outputs")
OUT_DIR  = os.path.join(_HERE, "outputs", "notebook_plots")
GT_KEY   = "nonlin_additive_pairwise"

COLOR_T2 = "#4C72B0"
COLOR_PW = "#DD8452"
S        = 3
ALPHA    = 0.25


def _stats(y_gt, y_hat):
    mask = np.isfinite(y_gt) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    rho = float(spearmanr(y_gt[mask], y_hat[mask])[0])
    r2  = float(pearsonr(y_gt[mask], y_hat[mask])[0]) ** 2
    return rho, r2


def _identity_lims(x, y, pad_frac=0.04):
    all_v = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    lo = float(np.nanpercentile(all_v, 0.5))
    hi = float(np.nanpercentile(all_v, 99.5))
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_key", default=GT_KEY,
                        choices=["additive", "additive_pairwise",
                                 "nonlin_additive", "nonlin_additive_pairwise"])
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "ssm_scatter.png"))
    args = parser.parse_args()

    instances = sorted(
        int(d.replace("instance_", ""))
        for d in os.listdir(INST_DIR)
        if d.startswith("instance_") and os.path.isdir(os.path.join(INST_DIR, d))
    )
    print(f"Instances: {instances}")

    y_gt_t2, y_ssm_t2 = [], []
    y_gt_pw, y_ssm_pw = [], []

    for k in instances:
        gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k,
                                   stem_sigma=STEM_SIGMA)
        wt_oh = gt.wt_one_hot()
        delta = make_ssm_deltas(wt_oh, gt)

        inst_dir = os.path.join(INST_DIR, f"instance_{k:02d}")

        t2_path = os.path.join(inst_dir, "type2.npz")
        if os.path.isfile(t2_path):
            ev     = np.load(t2_path)
            x_oh   = np.eye(4, dtype=np.float32)[ev["nuc_ids"].astype(int)]
            y_gt_t2.append(ev[f"scores_{args.gt_key}"].astype(float))
            y_ssm_t2.append(predict_ssm(x_oh, delta))

        pw_path = os.path.join(inst_dir, "pairwise_lib.npz")
        if os.path.isfile(pw_path):
            ev     = np.load(pw_path)
            x_oh   = np.eye(4, dtype=np.float32)[ev["nuc_ids"].astype(int)]
            y_gt_pw.append(ev[f"scores_{args.gt_key}"].astype(float))
            y_ssm_pw.append(predict_ssm(x_oh, delta))

    y_gt_t2  = np.concatenate(y_gt_t2)
    y_ssm_t2 = np.concatenate(y_ssm_t2)
    y_gt_pw  = np.concatenate(y_gt_pw)
    y_ssm_pw = np.concatenate(y_ssm_pw)

    rho_t2, r2_t2 = _stats(y_gt_t2, y_ssm_t2)
    rho_pw, r2_pw = _stats(y_gt_pw, y_ssm_pw)
    print(f"Type 2   ρ={rho_t2:.3f}  R²={r2_t2:.3f}  N={len(y_gt_t2):,}")
    print(f"Pairwise ρ={rho_pw:.3f}  R²={r2_pw:.3f}  N={len(y_gt_pw):,}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8),
                             gridspec_kw={"wspace": 0.30})

    panels = [
        (axes[0], y_gt_t2, y_ssm_t2, rho_t2, r2_t2, COLOR_T2, "Type 2 eval library"),
        (axes[1], y_gt_pw, y_ssm_pw, rho_pw, r2_pw, COLOR_PW, "Pairwise eval library"),
    ]

    for ax, y_gt, y_ssm, rho, r2, color, label in panels:
        mask = np.isfinite(y_gt) & np.isfinite(y_ssm)
        ax.scatter(y_gt[mask], y_ssm[mask],
                   s=S, alpha=ALPHA, color=color, rasterized=True)

        gt_finite = y_gt[np.isfinite(y_gt)]
        gt_lo = float(np.nanpercentile(gt_finite, 0.5))
        gt_hi = float(np.nanpercentile(gt_finite, 99.5))
        gt_pad = (gt_hi - gt_lo) * 0.04

        ssm_finite = y_ssm[np.isfinite(y_ssm)]
        ssm_unique = np.unique(ssm_finite)
        constant_ssm = len(ssm_unique) <= 1

        if constant_ssm:
            # SSM is flat (all zeros) — stem positions have no additive weights
            ssm_val = float(ssm_unique[0]) if len(ssm_unique) == 1 else 0.0
            x_range = [gt_lo - gt_pad, gt_hi + gt_pad]
            ax.plot(x_range, [ssm_val, ssm_val], color=color,
                    linewidth=1.5, zorder=5, label="SSM = 0 (constant)")
            ax.plot(x_range, x_range, "k--", linewidth=0.9, zorder=4, alpha=0.4)
            ax.set_xlim(x_range)
            ssm_pad = (gt_hi - gt_lo) * 0.30
            ax.set_ylim(gt_lo - ssm_pad, gt_hi + ssm_pad)
            stat_text = f"ρ = N/A\nR² = N/A\nN = {mask.sum():,}"
            note_text = ("SSM blind to pairwise effects:\n"
                         "stem positions have no\nadditive component (α=0)")
            ax.text(0.05, 0.35, note_text,
                    transform=ax.transAxes,
                    fontsize=8, va="top", ha="left", color="#880000",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF5F5",
                              edgecolor="#FFAAAA", alpha=0.9))
        else:
            lo, hi = _identity_lims(y_gt[mask], y_ssm[mask])
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, zorder=5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            stat_text = f"ρ = {rho:.3f}\nR² = {r2:.3f}\nN = {mask.sum():,}"

        ax.set_xlabel(f"GT score  ({args.gt_key.replace('_', ' ')})", fontsize=10)
        ax.set_ylabel("SSM prediction", fontsize=10)
        ax.set_title(label, fontsize=10, pad=6)

        ax.text(0.05, 0.95,
                stat_text,
                transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85))

    n_inst = len(instances)
    fig.suptitle(
        f"SSM baseline  |  {n_inst} instance{'s' if n_inst != 1 else ''}  "
        f"|  GT: {args.gt_key.replace('_', ' ')}",
        fontsize=11, y=1.01,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    svg_out = os.path.splitext(args.out)[0] + ".svg"
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.out} / {os.path.basename(svg_out)}")


if __name__ == "__main__":
    main()
