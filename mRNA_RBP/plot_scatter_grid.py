"""
mRNA_RBP/plot_scatter_grid.py

Generate outputs/notebook_plots/scatter_grid_type2.png:
  3-panel scatter (GT vs surrogate) for mut_rate = 5%, 10%, 25%
  at a fixed lib_size and instance.  Orange = type2 eval library,
  blue = training library subsample.

Usage:
    python mRNA_RBP/plot_scatter_grid.py
    python mRNA_RBP/plot_scatter_grid.py --lib_size 20000 --instance 0
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

from mRNA_RBP.generate_libraries import OUT_BASE

COEF_DIR = os.path.join(_HERE, "outputs", "surrogate_coefs")
OUT_DIR  = os.path.join(_HERE, "outputs", "notebook_plots")
GT_KEY   = "scores_nonlin_additive_pairwise"
MUT_RATES = [5, 10, 25]

COLOR_RAND = "#4C72B0"
COLOR_EVAL = "#DD8452"
SCATTER_S  = 2
ALPHA      = 0.25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--lib_size", type=int, default=20000)
    parser.add_argument("--eval_type", choices=["type2", "type3"], default="type2",
                        help="Eval library to use as the orange scatter set")
    args = parser.parse_args()

    inst_dir = os.path.join(OUT_BASE, f"instance_{args.instance:02d}")

    # First pass: load all data to determine a shared x-axis range
    panel_data = []
    all_x_vals = []
    for pct in MUT_RATES:
        coef_path = os.path.join(
            COEF_DIR,
            f"coefs_k{args.instance:02d}_mut{pct:02d}_lib{args.lib_size}"
            "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
        )
        if not os.path.isfile(coef_path):
            panel_data.append(None)
            continue

        coefs = np.load(coef_path)

        if "yhat_lib" not in coefs or "y_lib" not in coefs or \
           "yhat_t2"  not in coefs or "y_t2"  not in coefs:
            print(f"  [skip] {coef_path} missing yhat arrays — retrain with train_surrogate.py")
            panel_data.append(None)
            continue

        y_rand    = coefs["y_lib"].astype(float)
        yhat_rand = coefs["yhat_lib"].astype(float)
        y_eval    = coefs["y_t2"].astype(float)
        yhat_eval = coefs["yhat_t2"].astype(float)

        panel_data.append((y_rand, yhat_rand, y_eval, yhat_eval))
        all_x_vals.append(y_rand)
        all_x_vals.append(y_eval)

    # Compute shared x and y limits across all panels
    all_x = np.concatenate(all_x_vals) if all_x_vals else np.array([0.0, 1.0])
    x_lo = float(np.nanpercentile(all_x, 0.5))
    x_hi = float(np.nanpercentile(all_x, 99.5))
    x_pad = (x_hi - x_lo) * 0.04
    x_lim = [x_lo - x_pad, x_hi + x_pad]

    all_y_vals = [yhat for d in panel_data if d is not None for yhat in (d[1], d[2])]
    all_y = np.concatenate(all_y_vals) if all_y_vals else np.array([0.0, 1.0])
    y_lo = float(np.nanpercentile(all_y, 0.5))
    y_hi = float(np.nanpercentile(all_y, 99.5))
    y_pad = (y_hi - y_lo) * 0.04
    y_lim = [y_lo - y_pad, y_hi + y_pad]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    for col, (pct, data) in enumerate(zip(MUT_RATES, panel_data)):
        if data is None:
            axes[col].set_visible(False)
            continue

        y_rand, yhat_rand, y_eval, yhat_eval = data

        rho_r, _ = spearmanr(y_rand, yhat_rand)
        r2_r     = pearsonr(y_rand,  yhat_rand)[0] ** 2
        rho_e, _ = spearmanr(y_eval, yhat_eval)
        r2_e     = pearsonr(y_eval,  yhat_eval)[0] ** 2

        ax = axes[col]
        ax.scatter(y_rand, yhat_rand, s=SCATTER_S, alpha=ALPHA, color=COLOR_RAND,
                   label=f"random  ρ={rho_r:.3f}  R²={r2_r:.3f}", rasterized=True)
        eval_label = {"type2": "uniform eval", "type3": "type3"}.get(args.eval_type, args.eval_type)
        ax.scatter(y_eval, yhat_eval, s=SCATTER_S, alpha=ALPHA, color=COLOR_EVAL,
                   label=f"{eval_label}  ρ={rho_e:.3f}  R²={r2_e:.3f}", rasterized=True)

        ax.plot(x_lim, x_lim, "k--", linewidth=0.8)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("y  (GT score)", fontsize=9)
        if col == 0:
            ax.set_ylabel("ŷ  (surrogate)", fontsize=9)
        ax.legend(markerscale=8, fontsize=11, framealpha=0.85,
                  loc="upper left", borderpad=0.8, labelspacing=0.6,
                  handletextpad=0.5)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"scatter_grid_{args.eval_type}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"scatter_grid_{args.eval_type}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
