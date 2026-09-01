"""
mRNA_RBP/scripts/figures/core/plot_scatter_by_mutcount.py

3-panel scatter (GT vs surrogate) for mut_rate = 5%, 10%, 25% at lib_size=20000.
  - Blue:    random holdout from training library
  - Colored: activity-balanced library, colored by mutation count (rate_labels)

Oracle-agnostic: reads the prediction artifact produced by
generate_scatter_by_mutcount_predictions.py instead of retraining a
surrogate live (plotting should consume frozen artifacts, not train).

Usage:
    python mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py --oracle hur --wt_activity high
    python mRNA_RBP/scripts/figures/core/plot_scatter_by_mutcount.py --oracle hur --wt_activity high
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.src.oracles import MRNA_ORACLE, default_output_base, normalize_oracle_name

MUT_RATES = [5, 10, 25]
OUT_DIR_DEFAULT = os.path.join(_HERE, "outputs", "notebook_plots")

COLOR_RAND  = "#4C9BE8"
SCATTER_S   = 2
ALPHA_RAND  = 0.15
ALPHA_EVAL  = 0.35


def _r2(y, yhat):
    from scipy.stats import pearsonr
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return np.nan
    r = pearsonr(y[mask], yhat[mask])[0]
    return r ** 2


def _rmse(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 1:
        return np.nan
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2)))


def _rho(y, yhat):
    from scipy.stats import spearmanr
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(y[mask], yhat[mask])[0])


def _mut_count_colors(unique_counts):
    cmap = cm.get_cmap("plasma_r", len(unique_counts) + 2)
    return {m: cmap(i + 1) for i, m in enumerate(sorted(unique_counts))}


def main(oracle_name: str = MRNA_ORACLE, wt_activity: str = "high",
        out_base: str = None, predictions_path: str = None,
        out_dir: str = None, save_svg: bool = False):
    oracle_name = normalize_oracle_name(oracle_name)
    out_base = out_base or default_output_base(_HERE, oracle_name, wt_activity)
    out_dir = out_dir or OUT_DIR_DEFAULT
    predictions_path = predictions_path or os.path.join(out_base, "scatter_by_mutcount_predictions.npz")

    d = np.load(predictions_path)

    panel_data = []
    for pct in MUT_RATES:
        y_rand    = d[f"y_rand_{pct}"].astype(float)
        yhat_rand = d[f"yhat_rand_{pct}"].astype(float)
        y_t2      = d[f"y_activity_{pct}"].astype(float)
        yhat_t2   = d[f"yhat_activity_{pct}"].astype(float)
        rate_labels = d[f"rate_labels_{pct}"].astype(int)
        rho_rand = _rho(y_rand, yhat_rand)
        rho_t2   = _rho(y_t2, yhat_t2)
        r2_rand  = _r2(y_rand, yhat_rand)
        r2_t2    = _r2(y_t2, yhat_t2)
        rmse_rand = _rmse(y_rand, yhat_rand)
        rmse_t2   = _rmse(y_t2, yhat_t2)
        panel_data.append((y_rand, yhat_rand, y_t2, yhat_t2, rate_labels,
                           rho_rand, rho_t2, r2_rand, r2_t2, rmse_rand, rmse_t2))

    all_gt = np.concatenate([p[0] for p in panel_data] + [p[2] for p in panel_data])
    lo = float(np.nanpercentile(all_gt, 0.2))
    hi = float(np.nanpercentile(all_gt, 99.8))
    pad = (hi - lo) * 0.04
    lim = [lo - pad, hi + pad]

    all_counts = sorted(set(np.concatenate([p[4] for p in panel_data])))
    colors = _mut_count_colors(all_counts)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    for col, (pct, data) in enumerate(zip(MUT_RATES, panel_data)):
        (y_rand, yhat_rand, y_t2, yhat_t2, rate_labels,
         rho_rand, rho_t2, r2_rand, r2_t2, rmse_rand, rmse_t2) = data
        ax = axes[col]

        ax.scatter(y_rand, yhat_rand, s=SCATTER_S, alpha=ALPHA_RAND,
                   color=COLOR_RAND, rasterized=True, label="random holdout")

        for m in all_counts:
            mask = rate_labels == m
            if not mask.any():
                continue
            ax.scatter(y_t2[mask], yhat_t2[mask], s=SCATTER_S, alpha=ALPHA_EVAL,
                       color=colors[m], rasterized=True,
                       label=f"{m}-mut (n={mask.sum():,})")

        ax.plot(lim, lim, "k--", linewidth=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("y (GT score)", fontsize=10)
        if col == 0:
            ax.set_ylabel("ŷ (surrogate)", fontsize=10)
        ax.set_title(f"{pct}% mut rate", fontsize=10)

        stats_txt = (
            f"random holdout:  ρ={rho_rand:.3f}  R²={r2_rand:.3f}  RMSE={rmse_rand:.3f}\n"
            f"activity-balanced: ρ={rho_t2:.3f}  R²={r2_t2:.3f}  RMSE={rmse_t2:.3f}"
        )
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels, markerscale=5, fontsize=8,
                        framealpha=0.85, loc="lower right",
                        borderpad=0.6, labelspacing=0.4)
        for h in leg.legendHandles:
            h.set_alpha(1.0)

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "scatter_by_mutcount.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_png}")
    if save_svg:
        out_svg = os.path.join(out_dir, "scatter_by_mutcount.svg")
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"Saved → {out_svg}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=MRNA_ORACLE,
                        choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1",
                                 "hur", "residualbind_hur", "qki", "residualbind_qki",
                                 "twister", "twister_ribozyme", "deepsquid_hur", "deepsquid_vts1",
                                 "mrna_negative_control", "negative_control"])
    parser.add_argument("--wt_activity", choices=["high", "low"], default="high")
    parser.add_argument("--out_base", default=None)
    parser.add_argument("--predictions_path", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    main(oracle_name=args.oracle, wt_activity=args.wt_activity, out_base=args.out_base,
         predictions_path=args.predictions_path, out_dir=args.out_dir, save_svg=args.svg)
