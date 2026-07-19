"""
mRNA_RBP/plots/plot_scatter_by_mutcount.py

3-panel scatter (GT vs surrogate) for mut_rate = 5%, 10%, 25% at lib_size=20000.
  - Blue:    random holdout from training library
  - Colored: activity-balanced library, colored by mutation count (rate_labels)

Retrains the nonlinear additive+pairwise surrogate for each mutation rate.

Usage:
    PYTHONPATH=mRNA_RBP python mRNA_RBP/plots/plot_scatter_by_mutcount.py
    PYTHONPATH=mRNA_RBP python mRNA_RBP/plots/plot_scatter_by_mutcount.py --instance 0 --lib_size 20000
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr, pearsonr
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

import squid.surrogate_zoo
from mRNA_RBP.generate_libraries import (
    OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS, activity_balanced_path,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.lib_size_spearman import (
    nuc_ids_to_str, predict_chunked, _rho, _exclude_eval_seqs, SURROGATE_CONFIGS,
)

NUCS      = ["A", "C", "G", "U"]
MUT_RATES = [5, 10, 25]
GT_KEY    = "nonlin_additive_pairwise"
OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots")

COLOR_RAND  = "#4C9BE8"
SCATTER_S   = 2
ALPHA_RAND  = 0.15
ALPHA_EVAL  = 0.35


def _r2(y, yhat):
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

def _mut_count_colors(unique_counts):
    cmap = cm.get_cmap("plasma_r", len(unique_counts) + 2)
    return {m: cmap(i + 1) for i, m in enumerate(sorted(unique_counts))}


def train_nl_pair(X, y):
    cfg = SURROGATE_CONFIGS["nonlinear additive + pairwise"]
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 10),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    return model, test_df


def run_one(k, pct, n_lib):
    inst_dir   = os.path.join(OUT_BASE, f"instance_{k:02d}")
    rand_path  = os.path.join(inst_dir, f"mut{pct:02d}", f"lib_{n_lib}.npz")
    type2_path = activity_balanced_path(inst_dir)

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k)
    type2_ids = np.load(type2_path)["nuc_ids"]

    d       = np.load(rand_path)
    nuc_ids = d["nuc_ids"][:n_lib]
    nuc_ids, _ = _exclude_eval_seqs(nuc_ids, type2_ids)

    X     = np.eye(4, dtype=np.float32)[nuc_ids]
    y_all = gt.score_all(X)[GT_KEY].reshape(-1, 1)

    print(f"  training mut{pct:02d}% lib{n_lib} instance {k} (N={len(nuc_ids):,})...")
    model, test_df = train_nl_pair(X, y_all)

    # Random holdout predictions
    x_col = "x" if "x" in test_df.columns else "X"
    y_col = next(c for c in test_df.columns if c.startswith("y"))
    y_rand    = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_rand = predict_chunked(model, np.asarray(test_df[x_col]))

    # Activity-balanced predictions — recompute GT scores from scratch
    d2          = np.load(type2_path)
    type2_X     = np.eye(4, dtype=np.float32)[d2["nuc_ids"]]
    y_t2        = gt.score_all(type2_X)[GT_KEY].astype(float)
    rate_labels = d2["rate_labels"].astype(int)
    yhat_t2     = predict_chunked(model, nuc_ids_to_str(d2["nuc_ids"]))

    rho_rand = _rho(y_rand, yhat_rand)

    # GE model can converge with inverted sign — correct if flipped
    if rho_rand < 0:
        print(f"  [sign flip detected, correcting] rho_rand was {rho_rand:.3f}")
        yhat_rand = -yhat_rand
        yhat_t2   = -yhat_t2
        rho_rand  = _rho(y_rand, yhat_rand)

    rho_t2  = _rho(y_t2, yhat_t2)
    r2_rand = _r2(y_rand, yhat_rand)
    r2_t2   = _r2(y_t2, yhat_t2)
    rmse_rand = _rmse(y_rand, yhat_rand)
    rmse_t2   = _rmse(y_t2, yhat_t2)

    tf.keras.backend.clear_session()
    return (y_rand, yhat_rand, y_t2, yhat_t2, rate_labels,
            rho_rand, rho_t2, r2_rand, r2_t2, rmse_rand, rmse_t2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance",  type=int, default=0)
    parser.add_argument("--lib_size",  type=int, default=20000)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    panel_data = []
    for pct in MUT_RATES:
        data = run_one(args.instance, pct, args.lib_size)
        panel_data.append(data)

    # Shared axis limits across all panels
    all_gt = np.concatenate([d[0] for d in panel_data] + [d[2] for d in panel_data])
    all_yh = np.concatenate([d[1] for d in panel_data] + [d[3] for d in panel_data])
    lo = float(np.nanpercentile(all_gt, 0.2))
    hi = float(np.nanpercentile(all_gt, 99.8))
    pad = (hi - lo) * 0.04
    lim = [lo - pad, hi + pad]

    all_counts = sorted(set(np.concatenate([d[4] for d in panel_data])))
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
        ax.set_title(f"{pct}% mut rate  (lib={args.lib_size:,})", fontsize=10)

        # Stats box
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

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "scatter_by_mutcount.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_png}")
    if args.svg:
        out_svg = os.path.join(OUT_DIR, "scatter_by_mutcount.svg")
        fig.savefig(out_svg, bbox_inches="tight")
        print(f"Saved → {out_svg}")
    plt.close(fig)


if __name__ == "__main__":
    main()
