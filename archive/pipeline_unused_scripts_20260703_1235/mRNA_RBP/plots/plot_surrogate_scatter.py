"""
mRNA_RBP/plot_surrogate_scatter.py

For each of the 9 (mutation_rate × library_size) combinations:
  - Per instance (k=0..9): train pairwise GE surrogate on the random library,
    predict on the held-out test split (blue) and the activity-balanced library
    (orange), compute Spearman ρ and R².
  - Pool all 10 instances' test-split and eval scatter points.
  - Average ρ and R² across instances for the stats annotation.
  - Plot GT score vs ŷ scatter with marginal histograms of each distribution.

Output: <OUT_BASE>/plots/surrogate_scatter/mut{pct:02d}_lib{N}.png
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES,
)
import squid.surrogate_zoo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GT_KEY = "nonlin_additive_pairwise"
NUCS   = ["A", "C", "G", "U"]
_NUCS_A = np.array(list("ACGU"))

SURROGATE_CFG = {
    "gpmap":           "pairwise",
    "linearity":       "nonlinear",
    "regression_type": "GE",
    "noise":           "Gaussian",
    "noise_order":     2,
    "reg_strength":    0.0,
    "hidden_nodes":    50,
}

COLOR_RANDOM = "#4C9BE8"
COLOR_EVAL   = "#F5A623"
SCATTER_S     = 2
SCATTER_ALPHA = 0.12
MARGINAL_BINS = 60


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nuc_ids_to_str(nuc_ids: np.ndarray) -> np.ndarray:
    """(N, L) uint8 → (N,) object array of ACGU strings."""
    return np.array(["".join(_NUCS_A[row]) for row in nuc_ids], dtype=object)


def predict_chunked(model, x_str, chunk: int = 500) -> np.ndarray:
    """Chunk predictions to avoid GPU OOM on pairwise models."""
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def train_surrogate(X: np.ndarray, y: np.ndarray, label: str = ""):
    """Train pairwise GE surrogate. Returns (mavenn_model, test_df)."""
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG["hidden_nodes"],
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    print(f"    [{label}]  N={N:,}  lr={lr:.2e}  bsz={bsz}")
    return model, test_df


def _stats(y_gt, y_hat):
    """Spearman ρ and R² from paired arrays; nan if insufficient points."""
    mask = np.isfinite(y_gt) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return np.nan, np.nan
    rho = float(spearmanr(y_gt[mask], y_hat[mask])[0])
    r   = float(pearsonr(y_gt[mask],  y_hat[mask])[0])
    return rho, (r ** 2 if np.isfinite(r) else np.nan)


# ---------------------------------------------------------------------------
# Per-instance worker
# ---------------------------------------------------------------------------

def run_instance(k: int, pct: int, n_lib: int) -> dict:
    """Train surrogate for one instance and return scatter data + stats.

    Returns dict:
      y_rand, yhat_rand  — GT and ŷ for held-out test split
      y_eval, yhat_eval  — GT and yhat for activity-balanced library
      rho_rand, r2_rand  — stats on test split
      rho_eval, r2_eval  — stats on eval library
    """
    mut_dir = os.path.join(OUT_BASE, f"instance_{k:02d}", f"mut{pct:02d}")

    # ── Random library
    rand_path = os.path.join(mut_dir, f"lib_{n_lib}.npz")
    if not os.path.isfile(rand_path):
        raise FileNotFoundError(
            f"Missing: {rand_path}\nRun generate_libraries.py first."
        )
    d           = np.load(rand_path)
    nuc_ids     = d["nuc_ids"]                                       # (N, L) uint8
    y_all       = d[f"scores_{GT_KEY}"].astype(np.float32)          # (N,)
    X           = np.eye(4, dtype=np.float32)[nuc_ids]               # (N, L, 4)
    y_in        = y_all.reshape(-1, 1)

    # ── Train surrogate
    model, test_df = train_surrogate(X, y_in, label=f"k={k:02d}")

    # ── Test-split predictions
    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
    x_test_str = np.asarray(test_df[x_col])
    y_test_gt  = np.asarray(test_df[y_col], dtype=float).ravel()
    y_test_hat = predict_chunked(model, x_test_str)
    rho_rand, r2_rand = _stats(y_test_gt, y_test_hat)

    # ── Eval library predictions
    eval_path = os.path.join(mut_dir, f"eval_uniform_{GT_KEY}_lib{n_lib}.npz")
    d_ev       = np.load(eval_path)
    nuc_ids_ev = d_ev["nuc_ids"]
    y_eval_gt  = d_ev[f"scores_{GT_KEY}"].astype(float)
    x_eval_str = nuc_ids_to_str(nuc_ids_ev)
    y_eval_hat = predict_chunked(model, x_eval_str)
    rho_eval, r2_eval = _stats(y_eval_gt, y_eval_hat)

    return {
        "y_rand":    y_test_gt,
        "yhat_rand": y_test_hat,
        "y_eval":    y_eval_gt,
        "yhat_eval": y_eval_hat,
        "rho_rand":  rho_rand,
        "r2_rand":   r2_rand,
        "rho_eval":  rho_eval,
        "r2_eval":   r2_eval,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _shared_bins(arrays, n_bins: int = MARGINAL_BINS) -> np.ndarray:
    """Shared bin edges from concatenated finite values (0.5–99.5 percentile)."""
    all_v = np.concatenate([v[np.isfinite(v)] for v in arrays if len(v) > 0])
    lo, hi = np.percentile(all_v, [0.5, 99.5])
    return np.linspace(lo, hi, n_bins + 1)


def plot_scatter(all_results: list, pct: int, n_lib: int, out_path: str,
                  save_svg: bool = False):
    """Build scatter + marginal plot from pooled instance results."""
    # Pool scatter points
    y_rand_pool    = np.concatenate([r["y_rand"]    for r in all_results])
    yhat_rand_pool = np.concatenate([r["yhat_rand"] for r in all_results])
    y_eval_pool    = np.concatenate([r["y_eval"]    for r in all_results])
    yhat_eval_pool = np.concatenate([r["yhat_eval"] for r in all_results])

    # Average statistics across instances
    rho_rand = float(np.nanmean([r["rho_rand"] for r in all_results]))
    r2_rand  = float(np.nanmean([r["r2_rand"]  for r in all_results]))
    rho_eval = float(np.nanmean([r["rho_eval"] for r in all_results]))
    r2_eval  = float(np.nanmean([r["r2_eval"]  for r in all_results]))

    # ── Figure layout: top (ŷ marginal), main scatter, right (GT marginal)
    fig = plt.figure(figsize=(7.5, 7.5))
    gs  = fig.add_gridspec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.04, wspace=0.04,
    )
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_top     = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_right   = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # ── Finite masks
    m_r = np.isfinite(y_rand_pool)    & np.isfinite(yhat_rand_pool)
    m_e = np.isfinite(y_eval_pool)    & np.isfinite(yhat_eval_pool)

    # ── Main scatter: ŷ on x-axis, GT on y-axis
    ax_scatter.scatter(
        yhat_rand_pool[m_r], y_rand_pool[m_r],
        s=SCATTER_S, alpha=SCATTER_ALPHA, color=COLOR_RANDOM, rasterized=True,
        label=f"Random   ρ={rho_rand:.3f}  R²={r2_rand:.3f}",
    )
    ax_scatter.scatter(
        yhat_eval_pool[m_e], y_eval_pool[m_e],
        s=SCATTER_S, alpha=SCATTER_ALPHA, color=COLOR_EVAL, rasterized=True,
        label=f"Eval       ρ={rho_eval:.3f}  R²={r2_eval:.3f}",
    )
    # Identity line on shared axis range
    all_finite = np.concatenate([
        yhat_rand_pool[m_r], y_rand_pool[m_r],
        yhat_eval_pool[m_e], y_eval_pool[m_e],
    ])
    lo  = float(np.nanpercentile(all_finite, 0.5))
    hi  = float(np.nanpercentile(all_finite, 99.5))
    pad = (hi - lo) * 0.04
    lm  = [lo - pad, hi + pad]
    ax_scatter.plot(lm, lm, "k--", linewidth=0.9, zorder=10)
    ax_scatter.set_xlim(lm)
    ax_scatter.set_ylim(lm)
    ax_scatter.set_xlabel("ŷ  (surrogate prediction)", fontsize=11)
    ax_scatter.set_ylabel(f"GT score  ({GT_KEY.replace('_', ' ')})", fontsize=11)
    h, l = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(h, l, markerscale=5, fontsize=9,
                      loc="upper left", framealpha=0.75)

    # ── Top marginal: distribution of ŷ
    bins_x = _shared_bins([yhat_rand_pool[m_r], yhat_eval_pool[m_e]])
    ax_top.hist(yhat_rand_pool[m_r], bins=bins_x, density=True,
                alpha=0.55, color=COLOR_RANDOM,
                label=f"Random  (N={m_r.sum():,})")
    ax_top.hist(yhat_eval_pool[m_e], bins=bins_x, density=True,
                alpha=0.55, color=COLOR_EVAL,
                label=f"Eval      (N={m_e.sum():,})")
    ax_top.set_ylabel("Density", fontsize=8)
    ax_top.tick_params(axis="x", labelbottom=False, labelsize=7)
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.legend(fontsize=7, framealpha=0.7)
    # ── Right marginal: distribution of GT score (horizontal)
    bins_y = _shared_bins([y_rand_pool[m_r], y_eval_pool[m_e]])
    ax_right.hist(y_rand_pool[m_r], bins=bins_y, density=True,
                  alpha=0.55, color=COLOR_RANDOM, orientation="horizontal")
    ax_right.hist(y_eval_pool[m_e], bins=bins_y, density=True,
                  alpha=0.55, color=COLOR_EVAL,   orientation="horizontal")
    ax_right.set_xlabel("Density", fontsize=8)
    ax_right.tick_params(axis="y", labelleft=False, labelsize=7)
    ax_right.tick_params(axis="x", labelsize=7)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    msg = out_path
    if save_svg:
        svg_path = os.path.splitext(out_path)[0] + ".svg"
        plt.savefig(svg_path, bbox_inches="tight")
        msg = f"{out_path} / {os.path.basename(svg_path)}"
    plt.close(fig)
    print(f"  → {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scatter(save_svg: bool = False):
    plots_dir = os.path.join(OUT_BASE, "plots", "surrogate_scatter")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"GT score:  {GT_KEY}")
    print(f"Surrogate: gpmap={SURROGATE_CFG['gpmap']}  "
          f"linearity={SURROGATE_CFG['linearity']}  "
          f"noise={SURROGATE_CFG['noise']}")
    print(f"Instances: {N_INSTANCES}\n")

    for pct in MUT_RATES_PCT:
        for n_lib in LIB_SIZES:
            print(f"\n{'='*60}")
            print(f"  mut{pct:02d}%   lib_size = {n_lib:,}")
            print(f"{'='*60}")

            all_results = []
            for k in range(N_INSTANCES):
                print(f"\n  instance {k:02d} …")
                try:
                    res = run_instance(k, pct, n_lib)
                    all_results.append(res)
                    print(f"    rho_rand={res['rho_rand']:+.3f}  R²={res['r2_rand']:.3f}  "
                          f"rho_eval={res['rho_eval']:+.3f}  R²={res['r2_eval']:.3f}")
                except Exception as exc:
                    print(f"    FAILED: {exc}")

            if not all_results:
                print("  No successful instances — skipping plot.")
                continue

            out_path = os.path.join(plots_dir, f"mut{pct:02d}_lib{n_lib}.png")
            plot_scatter(all_results, pct, n_lib, out_path, save_svg=save_svg)

            rho_r = float(np.nanmean([r["rho_rand"] for r in all_results]))
            rho_e = float(np.nanmean([r["rho_eval"] for r in all_results]))
            r2_r  = float(np.nanmean([r["r2_rand"]  for r in all_results]))
            r2_e  = float(np.nanmean([r["r2_eval"]  for r in all_results]))
            print(f"\n  [avg] rand ρ={rho_r:+.3f} R²={r2_r:.3f}  "
                  f"eval ρ={rho_e:+.3f} R²={r2_e:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    run_scatter(save_svg=args.svg)
