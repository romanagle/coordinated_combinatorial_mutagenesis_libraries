"""
mRNA_RBP/plots/plot_oracle_scatter_by_mutcount.py

Oracle score vs surrogate prediction scatter for MSI1 and VTS1.

Two output figures (one per RBP), each with 3 panels (5%, 10%, 25% mut rate):
  - Blue:    random holdout from training library (oracle-labeled)
  - Colored: activity-balanced library, colored by mutation count

Surrogate: nonlinear additive + pairwise MAVE-NN (same as scatter_by_mutcount.py)
Oracle labels: ResidualBind ensemble (pre-scored; reads from pre-scored files / cache)

Pre-requisites:
  1. Run plot_oracle_lib_distributions.py (toehold_gpu env) to generate
     mRNA_RBP/outputs/notebook_plots/cache_msi1_lib_scores.npz
  2. Run this script in the squid env (MAVE-NN / TensorFlow)

Usage:
    python mRNA_RBP/plots/plot_oracle_scatter_by_mutcount.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
import tensorflow as tf

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "mRNA_RBP"))

from mRNA_RBP.generate_libraries import activity_balanced_path
sys.path.insert(0, str(_REPO / "squid-nn"))
sys.path.insert(0, str(_REPO / "squid-manuscript" / "squid"))

import squid.surrogate_zoo
from mRNA_RBP.lib_size_spearman import (
    nuc_ids_to_str, predict_chunked, _rho, SURROGATE_CONFIGS,
)
from mRNA_RBP.sequence_configs import VTS1_SEQ

NUCS      = ["A", "C", "G", "U"]
MUT_RATES = [5, 10, 25]

_OUT = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
_OUT.mkdir(parents=True, exist_ok=True)

MSI1_RB_BASE = (_REPO / "mRNA_RBP" / "outputs_vts1_residualbind"
                / "outputs_residualbind_ensemble" / "instance_00")
VTS1_LIB_BASE = _REPO / "mRNA_RBP" / "outputs_vts1_residualbind" / "instance_00"
MSI1_CACHE    = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_msi1_lib_scores.npz"
VTS1_CACHE    = _REPO / "mRNA_RBP" / "outputs" / "notebook_plots" / "cache_vts1_lib_scores.npz"
SYNTH_INST    = _REPO / "mRNA_RBP" / "outputs" / "instance_00"

COLOR_RAND = "#4C9BE8"
SCATTER_S  = 2
ALPHA_RAND = 0.15
ALPHA_EVAL = 0.35


def _r2(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return float("nan")
    return float(pearsonr(y[mask], yhat[mask])[0] ** 2)


def _rmse(y, yhat):
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 1:
        return float("nan")
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2)))


def _exclude_mask(train_nids, eval_nids):
    """Boolean mask: True = keep (sequence not in eval set)."""
    eval_set = {row.tobytes() for row in eval_nids}
    return np.array([r.tobytes() not in eval_set for r in train_nids], dtype=bool)


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


def run_one(nuc_ids, oracle_scores, type2_nids, type2_oracle, label):
    """Train surrogate on oracle-labeled lib; return holdout + type2 panel data."""
    mask       = _exclude_mask(nuc_ids, type2_nids)
    nuc_ids_f  = nuc_ids[mask]
    oracle_f   = oracle_scores[mask]

    X     = np.eye(4, dtype=np.float32)[nuc_ids_f]
    y_all = oracle_f.reshape(-1, 1)

    print(f"  {label}: training on N={len(X):,} …")
    model, test_df = train_nl_pair(X, y_all)

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = next(c for c in test_df.columns if c.startswith("y"))
    y_rand    = np.asarray(test_df[y_col],     dtype=float).ravel()
    yhat_rand = predict_chunked(model, np.asarray(test_df[x_col]))
    yhat_t2   = predict_chunked(model, nuc_ids_to_str(type2_nids))

    rho_rand = _rho(y_rand, yhat_rand)
    if rho_rand < 0:
        print(f"    [sign flip corrected; was {rho_rand:.3f}]")
        yhat_rand = -yhat_rand
        yhat_t2   = -yhat_t2
        rho_rand  = _rho(y_rand, yhat_rand)

    rho_t2    = _rho(type2_oracle, yhat_t2)
    r2_rand   = _r2(y_rand, yhat_rand)
    r2_t2     = _r2(type2_oracle, yhat_t2)
    rmse_rand = _rmse(y_rand, yhat_rand)
    rmse_t2   = _rmse(type2_oracle, yhat_t2)

    tf.keras.backend.clear_session()
    return (y_rand, yhat_rand, type2_oracle, yhat_t2,
            rho_rand, rho_t2, r2_rand, r2_t2, rmse_rand, rmse_t2)


def make_figure(panel_data, type2_labels_list, rbp_label, out_path,
                score_label="Oracle score", title_label="oracle labels"):
    all_gt = np.concatenate([d[0] for d in panel_data] + [d[2] for d in panel_data])
    lo = float(np.nanpercentile(all_gt, 0.2))
    hi = float(np.nanpercentile(all_gt, 99.8))
    pad = (hi - lo) * 0.04
    lim = [lo - pad, hi + pad]

    all_counts = sorted(set(np.concatenate(type2_labels_list)))
    colors = _mut_count_colors(all_counts)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    for col, (pct, data, t2_labels) in enumerate(
            zip(MUT_RATES, panel_data, type2_labels_list)):
        (y_rand, yhat_rand, y_t2, yhat_t2,
         rho_rand, rho_t2, r2_rand, r2_t2, rmse_rand, rmse_t2) = data
        ax = axes[col]

        ax.scatter(y_rand, yhat_rand, s=SCATTER_S, alpha=ALPHA_RAND,
                   color=COLOR_RAND, rasterized=True, label="random holdout")

        for m in all_counts:
            mask = t2_labels == m
            if not mask.any():
                continue
            ax.scatter(y_t2[mask], yhat_t2[mask], s=SCATTER_S, alpha=ALPHA_EVAL,
                       color=colors[m], rasterized=True,
                       label=f"{m}-mut (n={mask.sum():,})")

        ax.plot(lim, lim, "k--", linewidth=0.9)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{score_label} (Δ activity, {rbp_label})", fontsize=10)
        if col == 0:
            ax.set_ylabel("ŷ (surrogate)", fontsize=10)
        ax.set_title(f"{pct}% mut rate  (lib=20,000)", fontsize=10)

        stats_txt = (
            f"random holdout:  ρ={rho_rand:.3f}  R²={r2_rand:.3f}  RMSE={rmse_rand:.3f}\n"
            f"activity-balanced: ρ={rho_t2:.3f}  R²={r2_t2:.3f}  RMSE={rmse_t2:.3f}"
        )
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        handles, labels_leg = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels_leg, markerscale=5, fontsize=8,
                        framealpha=0.85, loc="lower right",
                        borderpad=0.6, labelspacing=0.4)
        for h in leg.legendHandles:
            h.set_alpha(1.0)

    fig.suptitle(f"Surrogate trained on {rbp_label} {title_label}", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def run_msi1():
    print("\n=== MSI1 ===")
    if not MSI1_CACHE.exists():
        raise RuntimeError(
            f"MSI1 cache not found: {MSI1_CACHE}\n"
            "Run plot_oracle_lib_distributions.py (toehold_gpu env) first."
        )

    msi1_cache   = np.load(MSI1_CACHE)
    d_t2         = np.load(activity_balanced_path(str(SYNTH_INST)))
    type2_nids   = d_t2["nuc_ids"].astype(np.int32)
    type2_labels = d_t2["rate_labels"].astype(int)
    type2_oracle = msi1_cache["type2"].astype(np.float64)

    panel_data, labels_list = [], []
    for pct in MUT_RATES:
        d = np.load(MSI1_RB_BASE / f"mut{pct:02d}" / "lib_20000.npz")
        data = run_one(
            d["nuc_ids"].astype(np.int32),
            d["scores_residualbind_ensemble"].astype(np.float64),
            type2_nids, type2_oracle,
            f"MSI1 mut{pct:02d}%",
        )
        panel_data.append(data)
        labels_list.append(type2_labels)

    make_figure(panel_data, labels_list, "MSI1",
                _OUT / "scatter_by_mutcount_msi1_oracle.png")


def run_vts1():
    print("\n=== VTS1 ===")
    vts1_cache   = np.load(VTS1_CACHE)
    if "wt_seq" not in vts1_cache or str(vts1_cache["wt_seq"].item()) != VTS1_SEQ:
        raise RuntimeError(
            f"{VTS1_CACHE} is stale or not VTS1-sequence native. "
            "Run plot_oracle_lib_distributions.py first."
        )
    type2_nids   = vts1_cache["type2_nids"].astype(np.int32)
    type2_labels = vts1_cache["type2_labels"].astype(int)
    type2_oracle = vts1_cache["type2"].astype(np.float64)

    d_r10 = np.load(VTS1_LIB_BASE / "vts1_mut10" / "lib_20000.npz")
    vts1_libs = {
        5:  (vts1_cache["rand05_nids"].astype(np.int32),
             vts1_cache["rand05"].astype(np.float64)),
        10: (d_r10["nuc_ids"].astype(np.int32),
             d_r10["scores_vts1_residualbind"].astype(np.float64)),
        25: (vts1_cache["rand25_nids"].astype(np.int32),
             vts1_cache["rand25"].astype(np.float64)),
    }

    panel_data, labels_list = [], []
    for pct in MUT_RATES:
        nuc_ids, oracle_scores = vts1_libs[pct]
        data = run_one(nuc_ids, oracle_scores, type2_nids, type2_oracle,
                       f"VTS1 mut{pct:02d}%")
        panel_data.append(data)
        labels_list.append(type2_labels)

    make_figure(panel_data, labels_list, "VTS1",
                _OUT / "scatter_by_mutcount_vts1_oracle.png")


def main():
    run_msi1()
    run_vts1()
    print("\nDone.")


if __name__ == "__main__":
    main()
