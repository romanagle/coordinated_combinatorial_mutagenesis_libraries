"""matched_pair_scatter.py

Produce 5 scatter plots — one per matched (surrogate complexity, GT complexity) pair:

  1. Additive surrogate      × Additive GT
  2. Pairwise surrogate      × Additive Pairwise GT
  3. Additive GE surrogate   × Nonlinear Additive GT
  4. Pairwise GE surrogate   × Nonlinear Additive Pairwise GT
  5. Pairwise GE surrogate   × Surrogate GT

Each figure has a central scatter (ŷ vs y) plus marginal histograms on both axes.

Usage:
    python scripts/matched_pair_scatter.py \\
        --out_dir outputs/surrogate_gt_pipeline \\
        --seed 42
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT, init_pairwise_potts_optionA, init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts, additive_affinity_noWT,
    uniformize_by_histogram,
)

# ---------------------------------------------------------------------------
# Config — must match surrogate_gt_full_pipeline.py exactly
# ---------------------------------------------------------------------------

NUCS   = ['A', 'C', 'G', 'U']
_NUCS  = np.array(list("ACGU"))

SURROGATE_CONFIGS = {
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 12,
    },
    "pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "additive_GE": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 12, "hidden_nodes": 50,
    },
    "pairwise_GE": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.1, "hidden_nodes": 50,
    },
}

SURROGATE_GT_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}

# (surrogate_cfg_name, gt_key, plot title)
MATCHED_PAIRS = [
    ("additive",    "additive",                  "Additive Surrogate\nAdditive GT"),
    ("pairwise",    "additive_pairwise",          "Pairwise Surrogate\nAdditive Pairwise GT"),
    ("additive_GE", "nonlin_additive",            "Additive GE Surrogate\nNonlinear Additive GT"),
    ("pairwise_GE", "nonlin_additive_pairwise",   "Pairwise GE Surrogate\nNonlinear Additive Pairwise GT"),
    ("pairwise_GE", "surrogate_gt",               "Pairwise GE Surrogate\nSurrogate GT"),
]

# ---------------------------------------------------------------------------
# Utilities (copied from surrogate_gt_full_pipeline.py for self-containment)
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=1_000):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc      = min(CHUNK, n_seqs - start)
        noise   = rng.random((nc, L))
        pos     = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X       = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx   = np.repeat(np.arange(nc), exact_mut_count)
        p_idx   = pos.ravel()
        wt_at   = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def train_mavenn(X, y, cfg, label="", verbose=0):
    N       = X.shape[0]
    bsz     = max(32, min(N // 150, 2048))
    lr      = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp  = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs  = 1000 if is_nlp else 500
    patience = 50  if is_nlp else 25
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y, learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=verbose,
    )
    print(f"    [{label}]  lr={lr:.2e}  bsz={bsz}  ep={epochs}", flush=True)
    return model, test_df


# ---------------------------------------------------------------------------
# GT initialisation
# ---------------------------------------------------------------------------

def init_gt(wt_onehot, exact_mut_count, seed):
    """Returns (W_mut, mut_map, b0, edges, J, nonlin_kw)."""
    rng = np.random.default_rng(seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )
    rng_ref = np.random.default_rng(seed + 1)
    X_ref   = generate_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add   = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nk      = init_sigmoid_nonlin(s_add)
    del X_ref, s_add
    return W_mut, mut_map, b0, edges, J, nk


def score_analytic(X, W_mut, mut_map, b0, edges, J, nk):
    return compute_gt_scores_for_library_potts(
        X, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nk,
        edges=edges, J=J,
    )


def train_surrogate_gt(wt_onehot, exact_mut_count, seed, W_mut, mut_map, b0, edges, J, nk):
    """Train the SurrogateGT on nonlin_additive_pairwise scores, same as pipeline stage 1."""
    rng_lib = np.random.default_rng(seed + 2)
    X_lib   = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)
    gt_sc   = score_analytic(X_lib, W_mut, mut_map, b0, edges, J, nk)
    y       = gt_sc["nonlin_additive_pairwise"].astype(float).reshape(-1, 1)
    model, _ = train_mavenn(X_lib, y, SURROGATE_GT_CFG, label="SurrogateGT", verbose=1)
    return model


# ---------------------------------------------------------------------------
# Plotting — scatter with marginals
# ---------------------------------------------------------------------------

BINS = 60


def _marginal_scatter(ax, ax_top, ax_right, yhat, y, color_rand, color_eval,
                      yhat_rand, y_rand, yhat_eval, y_eval, title):
    """Draw scatter on ax with marginal histograms on ax_top (ŷ) and ax_right (y)."""

    # Shared axis limits
    all_vals = np.concatenate([yhat_rand, y_rand, yhat_eval, y_eval])
    all_vals = all_vals[np.isfinite(all_vals)]
    lo, hi   = np.percentile(all_vals, [0.5, 99.5])
    pad      = (hi - lo) * 0.05
    lim      = [lo - pad, hi + pad]

    # Scatter — x=y (GT score), y=ŷ (surrogate prediction)
    ax.scatter(y_rand,  yhat_rand, s=1, alpha=0.07, color=color_rand,
               rasterized=True, label="random test split")
    ax.scatter(y_eval,  yhat_eval, s=1, alpha=0.07, color=color_eval,
               rasterized=True, label="eval library")
    ax.plot(lim, lim, "k--", linewidth=0.8, zorder=10)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("y  (GT score)", fontsize=9)
    ax.set_ylabel("ŷ  (surrogate prediction)", fontsize=9)
    ax.tick_params(labelsize=7)

    # Stats annotations
    def _stats(yh, yt):
        m = np.isfinite(yh) & np.isfinite(yt)
        if m.sum() < 3:
            return np.nan, np.nan, np.nan
        rho = float(spearmanr(yh[m], yt[m])[0])
        r   = float(pearsonr( yh[m], yt[m])[0])
        return rho, r, r**2

    rho_r, r_r, r2_r = _stats(yhat_rand, y_rand)
    rho_e, r_e, r2_e = _stats(yhat_eval, y_eval)
    ax.text(0.03, 0.97,
            f"random   ρ={rho_r:.3f}  r²={r2_r:.3f}\n"
            f"eval      ρ={rho_e:.3f}  r²={r2_e:.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    # Title above top marginal so it doesn't overlap the distributions
    ax_top.set_title(title, fontsize=10, pad=6)

    bins = np.linspace(lim[0], lim[1], BINS + 1)

    # Top marginal — y (GT score) distribution
    ax_top.hist(y_rand,  bins=bins, density=True, color=color_rand, alpha=0.55)
    ax_top.hist(y_eval,  bins=bins, density=True, color=color_eval, alpha=0.55)
    ax_top.set_xlim(lim)
    ax_top.set_ylabel("density", fontsize=7)
    ax_top.tick_params(labelsize=6, labelbottom=False)
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # Right marginal — ŷ (surrogate prediction) distribution
    ax_right.hist(yhat_rand, bins=bins, density=True, color=color_rand, alpha=0.55,
                  orientation="horizontal")
    ax_right.hist(yhat_eval, bins=bins, density=True, color=color_eval, alpha=0.55,
                  orientation="horizontal")
    ax_right.set_ylim(lim)
    ax_right.set_xlabel("density", fontsize=7)
    ax_right.tick_params(labelsize=6, labelleft=False)
    ax_right.grid(True, linestyle="--", alpha=0.3)


def plot_matched_pair(cfg_name, gt_key, title, model, test_df,
                      y_eval, yhat_eval, out_path):
    """One figure: scatter + marginals."""
    COLOR_RAND = "#4C72B0"
    COLOR_EVAL = "#DD8452"

    cols  = list(test_df.columns)
    x_col = "x" if "x" in cols else "X"
    y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
    X_test = np.asarray(test_df[x_col])
    y_rand = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_rand = _predict_chunked(model, X_test)

    fig = plt.figure(figsize=(6.5, 6.5))
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 1]).set_visible(False)   # empty corner

    _marginal_scatter(
        ax_main, ax_top, ax_right,
        None, None,
        COLOR_RAND, COLOR_EVAL,
        yhat_rand, y_rand,
        yhat_eval, y_eval,
        title,
    )

    # Legend on main axes
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_RAND,
                   markersize=6, label="random MAVE test split"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_EVAL,
                   markersize=6, label="uniform eval library"),
    ]
    ax_main.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.7)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",     default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--lib_size", type=int, default=20_000)
    parser.add_argument("--n_pool",   type=int, default=50_000)
    parser.add_argument("--target_n", type=int, default=5_000)
    parser.add_argument("--out_dir",  default="outputs/surrogate_gt_pipeline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq       = args.seq.upper().replace("T", "U")
    oh_seq    = rna_to_one_hot(seq)
    wt_oh, _  = remove_padding(oh_seq)
    L         = wt_oh.shape[0]
    exact_mut = min(4, max(1, L // 2))
    print(f"[init] L={L}  exact_mut={exact_mut}  seed={args.seed}", flush=True)

    # --- GT parameters (same seed as full pipeline) ---
    print("\n[1/3] Initialising GT parameters…", flush=True)
    W_mut, mut_map, b0, edges, J, nk = init_gt(wt_oh, exact_mut, args.seed)

    # --- Train SurrogateGT (same as pipeline stage 1) ---
    print("\n[2/3] Training SurrogateGT…", flush=True)
    sgt_model = train_surrogate_gt(wt_oh, exact_mut, args.seed,
                                   W_mut, mut_map, b0, edges, J, nk)

    # --- Generate one shared library and score everything ---
    print("\n[3/3] Generating shared library and scoring…", flush=True)
    rng_lib = np.random.default_rng(args.seed + 200)
    X_lib   = generate_library(wt_oh, args.lib_size, exact_mut, rng_lib)
    print(f"  library: N={len(X_lib):,}", flush=True)

    analytic = score_analytic(X_lib, W_mut, mut_map, b0, edges, J, nk)
    sgt_str  = _to_str(X_lib)
    analytic["surrogate_gt"] = _predict_chunked(sgt_model, sgt_str)

    # Build eval libraries — one per GT key via uniformisation
    print("  building eval libraries…", flush=True)
    rng_pool = np.random.default_rng(args.seed + 99_999)
    X_pool   = generate_library(wt_oh, args.n_pool, exact_mut, rng_pool)

    analytic_pool = score_analytic(X_pool, W_mut, mut_map, b0, edges, J, nk)
    analytic_pool["surrogate_gt"] = _predict_chunked(sgt_model, _to_str(X_pool))

    eval_data = {}   # gt_key → (X_eval, y_eval)
    for gt_key in analytic_pool:
        y_pool_k = analytic_pool[gt_key].astype(float)
        y_uni, keep = uniformize_by_histogram(
            y_pool_k, X=None, n_bins=200, clip_hi=98,
            target_n=args.target_n, seed=args.seed,
        )
        eval_data[gt_key] = (X_pool[keep], y_uni.astype(float))
        print(f"    {gt_key}: eval N={len(y_uni):,}  "
              f"range=[{y_uni.min():.4f}, {y_uni.max():.4f}]", flush=True)

    # --- Train and plot each matched pair ---
    print("\n[4/4] Training matched surrogates and plotting…", flush=True)
    for cfg_name, gt_key, title in MATCHED_PAIRS:
        print(f"\n  pair: {cfg_name} × {gt_key}", flush=True)
        y_train = analytic[gt_key].astype(float).reshape(-1, 1)
        model, test_df = train_mavenn(
            X_lib, y_train, SURROGATE_CONFIGS[cfg_name],
            label=f"{cfg_name}×{gt_key}", verbose=1,
        )

        X_eval, y_eval_raw = eval_data[gt_key]
        yhat_eval = _predict_chunked(model, _to_str(X_eval))

        out_name = f"matched_{cfg_name}__{gt_key.replace('/', '_')}.png"
        plot_matched_pair(
            cfg_name, gt_key, title,
            model, test_df,
            y_eval_raw, yhat_eval,
            os.path.join(args.out_dir, out_name),
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
