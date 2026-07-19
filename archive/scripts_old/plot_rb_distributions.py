"""plot_rb_distributions.py

For the ResidualBind Surrogate GT pipeline, plot four overlaid histograms
per surrogate config showing:
  - y_rand  : oracle GT scores for the random training library (MAVE test split)
  - y_eval  : oracle GT scores for the uniformized eval library
  - yhat_rand: surrogate predictions on the random test split
  - yhat_eval: surrogate predictions on the eval library

Layout: 2 rows × 4 cols
  Row 0: oracle GT distributions (same for all 4 — just shown per col for alignment)
  Row 1: surrogate prediction distributions

Usage:
    python scripts/plot_rb_distributions.py \\
        --seq    AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --experiment RNCMPT00111 \\
        --seed   42 \\
        --out    outputs/postabrcms/vts1high/rb_distributions.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import uniformize_by_histogram
import helper
from residualbind import ResidualBind

NUCS    = ['A', 'C', 'G', 'U']
_NUCS_A = np.array(list("ACGU"))

SURROGATE_CONFIGS = {
    "additive":    {"gpmap": "additive",  "linearity": "linear",    "regression_type": "GE",
                    "noise": "Gaussian",  "noise_order": 0, "reg_strength": 12},
    "pairwise":    {"gpmap": "pairwise",  "linearity": "linear",    "regression_type": "GE",
                    "noise": "Gaussian",  "noise_order": 0, "reg_strength": 0.1},
    "additive_GE": {"gpmap": "additive",  "linearity": "nonlinear", "regression_type": "GE",
                    "noise": "SkewedT",   "noise_order": 2, "reg_strength": 12,  "hidden_nodes": 50},
    "pairwise_GE": {"gpmap": "pairwise",  "linearity": "nonlinear", "regression_type": "GE",
                    "noise": "SkewedT",   "noise_order": 2, "reg_strength": 0.1, "hidden_nodes": 50},
}

CFG_LABELS = {"additive": "Additive", "pairwise": "Pairwise",
              "additive_GE": "Add GE", "pairwise_GE": "Pair GE"}

SURROGATE_GT_RB_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear", "regression_type": "GE",
    "noise": "Gaussian",  "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#2ca02c"
RED    = "#d62728"


def _to_str(X):
    return np.array(["".join(_NUCS_A[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=1_000):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L, wt_idx = wt_onehot.shape[0], np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc    = min(CHUNK, n_seqs - start)
        pos   = np.argpartition(rng.random((nc, L)), exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_idx[p_idx], rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def pad_to_41(X_lib):
    N, L, A = X_lib.shape
    if L == 41:
        return X_lib
    pad = 41 - L
    return np.concatenate([
        np.zeros((N, pad // 2, A),       dtype=np.float32),
        X_lib,
        np.zeros((N, pad - pad // 2, A), dtype=np.float32),
    ], axis=1)


def rb_score_library(X_lib, residbind, batch_size=512):
    X_padded = pad_to_41(X_lib)
    parts = []
    for i in range(0, len(X_padded), batch_size):
        parts.append(np.asarray(residbind.predict(X_padded[i:i + batch_size]),
                                dtype=float).ravel())
    return np.concatenate(parts)


def train_mavenn(X, y, cfg, label="", verbose=0):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp = cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear"
    epochs, patience = (1000, 50) if is_nlp else (500, 25)
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
    print(f"  [{label}] trained  lr={lr:.2e}  bsz={bsz}")
    return model, test_df


def _kde_plot(ax, values, color, label, linestyle="-", bw_adjust=0.6):
    v = values[np.isfinite(values)]
    if v.size < 3:
        return
    kde = gaussian_kde(v, bw_method=bw_adjust)
    xs  = np.linspace(v.min(), v.max(), 400)
    ax.plot(xs, kde(xs), color=color, linestyle=linestyle, linewidth=1.8, label=label)
    ax.fill_between(xs, kde(xs), alpha=0.18, color=color)


def plot_distributions(results, y_rand_all, y_eval, out_path, experiment=""):
    """
    results   : {cfg_name: {"model": ..., "test_df": ..., "yhat_rand": ..., "yhat_eval": ...}}
    y_rand_all: oracle GT scores for the full random training library
    y_eval    : oracle GT scores for the uniformized eval library
    """
    cfg_names = list(SURROGATE_CONFIGS.keys())
    n_cols    = len(cfg_names)

    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7.5))
    fig.suptitle(
        f"{experiment} — RB Surrogate GT score & prediction distributions\n"
        "Row 0: oracle GT labels (histogram)   |   Row 1: surrogate ŷ predictions (KDE)",
        fontsize=12,
    )

    # Shared bin edges for row 0 (oracle GT scores) — same range for both libraries
    all_gt   = np.concatenate([y_rand_all[np.isfinite(y_rand_all)],
                                y_eval[np.isfinite(y_eval)]])
    gt_edges = np.linspace(np.nanpercentile(all_gt, 0.5),
                           np.nanpercentile(all_gt, 99.5), 51)

    for c, cfg_name in enumerate(cfg_names):
        entry = results[cfg_name]

        # ── Row 0: oracle GT label distributions (histogram) ───────────────
        ax0 = axes[0, c]
        ax0.hist(y_rand_all[np.isfinite(y_rand_all)], bins=gt_edges,
                 density=True, alpha=0.5, color=BLUE,   label="random lib y (GT)")
        ax0.hist(y_eval[np.isfinite(y_eval)],         bins=gt_edges,
                 density=True, alpha=0.5, color=ORANGE, label="eval lib y (uniform)")
        ax0.set_title(CFG_LABELS[cfg_name], fontsize=10, fontweight="bold")
        ax0.set_xlabel("Oracle GT score", fontsize=8)
        ax0.set_ylabel("Density", fontsize=8)
        ax0.tick_params(labelsize=7)
        if c == 0:
            ax0.legend(fontsize=7, loc="upper left")

        # ── Row 1: surrogate ŷ prediction distributions (KDE) ──────────────
        ax1 = axes[1, c]
        _kde_plot(ax1, entry["yhat_rand"], BLUE,   "ŷ on random test split", linestyle="-")
        _kde_plot(ax1, entry["yhat_eval"], ORANGE, "ŷ on eval library",      linestyle="-")
        ax1.set_xlabel("Surrogate ŷ", fontsize=8)
        ax1.set_ylabel("Density", fontsize=8)
        ax1.tick_params(labelsize=7)
        if c == 0:
            ax1.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--n_train",    type=int, default=20_000)
    parser.add_argument("--n_pool",     type=int, default=50_000)
    parser.add_argument("--target_n",   type=int, default=5_000)
    parser.add_argument("--out",
        default="outputs/postabrcms/vts1high/rb_distributions.png")
    args = parser.parse_args()

    # ── Load ResidualBind ─────────────────────────────────────────────────
    data_path   = os.path.expanduser("~/residualbind/data/RNAcompete_2013/rnacompete2013.h5")
    save_path   = "/home/nagle/residualbind/weights/log_norm_seq"
    rbp_index   = helper.find_experiment_index(data_path, args.experiment)
    train, _, _ = helper.load_rnacompete_data(
        data_path, ss_type="seq", normalization="log_norm", rbp_index=rbp_index,
    )
    input_shape  = list(train["inputs"].shape)[1:]
    residbind    = ResidualBind(input_shape, 1,
                                os.path.join(save_path, args.experiment + "_weights.hdf5"))
    residbind.load_weights()
    print(f"ResidualBind loaded ({args.experiment})")

    seq         = args.seq.upper().replace("T", "U")
    wt_oh, _    = remove_padding(rna_to_one_hot(seq))
    L           = wt_oh.shape[0]
    exact_mut   = min(4, max(1, L // 2))
    print(f"Sequence: {seq}  L={L}  exact_mut={exact_mut}")

    # ── Stage 1: Train SurrogateGT_RB ─────────────────────────────────────
    print(f"\n{'='*60}\nStage 1: Train SurrogateGT_RB\n{'='*60}")
    rng1  = np.random.default_rng(args.seed)
    X1    = generate_library(wt_oh, args.n_train, exact_mut, rng1)
    y1    = rb_score_library(X1, residbind)
    print(f"RB scores: [{y1.min():.4f}, {y1.max():.4f}]  std={y1.std():.4f}")
    rb_sgt, _ = train_mavenn(X1, y1.reshape(-1, 1), SURROGATE_GT_RB_CFG,
                              label="SurrogateGT_RB", verbose=1)
    print("SurrogateGT_RB trained.")

    # ── Stage 2: Build eval library + train 4 surrogates ──────────────────
    print(f"\n{'='*60}\nStage 2: Eval recovery\n{'='*60}")
    rng_lib  = np.random.default_rng(args.seed + 500)
    X_lib    = generate_library(wt_oh, args.n_train, exact_mut, rng_lib)
    y_lib    = _predict_chunked(rb_sgt, _to_str(X_lib))
    print(f"GT scores (random lib): [{y_lib.min():.4f}, {y_lib.max():.4f}]")

    rng_pool = np.random.default_rng(args.seed + 500 + 99_999)
    X_pool   = generate_library(wt_oh, args.n_pool, exact_mut, rng_pool)
    y_pool   = _predict_chunked(rb_sgt, _to_str(X_pool))
    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98,
        target_n=args.target_n, seed=args.seed,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    del X_pool, y_pool
    print(f"Eval library: {len(y_eval):,} seqs  [{y_eval.min():.4f}, {y_eval.max():.4f}]")

    results = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"\nTraining {cfg_name} …")
        model, test_df = train_mavenn(X_lib, y_lib.reshape(-1, 1), cfg, label=cfg_name)

        cols  = list(test_df.columns)
        x_col = "x" if "x" in cols else "X"
        y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        x_rs  = np.asarray(test_df[x_col])
        y_rs  = np.asarray(test_df[y_col], dtype=float).ravel()

        yhat_rand = _predict_chunked(model, x_rs)
        yhat_eval = _predict_chunked(model, _to_str(X_eval))

        from scipy.stats import spearmanr
        m_r = np.isfinite(y_rs)   & np.isfinite(yhat_rand)
        m_e = np.isfinite(y_eval) & np.isfinite(yhat_eval)
        rho_r = float(spearmanr(yhat_rand[m_r], y_rs[m_r])[0])   if m_r.sum() >= 3 else np.nan
        rho_e = float(spearmanr(yhat_eval[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
        print(f"  ρ_rand={rho_r:.4f}  ρ_eval={rho_e:.4f}")

        results[cfg_name] = {
            "model": model, "test_df": test_df,
            "yhat_rand": yhat_rand, "yhat_eval": yhat_eval,
            "y_rs": y_rs,
        }

    plot_distributions(results, y_lib, y_eval, args.out, experiment=args.experiment)


if __name__ == "__main__":
    main()
