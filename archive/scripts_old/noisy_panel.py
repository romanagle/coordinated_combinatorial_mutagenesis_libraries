"""noisy_panel.py

Regenerates the full 5×4 y-vs-ŷ panel with Gaussian noise added to all 4
synthetic GT scores before surrogate training.  Also produces overlaid
random-vs-eval distribution plots for each GT.

Rows  (GT oracle):  additive | additive_pairwise | nonlin_additive |
                    nonlin_additive_pairwise | RB Surrogate GT
Cols  (surrogate):  additive | pairwise | additive_GE | pairwise_GE

The 5th row uses ResidualBind as the biological oracle: a 20 k library is
scored by ResidualBind, a nonlinear pairwise GE surrogate (SurrogateGT_RB)
is trained on those scores, then a fresh 20 k library is scored by
SurrogateGT_RB and used to train/evaluate the four surrogate configs.

Usage:
    python scripts/noisy_panel.py \\
        --noise_sigma 0.05 \\
        --experiment  RNCMPT00111 \\
        --out_dir     outputs/surrogate_gt_pipeline/noisy_panel_v2
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    pairwise_potts_energy, uniformize_by_histogram,
)
import helper
from residualbind import ResidualBind

# ---------------------------------------------------------------------------
# Config
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

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
    "surrogate_gt",
]

GT_LABELS = {
    "additive":                  "Additive GT",
    "additive_pairwise":         "Add+Pair GT",
    "nonlin_additive":           "Nonlin Add GT",
    "nonlin_additive_pairwise":  "Nonlin Add+Pair GT",
    "surrogate_gt":              "RB Surrogate GT",
}

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Additive GE",
    "pairwise_GE": "Pairwise GE",
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=1_000):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s+chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def generate_random_pool(L, n_seqs, rng):
    """Fully random one-hot sequences (uniform over all 4 nucleotides)."""
    nuc_ids = rng.integers(0, 4, size=(n_seqs, L), dtype=np.uint8)
    return np.eye(4, dtype=np.float32)[nuc_ids]


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L, wt_idx = wt_onehot.shape[0], np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc       = min(CHUNK, n_seqs - start)
        pos      = np.argpartition(rng.random((nc, L)), exact_mut_count, axis=1)[:, :exact_mut_count]
        X        = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_idx[p_idx], rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def train_mavenn(X, y, cfg, label="", verbose=0):
    N        = X.shape[0]
    bsz      = max(32, min(N // 150, 2048))
    lr       = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp   = cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear"
    epochs   = 1000 if is_nlp else 500
    patience = 50   if is_nlp else 25
    wrapper  = squid.surrogate_zoo.SurrogateMAVENN(
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
    print(f"      [{label}]  lr={lr:.2e}  bsz={bsz}")
    return model, test_df


# ---------------------------------------------------------------------------
# GT init (shared across all synthetic GT rows)
# ---------------------------------------------------------------------------

def init_gt_params(wt_onehot, exact_mut_count, seed):
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
    # Store reference std for the combined additive+pairwise energy.
    # Potts couplings make std(s_addpair) >> std(s_add); using s_add's std to
    # normalise s_addpair pushes z-scores off the sigmoid's active range.
    s_pair  = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
    nk["_norm_std_addpair"] = float(np.std(s_add + s_pair)) + 1e-8
    del X_ref, s_add, s_pair
    print(f"  sigmoid ref_std={nk['_norm_std']:.4f}  "
          f"ref_std_addpair={nk['_norm_std_addpair']:.4f}")
    return {"W_mut": W_mut, "mut_map": mut_map, "b0": b0,
            "edges": edges, "J": J, "nk": nk}


def pad_to_41(X_lib):
    """Zero-pad library sequences to ResidualBind's required 41 nt."""
    N, L, A = X_lib.shape
    if L == 41:
        return X_lib
    pad = 41 - L
    beg = np.zeros((N, pad // 2,       A), dtype=np.float32)
    end = np.zeros((N, pad - pad // 2, A), dtype=np.float32)
    return np.concatenate([beg, X_lib, end], axis=1)


def rb_score_library(X_lib, residbind, batch_size=512):
    """Score a library with ResidualBind (auto-pads to 41 nt)."""
    X_padded = pad_to_41(X_lib)
    parts = []
    for i in range(0, len(X_padded), batch_size):
        pred = residbind.predict(X_padded[i:i + batch_size])
        parts.append(np.asarray(pred, dtype=float).ravel())
    return np.concatenate(parts)


def train_surrogate_gt(wt_onehot, exact_mut_count, seed, residbind, verbose=1):
    """Train SurrogateGT_RB: 20 k library → ResidualBind → pairwise nonlinear GE."""
    rng   = np.random.default_rng(seed + 2)
    X_lib = generate_library(wt_onehot, 20_000, exact_mut_count, rng)
    y_lib = rb_score_library(X_lib, residbind)
    print(f"  SurrogateGT_RB training library: "
          f"y=[{y_lib.min():.4f}, {y_lib.max():.4f}]  std={y_lib.std():.4f}")
    model, _ = train_mavenn(X_lib, y_lib.reshape(-1, 1),
                            SURROGATE_GT_CFG, label="SurrogateGT_RB", verbose=verbose)
    return model


# ---------------------------------------------------------------------------
# Per-GT: score library + build eval + train surrogates
# ---------------------------------------------------------------------------

def process_synthetic_gt(gt_key, gt_params, wt_onehot, exact_mut_count,
                          seed, noise_sigma):
    """Score 20k library for one synthetic GT row; build eval; train 4 surrogates."""
    print(f"\n  -- {GT_LABELS[gt_key]} --")

    # Random training library
    rng_lib = np.random.default_rng(seed + hash(gt_key) % 100_000)
    X_lib   = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)
    gt_sc   = compute_gt_scores_for_library_potts(
        X_lib,
        W_mut=gt_params["W_mut"], mut_map=gt_params["mut_map"], b0=gt_params["b0"],
        nonlin_name="sigmoid", nonlin_kwargs=gt_params["nk"],
        edges=gt_params["edges"], J=gt_params["J"],
        noise_sigma=noise_sigma, noise_seed=seed + hash(gt_key) % 10_000,
    )
    y = gt_sc[gt_key].astype(float).reshape(-1, 1)

    # Eval library (larger pool at same mut_count keeps scores in training regime)
    rng_pool = np.random.default_rng(seed + hash(gt_key) % 100_000 + 99_999)
    X_pool   = generate_library(wt_onehot, 200_000, exact_mut_count, rng_pool)
    gt_pool  = compute_gt_scores_for_library_potts(
        X_pool,
        W_mut=gt_params["W_mut"], mut_map=gt_params["mut_map"], b0=gt_params["b0"],
        nonlin_name="sigmoid", nonlin_kwargs=gt_params["nk"],
        edges=gt_params["edges"], J=gt_params["J"],
        noise_sigma=noise_sigma, noise_seed=seed + hash(gt_key) % 10_000 + 1,
    )
    y_pool = gt_pool[gt_key].astype(float)
    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98,
        target_n=20_000, seed=seed + hash(gt_key) % 10_000 + 2,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    del X_pool, gt_pool

    print(f"    y_rand=[{y.min():.3f}, {y.max():.3f}]  "
          f"y_eval=[{y_eval.min():.3f}, {y_eval.max():.3f}]  "
          f"n_eval={len(y_eval):,}")

    # Train all 4 surrogates
    row = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"    training {cfg_name} …", end=" ", flush=True)
        model, test_df = train_mavenn(X_lib, y, cfg, label=cfg_name)
        row[cfg_name] = {"model": model, "test_df": test_df,
                         "X_eval": X_eval, "y_eval": y_eval,
                         "y_rand_all": y.ravel()}
    return row


def process_surrogate_gt(sgt_model, wt_onehot, exact_mut_count, seed):
    """Score a fresh 20 k library with SurrogateGT_RB; build eval; train 4 surrogates."""
    print(f"\n  -- RB Surrogate GT --")

    rng_lib  = np.random.default_rng(seed + 200)
    X_lib    = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)
    y_lib    = _predict_chunked(sgt_model, _to_str(X_lib)).reshape(-1, 1)

    rng_pool = np.random.default_rng(seed + 200 + 99_999)
    X_pool   = generate_library(wt_onehot, 200_000, exact_mut_count, rng_pool)
    y_pool   = _predict_chunked(sgt_model, _to_str(X_pool))
    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=200, clip_hi=98, target_n=20_000, seed=seed + 200,
    )
    X_eval = X_pool[keep_idx]
    y_eval = y_uni.astype(float)
    del X_pool, y_pool
    print(f"    y_rand=[{y_lib.min():.3f}, {y_lib.max():.3f}]  "
          f"y_eval=[{y_eval.min():.3f}, {y_eval.max():.3f}]  n_eval={len(y_eval):,}")

    row = {}
    for cfg_name, cfg in SURROGATE_CONFIGS.items():
        print(f"    training {cfg_name} …", end=" ", flush=True)
        model, test_df = train_mavenn(X_lib, y_lib, cfg, label=cfg_name)
        row[cfg_name] = {"model": model, "test_df": test_df,
                         "X_eval": X_eval, "y_eval": y_eval,
                         "y_rand_all": y_lib.ravel()}
    return row


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(grid_results, out_path, noise_sigma):
    """5-row × 4-col y-vs-ŷ grid."""
    n_rows = len(GT_KEYS)
    n_cols = len(SURROGATE_CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))

    for r, gt_key in enumerate(GT_KEYS):
        row_lo, row_hi = np.inf, -np.inf
        panel_data = []

        for c, cfg_name in enumerate(SURROGATE_CONFIGS.keys()):
            entry    = grid_results[gt_key][cfg_name]
            model    = entry["model"]
            df       = entry["test_df"]
            X_eval   = entry["X_eval"]
            y_eval   = entry["y_eval"].ravel()

            cols  = list(df.columns)
            x_col = "x" if "x" in cols else "X"
            y_col = "y" if "y" in cols else next(c2 for c2 in cols if c2.startswith("y"))

            x_rs   = np.asarray(df[x_col])
            y_rs   = np.asarray(df[y_col], dtype=float).ravel()
            yh_rs  = _predict_chunked(model, x_rs)
            yh_ev  = _predict_chunked(model, _to_str(X_eval))

            m_r = np.isfinite(y_rs)   & np.isfinite(yh_rs)
            m_e = np.isfinite(y_eval) & np.isfinite(yh_ev)
            rho_r = float(spearmanr(yh_rs[m_r], y_rs[m_r])[0])   if m_r.sum() >= 3 else np.nan
            rho_e = float(spearmanr(yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan
            r_r   = float(pearsonr( yh_rs[m_r], y_rs[m_r])[0])   if m_r.sum() >= 3 else np.nan
            r_e   = float(pearsonr( yh_ev[m_e], y_eval[m_e])[0]) if m_e.sum() >= 3 else np.nan

            all_v = np.concatenate([yh_rs[m_r], yh_ev[m_e], y_rs[m_r], y_eval[m_e]])
            if all_v.size:
                row_lo = min(row_lo, float(all_v.min()))
                row_hi = max(row_hi, float(all_v.max()))

            panel_data.append((c, cfg_name, yh_rs, y_rs, yh_ev, y_eval,
                                rho_r, rho_e, r_r, r_e))

        for (c, cfg_name, yh_rs, y_rs, yh_ev, y_eval,
             rho_r, rho_e, r_r, r_e) in panel_data:
            ax  = axes[r, c]
            r2r = r_r**2 if np.isfinite(r_r) else np.nan
            r2e = r_e**2 if np.isfinite(r_e) else np.nan
            ax.scatter(yh_rs, y_rs,   s=1, alpha=0.08, color="C0", rasterized=True,
                       label=f"random  ρ={rho_r:.2f}  R²={r2r:.2f}")
            ax.scatter(yh_ev, y_eval, s=1, alpha=0.08, color="C1", rasterized=True,
                       label=f"eval     ρ={rho_e:.2f}  R²={r2e:.2f}")
            ax.set_xlabel("ŷ", fontsize=8)
            ax.set_ylabel("y", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_title(f"{CFG_LABELS[cfg_name]} / {GT_LABELS[gt_key]}", fontsize=7.5)
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, l, markerscale=6, fontsize=5.5,
                      loc="upper left", framealpha=0.6)
            if np.isfinite(row_lo) and np.isfinite(row_hi) and row_lo < row_hi:
                pad = (row_hi - row_lo) * 0.05
                lm  = [row_lo - pad, row_hi + pad]
                ax.set_xlim(lm); ax.set_ylim(lm)
                ax.plot(lm, lm, "k--", linewidth=0.8, zorder=10)

    fig.suptitle(
        f"y vs ŷ — all surrogate × GT combinations  (noise σ={noise_sigma})\n"
        "blue = random MAVE test split    orange = uniform eval library",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] 5×4 panel → {out_path}")


def plot_distributions(grid_results, out_dir, noise_sigma):
    """One overlaid random-vs-eval histogram per GT row."""
    os.makedirs(out_dir, exist_ok=True)
    for gt_key in GT_KEYS:
        # grab y_rand and y_eval from the first cfg (same data for all cfgs in that row)
        first_cfg = next(iter(SURROGATE_CONFIGS))
        entry  = grid_results[gt_key][first_cfg]
        y_rand = entry["y_rand_all"][np.isfinite(entry["y_rand_all"])]
        y_eval = entry["y_eval"][np.isfinite(entry["y_eval"])]

        lo, hi = np.percentile(np.concatenate([y_rand, y_eval]), [0.5, 99.5])
        if lo == hi:
            lo -= 1e-6; hi += 1e-6
        bins = np.linspace(lo, hi, 121)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.hist(y_rand, bins=bins, density=True, alpha=0.45,
                label=f"Random library (N={len(y_rand):,})")
        ax.hist(y_eval, bins=bins, density=True, alpha=0.45,
                label=f"Eval library (N={len(y_eval):,})")
        ax.set_title(f"{GT_LABELS[gt_key]} — random vs eval  (noise σ={noise_sigma})")
        ax.set_xlabel("GT score")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, f"{gt_key}_random_vs_eval.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] dist → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",         default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--noise_sigma", type=float, default=0.05,
                        help="Std of Gaussian noise added to synthetic GT scores")
    parser.add_argument("--experiment",  default="RNCMPT00111",
                        help="RNACompete experiment ID for ResidualBind")
    parser.add_argument("--out_dir",     default="outputs/surrogate_gt_pipeline/noisy_panel_v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq          = args.seq.upper().replace("T", "U")
    oh_seq       = rna_to_one_hot(seq)
    wt_onehot, _ = remove_padding(oh_seq)
    L            = wt_onehot.shape[0]
    exact_mut    = min(4, max(1, L // 2))
    print(f"Sequence: {seq}  L={L}  exact_mut={exact_mut}  noise_sigma={args.noise_sigma}")

    # Load ResidualBind
    print(f"\n[init] Loading ResidualBind ({args.experiment}) …")
    data_path    = os.path.expanduser(
        "~/residualbind/data/RNAcompete_2013/rnacompete2013.h5"
    )
    save_path    = "/home/nagle/residualbind/weights/log_norm_seq"
    rbp_index    = helper.find_experiment_index(data_path, args.experiment)
    train_data, _, _ = helper.load_rnacompete_data(
        data_path, ss_type="seq", normalization="log_norm", rbp_index=rbp_index,
    )
    input_shape  = list(train_data["inputs"].shape)[1:]
    weights_path = os.path.join(save_path, args.experiment + "_weights.hdf5")
    residbind    = ResidualBind(input_shape, 1, weights_path)
    residbind.load_weights()
    print(f"[init] ResidualBind loaded.")

    # Shared synthetic GT params (rows 1–4)
    print("\n[init] GT params …")
    gt_params = init_gt_params(wt_onehot, exact_mut, args.seed)

    # Train SurrogateGT_RB (row 5): 20 k → ResidualBind → pairwise nonlinear GE
    print("\n[init] Training SurrogateGT_RB …")
    sgt_model = train_surrogate_gt(wt_onehot, exact_mut, args.seed, residbind)

    # Build grid_results for all 5 GT rows
    grid_results = {}

    synthetic_gt_keys = [
        "additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"
    ]
    for gt_key in synthetic_gt_keys:
        grid_results[gt_key] = process_synthetic_gt(
            gt_key, gt_params, wt_onehot, exact_mut,
            args.seed, args.noise_sigma,
        )

    grid_results["surrogate_gt"] = process_surrogate_gt(
        sgt_model, wt_onehot, exact_mut, args.seed,
    )

    # Plots
    panel_path = os.path.join(args.out_dir, f"panel_noise{args.noise_sigma}.png")
    plot_panel(grid_results, panel_path, args.noise_sigma)
    plot_distributions(grid_results, args.out_dir, args.noise_sigma)

    print("\nDone.")


if __name__ == "__main__":
    main()
