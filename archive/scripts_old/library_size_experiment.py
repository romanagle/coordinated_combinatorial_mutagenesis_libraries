"""library_size_experiment.py

Sweeps training library size from 200 → 2,000,000 and records Spearman
correlation between surrogate predictions and ground-truth scores on both
a random held-out test split and a curated uniformized eval library.

Six ground truths:
  1. additive
  2. additive_pairwise           (Potts couplings)
  3. nonlin_additive
  4. nonlin_additive_pairwise    (Potts couplings)
  5. additive_pairwise_gaussian  (Gaussian couplings – negative control)
  6. nonlin_additive_pairwise_gaussian

Usage:
    python scripts/library_size_experiment.py \\
        --seq <RNA_SEQ> --experiment RNCMPT00111 \\
        --mut_rate 4 --seed 0 \\
        --out_dir outputs/lib_size_experiment
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import tensorflow as tf
import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT,
    init_pairwise_potts_optionA,
    init_pairwise_gaussian,
    init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
    additive_affinity_noWT,
    pairwise_potts_energy,
    apply_global_nonlin,
)
from stats_utils import spearman_on_testdf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUCS = ['A', 'C', 'G', 'U']

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
    "additive_pairwise_gaussian",
    "nonlin_additive_pairwise_gaussian",
]

GT_TITLES = {
    "additive":                          "Additive",
    "additive_pairwise":                 "Additive + Pairwise (Potts)",
    "nonlin_additive":                   "Nonlinear Additive",
    "nonlin_additive_pairwise":          "Nonlinear Additive + Pairwise (Potts)",
    "additive_pairwise_gaussian":        "Additive + Pairwise (Gaussian)",
    "nonlin_additive_pairwise_gaussian": "Nonlinear Additive + Pairwise (Gaussian)",
}

# Nonlinear pairwise uses Gaussian noise for training stability at large N.
# SkewedT was found to diverge at N=2M; Gaussian noise achieves rho≈0.99.
SURROGATE_CONFIGS = {
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 12,
    },
    "additive + pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "nonlinear additive": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 12, "hidden_nodes": 50,
    },
    "nonlinear additive + pairwise": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
    },
}

SURROGATE_COLORS = {
    "additive":                      "#1f77b4",
    "additive + pairwise":           "#ff7f0e",
    "nonlinear additive":            "#2ca02c",
    "nonlinear additive + pairwise": "#d62728",
}

LIBRARY_SIZES = [200, 2_000, 20_000, 200_000, 2_000_000]
NONLIN_NAME   = "sigmoid"

# ---------------------------------------------------------------------------
# Random library generation
# ---------------------------------------------------------------------------

def generate_random_library(wt_onehot, n_seqs, exact_mut_count, rng):
    """Vectorised: mutate exactly `exact_mut_count` distinct positions per sequence."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)

    CHUNK = 50_000
    parts = []
    for start in range(0, n_seqs, CHUNK):
        nc = min(CHUNK, n_seqs - start)
        noise = rng.random((nc, L))
        pos   = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)

    return np.concatenate(parts, axis=0)


def onehot_to_str(x):
    return "".join(np.array(list("ACGU"))[np.argmax(x, axis=1)])


# ---------------------------------------------------------------------------
# GT scoring — both Potts and Gaussian pairwise
# ---------------------------------------------------------------------------

def score_library(X, gt_params_potts, gt_params_gaussian, nonlin_kwargs):
    """Score all N sequences against all 6 GT keys.

    Calls compute_gt_scores_for_library_potts twice (once per J matrix) and
    merges. The additive terms are identical; only the pairwise column differs.
    """
    potts_scores = compute_gt_scores_for_library_potts(
        X,
        W_mut=gt_params_potts["W_mut"],
        mut_map=gt_params_potts["mut_map"],
        b0=gt_params_potts["b"],
        nonlin_name=NONLIN_NAME,
        nonlin_kwargs=nonlin_kwargs,
        edges=gt_params_potts["edges"],
        J=gt_params_potts["J"],
    )
    gauss_scores = compute_gt_scores_for_library_potts(
        X,
        W_mut=gt_params_gaussian["W_mut"],
        mut_map=gt_params_gaussian["mut_map"],
        b0=gt_params_gaussian["b"],
        nonlin_name=NONLIN_NAME,
        nonlin_kwargs=nonlin_kwargs,
        edges=gt_params_gaussian["edges"],
        J=gt_params_gaussian["J"],
    )
    return {
        "additive":                          potts_scores["additive"],
        "additive_pairwise":                 potts_scores["additive_pairwise"],
        "nonlin_additive":                   potts_scores["nonlin_additive"],
        "nonlin_additive_pairwise":          potts_scores["nonlin_additive_pairwise"],
        "additive_pairwise_gaussian":        gauss_scores["additive_pairwise"],
        "nonlin_additive_pairwise_gaussian": gauss_scores["nonlin_additive_pairwise"],
    }


# ---------------------------------------------------------------------------
# Eval library builder
# ---------------------------------------------------------------------------

def build_eval_libraries(wt_onehot, gt_params_potts, gt_params_gaussian, nonlin_kwargs,
                          exact_mut_count, n_pool=500_000,
                          target_n=5_000, n_bins=200, clip_hi=98, seed=42):
    """Build one uniformized eval library per GT key (all 6).

    Scores the pool once against both Potts J and Gaussian J, then
    uniformizes each of the 6 score vectors independently.
    """
    W_mut   = gt_params_potts["W_mut"]
    mut_map = gt_params_potts["mut_map"]
    b0      = gt_params_potts["b"]
    edges_p = gt_params_potts["edges"]
    J_p     = gt_params_potts["J"]
    edges_g = gt_params_gaussian["edges"]
    J_g     = gt_params_gaussian["J"]
    L, A    = wt_onehot.shape
    wt_idx  = np.argmax(wt_onehot, axis=1).astype(np.uint8)

    rng_pool = np.random.default_rng(seed + 99_999)

    # ── Generate pool
    print(f"[eval] Generating {n_pool:,} pool sequences (exact {exact_mut_count} muts) …")
    nuc_ids = np.tile(wt_idx[None], (n_pool, 1)).astype(np.uint8)
    CHUNK = 100_000
    for start in range(0, n_pool, CHUNK):
        nc      = min(CHUNK, n_pool - start)
        noise   = rng_pool.random((nc, L))
        pos     = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        n_idx   = np.repeat(np.arange(nc), exact_mut_count)
        p_idx   = pos.ravel()
        wt_at   = wt_idx[p_idx]
        rand_nuc = rng_pool.integers(0, 3, size=nc * exact_mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
        nuc_ids[start:start + nc][n_idx, p_idx] = new_nucs

    # ── Pass 1: collect raw energies
    print("[eval] Scoring pool (pass 1) …")
    s_add_all      = np.empty(n_pool, dtype=np.float32)
    s_addpair_p    = np.empty(n_pool, dtype=np.float32)   # Potts pairwise
    s_addpair_g    = np.empty(n_pool, dtype=np.float32)   # Gaussian pairwise
    for start in range(0, n_pool, CHUNK):
        end     = min(start + CHUNK, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa  = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp  = pairwise_potts_energy(X_chunk, edges_p, J_p, b=0.0).reshape(-1)
        sg  = pairwise_potts_energy(X_chunk, edges_g, J_g, b=0.0).reshape(-1)
        s_add_all[start:end]   = sa
        s_addpair_p[start:end] = sa + sp
        s_addpair_g[start:end] = sa + sg

    ref_std = float(nonlin_kwargs.get("_norm_std", float(np.std(s_add_all)) + 1e-8))
    print(f"[eval] using ref_std={ref_std:.4f} (from nonlin_kwargs)")

    # ── Pass 2: apply normalization + nonlinearity
    print("[eval] Scoring pool (pass 2) …")
    pool_scores = {k: np.empty(n_pool, dtype=np.float32) for k in GT_KEYS}
    for start in range(0, n_pool, CHUNK):
        end = min(start + CHUNK, n_pool)
        sa  = s_add_all[start:end].astype(float)
        sap = s_addpair_p[start:end].astype(float)
        sag = s_addpair_g[start:end].astype(float)
        pool_scores["additive"][start:end]             = sa
        pool_scores["additive_pairwise"][start:end]    = sap
        pool_scores["additive_pairwise_gaussian"][start:end] = sag
        pool_scores["nonlin_additive"][start:end] = apply_global_nonlin(
            sa / ref_std, NONLIN_NAME, nonlin_kwargs,
        ).reshape(-1)
        pool_scores["nonlin_additive_pairwise"][start:end] = apply_global_nonlin(
            sap / ref_std, NONLIN_NAME, nonlin_kwargs,
        ).reshape(-1)
        pool_scores["nonlin_additive_pairwise_gaussian"][start:end] = apply_global_nonlin(
            sag / ref_std, NONLIN_NAME, nonlin_kwargs,
        ).reshape(-1)

    # Anchor WT → 0 for nonlinear keys
    wt_nl = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))
    pool_scores["nonlin_additive"]                   -= wt_nl
    pool_scores["nonlin_additive_pairwise"]          -= wt_nl
    pool_scores["nonlin_additive_pairwise_gaussian"] -= wt_nl

    del s_add_all, s_addpair_p, s_addpair_g

    # ── Uniformize each GT key
    eval_libs = {}
    for k in GT_KEYS:
        yk = pool_scores[k].astype(float)
        y_uni, keep_idx = uniformize_by_histogram(
            yk, X=None, n_bins=n_bins, clip_hi=clip_hi,
            target_n=target_n, seed=seed + hash(k) % 10_000,
        )
        eval_libs[k] = {
            "X_eval": np.eye(A, dtype=np.float32)[nuc_ids[keep_idx]],
            "y_eval": y_uni.astype(float),
        }
        print(f"[eval] {k}: {len(y_uni):,} seqs  "
              f"range [{y_uni.min():.3f}, {y_uni.max():.3f}]")

    del nuc_ids, pool_scores
    return eval_libs


# ---------------------------------------------------------------------------
# Surrogate fitting
# ---------------------------------------------------------------------------

def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for start in range(0, len(x_str), chunk):
        parts.append(
            np.asarray(model.x_to_yhat(x_str[start:start + chunk]), dtype=float).reshape(-1)
        )
    return np.concatenate(parts)


def fit_surrogate(x_lib, y, cfg_name, x_eval_oh, y_eval):
    """Train one surrogate; return (rho_random, rho_eval, pearson_r_random, r2_random, pearson_r_eval, r2_eval, history)."""
    cfg  = SURROGATE_CONFIGS[cfg_name]
    N    = x_lib.shape[0]
    bsz  = max(32, min(N // 150, 2048))
    lr   = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    # Nonlinear pairwise needs more epochs at large N to converge fully
    is_nl_pairwise = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs  = 1000 if is_nl_pairwise else 500
    patience = 50  if is_nl_pairwise else 25

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        x_lib.shape,
        num_tasks=1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS,
        deduplicate=True,
        gpu=True,
    )

    model, _, test_df = wrapper.train(
        x_lib, y,
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    print(f"      [lr={lr:.2e}  bsz={bsz}  ep={epochs}]", end=" ")

    # Spearman + Pearson on random held-out split
    rho_random = np.nan
    pearson_r_random = np.nan
    r2_random = np.nan
    try:
        cols  = list(test_df.columns)
        X_col = "x" if "x" in cols else "X"
        y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        X_test_str = np.asarray(test_df[X_col])
        y_test     = np.asarray(test_df[y_col], dtype=float).reshape(-1)
        y_hat_test = _predict_chunked(model, X_test_str)
        nan_frac = np.mean(~np.isfinite(y_hat_test))
        if nan_frac > 0.1:
            print(f"      [warn] {nan_frac:.0%} non-finite predictions on random test split")
        m = np.isfinite(y_test) & np.isfinite(y_hat_test)
        if m.sum() >= 3:
            rho_random, _ = spearmanr(y_test[m], y_hat_test[m])
            pearson_r_random, _ = pearsonr(y_test[m], y_hat_test[m])
            ss_res = np.sum((y_test[m] - y_hat_test[m]) ** 2)
            ss_tot = np.sum((y_test[m] - np.mean(y_test[m])) ** 2)
            r2_random = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    except Exception as e:
        print(f"      [warn] rho_random: {e}")

    # Spearman + Pearson on curated eval library
    rho_eval = np.nan
    pearson_r_eval = np.nan
    r2_eval = np.nan
    if x_eval_oh is not None and y_eval is not None:
        try:
            x_eval_str = np.array([onehot_to_str(xi) for xi in x_eval_oh], dtype=object)
            y_hat_eval = _predict_chunked(model, x_eval_str)
            m = np.isfinite(y_eval) & np.isfinite(y_hat_eval)
            if m.sum() >= 3:
                rho_eval, _ = spearmanr(y_eval[m], y_hat_eval[m])
                pearson_r_eval, _ = pearsonr(y_eval[m], y_hat_eval[m])
                ss_res = np.sum((y_eval[m] - y_hat_eval[m]) ** 2)
                ss_tot = np.sum((y_eval[m] - np.mean(y_eval[m])) ** 2)
                r2_eval = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception as e:
            print(f"      [warn] rho_eval: {e}")

    history = {}
    try:
        h = model.history
        for key in ("loss", "val_loss", "I_var", "val_I_var"):
            if key in h:
                history[key] = np.array(h[key], dtype=float)
    except Exception:
        pass

    tf.keras.backend.clear_session()
    return (float(rho_random), float(rho_eval),
            float(pearson_r_random), float(r2_random),
            float(pearson_r_eval), float(r2_eval),
            history)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, library_sizes, out_dir):
    sizes = np.array(library_sizes)

    for gt_key in GT_KEYS:
        fig, ax = plt.subplots(figsize=(7, 5))

        for cfg_name, color in SURROGATE_COLORS.items():
            rhos_random = np.array(results[gt_key][cfg_name]["random"], dtype=float)
            rhos_eval   = np.array(results[gt_key][cfg_name]["eval"],   dtype=float)

            ax.semilogx(sizes, rhos_random, marker="o", color=color,
                        linestyle="-",  linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (random)")
            ax.semilogx(sizes, rhos_eval,   marker="s", color=color,
                        linestyle="--", linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (eval)")

        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(GT_TITLES[gt_key], fontsize=12)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.25)

        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_spearman_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


def plot_pearson_r2(results, library_sizes, out_dir):
    """Save Pearson r and R² figures (same layout as Spearman plots)."""
    sizes = np.array(library_sizes)

    for gt_key in GT_KEYS:
        # ── Pearson r
        fig, ax = plt.subplots(figsize=(7, 5))
        for cfg_name, color in SURROGATE_COLORS.items():
            pr_random = np.array(results[gt_key][cfg_name]["pearson_random"], dtype=float)
            pr_eval   = np.array(results[gt_key][cfg_name]["pearson_eval"],   dtype=float)
            ax.semilogx(sizes, pr_random, marker="o", color=color,
                        linestyle="-",  linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (random)")
            ax.semilogx(sizes, pr_eval,   marker="s", color=color,
                        linestyle="--", linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (eval)")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title(GT_TITLES[gt_key], fontsize=12)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_pearson_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")

        # ── R²
        fig, ax = plt.subplots(figsize=(7, 5))
        all_r2 = []
        for cfg_name, color in SURROGATE_COLORS.items():
            r2_random = np.array(results[gt_key][cfg_name]["r2_random"], dtype=float)
            r2_eval   = np.array(results[gt_key][cfg_name]["r2_eval"],   dtype=float)
            ax.semilogx(sizes, r2_random, marker="o", color=color,
                        linestyle="-",  linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (random)")
            ax.semilogx(sizes, r2_eval,   marker="s", color=color,
                        linestyle="--", linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (eval)")
            all_r2.extend(r2_random[np.isfinite(r2_random)].tolist())
            all_r2.extend(r2_eval[np.isfinite(r2_eval)].tolist())
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("R²", fontsize=11)
        ax.set_title(GT_TITLES[gt_key], fontsize=12)
        ymin = min(all_r2) if all_r2 else -0.1
        ymin = min(ymin - 0.05, -0.1)
        ax.set_ylim(ymin, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_r2_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",           type=str, required=True)
    parser.add_argument("--experiment",    type=str, default="RNCMPT00111")
    parser.add_argument("--mut_rate",      type=int, default=4)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--out_dir",       type=str,
                        default="outputs/lib_size_experiment")
    parser.add_argument("--eval_n_pool",   type=int, default=500_000)
    parser.add_argument("--eval_target_n", type=int, default=5_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Parse WT
    oh_seq            = rna_to_one_hot(args.seq)
    no_padding_seq, _ = remove_padding(oh_seq)
    wt_onehot         = no_padding_seq
    L                 = wt_onehot.shape[0]
    exact_mut_count   = args.mut_rate
    print(f"[init] WT length L={L},  exact_mut_count={exact_mut_count}")

    # ── 2. Shared additive params + two pairwise coupling sets
    rng_gt = np.random.default_rng(args.seed)
    print("[init] Initializing additive GT parameters …")
    W_mut, mut_map, b0 = init_additive_noWT(
        rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0,
    )

    print("[init] Initializing Potts pairwise couplings …")
    edges_p, J_p = init_pairwise_potts_optionA(
        rng_gt, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    print("[init] Initializing Gaussian pairwise couplings (negative control) …")
    edges_g, J_g = init_pairwise_gaussian(
        rng_gt, wt_onehot, sigma=0.3, wt_rowcol_zero=True,
    )
    print(f"[init] Potts edges: {len(edges_p):,}   Gaussian edges: {len(edges_g):,}")

    gt_params_potts = {
        "W_mut": W_mut, "mut_map": mut_map, "b": b0,
        "edges": edges_p, "J": J_p,
    }
    gt_params_gaussian = {
        "W_mut": W_mut, "mut_map": mut_map, "b": b0,
        "edges": edges_g, "J": J_g,
    }

    # ── 3. Sigmoid nonlin params (fixed 200K reference library)
    print("[init] Computing sigmoid ref_std from 200K reference library …")
    rng_ref = np.random.default_rng(args.seed + 1)
    X_ref   = generate_random_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add_ref     = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"[init] sigmoid ref_std={nonlin_kwargs['_norm_std']:.4f}  "
          f"sig_z0={nonlin_kwargs['sig_z0']:.4f}")

    # ── 4. Fixed eval libraries
    print("\n[init] Building eval libraries …")
    eval_libs = build_eval_libraries(
        wt_onehot, gt_params_potts, gt_params_gaussian, nonlin_kwargs,
        exact_mut_count=exact_mut_count,
        n_pool=args.eval_n_pool,
        target_n=args.eval_target_n,
        seed=args.seed,
    )

    # ── 5. Result storage
    results = {
        gt_key: {cfg: {"random": [], "eval": [], "pearson_random": [], "r2_random": [], "pearson_eval": [], "r2_eval": []} for cfg in SURROGATE_CONFIGS}
        for gt_key in GT_KEYS
    }
    loss_history = {
        gt_key: {cfg: {} for cfg in SURROGATE_CONFIGS}
        for gt_key in GT_KEYS
    }

    # ── 6. Sweep library sizes
    for N in LIBRARY_SIZES:
        print(f"\n{'='*62}")
        print(f"  Library size N = {N:,}")
        print(f"{'='*62}")

        rng_lib   = np.random.default_rng(args.seed + N)
        X_lib     = generate_random_library(wt_onehot, N, exact_mut_count, rng_lib)
        gt_scores = score_library(X_lib, gt_params_potts, gt_params_gaussian, nonlin_kwargs)

        for gt_key in GT_KEYS:
            y    = gt_scores[gt_key].astype(float).reshape(-1, 1)
            mask = np.isfinite(y[:, 0])
            x_use, y_use = X_lib[mask], y[mask]

            x_eval_oh = eval_libs[gt_key]["X_eval"]
            y_eval    = eval_libs[gt_key]["y_eval"]

            print(f"\n  [{gt_key}]  N_valid={mask.sum():,}  "
                  f"score range [{y_use.min():.3f}, {y_use.max():.3f}]")

            for cfg_name in SURROGATE_CONFIGS:
                print(f"    {cfg_name:<32} … ", end="", flush=True)
                rho_rand, rho_ev, pr_rand, r2_rand, pr_ev, r2_ev, hist = fit_surrogate(
                    x_use, y_use, cfg_name, x_eval_oh, y_eval,
                )
                results[gt_key][cfg_name]["random"].append(rho_rand)
                results[gt_key][cfg_name]["eval"].append(rho_ev)
                results[gt_key][cfg_name]["pearson_random"].append(pr_rand)
                results[gt_key][cfg_name]["r2_random"].append(r2_rand)
                results[gt_key][cfg_name]["pearson_eval"].append(pr_ev)
                results[gt_key][cfg_name]["r2_eval"].append(r2_ev)
                loss_history[gt_key][cfg_name][N] = hist
                print(f"rho_random={rho_rand:+.3f}   rho_eval={rho_ev:+.3f}   pearson_random={pr_rand:+.3f}   r2_random={r2_rand:+.3f}")

        del X_lib, gt_scores

    # ── 7. Save CSV
    rows = []
    for gt_key in GT_KEYS:
        for cfg_name in SURROGATE_CONFIGS:
            for i, N in enumerate(LIBRARY_SIZES):
                rows.append({
                    "gt_key":          gt_key,
                    "surrogate":       cfg_name,
                    "library_size":    N,
                    "rho_random":      results[gt_key][cfg_name]["random"][i],
                    "rho_eval":        results[gt_key][cfg_name]["eval"][i],
                    "pearson_random":  results[gt_key][cfg_name]["pearson_random"][i],
                    "r2_random":       results[gt_key][cfg_name]["r2_random"][i],
                    "pearson_eval":    results[gt_key][cfg_name]["pearson_eval"][i],
                    "r2_eval":         results[gt_key][cfg_name]["r2_eval"][i],
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "lib_size_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[results] saved → {csv_path}")
    print(df.to_string(index=False))

    npy_path = os.path.join(args.out_dir, "loss_history.npy")
    np.save(npy_path, loss_history, allow_pickle=True)
    print(f"[results] loss history saved → {npy_path}")

    # ── 8. Plot
    plot_results(results, LIBRARY_SIZES, args.out_dir)
    plot_pearson_r2(results, LIBRARY_SIZES, args.out_dir)


if __name__ == "__main__":
    main()
