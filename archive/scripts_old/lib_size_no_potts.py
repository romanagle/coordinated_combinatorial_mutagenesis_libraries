"""lib_size_no_potts.py

Same library-size sweep as library_size_experiment.py, but the
nonlin_additive_pairwise ground truth has the Potts pairwise term
zeroed out (s_pair = 0).  Only the nonlin_additive_pairwise GT key
is computed, so the run is ~4x faster.

Saves results to outputs/lib_size_experiment/no_potts/ and writes a
2-panel comparison figure (with Potts vs without Potts) alongside the
per-panel random-split figure.

Usage:
    python scripts/lib_size_no_potts.py \\
        --seq <RNA_SEQ> --experiment RNCMPT00111 \\
        --mut_rate 4 --seed 0 \\
        --original_csv outputs/lib_size_experiment/lib_size_results.csv \\
        --out_dir outputs/lib_size_experiment/no_potts
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
    init_sigmoid_nonlin,
    uniformize_by_histogram,
    additive_affinity_noWT,
    pairwise_potts_energy,
    apply_global_nonlin,
)

# ---------------------------------------------------------------------------
# Config  (must match library_size_experiment.py exactly)
# ---------------------------------------------------------------------------

NUCS = ['A', 'C', 'G', 'U']
GT_KEY = "nonlin_additive_pairwise"

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

CHUNK = 50_000

# ---------------------------------------------------------------------------
# Helpers (copied from library_size_experiment.py verbatim)
# ---------------------------------------------------------------------------

def generate_random_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    parts  = []
    for start in range(0, n_seqs, CHUNK):
        nc    = min(CHUNK, n_seqs - start)
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
# GT scoring — Potts term is zeroed out
# ---------------------------------------------------------------------------

def score_library_no_potts(X, W_mut, mut_map, b0, nonlin_kwargs):
    """Score nonlin_additive_pairwise with s_pair = 0."""
    s_add = additive_affinity_noWT(X, W_mut, mut_map, b=b0).reshape(-1).astype(float)
    # Potts deliberately omitted: s_pair = 0
    ref_std = float(nonlin_kwargs.get("_norm_std", float(np.std(s_add)) + 1e-8))
    y = apply_global_nonlin(s_add / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1)
    wt_nl = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))
    return y - wt_nl


# ---------------------------------------------------------------------------
# Eval library builder — Potts term zeroed out
# ---------------------------------------------------------------------------

def build_eval_library_no_potts(wt_onehot, W_mut, mut_map, b0, nonlin_kwargs,
                                 exact_mut_count, n_pool=500_000,
                                 target_n=5_000, n_bins=200, clip_hi=98, seed=42):
    L, A   = wt_onehot.shape
    wt_idx = np.argmax(wt_onehot, axis=1).astype(np.uint8)

    rng_pool = np.random.default_rng(seed + 99_999)
    print(f"[eval] Generating {n_pool:,} pool sequences …")
    nuc_ids = np.tile(wt_idx[None], (n_pool, 1)).astype(np.uint8)
    for start in range(0, n_pool, 100_000):
        nc       = min(100_000, n_pool - start)
        noise    = rng_pool.random((nc, L))
        pos      = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng_pool.integers(0, 3, size=nc * exact_mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
        nuc_ids[start:start + nc][n_idx, p_idx] = new_nucs

    print(f"[eval] Scoring pool …")
    ref_std = float(nonlin_kwargs.get("_norm_std", 1.0))
    wt_nl   = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))
    pool_scores = np.empty(n_pool, dtype=np.float32)
    for start in range(0, n_pool, 100_000):
        end     = min(start + 100_000, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa      = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1).astype(float)
        pool_scores[start:end] = (
            apply_global_nonlin(sa / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
        )

    y_uni, keep_idx = uniformize_by_histogram(
        pool_scores.astype(float), X=None,
        n_bins=n_bins, clip_hi=clip_hi, target_n=target_n,
        seed=seed + hash(GT_KEY) % 10_000,
    )
    X_eval = np.eye(A, dtype=np.float32)[nuc_ids[keep_idx]]
    print(f"[eval] {GT_KEY} (no Potts): {len(y_uni):,} seqs  "
          f"range [{y_uni.min():.3f}, {y_uni.max():.3f}]")
    del nuc_ids, pool_scores
    return X_eval, y_uni.astype(float)


# ---------------------------------------------------------------------------
# Surrogate fitting (identical to library_size_experiment.py)
# ---------------------------------------------------------------------------

def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for start in range(0, len(x_str), chunk):
        parts.append(
            np.asarray(model.x_to_yhat(x_str[start:start + chunk]), dtype=float).reshape(-1)
        )
    return np.concatenate(parts)


def fit_surrogate(x_lib, y, cfg_name, x_eval_oh, y_eval):
    cfg  = SURROGATE_CONFIGS[cfg_name]
    N    = x_lib.shape[0]
    bsz  = max(32, min(N // 150, 2048))
    lr   = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

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

    rho_random = np.nan
    try:
        cols      = list(test_df.columns)
        X_col     = "x" if "x" in cols else "X"
        y_col     = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        y_test    = np.asarray(test_df[y_col], dtype=float).reshape(-1)
        y_hat_test = _predict_chunked(model, np.asarray(test_df[X_col]))
        m = np.isfinite(y_test) & np.isfinite(y_hat_test)
        if m.sum() >= 3:
            rho_random, _ = spearmanr(y_test[m], y_hat_test[m])
    except Exception as e:
        print(f"      [warn] rho_random: {e}")

    rho_eval = np.nan
    if x_eval_oh is not None and y_eval is not None:
        try:
            x_eval_str = np.array([onehot_to_str(xi) for xi in x_eval_oh], dtype=object)
            y_hat_eval = _predict_chunked(model, x_eval_str)
            m = np.isfinite(y_eval) & np.isfinite(y_hat_eval)
            if m.sum() >= 3:
                rho_eval, _ = spearmanr(y_eval[m], y_hat_eval[m])
        except Exception as e:
            print(f"      [warn] rho_eval: {e}")

    tf.keras.backend.clear_session()
    return float(rho_random), float(rho_eval)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single(results_no_potts, library_sizes, out_dir, split="random"):
    """One figure: nonlin_additive_pairwise without Potts, all 4 surrogates."""
    sizes = np.array(library_sizes)
    fig, ax = plt.subplots(figsize=(7, 5))
    rho_key = "random" if split == "random" else "eval"

    for cfg_name, color in SURROGATE_COLORS.items():
        rhos = np.array(results_no_potts[cfg_name][rho_key], dtype=float)
        ax.semilogx(sizes, rhos, marker="o", color=color, linestyle="-",
                    linewidth=1.8, markersize=5, label=cfg_name)

    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("Nonlinear Additive + Pairwise  (no Potts)", fontsize=12)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, f"lib_size_no_potts_{split}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")


def plot_comparison(results_no_potts, original_csv, library_sizes, out_dir, split="random"):
    """2-panel: left = with Potts (original), right = without Potts."""
    sizes    = np.array(library_sizes)
    rho_key  = "random" if split == "random" else "eval"
    col_key  = "rho_random" if split == "random" else "rho_eval"

    # Load original results for nonlin_additive_pairwise
    orig_df  = pd.read_csv(original_csv)
    orig_sub = orig_df[orig_df["gt_key"] == GT_KEY].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (title, source) in zip(axes, [
        ("With Potts",    "original"),
        ("Without Potts", "no_potts"),
    ]):
        for cfg_name, color in SURROGATE_COLORS.items():
            if source == "original":
                row  = orig_sub[orig_sub["surrogate"] == cfg_name].sort_values("library_size")
                rhos = row[col_key].to_numpy(dtype=float)
                szs  = row["library_size"].to_numpy()
            else:
                rhos = np.array(results_no_potts[cfg_name][rho_key], dtype=float)
                szs  = sizes

            ax.semilogx(szs, rhos, marker="o", color=color, linestyle="-",
                        linewidth=1.8, markersize=5, label=cfg_name)

        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_title(f"Nonlinear Additive + Pairwise\n({title})", fontsize=11)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Spearman ρ", fontsize=11)
    fig.suptitle(f"Surrogate reconstruction  ({split} split)", fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"lib_size_comparison_potts_vs_nopotts_{split}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",          type=str, required=True)
    parser.add_argument("--experiment",   type=str, default="RNCMPT00111")
    parser.add_argument("--mut_rate",     type=int, default=4)
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--out_dir",      type=str,
                        default="outputs/lib_size_experiment/no_potts")
    parser.add_argument("--original_csv", type=str,
                        default="outputs/lib_size_experiment/lib_size_results.csv",
                        help="CSV from library_size_experiment.py for the comparison figure")
    parser.add_argument("--eval_n_pool",   type=int, default=500_000)
    parser.add_argument("--eval_target_n", type=int, default=5_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Parse WT sequence
    oh_seq            = rna_to_one_hot(args.seq)
    no_padding_seq, _ = remove_padding(oh_seq)
    wt_onehot         = no_padding_seq
    L                 = wt_onehot.shape[0]
    exact_mut_count   = args.mut_rate
    print(f"[init] WT length L={L},  exact_mut_count={exact_mut_count}")

    # ── 2. Fixed GT params (same seed as library_size_experiment.py)
    rng_gt = np.random.default_rng(args.seed)
    print("[init] Initializing GT parameters …")
    W_mut, mut_map, b0 = init_additive_noWT(
        rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0,
    )
    # Potts params are initialized with same seed for reproducibility,
    # but they are NOT used in scoring (s_pair = 0 throughout).
    _, _ = init_pairwise_potts_optionA(
        rng_gt, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    # ── 3. Sigmoid nonlin from 200K reference library (same as original)
    print("[init] Computing sigmoid ref_std from 200K reference library …")
    rng_ref  = np.random.default_rng(args.seed + 1)
    X_ref    = generate_random_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"[init] sigmoid ref_std={nonlin_kwargs['_norm_std']:.4f}  "
          f"sig_z0={nonlin_kwargs['sig_z0']:.4f}")

    # ── 4. Fixed eval library (no Potts)
    print("\n[init] Building eval library (no Potts) …")
    X_eval, y_eval = build_eval_library_no_potts(
        wt_onehot, W_mut, mut_map, b0, nonlin_kwargs,
        exact_mut_count=exact_mut_count,
        n_pool=args.eval_n_pool,
        target_n=args.eval_target_n,
        seed=args.seed,
    )

    # ── 5. Result storage
    results = {cfg: {"random": [], "eval": []} for cfg in SURROGATE_CONFIGS}

    # ── 6. Sweep library sizes
    for N in LIBRARY_SIZES:
        print(f"\n{'='*62}")
        print(f"  Library size N = {N:,}")
        print(f"{'='*62}")

        rng_lib  = np.random.default_rng(args.seed + N)
        X_lib    = generate_random_library(wt_onehot, N, exact_mut_count, rng_lib)
        y_scores = score_library_no_potts(X_lib, W_mut, mut_map, b0, nonlin_kwargs)
        y        = y_scores.reshape(-1, 1)
        mask     = np.isfinite(y[:, 0])
        x_use, y_use = X_lib[mask], y[mask]
        print(f"  N_valid={mask.sum():,}  "
              f"score range [{y_use.min():.3f}, {y_use.max():.3f}]")

        for cfg_name in SURROGATE_CONFIGS:
            print(f"    {cfg_name:<36} … ", end="", flush=True)
            rho_rand, rho_ev = fit_surrogate(
                x_use, y_use, cfg_name, X_eval, y_eval,
            )
            results[cfg_name]["random"].append(rho_rand)
            results[cfg_name]["eval"].append(rho_ev)
            print(f"rho_random={rho_rand:+.3f}   rho_eval={rho_ev:+.3f}")

        del X_lib, y_scores

    # ── 7. Save CSV
    rows = []
    for cfg_name in SURROGATE_CONFIGS:
        for i, N in enumerate(LIBRARY_SIZES):
            rows.append({
                "gt_key":       GT_KEY + "_no_potts",
                "surrogate":    cfg_name,
                "library_size": N,
                "rho_random":   results[cfg_name]["random"][i],
                "rho_eval":     results[cfg_name]["eval"][i],
            })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "lib_size_results_no_potts.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[results] saved → {csv_path}")
    print(df.to_string(index=False))

    # ── 8. Figures
    for split in ("random", "eval"):
        plot_single(results, LIBRARY_SIZES, args.out_dir, split=split)
        plot_comparison(results, args.original_csv, LIBRARY_SIZES, args.out_dir, split=split)


if __name__ == "__main__":
    main()
