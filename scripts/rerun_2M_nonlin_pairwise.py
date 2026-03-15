"""rerun_2M_nonlin_pairwise.py

Reruns ONLY the N=2,000,000 point for the 'nonlinear additive + pairwise'
surrogate against the 'nonlin_additive_pairwise' GT, with:
  - reg_strength = 0  (no regularisation)
  - LR scaled down:   5e-4 * sqrt(20_000 / 2_000_000) = 5e-5
  - NaN guard on predictions

Patches the existing lib_size_results.csv in-place and regenerates the
random-split figure for nonlin_additive_pairwise.

Usage:
    python scripts/rerun_2M_nonlin_pairwise.py \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --csv outputs/lib_size_experiment/lib_size_results.csv \\
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
    init_pairwise_gaussian,
    init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
    additive_affinity_noWT,
    pairwise_potts_energy,
    apply_global_nonlin,
)

NUCS         = ['A', 'C', 'G', 'U']
N_TARGET     = 2_000_000
GT_KEY       = "nonlin_additive_pairwise"
SURROGATE    = "nonlinear additive + pairwise"
NONLIN_NAME  = "sigmoid"
SEED         = 0

SURROGATE_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.0, "hidden_nodes": 50,
}

SURROGATE_COLORS = {
    "additive":                      "#1f77b4",
    "additive + pairwise":           "#ff7f0e",
    "nonlinear additive":            "#2ca02c",
    "nonlinear additive + pairwise": "#d62728",
}

CHUNK = 50_000


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


def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for start in range(0, len(x_str), chunk):
        parts.append(
            np.asarray(model.x_to_yhat(x_str[start:start + chunk]), dtype=float).reshape(-1)
        )
    return np.concatenate(parts)


def plot_random_split(df, out_dir):
    """Regenerate the random-split figure for nonlin_additive_pairwise."""
    sub   = df[df["gt_key"] == GT_KEY].copy()
    sizes = sorted(sub["library_size"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg_name, color in SURROGATE_COLORS.items():
        row = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
        ax.semilogx(
            row["library_size"], row["rho_random"],
            marker="o", color=color, linestyle="-",
            linewidth=1.8, markersize=5, label=cfg_name,
        )
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title("Nonlinear Additive + Pairwise  (random split)", fontsize=12)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "lib_size_spearman_random_nonlin_additive_pairwise.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",     type=str, required=True)
    parser.add_argument("--mut_rate", type=int, default=4)
    parser.add_argument("--csv",     type=str,
                        default="outputs/lib_size_experiment/lib_size_results.csv")
    parser.add_argument("--out_dir", type=str,
                        default="outputs/lib_size_experiment")
    args = parser.parse_args()

    # ── 1. Parse WT
    oh_seq            = rna_to_one_hot(args.seq)
    no_padding_seq, _ = remove_padding(oh_seq)
    wt_onehot         = no_padding_seq
    L                 = wt_onehot.shape[0]
    exact_mut_count   = args.mut_rate
    print(f"[init] L={L}  mut_rate={exact_mut_count}")

    # ── 2. GT params (same seed as original experiment)
    rng_gt = np.random.default_rng(SEED)
    W_mut, mut_map, b0 = init_additive_noWT(
        rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0,
    )
    edges, J = init_pairwise_potts_optionA(
        rng_gt, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    # ── 3. Sigmoid params from 200K reference library (same seed as original)
    print("[init] Computing sigmoid ref_std …")
    rng_ref   = np.random.default_rng(SEED + 1)
    X_ref     = generate_random_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"[init] ref_std={nonlin_kwargs['_norm_std']:.4f}")

    # ── 4. Generate N=2M library (same seed as original: seed + N)
    print(f"\n[run] Generating N={N_TARGET:,} library …")
    rng_lib  = np.random.default_rng(SEED + N_TARGET)
    X_lib    = generate_random_library(wt_onehot, N_TARGET, exact_mut_count, rng_lib)
    gt_scores = compute_gt_scores_for_library_potts(
        X_lib,
        W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name=NONLIN_NAME, nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )
    y    = gt_scores[GT_KEY].astype(float).reshape(-1, 1)
    mask = np.isfinite(y[:, 0])
    x_use, y_use = X_lib[mask], y[mask]
    print(f"[run] N_valid={mask.sum():,}  "
          f"score range [{y_use.min():.3f}, {y_use.max():.3f}]")
    del X_lib, gt_scores

    # ── 5. Train surrogate
    N   = x_use.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    print(f"[train] surrogate='{SURROGATE}'  lr={lr:.2e}  bsz={bsz}  reg=0")

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        x_use.shape,
        num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG["hidden_nodes"],
        alphabet=NUCS,
        deduplicate=True,
        gpu=True,
    )

    model, _, test_df = wrapper.train(
        x_use, y_use,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50, restore_best_weights=True,
        save_dir=None, verbose=0,
    )

    # ── 6. Evaluate
    rho_random = np.nan
    try:
        cols       = list(test_df.columns)
        X_col      = "x" if "x" in cols else "X"
        y_col      = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        y_test     = np.asarray(test_df[y_col], dtype=float).reshape(-1)
        y_hat_test = _predict_chunked(model, np.asarray(test_df[X_col]))
        nan_frac   = np.mean(~np.isfinite(y_hat_test))
        if nan_frac > 0.1:
            print(f"[warn] {nan_frac:.0%} non-finite predictions — model may have diverged")
        m = np.isfinite(y_test) & np.isfinite(y_hat_test)
        if m.sum() >= 3:
            rho_random, _ = spearmanr(y_test[m], y_hat_test[m])
    except Exception as e:
        print(f"[warn] rho_random: {e}")

    tf.keras.backend.clear_session()
    print(f"\n[result] rho_random = {rho_random:+.4f}  (was -0.6367 with reg=0.1 + lr=5e-4)")

    # ── 7. Patch CSV
    df = pd.read_csv(args.csv)
    mask_row = (
        (df["gt_key"]       == GT_KEY)     &
        (df["surrogate"]    == SURROGATE)  &
        (df["library_size"] == N_TARGET)
    )
    if mask_row.sum() == 0:
        print("[warn] no matching row found in CSV — appending new row")
        new_row = pd.DataFrame([{
            "gt_key": GT_KEY, "surrogate": SURROGATE,
            "library_size": N_TARGET, "rho_random": rho_random, "rho_eval": np.nan,
        }])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[mask_row, "rho_random"] = rho_random
        print(f"[csv] patched {mask_row.sum()} row(s)")

    df.to_csv(args.csv, index=False)
    print(f"[csv] saved → {args.csv}")

    # ── 8. Regenerate figure
    plot_random_split(df, args.out_dir)


if __name__ == "__main__":
    main()
