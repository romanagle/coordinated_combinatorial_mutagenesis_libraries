"""scripts/single_seq_lib_size_graph.py

Trains a nonlinear additive+pairwise MAVE-NN surrogate at 9 conditions
(3 mutation rates × 3 library sizes) for one RNA sequence using the mRNA_RBP
ground-truth model, then evaluates on three eval sets and plots Option B:

  Solid lines   = Spearman ρ on MAVE-NN 20% random test split
  Dashed lines  = Spearman ρ on Type 1 uniform eval library (matched lib_size)
  Dotted lines  = Spearman ρ on Type 2 uniform eval library (shared 20K)
  Thick black dashed = saturated mutagenesis ρ on Type 2

Colors encode mutation rate (5% / 10% / 25%).

Type 1 target_n = int(0.2 × lib_size)  (matches MAVE-NN's 20% test split)
Type 2 target_n = 20,000

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/single_seq_lib_size_graph.py \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --out_dir outputs/single_seq_lib_size
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mRNA_RBP"))

import tensorflow as tf
import squid.surrogate_zoo

from seq_utils import rna_to_one_hot
from ground_truth import (
    init_additive_noWT, init_sigmoid_nonlin,
    uniformize_by_histogram, additive_affinity_noWT,
    pairwise_potts_energy, apply_global_nonlin,
)
from gt_init import init_wc_pairwise_sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NONLIN_NAME     = "sigmoid"
MUT_RATES_PCT   = [5, 10, 25]
LIB_SIZES       = [200, 2_000, 20_000]
TRAIN_POOL_SIZE = 200_000
TYPE2_TARGET_N  = 20_000
N_POOL_PER_RATE = 300_000

MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}
MUT_LABELS = {5: "5% mut", 10: "10% mut", 25: "25% mut"}

SURROGATE_CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}
NUCS = ['A', 'C', 'G', 'U']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_mut_count(pct, L):
    return max(1, round(pct / 100.0 * L))


def _gen_pool(wt_onehot, n, exact_mut_count, rng):
    """Generate n mutant sequences as one-hot (n, L, A) float32."""
    L, A   = wt_onehot.shape
    wt_idx = np.argmax(wt_onehot, axis=1)
    CHUNK  = 50_000
    parts  = []
    for start in range(0, n, CHUNK):
        nc    = min(CHUNK, n - start)
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


def _score_X(X, gt_params, nonlin_kwargs):
    """Score library under nonlin_additive_pairwise GT, return (N,) float64."""
    W_mut      = gt_params["W_mut"]
    mut_map    = gt_params["mut_map"]
    b0         = gt_params["b"]
    edges      = gt_params["edges"]
    J          = gt_params["J"]
    ref_std_ap = float(nonlin_kwargs["_norm_std_addpair"])
    wt_nl      = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))
    CHUNK = 50_000
    N     = len(X)
    y     = np.empty(N, dtype=np.float32)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        sa  = additive_affinity_noWT(X[start:end], W_mut, mut_map, b=b0).reshape(-1)
        sp  = pairwise_potts_energy(X[start:end], edges, J, b=0.0).reshape(-1)
        sap = sa + sp
        y[start:end] = (
            apply_global_nonlin(sap / ref_std_ap, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
        )
    return y.astype(np.float64)


# ---------------------------------------------------------------------------
# GT initialiser
# ---------------------------------------------------------------------------

def init_gt(seq, seq_seed):
    """Initialise mRNA_RBP ground truth for one sequence."""
    wt_oh = rna_to_one_hot(seq)
    L, A  = wt_oh.shape

    rng_gt = np.random.default_rng(seq_seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_oh, sigma=0.5, l1_w=0.03, bias=0.0)
    edges, J = init_wc_pairwise_sparse(
        rng_gt, wt_oh, seq,
        p_edge=0.15, sigma_P=0.30, l1_P=0.40,
        edge_seed=int(seq_seed),
    )
    gt_params = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges, "J": J}

    calib_emc = _exact_mut_count(10, L)
    rng_ref   = np.random.default_rng(seq_seed + 1)
    X_cal     = _gen_pool(wt_oh, 200_000, calib_emc, rng_ref)
    sa_ref    = additive_affinity_noWT(X_cal, W_mut, mut_map, b=b0).reshape(-1)
    sp_ref    = pairwise_potts_energy(X_cal, edges, J, b=0.0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(sa_ref)
    nonlin_kwargs["_norm_std_addpair"] = float(np.std(sa_ref + sp_ref)) + 1e-8
    print(f"  [gt] sigmoid std={nonlin_kwargs['_norm_std']:.4f}  "
          f"addpair_std={nonlin_kwargs['_norm_std_addpair']:.4f}")
    del X_cal, sa_ref, sp_ref

    return wt_oh, gt_params, nonlin_kwargs


# ---------------------------------------------------------------------------
# Eval library builders
# ---------------------------------------------------------------------------

def build_eval_pool(wt_oh, gt_params, nonlin_kwargs, mut_rates_exact, seed):
    """Generate shared eval pool (3 × N_POOL_PER_RATE seqs), score, return arrays."""
    rng = np.random.default_rng(seed)
    all_X, all_labels = [], []
    for pct, emc in mut_rates_exact:
        Xi = _gen_pool(wt_oh, N_POOL_PER_RATE, emc, rng)
        all_X.append(Xi)
        all_labels.append(np.full(N_POOL_PER_RATE, pct, dtype=np.uint8))
    X_pool = np.concatenate(all_X, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    del all_X
    print(f"  [eval pool] {len(X_pool):,} seqs — scoring …")
    y_pool = _score_X(X_pool, gt_params, nonlin_kwargs)
    print(f"  [eval pool] range [{y_pool.min():.3f}, {y_pool.max():.3f}]")
    return X_pool, y_pool, labels


def uniformize_eval(X_pool, y_pool, target_n, seed):
    n_bins = min(200, target_n)
    y_uni, keep_idx = uniformize_by_histogram(
        y_pool, X=None, n_bins=n_bins, clip_hi=98,
        target_n=target_n, seed=seed,
    )
    return X_pool[keep_idx], y_uni.astype(np.float64)


# ---------------------------------------------------------------------------
# Saturated ρ
# ---------------------------------------------------------------------------

def compute_saturated_rho(wt_oh, gt_params, nonlin_kwargs, X_type2, y_type2):
    """Fit additive delta_W from all 3L single-mutant GT scores, predict on Type2."""
    L, A   = wt_oh.shape
    wt_idx = np.argmax(wt_oh, axis=1)

    X_sat = []
    for pos in range(L):
        for nuc in range(A):
            if nuc == wt_idx[pos]:
                continue
            x = wt_oh.copy()
            x[pos]      = 0.0
            x[pos, nuc] = 1.0
            X_sat.append(x)
    X_sat = np.stack(X_sat).astype(np.float32)

    y_sat   = _score_X(X_sat, gt_params, nonlin_kwargs)
    delta_W = np.zeros((A, L), dtype=np.float64)
    idx_s   = 0
    for pos in range(L):
        for nuc in range(A):
            if nuc == wt_idx[pos]:
                continue
            delta_W[nuc, pos] = float(y_sat[idx_s]) if np.isfinite(y_sat[idx_s]) else 0.0
            idx_s += 1

    y_hat = np.einsum('nla,al->n', X_type2.astype(np.float64), delta_W)
    m     = np.isfinite(y_type2) & np.isfinite(y_hat)
    if m.sum() < 3:
        return np.nan
    rho, _ = spearmanr(y_type2[m], y_hat[m])
    return float(rho)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def _onehot_to_str(x):
    return "".join(np.array(list("ACGU"))[np.argmax(x, axis=1)])


def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for start in range(0, len(x_str), chunk):
        parts.append(
            np.asarray(model.x_to_yhat(x_str[start:start + chunk]), dtype=float).reshape(-1)
        )
    return np.concatenate(parts)


def fit_surrogate(X_lib, y_lib, X_type1, y_type1, X_type2, y_type2):
    """Train nonlin add+pair surrogate; return (rho_random, rho_type1, rho_type2)."""
    N   = X_lib.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X_lib.shape,
        num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG.get("hidden_nodes", 10),
        alphabet=NUCS,
        deduplicate=True,
        gpu=True,
    )

    try:
        model, _, test_df = wrapper.train(
            X_lib, y_lib,
            learning_rate=lr, epochs=1000, batch_size=bsz,
            early_stopping=True, patience=50, restore_best_weights=True,
            save_dir=None, verbose=0,
        )
    except Exception as e:
        print(f"      [SKIP] train failed: {e}")
        tf.keras.backend.clear_session()
        return np.nan, np.nan, np.nan

    rho_random = np.nan
    try:
        cols      = list(test_df.columns)
        X_col     = "x" if "x" in cols else "X"
        y_col     = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        x_test_st = np.asarray(test_df[X_col])
        y_test    = np.asarray(test_df[y_col], dtype=float).reshape(-1)
        y_hat_rnd = _predict_chunked(model, x_test_st)
        m = np.isfinite(y_test) & np.isfinite(y_hat_rnd)
        if m.sum() >= 3:
            rho_random, _ = spearmanr(y_test[m], y_hat_rnd[m])
    except Exception as e:
        print(f"      [warn] rho_random: {e}")

    def _eval_on(X_ev, y_ev, label):
        if X_ev is None or y_ev is None:
            return np.nan
        try:
            x_str  = np.array([_onehot_to_str(xi) for xi in X_ev], dtype=object)
            y_hat  = _predict_chunked(model, x_str)
            m      = np.isfinite(y_ev) & np.isfinite(y_hat)
            if m.sum() < 3:
                return np.nan
            r, _   = spearmanr(y_ev[m], y_hat[m])
            return float(r)
        except Exception as e:
            print(f"      [warn] {label}: {e}")
            return np.nan

    rho_type1 = _eval_on(X_type1, y_type1, "rho_type1")
    rho_type2 = _eval_on(X_type2, y_type2, "rho_type2")

    tf.keras.backend.clear_session()
    return float(rho_random), rho_type1, rho_type2


# ---------------------------------------------------------------------------
# Plotting — Option B
# ---------------------------------------------------------------------------

def plot_option_b(results, rho_saturated, out_path, seq):
    """
    results: dict[(pct, lib_size)] = {"rho_random": float, "rho_type1": float, "rho_type2": float}
    rho_saturated: float (evaluated on Type 2)
    Option B: 1 panel, line styles = eval set, colors = mut rate, black dashed = saturated.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x_vals  = np.array(LIB_SIZES, dtype=float)

    for pct in MUT_RATES_PCT:
        color     = MUT_COLORS[pct]
        rho_rand  = [results.get((pct, n), {}).get("rho_random", np.nan) for n in LIB_SIZES]
        rho_type1 = [results.get((pct, n), {}).get("rho_type1",  np.nan) for n in LIB_SIZES]
        rho_type2 = [results.get((pct, n), {}).get("rho_type2",  np.nan) for n in LIB_SIZES]

        ax.plot(x_vals, rho_rand,  color=color, lw=1.8, ls="-",  marker="o", ms=5)
        ax.plot(x_vals, rho_type1, color=color, lw=1.8, ls="--", marker="s", ms=5)
        ax.plot(x_vals, rho_type2, color=color, lw=1.8, ls=":",  marker="^", ms=5)

    if np.isfinite(rho_saturated):
        ax.axhline(rho_saturated, color="black", lw=2.5, ls="--",
                   zorder=5, label=f"Saturated mut. ρ={rho_saturated:.3f}")

    # Legend: colour patches + linestyle patches
    color_handles = [
        Line2D([0], [0], color=MUT_COLORS[p], lw=2, label=MUT_LABELS[p])
        for p in MUT_RATES_PCT
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2, ls="-",  label="Random test split"),
        Line2D([0], [0], color="gray", lw=2, ls="--", label="Type 1 eval (per lib size)"),
        Line2D([0], [0], color="gray", lw=2, ls=":",  label="Type 2 eval (20K shared)"),
    ]
    sat_handle = [
        Line2D([0], [0], color="black", lw=2.5, ls="--",
               label=f"Saturated ρ={rho_saturated:.3f}" if np.isfinite(rho_saturated) else "Saturated mut.")
    ]
    ax.legend(handles=color_handles + style_handles + sat_handle,
              fontsize=8, loc="lower right", framealpha=0.9, ncol=2)

    ax.set_xscale("log")
    ax.set_xticks(LIB_SIZES)
    ax.set_xticklabels(["200", "2K", "20K"], fontsize=10)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="gray", lw=0.7, ls=":", alpha=0.5)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.set_title(
        f"Nonlinear Add+Pair surrogate — GT: nonlin add+pair (mRNA_RBP)\n"
        f"seq={seq[:35]}… (L={len(seq)})",
        fontsize=10, pad=8,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA",
        help="RNA sequence (default: project main seq, L=41)",
    )
    parser.add_argument("--seq_seed", type=int, default=0,
                        help="Seed for GT parameter initialization")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Seed for library generation")
    parser.add_argument("--out_dir",  type=str, default="outputs/single_seq_lib_size")
    args = parser.parse_args()

    seq = args.seq.strip().upper().replace("T", "U")
    L   = len(seq)
    print(f"[init] seq={seq}  L={L}")
    os.makedirs(args.out_dir, exist_ok=True)

    # GT init
    print("\n[gt] Initialising mRNA_RBP ground truth …")
    wt_oh, gt_params, nonlin_kwargs = init_gt(seq, args.seq_seed)

    mut_rates_exact = [(pct, _exact_mut_count(pct, L)) for pct in MUT_RATES_PCT]
    print(f"[init] mut_rates_exact={mut_rates_exact}")

    # Build shared eval pool (generated once, kept in memory for all eval libraries)
    print("\n[eval pool] Building shared eval pool …")
    X_pool, y_pool, pool_labels = build_eval_pool(
        wt_oh, gt_params, nonlin_kwargs, mut_rates_exact, seed=args.seed,
    )

    # Type 1 eval libraries (one per lib_size, from shared pool)
    print("\n[eval] Uniformizing Type 1 libraries …")
    type1_libs = {}
    for lib_size in LIB_SIZES:
        target_n = max(1, int(0.2 * lib_size))
        X_t1, y_t1 = uniformize_eval(X_pool, y_pool, target_n, seed=args.seed + lib_size)
        type1_libs[lib_size] = (X_t1, y_t1)
        print(f"  lib_size={lib_size:,}  target_n={target_n}  got={len(y_t1):,}  "
              f"range [{y_t1.min():.3f}, {y_t1.max():.3f}]")

    # Type 2 eval library (shared 20K, from same pool)
    print(f"\n[eval] Uniformizing Type 2 library (target_n={TYPE2_TARGET_N:,}) …")
    X_type2, y_type2 = uniformize_eval(X_pool, y_pool, TYPE2_TARGET_N,
                                        seed=args.seed + 999_999)
    print(f"  Type2: {len(y_type2):,} seqs  range [{y_type2.min():.3f}, {y_type2.max():.3f}]")

    # Free pool — no longer needed
    del X_pool, y_pool, pool_labels

    # Saturated ρ on Type 2
    print("\n[saturated] Computing saturated mutagenesis ρ on Type 2 …")
    rho_sat = compute_saturated_rho(wt_oh, gt_params, nonlin_kwargs, X_type2, y_type2)
    print(f"  rho_saturated = {rho_sat:.4f}")

    # Training loop
    print("\n[train] Starting 9-condition surrogate training …")
    results = {}

    for pct, emc in mut_rates_exact:
        print(f"\n  === mut_rate={pct}%  exact_mut_count={emc} ===")

        # Generate training pool for this mutation rate, then subsample per lib_size
        rng_pool = np.random.default_rng(args.seed + pct)
        X_train_pool = _gen_pool(wt_oh, TRAIN_POOL_SIZE, emc, rng_pool)
        y_train_pool = _score_X(X_train_pool, gt_params, nonlin_kwargs)
        rng_sub = np.random.default_rng(args.seed + pct + 1_000)

        for lib_size in LIB_SIZES:
            print(f"\n  lib_size={lib_size:,}  mut_rate={pct}%")

            idx    = rng_sub.choice(TRAIN_POOL_SIZE, size=lib_size, replace=False)
            X_lib  = X_train_pool[idx]
            y_lib  = y_train_pool[idx].reshape(-1, 1)

            X_t1, y_t1 = type1_libs[lib_size]

            print(f"    Training nonlin add+pair  N={lib_size}  "
                  f"lr={5e-4 * min(1.0, (20_000/lib_size)**0.5):.2e}  "
                  f"type1_n={len(y_t1)}  type2_n={len(y_type2)}")

            rho_rand, rho_t1, rho_t2 = fit_surrogate(
                X_lib, y_lib, X_t1, y_t1, X_type2, y_type2,
            )
            print(f"    rho_random={rho_rand:+.3f}  rho_type1={rho_t1:+.3f}  rho_type2={rho_t2:+.3f}")

            results[(pct, lib_size)] = {
                "rho_random": rho_rand,
                "rho_type1":  rho_t1,
                "rho_type2":  rho_t2,
            }

        del X_train_pool, y_train_pool

    # Plot Option B
    out_path = os.path.join(args.out_dir, "single_seq_lib_size_optionB.png")
    plot_option_b(results, rho_sat, out_path, seq)

    print("\n[done]")


if __name__ == "__main__":
    main()
