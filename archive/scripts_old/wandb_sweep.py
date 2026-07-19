"""wandb_sweep.py – Hyperparameter sweep for surrogate modelling of GT toy models.

No oracle model (ResidualBind or otherwise) is needed — the library is generated
by direct random mutagenesis from the WT sequence and scored entirely by the
analytic GT toy models.

Sweeps over:
  mut_rate_int  – expected number of mutations per sequence
                  (per-position rate = mut_rate_int / seq_length)
  seq_length    – prefix length of --seq to use
  potts_p_edge  – probability a pair (i,j) gets a non-zero Potts coupling
  potts_df      – t-distribution degrees of freedom for coupling magnitudes
  potts_lambda_J – global scale of Potts coupling matrices
  potts_p_rescue – fraction of coupling entries zeroed after sampling
  seed          – RNG seed

Usage
-----
# 1. Create the sweep (once):
    wandb sweep scripts/sweep_config.yaml

# 2. Launch one or more agents:
    wandb agent <entity>/<project>/<sweep_id>

# 3. Single local trial with wandb defaults:
    python scripts/wandb_sweep.py --seq <RNA_SEQ>

Fixed hyperparameters are CLI flags; swept ones come from wandb.config.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from scipy.stats import spearmanr

sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')

import squid
import mavenn

from seq_utils import rna_to_one_hot, onehot_to_seq
from ground_truth import (
    additive_affinity_noWT,
    pairwise_potts_energy,
    apply_global_nonlin,
    init_additive_noWT,
    init_pairwise_potts_optionA,
    compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
    init_sigmoid_nonlin,
)
from stats_utils import spearman_on_testdf
from plotting import plot_y_vs_yhat


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUCS    = ['A', 'C', 'G', 'U']

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

GT_KEYS = ["additive", "additive_pairwise", "nonlin_additive", "nonlin_additive_pairwise"]


# ---------------------------------------------------------------------------
# CLI args – fixed across all sweep trials
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="wandb sweep agent for GT surrogate modelling")
parser.add_argument("--seq",           required=True,
                    help="Base RNA sequence; trials truncate it to seq_length")
parser.add_argument("--lib_size",      type=int, default=5000,
                    help="Training library size per trial")
parser.add_argument("--eval_pool",     type=int, default=200_000,
                    help="Pool size for eval library generation")
parser.add_argument("--eval_target_n", type=int, default=5000,
                    help="Sequences per eval library after uniformisation")
parser.add_argument("--epochs",        type=int, default=200,
                    help="Max surrogate training epochs per trial")
cli_args, _ = parser.parse_known_args()  # swept params injected by wandb; ignore them here


# ---------------------------------------------------------------------------
# Library generation (no oracle model needed)
# ---------------------------------------------------------------------------

def _generate_library(wt_onehot, mut_rate, lib_size, rng):
    """Random mutagenesis from WT — no oracle predictor required.

    At each position independently, mutates to a uniformly random non-WT
    nucleotide with probability `mut_rate`.

    Returns x_mut: (lib_size, L, 4) float32 one-hot.
    """
    L, A       = wt_onehot.shape
    wt_idx     = np.argmax(wt_onehot, axis=1)          # (L,)
    mut_mask   = rng.random(size=(lib_size, L)) < mut_rate
    offsets    = rng.integers(1, A, size=(lib_size, L))  # 1, 2, or 3
    mut_idx    = (wt_idx[None, :].astype(np.int32) + offsets) % A
    nuc_ids    = np.where(mut_mask, mut_idx, wt_idx[None, :]).astype(np.uint8)
    return np.eye(A, dtype=np.float32)[nuc_ids]          # (N, L, 4)


# ---------------------------------------------------------------------------
# Surrogate training helper
# ---------------------------------------------------------------------------

def _run_surrogate(x_lib, y, *, cfg_name, X_eval, y_eval, epochs):
    """Train one MAVENN surrogate config; return (rho_random, rho_curated)."""
    cfg = SURROGATE_CONFIGS[cfg_name]

    surrogate_wrapper = squid.surrogate_zoo.SurrogateMAVENN(
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

    mavenn_model, mave_df, test_df = surrogate_wrapper.train(
        x_lib, y,
        learning_rate=5e-4,
        epochs=epochs,
        batch_size=100,
        early_stopping=True,
        patience=25,
        restore_best_weights=True,
        save_dir=None,
        verbose=0,
    )

    try:
        rho_random = float(spearman_on_testdf(mavenn_model, test_df))
    except Exception:
        rho_random = float('nan')

    rho_curated = float('nan')
    if X_eval is not None and y_eval is not None:
        nucs  = np.array(list("ACGU"))
        X_str = np.array(
            ["".join(nucs[np.argmax(x, axis=1)]) for x in X_eval], dtype=object
        )
        eval_df = pd.DataFrame({"x": X_str, "y": y_eval.ravel()})
        try:
            fig, _, _, rho_curated = plot_y_vs_yhat(mavenn_model, eval_df)
            plt.close(fig)
            rho_curated = float(rho_curated)
        except Exception:
            rho_curated = float('nan')

    return rho_random, rho_curated


# ---------------------------------------------------------------------------
# Eval library builder
# ---------------------------------------------------------------------------

def _make_eval_libraries(
    *, wt_onehot, input_seq_str, gt_params, nonlin_name, nonlin_kwargs,
    n_pool, target_n, seed, mut_rate,
):
    """Generate 4 uniformly-scored GT eval libraries from a pool of n_pool sequences."""
    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]

    L, A     = wt_onehot.shape
    rng_pool = np.random.default_rng(seed + 99_999)

    # Pool sequences via the same mutagenesis as the training library
    nuc_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    wt_nuc_ids = np.array([nuc_to_idx[c] for c in input_seq_str[:L]], dtype=np.uint8)
    mut_mask    = rng_pool.random(size=(n_pool, L)) < mut_rate
    mut_offsets = rng_pool.integers(1, A, size=(n_pool, L), dtype=np.uint8)
    nuc_ids = np.where(
        mut_mask,
        (wt_nuc_ids[None, :].astype(np.uint16) + mut_offsets) % A,
        wt_nuc_ids[None, :],
    ).astype(np.uint8)

    chunk_size    = 50_000
    s_add_all     = np.empty(n_pool, dtype=np.float32)
    s_addpair_all = np.empty(n_pool, dtype=np.float32)

    for start in range(0, n_pool, chunk_size):
        end     = min(start + chunk_size, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa      = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp      = pairwise_potts_energy(X_chunk, edges, J, b=0.0).reshape(-1)
        s_add_all[start:end]     = sa
        s_addpair_all[start:end] = sa + sp

    mean_add     = float(np.mean(s_add_all))
    mean_addpair = float(np.mean(s_addpair_all))
    std_add      = float(np.std(s_add_all))     + 1e-8
    std_addpair  = float(np.std(s_addpair_all)) + 1e-8

    pool_scores = {k: np.empty(n_pool, dtype=np.float32) for k in GT_KEYS}
    for start in range(0, n_pool, chunk_size):
        end = min(start + chunk_size, n_pool)
        sa  = s_add_all[start:end].astype(float)
        sap = s_addpair_all[start:end].astype(float)
        pool_scores["additive"][start:end]          = sa
        pool_scores["additive_pairwise"][start:end] = sap
        if nonlin_name == "sigmoid":
            pool_scores["nonlin_additive"][start:end] = apply_global_nonlin(
                sa / std_add, nonlin_name, nonlin_kwargs).reshape(-1)
            pool_scores["nonlin_additive_pairwise"][start:end] = apply_global_nonlin(
                sap / std_add, nonlin_name, nonlin_kwargs).reshape(-1)
        else:
            pool_scores["nonlin_additive"][start:end] = apply_global_nonlin(
                (sa - mean_add) / std_add, nonlin_name, nonlin_kwargs).reshape(-1)
            pool_scores["nonlin_additive_pairwise"][start:end] = apply_global_nonlin(
                (sap - mean_add) / std_add, nonlin_name, nonlin_kwargs).reshape(-1)

    if nonlin_name == "sigmoid":
        wt_z_add = wt_z_addpair = 0.0
    else:
        wt_z_add     = (0.0 - mean_add)     / std_add
        wt_z_addpair = (0.0 - mean_addpair) / std_addpair

    wt_nl_add     = float(apply_global_nonlin(np.array([[wt_z_add]]),     nonlin_name, nonlin_kwargs))
    wt_nl_addpair = float(apply_global_nonlin(np.array([[wt_z_addpair]]), nonlin_name, nonlin_kwargs))
    pool_scores["nonlin_additive"]          -= wt_nl_add
    pool_scores["nonlin_additive_pairwise"] -= wt_nl_addpair
    del s_add_all, s_addpair_all

    eval_libs = {}
    for k in GT_KEYS:
        y_uni, keep  = uniformize_by_histogram(
            pool_scores[k].astype(float), X=None,
            n_bins=100, clip_hi=98, target_n=target_n,
            seed=seed + hash(k) % 10_000,
        )
        eval_libs[k] = {
            "X_eval": np.eye(A, dtype=np.float32)[nuc_ids[keep]],
            "y_eval": y_uni.astype(float),
        }

    return eval_libs


# ---------------------------------------------------------------------------
# Main sweep trial
# ---------------------------------------------------------------------------

def run_trial(config=None):
    with wandb.init(config=config) as run:
        cfg = wandb.config

        # ── Swept hyperparameters ─────────────────────────────────────────
        mut_rate_int   = int(cfg.mut_rate_int)
        seed           = int(getattr(cfg, "seed", 0))
        potts_p_edge   = float(cfg.potts_p_edge)
        potts_df       = float(cfg.potts_df)
        potts_lambda_J = float(cfg.potts_lambda_J)
        potts_p_rescue = float(cfg.potts_p_rescue)

        # ── Sequence setup ────────────────────────────────────────────────
        seq        = cli_args.seq[:int(cfg.seq_length)]
        seq_length = len(seq)

        # mut_rate as fraction of seq_length (expected mutations / seq_length)
        mut_rate_int = min(mut_rate_int, seq_length)
        mut_rate     = mut_rate_int / seq_length

        wandb.config.update({
            "seq": seq,
            "seq_length_actual": seq_length,
            "mut_rate": mut_rate,
            "mut_rate_int": mut_rate_int,
        }, allow_val_change=True)

        print(f"\n[trial] len={seq_length}  mut_rate_int={mut_rate_int}  "
              f"p_edge={potts_p_edge:.2f}  df={potts_df}  "
              f"lambda_J={potts_lambda_J:.2f}  p_rescue={potts_p_rescue:.2f}  seed={seed}")

        rng    = np.random.default_rng(seed)
        wt_oh  = rna_to_one_hot(seq)    # (L, 4)

        # ── Generate library directly — no oracle needed ──────────────────
        x_mut = _generate_library(wt_oh, mut_rate, cli_args.lib_size, rng)
        print(f"[trial] library: {x_mut.shape}")

        # ── Initialise GT parameters ──────────────────────────────────────
        W_mut, mut_map, b0 = init_additive_noWT(rng, wt_oh, sigma=0.5, l1_w=0.1)
        edges, J = init_pairwise_potts_optionA(
            rng, wt_oh,
            p_edge=potts_p_edge,
            df=potts_df,
            lambda_J=potts_lambda_J,
            p_rescue=potts_p_rescue,
            wt_rowcol_zero=True,
        )

        n_possible = seq_length * (seq_length - 1) // 2
        J_flat     = J[J != 0].ravel()
        wandb.log({
            "potts/n_edges":      len(edges),
            "potts/edge_density": len(edges) / max(n_possible, 1),
            "potts/J_mean_abs":   float(np.mean(np.abs(J_flat))) if J_flat.size else 0.0,
            "potts/J_std":        float(np.std(J_flat))          if J_flat.size else 0.0,
        })

        # ── GT scoring ────────────────────────────────────────────────────
        s_add_raw     = additive_affinity_noWT(x_mut, W_mut, mut_map, b0)
        nonlin_kwargs = init_sigmoid_nonlin(s_add_raw)
        nonlin_name   = "sigmoid"
        gt_params     = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges, "J": J}

        gt_scores = compute_gt_scores_for_library_potts(
            x_mut, W_mut=W_mut, mut_map=mut_map, b0=b0,
            nonlin_name=nonlin_name, nonlin_kwargs=nonlin_kwargs,
            edges=edges, J=J,
        )

        for k, arr in gt_scores.items():
            finite = np.asarray(arr, dtype=float)
            finite = finite[np.isfinite(finite)]
            wandb.log({
                f"gt_scores/{k}/std":  float(np.std(finite))           if finite.size else float('nan'),
                f"gt_scores/{k}/min":  float(np.min(finite))           if finite.size else float('nan'),
                f"gt_scores/{k}/pct5": float(np.percentile(finite, 5)) if finite.size else float('nan'),
            })

        # ── Eval libraries ────────────────────────────────────────────────
        eval_libs = _make_eval_libraries(
            wt_onehot=wt_oh,
            input_seq_str=seq,
            gt_params=gt_params,
            nonlin_name=nonlin_name,
            nonlin_kwargs=nonlin_kwargs,
            n_pool=cli_args.eval_pool,
            target_n=cli_args.eval_target_n,
            seed=seed,
            mut_rate=mut_rate,
        )
        print("[trial] eval libs: "
              + ", ".join(f"{k}={v['X_eval'].shape[0]}" for k, v in eval_libs.items()))

        # ── Train surrogates ──────────────────────────────────────────────
        all_rho_curated, all_rho_random = [], []

        for gt_key in GT_KEYS:
            y_true = np.asarray(gt_scores[gt_key], dtype=float).reshape(-1, 1)
            finite = np.isfinite(y_true[:, 0])
            x_use, y_use = x_mut[finite], y_true[finite]

            if y_use.shape[0] < 50 or np.nanstd(y_use) < 1e-8:
                print(f"[trial] skipping {gt_key}")
                for cfg_name in SURROGATE_CONFIGS:
                    wandb.log({
                        f"rho_curated/{cfg_name}/{gt_key}": float('nan'),
                        f"rho_random/{cfg_name}/{gt_key}":  float('nan'),
                    })
                continue

            for cfg_name in SURROGATE_CONFIGS:
                print(f"[trial] {cfg_name} × {gt_key}")
                rho_r, rho_c = _run_surrogate(
                    x_use, y_use,
                    cfg_name=cfg_name,
                    X_eval=eval_libs[gt_key]["X_eval"],
                    y_eval=eval_libs[gt_key]["y_eval"],
                    epochs=cli_args.epochs,
                )
                wandb.log({
                    f"rho_curated/{cfg_name}/{gt_key}": rho_c,
                    f"rho_random/{cfg_name}/{gt_key}":  rho_r,
                })
                if np.isfinite(rho_c): all_rho_curated.append(rho_c)
                if np.isfinite(rho_r): all_rho_random.append(rho_r)

        rho_curated_mean = float(np.mean(all_rho_curated)) if all_rho_curated else float('nan')
        rho_random_mean  = float(np.mean(all_rho_random))  if all_rho_random  else float('nan')
        wandb.log({"rho_curated_mean": rho_curated_mean, "rho_random_mean": rho_random_mean})
        print(f"[trial] rho_curated_mean={rho_curated_mean:.4f}  rho_random_mean={rho_random_mean:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_trial()
