"""spearman_violin.py

Generate side-by-side Spearman ρ violin plots with error bars.

For each of N randomly generated RNA sequences of the same length as the
main sequence, the full pipeline is run:
  • Random GT parameters (additive weights + Potts pairwise interactions)
  • 20k random library (mutating from that sequence)
  • Fresh eval library (uniformized from a 50k pool)
  • All 4 × 4 = 16 surrogate models trained; Spearman ρ recorded

Each violin shows the distribution of ρ across the N sequences.
Error bars (mean ± SEM) are overlaid on every violin body.

Layout — 2×2 grid (left col = additive GT, right col = nonlinear GT):
  [0,0] Additive GT          [0,1] Nonlin Additive GT
  [1,0] Additive+Pairwise GT [1,1] Nonlin Add+Pairwise GT

Each subplot: 8 violins (4 surrogate configs × 2 library types)
  blue   = random MAVE test split
  orange = curated eval library

Usage (include the main sequence plus 9 random ones):
    python scripts/spearman_violin.py \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --experiment RNCMPT00111 \\
        --n_seqs 10 \\
        --include_main_seq \\
        --out_path outputs/postabrcms/vts1high/spearman_violin_run0.png \\
        --seed 0

Usage (10 fully random sequences):
    python scripts/spearman_violin.py \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --n_seqs 10 \\
        --out_path outputs/postabrcms/vts1high/spearman_violin_run0.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.insert(0, '/home/nagle/final_version/squid-manuscript/squid')
sys.path.insert(0, '/home/nagle/final_version/squid-nn/squid')
sys.path.insert(0, '/home/nagle/final_version/residualbind')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]

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

GT_LABELS = {
    "additive":                 "Additive GT",
    "additive_pairwise":        "Additive+Pair GT",
    "nonlin_additive":          "Nonlin Add GT",
    "nonlin_additive_pairwise": "Nonlin Add+Pair GT",
}

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Add GE",
    "pairwise_GE": "Pair GE",
}

NUCS      = ['A', 'C', 'G', 'U']
_NUCS_ARR = np.array(list("ACGU"))

# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_ARR[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _random_rna_sequences(n, L, rng):
    """Generate n uniform-random RNA sequences of length L as one-hot arrays."""
    nuc_ids = rng.integers(0, 4, size=(n, L), dtype=np.uint8)
    return np.eye(4, dtype=np.float32)[nuc_ids]   # (n, L, 4)


def _onehot_from_str(seq):
    """Convert a single RNA string to one-hot (L, 4)."""
    idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    L   = len(seq)
    X   = np.zeros((L, 4), dtype=np.float32)
    for j, c in enumerate(seq):
        X[j, idx.get(c, 0)] = 1.0
    return X


def _mutate_from_wt(wt_oh, n_seqs, mut_rate, rng):
    """Generate n_seqs sequences by point-mutating wt_oh at per-position rate."""
    L, A = wt_oh.shape
    wt_ids  = np.argmax(wt_oh, axis=1)
    mask    = rng.random(size=(n_seqs, L)) < mut_rate
    offsets = rng.integers(1, 4, size=(n_seqs, L), dtype=np.uint8)
    nuc_ids = np.where(mask, (wt_ids[None, :] + offsets) % A, wt_ids[None, :]).astype(np.uint8)
    return np.eye(A, dtype=np.float32)[nuc_ids]

# ---------------------------------------------------------------------------
# GT initialisation and scoring
# ---------------------------------------------------------------------------

def _init_gt(rng, wt_oh):
    from ground_truth import (
        init_additive_noWT, init_pairwise_potts_optionA,
        init_sigmoid_nonlin, additive_affinity_noWT,
    )
    print("      [GT] initialising additive weights…", flush=True)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_oh, sigma=0.5, l1_w=0.1, bias=0.0)
    print("      [GT] initialising Potts pairwise interactions…", flush=True)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_oh, p_edge=0.70, df=5.0, lambda_J=2.0,
        p_rescue=0.10, wt_rowcol_zero=True,
    )
    # Calibrate sigmoid nonlin on a reference batch for this sequence
    L, A = wt_oh.shape
    print("      [GT] calibrating sigmoid nonlinearity (50k ref batch)…", flush=True)
    rng_ref = np.random.default_rng(int(rng.integers(0, 2**31)))
    nuc_ids_ref = rng_ref.integers(0, A, size=(50_000, L), dtype=np.uint8)
    X_ref = np.eye(A, dtype=np.float32)[nuc_ids_ref]
    s_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kw = init_sigmoid_nonlin(s_ref)
    del X_ref, s_ref
    print("      [GT] done", flush=True)
    return W_mut, mut_map, b0, edges, J, nonlin_kw


def _raw_energies(X, W_mut, mut_map, b0, edges, J):
    """Return (s_add, s_pair) raw energy arrays before any nonlinearity."""
    from ground_truth import additive_affinity_noWT, pairwise_potts_energy
    s_add  = additive_affinity_noWT(X, W_mut, mut_map, b=b0).reshape(-1)
    s_pair = pairwise_potts_energy(X, edges, J, b=0.0).reshape(-1)
    return s_add, s_pair


def _score_library(X, W_mut, mut_map, b0, edges, J, nonlin_kwargs):
    from ground_truth import compute_gt_scores_for_library_potts
    return compute_gt_scores_for_library_potts(
        X, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )


def _build_eval_library(wt_oh, mut_rate, W_mut, mut_map, b0, edges, J, nonlin_kw,
                         n_pool, target_n, seed):
    """Generate an eval library (uniformized GT scores) for one sequence.

    Returns (eval_libs, X_pool, s_add_pool, s_pair_pool) so the caller can
    save the raw pool energies for future GT types.
    """
    from ground_truth import uniformize_by_histogram
    L, A = wt_oh.shape
    print(f"      [eval lib] generating {n_pool:,}-seq pool…", flush=True)
    rng = np.random.default_rng(seed)
    X_pool = _mutate_from_wt(wt_oh, n_pool, mut_rate, rng)
    print(f"      [eval lib] computing raw energies for pool…", flush=True)
    s_add_pool, s_pair_pool = _raw_energies(X_pool, W_mut, mut_map, b0, edges, J)
    print(f"      [eval lib] scoring pool with all 4 GT functions…", flush=True)
    gt_pool = _score_library(X_pool, W_mut, mut_map, b0, edges, J, nonlin_kw)
    eval_libs = {}
    for gt_key in GT_KEYS:
        y_pool = np.asarray(gt_pool[gt_key], dtype=float).ravel()
        y_uni, keep_idx = uniformize_by_histogram(
            y_pool, X=None, n_bins=200, clip_hi=98,
            target_n=target_n, seed=seed + abs(hash(gt_key)) % 10_000,
        )
        eval_libs[gt_key] = (X_pool[keep_idx], y_uni.astype(float))
        print(f"      [eval lib] {gt_key}: {len(keep_idx):,} seqs selected", flush=True)
    return eval_libs, X_pool, s_add_pool, s_pair_pool


# ---------------------------------------------------------------------------
# Save / load per-sequence data
# ---------------------------------------------------------------------------

def _save_seq_data(seq_dir, seq_oh, X_rand, s_add_rand, s_pair_rand,
                   gt_scores_rand, X_pool, s_add_pool, s_pair_pool,
                   gt_scores_pool, nonlin_kw,
                   W_mut=None, mut_map=None, b0=None, edges=None, J=None):
    """Save all data needed to score future GT types without rerunning.

    Saved files
    -----------
    sequence_onehot.npy     (L, 4) float32 — the sequence itself
    X_rand_nucids.npy       (N, L) uint8   — random lib as nucleotide indices
    s_add_rand.npy          (N,)   float32 — additive energies, random lib
    s_pair_rand.npy         (N,)   float32 — pairwise energies, random lib
    gt_scores_rand.npz      dict of (N,) float32 — GT scores, random lib
    X_pool_nucids.npy       (P, L) uint8   — eval pool as nucleotide indices
    s_add_pool.npy          (P,)   float32 — additive energies, eval pool
    s_pair_pool.npy         (P,)   float32 — pairwise energies, eval pool
    gt_scores_pool.npz      dict of (P,) float32 — GT scores, eval pool
    nonlin_kwargs.json      sigmoid calibration scalars
    gt_params/W_mut.npy     (L, 3) float32 — additive weights
    gt_params/mut_map.npy   (L, 3) int     — non-WT nucleotide index map
    gt_params/edges.npy     (M, 2) int     — Potts edge pairs
    gt_params/J.npy         (L,L,4,4) float32 — Potts coupling tensor
    gt_params/params.json   b0 scalar + per-param summary stats for quick comparison

    To add a new GT for this sequence:
      1. Load X_rand_nucids → one-hot → train surrogate on (X_rand, new_y_rand)
      2. Compute new_y_rand = new_gt(s_add_rand, s_pair_rand)
      3. Compute new_y_pool = new_gt(s_add_pool, s_pair_pool)
      4. Uniformize new_y_pool → new eval library
      5. Predict on eval lib → compute Spearman ρ
    """
    import json
    os.makedirs(seq_dir, exist_ok=True)

    np.save(os.path.join(seq_dir, "sequence_onehot.npy"), seq_oh)
    np.save(os.path.join(seq_dir, "X_rand_nucids.npy"),
            np.argmax(X_rand, axis=-1).astype(np.uint8))
    np.save(os.path.join(seq_dir, "s_add_rand.npy"),  s_add_rand.astype(np.float32))
    np.save(os.path.join(seq_dir, "s_pair_rand.npy"), s_pair_rand.astype(np.float32))
    np.savez_compressed(os.path.join(seq_dir, "gt_scores_rand.npz"),
                        **{k: np.asarray(v, dtype=np.float32)
                           for k, v in gt_scores_rand.items()})

    np.save(os.path.join(seq_dir, "X_pool_nucids.npy"),
            np.argmax(X_pool, axis=-1).astype(np.uint8))
    np.save(os.path.join(seq_dir, "s_add_pool.npy"),  s_add_pool.astype(np.float32))
    np.save(os.path.join(seq_dir, "s_pair_pool.npy"), s_pair_pool.astype(np.float32))
    np.savez_compressed(os.path.join(seq_dir, "gt_scores_pool.npz"),
                        **{k: np.asarray(v, dtype=np.float32)
                           for k, v in gt_scores_pool.items()})

    # Serialize nonlin_kwargs (all values should be scalars or small arrays)
    nkw_serial = {}
    for k, v in nonlin_kw.items():
        if isinstance(v, np.ndarray):
            nkw_serial[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            nkw_serial[k] = float(v)
        else:
            nkw_serial[k] = v
    with open(os.path.join(seq_dir, "nonlin_kwargs.json"), "w") as f:
        json.dump(nkw_serial, f, indent=2)

    # Save raw GT parameters so sequences can be verified as independent
    if W_mut is not None:
        params_dir = os.path.join(seq_dir, "gt_params")
        os.makedirs(params_dir, exist_ok=True)
        np.save(os.path.join(params_dir, "W_mut.npy"),  W_mut.astype(np.float32))
        np.save(os.path.join(params_dir, "mut_map.npy"), mut_map.astype(np.int32))
        if edges is not None:
            np.save(os.path.join(params_dir, "edges.npy"), edges.astype(np.int32))
        if J is not None:
            np.save(os.path.join(params_dir, "J.npy"), J.astype(np.float32))

        # Human-readable summary for quick cross-sequence comparison
        summary = {
            "b0": float(b0) if b0 is not None else None,
            "W_mut_mean":  float(np.mean(W_mut)),
            "W_mut_std":   float(np.std(W_mut)),
            "W_mut_first5": W_mut.ravel()[:5].tolist(),
            "n_edges": int(len(edges)) if edges is not None else None,
            "J_mean":  float(np.mean(J))  if J is not None else None,
            "J_std":   float(np.std(J))   if J is not None else None,
            "J_first5": J[J != 0].ravel()[:5].tolist() if J is not None else None,
            "nonlin_norm_std": float(nonlin_kw.get("_norm_std", float("nan"))),
        }
        with open(os.path.join(params_dir, "params.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print(f"      [save] data written to {seq_dir}", flush=True)


def _save_rho_checkpoint(save_dir, accum):
    """Write accumulated ρ values to rho_results.json (called after each sequence)."""
    import json

    def _ser(obj):
        if isinstance(obj, dict):   return {k: _ser(v) for k, v in obj.items()}
        if isinstance(obj, list):   return [_ser(x) for x in obj]
        if isinstance(obj, float):  return None if (np.isnan(obj) or np.isinf(obj)) else obj
        if isinstance(obj, (np.floating,)):
            v = float(obj); return None if (np.isnan(v) or np.isinf(v)) else v
        return obj

    out = os.path.join(save_dir, "rho_results.json")
    with open(out, "w") as f:
        json.dump(_ser(accum), f, indent=2)
    print(f"  [checkpoint] rho_results saved → {out}", flush=True)

# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def _train_surrogate(X_rand, y_rand, cfg_name, cfg):
    import squid.surrogate_zoo
    N   = X_rand.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nl_pair = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs   = 1000 if is_nl_pair else 500
    patience = 50   if is_nl_pair else 25
    print(f"        → MAVENN {cfg['gpmap']}/{cfg['linearity']}  "
          f"N={N}  bsz={bsz}  lr={lr:.1e}  max_epochs={epochs}", flush=True)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X_rand.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X_rand, y_rand.reshape(-1, 1),
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    print(f"        → training done", flush=True)
    return model, test_df

# ---------------------------------------------------------------------------
# Per-sequence experiment
# ---------------------------------------------------------------------------

def _run_one_sequence(seq_oh, mut_rate, gt_seed, lib_size, n_pool, target_n,
                      seq_label="", save_dir=None):
    """Run the full GT+surrogate pipeline for one sequence.

    Returns {gt_key: {cfg_name: (rho_rand, rho_eval)}}
    If save_dir is set, raw energies + sequences are saved there so future
    GT types can be scored without re-running this function.
    """
    from scipy.stats import spearmanr

    rng_gt = np.random.default_rng(gt_seed)
    print(f"  [{seq_label}] initialising GT parameters…", flush=True)
    W_mut, mut_map, b0, edges, J, nonlin_kw = _init_gt(rng_gt, seq_oh)

    # Random training library
    print(f"  [{seq_label}] generating {lib_size:,}-seq random library…", flush=True)
    rng_lib = np.random.default_rng(gt_seed + 10_000)
    X_rand = _mutate_from_wt(seq_oh, lib_size, mut_rate, rng_lib)
    print(f"  [{seq_label}] computing raw energies for random library…", flush=True)
    s_add_rand, s_pair_rand = _raw_energies(X_rand, W_mut, mut_map, b0, edges, J)
    print(f"  [{seq_label}] scoring random library with all 4 GT functions…", flush=True)
    gt_rand = _score_library(X_rand, W_mut, mut_map, b0, edges, J, nonlin_kw)

    # Eval library (generated fresh for this sequence)
    print(f"  [{seq_label}] building eval library (pool={n_pool:,}, target={target_n:,})…", flush=True)
    eval_libs, X_pool, s_add_pool, s_pair_pool = _build_eval_library(
        seq_oh, mut_rate, W_mut, mut_map, b0, edges, J, nonlin_kw,
        n_pool=n_pool, target_n=target_n, seed=gt_seed + 20_000,
    )
    gt_pool = _score_library(X_pool, W_mut, mut_map, b0, edges, J, nonlin_kw)

    n_gt   = len(GT_KEYS)
    n_cfg  = len(SURROGATE_CONFIGS)
    total_models = n_gt * n_cfg
    model_count  = 0

    results = {}
    for gt_key in GT_KEYS:
        y_rand = np.asarray(gt_rand[gt_key], dtype=float).ravel()
        m = np.isfinite(y_rand)
        X_use, y_use = X_rand[m], y_rand[m]

        if y_use.size < 50 or np.nanstd(y_use) == 0:
            print(f"  [{seq_label}] SKIP {gt_key}: insufficient variance", flush=True)
            results.setdefault(gt_key, {})
            for cfg_name in SURROGATE_CONFIGS:
                results[gt_key][cfg_name] = (np.nan, np.nan)
            model_count += n_cfg
            continue

        X_eval, y_eval = eval_libs[gt_key]
        X_eval_str = _to_str(X_eval)
        results.setdefault(gt_key, {})

        for cfg_name, cfg in SURROGATE_CONFIGS.items():
            model_count += 1
            print(f"  [{seq_label}] model {model_count}/{total_models}  "
                  f"GT={gt_key}  surrogate={cfg_name}", flush=True)
            model, test_df = _train_surrogate(X_use, y_use, cfg_name, cfg)

            X_test_str = np.asarray(test_df["x"])
            y_test     = np.asarray(test_df["y"], dtype=float).ravel()

            def _rho(yhat, ytrue):
                m2 = np.isfinite(yhat) & np.isfinite(ytrue)
                return float(spearmanr(yhat[m2], ytrue[m2])[0]) if m2.sum() >= 3 else np.nan

            try:
                rho_r = _rho(np.asarray(model.x_to_yhat(X_test_str), dtype=float).ravel(), y_test)
            except Exception:
                rho_r = np.nan
            try:
                rho_e = _rho(np.asarray(model.x_to_yhat(X_eval_str), dtype=float).ravel(), y_eval)
            except Exception:
                rho_e = np.nan

            results[gt_key][cfg_name] = (rho_r, rho_e)
            print(f"        ρ_rand={rho_r:.4f}  ρ_eval={rho_e:.4f}", flush=True)

    print(f"  [{seq_label}] all {total_models} models done", flush=True)

    # Save raw data so future GT types can be added without re-running
    if save_dir is not None:
        print(f"  [{seq_label}] saving raw data…", flush=True)
        _save_seq_data(
            save_dir, seq_oh,
            X_rand, s_add_rand, s_pair_rand, gt_rand,
            X_pool, s_add_pool, s_pair_pool, gt_pool,
            nonlin_kw,
            W_mut=W_mut, mut_map=mut_map, b0=b0, edges=edges, J=J,
        )

    return results

# ---------------------------------------------------------------------------
# Collect results across N sequences
# ---------------------------------------------------------------------------

def run_all_sequences(sequences, mut_rate, lib_size, n_pool, target_n, base_seed,
                      save_dir=None):
    """
    sequences : list of (L,4) one-hot arrays
    Returns {gt_key: {cfg_name: {"rand": [rho,...], "eval": [rho,...]}}}
    """
    accum = {
        gt_key: {
            cfg_name: {"rand": [], "eval": []}
            for cfg_name in SURROGATE_CONFIGS
        }
        for gt_key in GT_KEYS
    }

    n_total = len(sequences)
    for s_idx, seq_oh in enumerate(sequences):
        seq_str = "".join(_NUCS_ARR[np.argmax(seq_oh, axis=1)])
        label   = f"seq {s_idx+1}/{n_total}"
        gt_seed = base_seed + s_idx * 100_000
        print(f"\n{'='*60}", flush=True)
        print(f"  {label}  seed={gt_seed}", flush=True)
        print(f"  sequence: {seq_str}", flush=True)
        print(f"{'='*60}", flush=True)

        seq_save_dir = (os.path.join(save_dir, f"seq_{s_idx:03d}") if save_dir else None)
        res = _run_one_sequence(
            seq_oh, mut_rate, gt_seed, lib_size, n_pool, target_n, label,
            save_dir=seq_save_dir,
        )
        for gt_key in GT_KEYS:
            for cfg_name in SURROGATE_CONFIGS:
                rho_r, rho_e = res.get(gt_key, {}).get(cfg_name, (np.nan, np.nan))
                accum[gt_key][cfg_name]["rand"].append(rho_r)
                accum[gt_key][cfg_name]["eval"].append(rho_e)

        print(f"\n  --- {label} summary ---", flush=True)
        for gt_key in GT_KEYS:
            vals = [(accum[gt_key][c]["rand"][-1], accum[gt_key][c]["eval"][-1])
                    for c in SURROGATE_CONFIGS]
            row = "  ".join(f"{c}  r={r:.3f}/e={e:.3f}"
                            for c, (r, e) in zip(SURROGATE_CONFIGS, vals))
            print(f"    {gt_key}: {row}", flush=True)
        print(f"  {s_idx+1}/{n_total} sequences complete\n", flush=True)

        # Checkpoint ρ results after every sequence so progress isn't lost
        if save_dir is not None:
            _save_rho_checkpoint(save_dir, accum)

    print(f"\n{'='*60}", flush=True)
    print(f"  ALL {n_total} SEQUENCES COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    return accum

# ---------------------------------------------------------------------------
# Violin plot with error bars
# ---------------------------------------------------------------------------

def plot_violin(accum, out_path, experiment="", n_seqs=10):
    """
    accum : {gt_key: {cfg_name: {"rand": [rho,...], "eval": [rho,...]}}}
    Each violin draws all data points plus a mean ± SEM error bar.
    """
    # Layout: left col = additive types, right col = nonlinear types
    #   [0,0]=GT_KEYS[0]  [0,1]=GT_KEYS[2]
    #   [1,0]=GT_KEYS[1]  [1,1]=GT_KEYS[3]
    subplot_order = [GT_KEYS[0], GT_KEYS[2], GT_KEYS[1], GT_KEYS[3]]
    cfg_names     = list(SURROGATE_CONFIGS.keys())

    BLUE   = "#4C72B0"
    ORANGE = "#DD8452"
    VIOLIN_W = 0.42

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for sp_idx, gt_key in enumerate(subplot_order):
        ax = axes.flatten()[sp_idx]
        positions  = []
        data_lists = []
        colors     = []
        xtick_pos  = []
        xtick_labs = []

        x_pos = 0.0
        for cfg_name in cfg_names:
            rho_rnd = np.asarray(accum[gt_key][cfg_name]["rand"], dtype=float)
            rho_cur = np.asarray(accum[gt_key][cfg_name]["eval"], dtype=float)
            rho_rnd = rho_rnd[np.isfinite(rho_rnd)]
            rho_cur = rho_cur[np.isfinite(rho_cur)]

            positions.extend([x_pos, x_pos + 0.55])
            data_lists.extend([rho_rnd, rho_cur])
            colors.extend([BLUE, ORANGE])
            xtick_pos.append(x_pos + 0.275)
            xtick_labs.append(CFG_LABELS.get(cfg_name, cfg_name))
            x_pos += 1.4

        # ── Draw violins ────────────────────────────────────────────────────
        for pos, data, color in zip(positions, data_lists, colors):
            if data.size == 0:
                continue
            if data.size < 3:
                # Too few points for KDE — scatter only
                ax.scatter(np.full(data.size, pos), data,
                           color=color, s=40, zorder=6, alpha=0.85)
            else:
                parts = ax.violinplot(
                    data, positions=[pos], widths=VIOLIN_W,
                    showmedians=False, showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.65)

            # Individual data points as jittered scatter
            rng_j = np.random.default_rng(42)
            jitter = rng_j.uniform(-0.08, 0.08, size=data.size)
            ax.scatter(np.full(data.size, pos) + jitter, data,
                       color=color, s=25, zorder=7, alpha=0.8, linewidths=0)

            # ── Error bar: mean ± SEM ────────────────────────────────────
            if data.size >= 2:
                mean = np.nanmean(data)
                sem  = np.nanstd(data, ddof=1) / np.sqrt(data.size)
                ax.errorbar(
                    [pos], [mean], yerr=[[sem], [sem]],
                    fmt="o", color="black",
                    markersize=5, linewidth=2.0,
                    capsize=5, capthick=2.0, zorder=10,
                )
            elif data.size == 1:
                ax.scatter([pos], data, color="black", s=40, zorder=10)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labs, fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_title(GT_LABELS.get(gt_key, gt_key), fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if xtick_pos:
            ax.set_xlim(-0.5, x_pos - 0.15)

    fig.legend(
        handles=[
            Patch(facecolor=BLUE,   alpha=0.65, label="Random (MAVE test split)"),
            Patch(facecolor=ORANGE, alpha=0.65, label="Eval (curated library)"),
            plt.Line2D([0], [0], color="black", marker="o", linewidth=2,
                       markersize=5, label="Mean ± SEM"),
        ],
        loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle(
        f"{experiment} — Spearman ρ  (N={n_seqs} random sequences, L=41)\n"
        "Left column: additive GT types   |   Right column: nonlinear GT types",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] violin plot → {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spearman ρ violin plots with error bars from N random sequences."
    )
    parser.add_argument("--seq",
                        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA",
                        help="Reference sequence (used for length; also included if "
                             "--include_main_seq is set)")
    parser.add_argument("--experiment",     default="RNCMPT00111")
    parser.add_argument("--n_seqs",         type=int, default=10,
                        help="Number of random sequences to run (default 10)")
    parser.add_argument("--include_main_seq", action="store_true",
                        help="Include the reference --seq as the first sequence "
                             "(counts toward --n_seqs)")
    parser.add_argument("--lib_size",       type=int, default=20_000,
                        help="Random library size per sequence (default 20000)")
    parser.add_argument("--eval_pool_size", type=int, default=200_000,
                        help="Pool size for eval library generation (default 200000)")
    parser.add_argument("--eval_target_n",  type=int, default=20_000,
                        help="Uniformized eval library size (default 20000)")
    parser.add_argument("--mut_rate",       type=float, default=None,
                        help="Per-position mutation rate (default 4/L)")
    parser.add_argument("--seed",           type=int, default=0,
                        help="Base random seed")
    parser.add_argument("--save_dir",       default=None,
                        help="Directory to save raw sequences, energies, GT scores, and ρ "
                             "results so future GT types can be added without re-running. "
                             "Each sequence gets a sub-folder seq_000/, seq_001/, … "
                             "and rho_results.json is checkpointed after every sequence.")
    parser.add_argument("--out_path",
                        default="outputs/postabrcms/vts1high/spearman_violin_run0.png")
    args = parser.parse_args()

    from seq_utils import rna_to_one_hot, remove_padding

    print(f"\n{'#'*60}", flush=True)
    print(f"  spearman_violin.py  —  starting up", flush=True)
    print(f"{'#'*60}\n", flush=True)

    seq   = args.seq.upper().replace("T", "U")
    oh    = rna_to_one_hot(seq)
    wt_oh, _ = remove_padding(oh)
    L     = wt_oh.shape[0]
    mut_rate = args.mut_rate if args.mut_rate is not None else 4.0 / L

    print(f"  Reference seq : {seq}", flush=True)
    print(f"  Length        : {L}", flush=True)
    print(f"  mut_rate      : {mut_rate:.4f}  ({mut_rate*L:.1f} muts/seq on average)", flush=True)
    print(f"  n_seqs        : {args.n_seqs}", flush=True)
    print(f"  include_main  : {args.include_main_seq}", flush=True)
    print(f"  lib_size      : {args.lib_size:,}", flush=True)
    print(f"  eval_pool     : {args.eval_pool_size:,}  →  target {args.eval_target_n:,}", flush=True)
    print(f"  base_seed     : {args.seed}", flush=True)
    print(f"  save_dir      : {args.save_dir or '(not saving)'}", flush=True)
    print(f"  out_path      : {args.out_path}", flush=True)
    total_models = args.n_seqs * len(GT_KEYS) * len(SURROGATE_CONFIGS)
    print(f"\n  Total models to train: {total_models}  "
          f"({args.n_seqs} seqs × {len(GT_KEYS)} GT types × {len(SURROGATE_CONFIGS)} surrogates)",
          flush=True)
    print(flush=True)

    # ── Build sequence list ──────────────────────────────────────────────────
    rng_seq = np.random.default_rng(args.seed + 1_000_000)
    sequences = []

    if args.include_main_seq:
        sequences.append(wt_oh)
        n_random = args.n_seqs - 1
    else:
        n_random = args.n_seqs

    if n_random > 0:
        rand_seqs = _random_rna_sequences(n_random, L, rng_seq)  # (n_random, L, 4)
        for i in range(n_random):
            sequences.append(rand_seqs[i])

    print(f"\nSequence list ({len(sequences)} total):")
    for i, s_oh in enumerate(sequences):
        s_str = "".join(_NUCS_ARR[np.argmax(s_oh, axis=1)])
        print(f"  [{i+1}] {s_str}")

    # ── Run experiments ──────────────────────────────────────────────────────
    accum = run_all_sequences(
        sequences, mut_rate,
        lib_size=args.lib_size,
        n_pool=args.eval_pool_size,
        target_n=args.eval_target_n,
        base_seed=args.seed,
        save_dir=args.save_dir,
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_violin(accum, args.out_path, experiment=args.experiment, n_seqs=args.n_seqs)


if __name__ == "__main__":
    main()
