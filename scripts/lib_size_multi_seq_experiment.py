"""lib_size_multi_seq_experiment.py

Runs the library-size Spearman sweep across multiple RNA sequences of
varying length and produces plots with mean ± std error bands across
sequences.

Parallelism: each sequence runs as an independent subprocess pinned to one
GPU via CUDA_VISIBLE_DEVICES (set before TF initializes).  A thread pool
recycles GPUs as sequences finish, so all 8 GPUs stay busy.

Usage:
    # Provide sequences in a file (one RNA sequence per line):
    python scripts/lib_size_multi_seq_experiment.py \\
        --seqs_file my_sequences.txt \\
        --mut_rate 4 --seed 0 --num_gpus 8 \\
        --out_dir outputs/lib_size_multi_seq

    # Auto-generate N random sequences spanning lengths min_len..max_len:
    python scripts/lib_size_multi_seq_experiment.py \\
        --num_seqs 10 --min_len 10 --max_len 80 \\
        --mut_rate 4 --seed 0 --num_gpus 8 \\
        --out_dir outputs/lib_size_multi_seq

    # Limit max library size for faster iteration:
    python scripts/lib_size_multi_seq_experiment.py \\
        --seqs_file seqs.txt --max_lib_size 20000 --num_gpus 8 \\
        --out_dir outputs/lib_size_multi_seq

    # Skip Gaussian negative-control GT keys (4 GT keys instead of 6):
    python scripts/lib_size_multi_seq_experiment.py \\
        --seqs_file seqs.txt --no_gaussian --num_gpus 8 \\
        --out_dir outputs/lib_size_multi_seq
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

# ---------------------------------------------------------------------------
# Worker-mode detection
# TF and SQUID are imported only in worker subprocesses, where
# CUDA_VISIBLE_DEVICES has already been set by the parent via the environment.
# The orchestrator never imports TF, so it never allocates GPU memory.
# ---------------------------------------------------------------------------
_IS_WORKER = "--_worker_seq_idx" in sys.argv

if _IS_WORKER:
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
else:
    # Orchestrator mode — only matplotlib/pandas needed for final plotting
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, '/home/nagle/final_version/squid-nn')
    sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
    sys.path.append('/home/nagle/final_version/squid-nn/squid')
    sys.path.append('/home/nagle/final_version/residualbind')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUCS = ['A', 'C', 'G', 'U']

ALL_GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
    "additive_pairwise_gaussian",
    "nonlin_additive_pairwise_gaussian",
]

CORE_GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
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

ALL_LIBRARY_SIZES = [200, 2_000, 20_000, 200_000, 2_000_000]
NONLIN_NAME = "sigmoid"


# ---------------------------------------------------------------------------
# Sequence utilities (used in orchestrator)
# ---------------------------------------------------------------------------

def generate_random_seqs(num_seqs, min_len, max_len, seed):
    """Generate `num_seqs` random RNA sequences with lengths geometrically
    spaced between min_len and max_len."""
    rng = np.random.default_rng(seed)
    lengths = np.unique(np.round(
        np.geomspace(min_len, max_len, num_seqs)
    ).astype(int))
    while len(lengths) < num_seqs:
        lengths = np.append(lengths, lengths[-1] + 1)
    lengths = lengths[:num_seqs]
    nucs = np.array(['A', 'C', 'G', 'U'])
    seqs = []
    for L in lengths:
        seq = "".join(nucs[rng.integers(0, 4, size=int(L))])
        seqs.append(seq)
    return seqs


def load_seqs_file(path):
    """Load sequences from a text file (one RNA sequence per line)."""
    seqs = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                seqs.append(s.upper().replace("T", "U"))
    return seqs


# ---------------------------------------------------------------------------
# Worker-only helpers (called only when _IS_WORKER=True)
# ---------------------------------------------------------------------------

def generate_random_library(wt_onehot, n_seqs, exact_mut_count, rng):
    """Vectorised: mutate exactly `exact_mut_count` distinct positions per seq."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    CHUNK  = 50_000
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


def score_library(X, gt_params_potts, gt_params_gaussian, nonlin_kwargs, gt_keys):
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
    need_gaussian = any(k.endswith("_gaussian") for k in gt_keys)
    gauss_scores  = {}
    if need_gaussian:
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
    out = {}
    for k in gt_keys:
        if k == "additive":
            out[k] = potts_scores["additive"]
        elif k == "additive_pairwise":
            out[k] = potts_scores["additive_pairwise"]
        elif k == "nonlin_additive":
            out[k] = potts_scores["nonlin_additive"]
        elif k == "nonlin_additive_pairwise":
            out[k] = potts_scores["nonlin_additive_pairwise"]
        elif k == "additive_pairwise_gaussian":
            out[k] = gauss_scores["additive_pairwise"]
        elif k == "nonlin_additive_pairwise_gaussian":
            out[k] = gauss_scores["nonlin_additive_pairwise"]
    return out


def build_eval_libraries(wt_onehot, gt_params_potts, gt_params_gaussian,
                          nonlin_kwargs, exact_mut_count, gt_keys,
                          n_pool=500_000, target_n=5_000,
                          n_bins=200, clip_hi=98, seed=42):
    W_mut    = gt_params_potts["W_mut"]
    mut_map  = gt_params_potts["mut_map"]
    b0       = gt_params_potts["b"]
    edges_p  = gt_params_potts["edges"]
    J_p      = gt_params_potts["J"]
    edges_g  = gt_params_gaussian["edges"]
    J_g      = gt_params_gaussian["J"]
    L, A     = wt_onehot.shape
    wt_idx   = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    rng_pool = np.random.default_rng(seed + 99_999)
    CHUNK    = 100_000

    print(f"  [eval] Generating {n_pool:,} pool seqs …")
    nuc_ids = np.tile(wt_idx[None], (n_pool, 1)).astype(np.uint8)
    for start in range(0, n_pool, CHUNK):
        nc       = min(CHUNK, n_pool - start)
        noise    = rng_pool.random((nc, L))
        pos      = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng_pool.integers(0, 3, size=nc * exact_mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
        nuc_ids[start:start + nc][n_idx, p_idx] = new_nucs

    print("  [eval] Scoring pool …")
    need_gaussian = any(k.endswith("_gaussian") for k in gt_keys)
    s_add_all   = np.empty(n_pool, dtype=np.float32)
    s_addpair_p = np.empty(n_pool, dtype=np.float32)
    s_addpair_g = np.empty(n_pool, dtype=np.float32) if need_gaussian else None
    for start in range(0, n_pool, CHUNK):
        end     = min(start + CHUNK, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa      = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp      = pairwise_potts_energy(X_chunk, edges_p, J_p, b=0.0).reshape(-1)
        s_add_all[start:end]   = sa
        s_addpair_p[start:end] = sa + sp
        if need_gaussian:
            sg = pairwise_potts_energy(X_chunk, edges_g, J_g, b=0.0).reshape(-1)
            s_addpair_g[start:end] = sa + sg

    ref_std = float(nonlin_kwargs.get("_norm_std", float(np.std(s_add_all)) + 1e-8))
    wt_nl   = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))

    pool_scores = {k: np.empty(n_pool, dtype=np.float32) for k in gt_keys}
    for start in range(0, n_pool, CHUNK):
        end = min(start + CHUNK, n_pool)
        sa  = s_add_all[start:end].astype(float)
        sap = s_addpair_p[start:end].astype(float)
        if need_gaussian:
            sag = s_addpair_g[start:end].astype(float)
        for k in gt_keys:
            if k == "additive":
                pool_scores[k][start:end] = sa
            elif k == "additive_pairwise":
                pool_scores[k][start:end] = sap
            elif k == "nonlin_additive":
                v = apply_global_nonlin(sa / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                pool_scores[k][start:end] = v
            elif k == "nonlin_additive_pairwise":
                v = apply_global_nonlin(sap / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                pool_scores[k][start:end] = v
            elif k == "additive_pairwise_gaussian":
                pool_scores[k][start:end] = sag
            elif k == "nonlin_additive_pairwise_gaussian":
                v = apply_global_nonlin(sag / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                pool_scores[k][start:end] = v

    eval_libs = {}
    for k in gt_keys:
        yk = pool_scores[k].astype(float)
        y_uni, keep_idx = uniformize_by_histogram(
            yk, X=None, n_bins=n_bins, clip_hi=clip_hi,
            target_n=target_n, seed=seed + hash(k) % 10_000,
        )
        eval_libs[k] = {
            "X_eval": np.eye(A, dtype=np.float32)[nuc_ids[keep_idx]],
            "y_eval": y_uni.astype(float),
        }
        print(f"  [eval] {k}: {len(y_uni):,} seqs  "
              f"range [{y_uni.min():.3f}, {y_uni.max():.3f}]")

    del nuc_ids, pool_scores, s_add_all, s_addpair_p
    if need_gaussian:
        del s_addpair_g
    return eval_libs


def _predict_chunked(model, x_str, chunk=10_000):
    parts = []
    for start in range(0, len(x_str), chunk):
        parts.append(
            np.asarray(model.x_to_yhat(x_str[start:start + chunk]), dtype=float).reshape(-1)
        )
    return np.concatenate(parts)


def fit_surrogate(x_lib, y, cfg_name, x_eval_oh, y_eval):
    """Train one surrogate; return (rho_random, rho_eval)."""
    cfg  = SURROGATE_CONFIGS[cfg_name]
    N    = x_lib.shape[0]
    bsz  = max(32, min(N // 150, 2048))
    lr   = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    is_nl_pairwise = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs   = 1000 if is_nl_pairwise else 500
    patience = 50   if is_nl_pairwise else 25

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
        cols  = list(test_df.columns)
        X_col = "x" if "x" in cols else "X"
        y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        X_test_str = np.asarray(test_df[X_col])
        y_test     = np.asarray(test_df[y_col], dtype=float).reshape(-1)
        y_hat_test = _predict_chunked(model, X_test_str)
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


def run_one_seq(seq_idx, seq, args, gt_keys, library_sizes):
    """Run the full lib-size sweep for one sequence. Returns list of result dicts."""
    seq_seed = args.seed + seq_idx * 1_000_007

    oh_seq            = rna_to_one_hot(seq)
    no_padding_seq, _ = remove_padding(oh_seq)
    wt_onehot         = no_padding_seq
    L                 = wt_onehot.shape[0]
    exact_mut_count   = min(args.mut_rate, max(1, L // 2))

    print(f"\n{'#'*70}")
    print(f"  Sequence {seq_idx}  length={L}  exact_mut_count={exact_mut_count}")
    print(f"  seq = {seq[:60]}{'...' if len(seq) > 60 else ''}")
    print(f"{'#'*70}")

    rng_gt = np.random.default_rng(seq_seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges_p, J_p = init_pairwise_potts_optionA(
        rng_gt, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )
    edges_g, J_g = init_pairwise_gaussian(rng_gt, wt_onehot, sigma=0.3, wt_rowcol_zero=True)

    gt_params_potts    = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges_p, "J": J_p}
    gt_params_gaussian = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges_g, "J": J_g}

    rng_ref   = np.random.default_rng(seq_seed + 1)
    X_ref     = generate_random_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"  [init] sigmoid ref_std={nonlin_kwargs['_norm_std']:.4f}")

    eval_libs = build_eval_libraries(
        wt_onehot, gt_params_potts, gt_params_gaussian, nonlin_kwargs,
        exact_mut_count=exact_mut_count,
        n_pool=args.eval_n_pool,
        target_n=args.eval_target_n,
        gt_keys=gt_keys,
        seed=seq_seed,
    )

    rows = []
    for N in library_sizes:
        print(f"\n  {'='*60}")
        print(f"  Library size N = {N:,}  (seq {seq_idx}, L={L})")
        print(f"  {'='*60}")

        rng_lib   = np.random.default_rng(seq_seed + N)
        X_lib     = generate_random_library(wt_onehot, N, exact_mut_count, rng_lib)
        gt_scores = score_library(X_lib, gt_params_potts, gt_params_gaussian, nonlin_kwargs, gt_keys)

        for gt_key in gt_keys:
            y    = gt_scores[gt_key].astype(float).reshape(-1, 1)
            mask = np.isfinite(y[:, 0])
            x_use, y_use = X_lib[mask], y[mask]
            x_eval_oh = eval_libs[gt_key]["X_eval"]
            y_eval    = eval_libs[gt_key]["y_eval"]

            print(f"\n  [{gt_key}]  N_valid={mask.sum():,}")
            for cfg_name in SURROGATE_CONFIGS:
                print(f"    {cfg_name:<32} … ", end="", flush=True)
                rho_rand, rho_ev = fit_surrogate(x_use, y_use, cfg_name, x_eval_oh, y_eval)
                print(f"rho_random={rho_rand:+.3f}   rho_eval={rho_ev:+.3f}")
                rows.append({
                    "seq_idx":      seq_idx,
                    "seq_len":      L,
                    "seq":          seq,
                    "gt_key":       gt_key,
                    "surrogate":    cfg_name,
                    "library_size": N,
                    "rho_random":   rho_rand,
                    "rho_eval":     rho_ev,
                })

        del X_lib, gt_scores

    return rows


# ---------------------------------------------------------------------------
# Worker entry point  (called when --_worker_seq_idx is present)
# ---------------------------------------------------------------------------

def worker_main(args):
    """Run one sequence and save results to {out_dir}/seq_{idx}_results.csv."""
    seq_idx    = args._worker_seq_idx
    seq        = args._worker_seq
    gt_keys    = CORE_GT_KEYS if args.no_gaussian else ALL_GT_KEYS
    lib_sizes  = [N for N in ALL_LIBRARY_SIZES if N <= args.max_lib_size]

    print(f"[worker] seq_idx={seq_idx}  GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}")

    rows = run_one_seq(seq_idx, seq, args, gt_keys, lib_sizes)

    out_csv = os.path.join(args.out_dir, f"seq_{seq_idx}_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[worker] saved → {out_csv}")


# ---------------------------------------------------------------------------
# Orchestrator — launches worker subprocesses, one per sequence
# ---------------------------------------------------------------------------

def _launch_worker(seq_idx, seq, gpu_id, args):
    """Spawn a child subprocess for one sequence pinned to `gpu_id`."""
    import subprocess
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))

    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--_worker_seq_idx", str(seq_idx),
        "--_worker_seq",     seq,
        "--out_dir",         args.out_dir,
        "--mut_rate",        str(args.mut_rate),
        "--seed",            str(args.seed),
        "--eval_n_pool",     str(args.eval_n_pool),
        "--eval_target_n",   str(args.eval_target_n),
        "--max_lib_size",    str(args.max_lib_size),
    ]
    if args.no_gaussian:
        cmd.append("--no_gaussian")

    log_path = os.path.join(args.out_dir, f"seq_{seq_idx}_gpu{gpu_id}.log")
    log_fh   = open(log_path, "w")
    p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"[orchestrator] launched seq {seq_idx:2d} on GPU {gpu_id}  "
          f"(pid={p.pid}, log={log_path})")
    return p, log_fh


def orchestrator_main(args, sequences, gt_keys, library_sizes):
    """Manage up to `num_gpus` concurrent worker subprocesses."""
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(args.out_dir, exist_ok=True)

    # Find sequences that still need to run
    pending = []
    for seq_idx, seq in enumerate(sequences):
        csv_path = os.path.join(args.out_dir, f"seq_{seq_idx}_results.csv")
        if os.path.exists(csv_path):
            print(f"[orchestrator] seq {seq_idx} already done — skipping")
        else:
            pending.append((seq_idx, seq))

    if not pending:
        print("[orchestrator] all sequences already complete; proceeding to aggregation.")
        return

    gpu_queue = Queue()
    for i in range(args.num_gpus):
        gpu_queue.put(i)

    def run_one(seq_idx_seq):
        seq_idx, seq = seq_idx_seq
        gpu_id = gpu_queue.get()
        try:
            p, log_fh = _launch_worker(seq_idx, seq, gpu_id, args)
            p.wait()
            log_fh.close()
            if p.returncode != 0:
                print(f"[orchestrator] WARNING: seq {seq_idx} exited with code {p.returncode}. "
                      f"Check {args.out_dir}/seq_{seq_idx}_gpu{gpu_id}.log")
            else:
                print(f"[orchestrator] seq {seq_idx} finished on GPU {gpu_id}")
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        list(pool.map(run_one, pending))

    print(f"\n[orchestrator] all sequences done.")


# ---------------------------------------------------------------------------
# Aggregation and plotting (orchestrator)
# ---------------------------------------------------------------------------

def merge_seq_csvs(out_dir, num_seqs):
    """Merge per-sequence CSVs into one DataFrame."""
    dfs = []
    for seq_idx in range(num_seqs):
        csv_path = os.path.join(out_dir, f"seq_{seq_idx}_results.csv")
        if os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
        else:
            print(f"[warn] missing {csv_path} — sequence {seq_idx} may have failed")
    if not dfs:
        raise RuntimeError(f"No per-sequence CSVs found in {out_dir}")
    return pd.concat(dfs, ignore_index=True)


def plot_aggregate(agg_df, library_sizes, gt_keys, out_dir):
    """One figure per GT key: mean ± std Spearman across sequences."""
    sizes = np.array(library_sizes)
    for gt_key in gt_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = agg_df[agg_df["gt_key"] == gt_key]
        for cfg_name, color in SURROGATE_COLORS.items():
            cfg_sub = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
            mu_rand = cfg_sub["rho_random_mean"].values
            sd_rand = cfg_sub["rho_random_std"].values
            mu_eval = cfg_sub["rho_eval_mean"].values
            sd_eval = cfg_sub["rho_eval_std"].values
            ax.semilogx(sizes, mu_rand, marker="o", color=color,
                        linestyle="-", linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (random)")
            ax.fill_between(sizes, mu_rand - sd_rand, mu_rand + sd_rand,
                            color=color, alpha=0.15)
            ax.semilogx(sizes, mu_eval, marker="s", color=color,
                        linestyle="--", linewidth=1.8, markersize=5,
                        label=f"{cfg_name} (eval)")
            ax.fill_between(sizes, mu_eval - sd_eval, mu_eval + sd_eval,
                            color=color, alpha=0.10)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        n_seqs = agg_df["seq_idx"].nunique()
        ax.set_title(
            f"{GT_TITLES[gt_key]}\n(mean ± std across {n_seqs} sequences)",
            fontsize=11,
        )
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_multi_seq_spearman_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


def plot_by_split(agg_df, library_sizes, gt_keys, out_dir, split="random"):
    sizes  = np.array(library_sizes)
    col_mu = f"rho_{split}_mean"
    col_sd = f"rho_{split}_std"
    marker = "o" if split == "random" else "s"
    for gt_key in gt_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = agg_df[agg_df["gt_key"] == gt_key]
        for cfg_name, color in SURROGATE_COLORS.items():
            cfg_sub = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
            mu = cfg_sub[col_mu].values
            sd = cfg_sub[col_sd].values
            ax.semilogx(sizes, mu, marker=marker, color=color,
                        linewidth=1.8, markersize=5, label=cfg_name)
            ax.fill_between(sizes, mu - sd, mu + sd, color=color, alpha=0.18)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        n_seqs = agg_df["seq_idx"].nunique()
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(
            f"{GT_TITLES[gt_key]} ({split} split)\n"
            f"mean ± std across {n_seqs} sequences",
            fontsize=11,
        )
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, f"lib_size_multi_seq_{split}_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


def plot_seq_lengths_overview(per_seq_df, library_sizes, gt_keys, out_dir):
    """Spearman at max library size vs. sequence length per surrogate."""
    max_N   = max(library_sizes)
    sub_max = per_seq_df[per_seq_df["library_size"] == max_N]
    for split in ("random", "eval"):
        col = f"rho_{split}"
        for gt_key in gt_keys:
            fig, ax = plt.subplots(figsize=(6, 4))
            sub = sub_max[sub_max["gt_key"] == gt_key]
            for cfg_name, color in SURROGATE_COLORS.items():
                cfg_sub = sub[sub["surrogate"] == cfg_name].sort_values("seq_len")
                ax.scatter(cfg_sub["seq_len"], cfg_sub[col], color=color,
                           label=cfg_name, s=40, zorder=3)
                ax.plot(cfg_sub["seq_len"], cfg_sub[col], color=color,
                        linewidth=1.0, alpha=0.5)
            ax.set_xlabel("Sequence length", fontsize=11)
            ax.set_ylabel("Spearman ρ", fontsize=11)
            ax.set_title(
                f"{GT_TITLES[gt_key]} — {split} split\n(N={max_N:,} library)",
                fontsize=10,
            )
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path = os.path.join(out_dir,
                                f"lib_size_multi_seq_vs_len_{split}_{gt_key}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] saved → {path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Library-size Spearman sweep across multiple RNA sequences"
    )
    seq_group = parser.add_mutually_exclusive_group()
    seq_group.add_argument("--seqs_file", type=str,
                           help="Text file with one RNA sequence per line")
    seq_group.add_argument("--num_seqs", type=int, default=10,
                           help="Number of random sequences to auto-generate (default: 10)")
    parser.add_argument("--min_len",       type=int, default=10)
    parser.add_argument("--max_len",       type=int, default=80)
    parser.add_argument("--mut_rate",      type=int, default=4)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--out_dir",       type=str, default="outputs/lib_size_multi_seq")
    parser.add_argument("--eval_n_pool",   type=int, default=500_000)
    parser.add_argument("--eval_target_n", type=int, default=5_000)
    parser.add_argument("--max_lib_size",  type=int, default=2_000_000,
                        help="Skip library sizes above this value")
    parser.add_argument("--no_gaussian",   action="store_true",
                        help="Skip Gaussian negative-control GT keys")
    parser.add_argument("--num_gpus",      type=int, default=8,
                        help="Number of GPUs to use in parallel (default: 8)")
    # Worker-mode args (internal use only — set by orchestrator)
    parser.add_argument("--_worker_seq_idx", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_worker_seq",     type=str, default=None,
                        help=argparse.SUPPRESS)
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Worker mode: run a single sequence, save CSV, exit
    if args._worker_seq_idx is not None:
        worker_main(args)
        return

    # ── Orchestrator mode
    if args.seqs_file:
        sequences = load_seqs_file(args.seqs_file)
        print(f"[init] Loaded {len(sequences)} sequences from {args.seqs_file}")
    else:
        sequences = generate_random_seqs(args.num_seqs, args.min_len, args.max_len, args.seed)
        print(f"[init] Generated {len(sequences)} random sequences, "
              f"lengths {[len(s) for s in sequences]}")

    gt_keys       = CORE_GT_KEYS if args.no_gaussian else ALL_GT_KEYS
    library_sizes = [N for N in ALL_LIBRARY_SIZES if N <= args.max_lib_size]
    print(f"[init] GT keys:        {gt_keys}")
    print(f"[init] Library sizes:  {library_sizes}")
    print(f"[init] GPUs:           {args.num_gpus}")
    print(f"[init] Sequences:      {len(sequences)}")

    # Save sequences so they can be referenced later
    seqs_log = os.path.join(args.out_dir, "sequences.txt")
    with open(seqs_log, "w") as fh:
        for i, s in enumerate(sequences):
            fh.write(f"# seq_{i}  len={len(s)}\n{s}\n")
    print(f"[init] sequences written → {seqs_log}")

    # Launch workers
    orchestrator_main(args, sequences, gt_keys, library_sizes)

    # Merge results
    per_seq_df = merge_seq_csvs(args.out_dir, len(sequences))
    csv_path   = os.path.join(args.out_dir, "per_seq_results.csv")
    per_seq_df.to_csv(csv_path, index=False)
    print(f"\n[results] per-sequence CSV saved → {csv_path}")

    # Aggregate: mean and std across sequences
    agg = (
        per_seq_df
        .groupby(["gt_key", "surrogate", "library_size"])[["rho_random", "rho_eval"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["gt_key", "surrogate", "library_size",
                   "rho_random_mean", "rho_random_std",
                   "rho_eval_mean",   "rho_eval_std"]
    agg_path = os.path.join(args.out_dir, "aggregate_results.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[results] aggregate CSV saved → {agg_path}")

    # Plots
    plot_aggregate(agg, library_sizes, gt_keys, args.out_dir)
    plot_by_split(agg, library_sizes, gt_keys, args.out_dir, split="random")
    plot_by_split(agg, library_sizes, gt_keys, args.out_dir, split="eval")
    plot_seq_lengths_overview(per_seq_df, library_sizes, gt_keys, args.out_dir)


if __name__ == "__main__":
    main()
