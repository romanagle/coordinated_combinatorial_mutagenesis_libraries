"""saturated_mut_experiment.py

Runs the complete single-mutant (saturated) mutagenesis library for each
sequence and records Spearman ρ on the eval library.

The saturated library (3×L single-mutant sequences + WT) is scored with each
GT function to build a (4,L) additive delta weight matrix.  Any query sequence
is then scored as the sum of delta weights at its mutated positions — no
surrogate training required.  Since the eval library is also single-mutant-only,
this predictor has exact information for every eval sequence.

Produces outputs/lib_size_saturated/
    seq_<i>_results.csv   — one row per gt_key
    aggregate_results.csv — mean ± std across sequences

Usage:
    python scripts/saturated_mut_experiment.py \
        --num_seqs 10 --seed 0 --eval_mut_count 4 \
        --out_dir outputs/lib_size_saturated
"""

import os, sys, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_IS_WORKER = "--_worker_seq_idx" in sys.argv

if _IS_WORKER:
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, '/home/nagle/final_version/squid-nn')
    sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
    sys.path.append('/home/nagle/final_version/squid-nn/squid')
    sys.path.append('/home/nagle/final_version/residualbind')

    from seq_utils import rna_to_one_hot
    from ground_truth import (
        init_additive_noWT,
        init_sigmoid_nonlin, compute_gt_scores_for_library_potts,
        uniformize_by_histogram, additive_affinity_noWT, pairwise_potts_energy,
        apply_global_nonlin,
    )
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mRNA_RBP"))
    from gt_init import init_wc_pairwise_sparse
else:
    sys.path.insert(0, os.path.dirname(__file__))

WORKER_PYTHON = "/home/nagle/miniconda3/envs/squid/bin/python"
NONLIN_NAME = "sigmoid"
POOL_SIZE   = 200_000

ALL_GT_KEYS = [
    "additive", "additive_pairwise",
    "nonlin_additive", "nonlin_additive_pairwise",
]

def build_additive_weights(wt_onehot, y_sat):
    """Build (4, L) delta weight matrix from single-mutant GT scores.

    y_sat: (3*L,) GT scores in the order produced by generate_saturated_library().
    WT score = 0 by GT construction, so delta_W[a, j] = y_sat directly.
    WT positions are left as 0.
    """
    L, A    = wt_onehot.shape
    wt_idx  = np.argmax(wt_onehot, axis=1)
    delta_W = np.zeros((A, L), dtype=np.float64)
    idx = 0
    for pos in range(L):
        for nuc in range(A):
            if nuc == wt_idx[pos]:
                continue
            delta_W[nuc, pos] = y_sat[idx] if np.isfinite(y_sat[idx]) else 0.0
            idx += 1
    return delta_W


def additive_predict(delta_W, X_query):
    """Score sequences via additive dot product with the delta weight matrix.

    delta_W:  (4, L)   — from build_additive_weights (WT=0 baseline implicit)
    X_query:  (N, L, 4) one-hot
    Returns:  (N,) predicted scores
    """
    return np.einsum('nla,al->n', X_query.astype(np.float64), delta_W)


# ---------------------------------------------------------------------------
# Shared helpers (worker only)
# ---------------------------------------------------------------------------

def generate_random_seqs(num_seqs, seq_len, seed):
    rng  = np.random.default_rng(seed)
    nucs = np.array(['A', 'C', 'G', 'U'])
    return ["".join(nucs[rng.integers(0, 4, size=seq_len)]) for _ in range(num_seqs)]


def generate_pool(wt_onehot, n, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
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


def generate_saturated_library(wt_onehot):
    """All 41 × 3 = 123 single-mutant sequences."""
    L, A    = wt_onehot.shape
    wt_idx  = np.argmax(wt_onehot, axis=1)
    seqs    = []
    for pos in range(L):
        for nuc in range(A):
            if nuc == wt_idx[pos]:
                continue
            x = wt_onehot.copy().astype(np.float32)
            x[pos, :]   = 0
            x[pos, nuc] = 1
            seqs.append(x)
    return np.stack(seqs)


def build_eval_libraries(wt_onehot, gt_params, nonlin_kwargs, exact_mut_count, gt_keys,
                          n_pool=500_000, target_n=5_000,
                          n_bins=200, clip_hi=98, seed=42):
    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]
    L, A    = wt_onehot.shape
    wt_idx  = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    rng_pool = np.random.default_rng(seed + 99_999)
    CHUNK    = 100_000

    nuc_ids = np.tile(wt_idx[None], (n_pool, 1)).astype(np.uint8)
    for start in range(0, n_pool, CHUNK):
        nc    = min(CHUNK, n_pool - start)
        noise = rng_pool.random((nc, L))
        pos   = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng_pool.integers(0, 3, size=nc * exact_mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
        nuc_ids[start:start + nc][n_idx, p_idx] = new_nucs

    s_add_all = np.empty(n_pool, dtype=np.float32)
    s_addpair = np.empty(n_pool, dtype=np.float32)
    for start in range(0, n_pool, CHUNK):
        end     = min(start + CHUNK, n_pool)
        X_chunk = np.eye(A, dtype=np.float32)[nuc_ids[start:end]]
        sa      = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp      = pairwise_potts_energy(X_chunk, edges, J, b=0.0).reshape(-1)
        s_add_all[start:end] = sa
        s_addpair[start:end] = sa + sp

    ref_std         = float(nonlin_kwargs.get("_norm_std",         float(np.std(s_add_all)) + 1e-8))
    ref_std_addpair = float(nonlin_kwargs.get("_norm_std_addpair", float(np.std(s_addpair))  + 1e-8))
    wt_nl = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))

    pool_scores = {k: np.empty(n_pool, dtype=np.float32) for k in gt_keys}
    for start in range(0, n_pool, CHUNK):
        end = min(start + CHUNK, n_pool)
        sa  = s_add_all[start:end].astype(float)
        sap = s_addpair[start:end].astype(float)
        for k in gt_keys:
            if k == "additive":
                pool_scores[k][start:end] = sa
            elif k == "additive_pairwise":
                pool_scores[k][start:end] = sap
            elif k == "nonlin_additive":
                pool_scores[k][start:end] = (
                    apply_global_nonlin(sa / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                )
            elif k == "nonlin_additive_pairwise":
                pool_scores[k][start:end] = (
                    apply_global_nonlin(sap / ref_std_addpair, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                )

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
    del nuc_ids, pool_scores, s_add_all, s_addpair
    return eval_libs




def run_one_seq(seq_idx, seq, seed, eval_n_pool, eval_target_n, eval_mut_count, gt_keys):
    seq_seed  = seed + seq_idx * 1_000_007
    wt_onehot = rna_to_one_hot(seq)
    L         = wt_onehot.shape[0]

    print(f"\n{'#'*60}")
    print(f"  Seq {seq_idx}  L={L}  saturated library ({L}×3={L*3} seqs)")
    print(f"{'#'*60}")

    rng_gt = np.random.default_rng(seq_seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.03, bias=0.0)
    edges, J = init_wc_pairwise_sparse(
        rng_gt, wt_onehot, seq,
        p_edge=0.15, sigma_P=0.30, l1_P=0.40,
        edge_seed=int(seq_seed),
    )
    gt_params = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges, "J": J}

    rng_ref    = np.random.default_rng(seq_seed + 1)
    X_ref      = generate_pool(wt_onehot, POOL_SIZE, eval_mut_count, rng_ref)
    s_add_ref  = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    s_pair_ref = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    nonlin_kwargs["_norm_std_addpair"] = float(np.std(s_add_ref + s_pair_ref)) + 1e-8
    del X_ref, s_add_ref, s_pair_ref

    eval_libs = build_eval_libraries(
        wt_onehot, gt_params, nonlin_kwargs,
        exact_mut_count=eval_mut_count,
        n_pool=eval_n_pool, target_n=eval_target_n,
        gt_keys=gt_keys, seed=seq_seed,
    )

    X_sat = generate_saturated_library(wt_onehot)
    scores = compute_gt_scores_for_library_potts(
        X_sat, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name=NONLIN_NAME, nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )

    rows = []
    for gt_key in gt_keys:
        y_sat_1d = scores[gt_key].astype(float).reshape(-1)   # (3L,)
        x_eval_oh  = eval_libs[gt_key]["X_eval"]
        y_eval     = eval_libs[gt_key]["y_eval"]

        delta_W    = build_additive_weights(wt_onehot, y_sat_1d)
        y_hat_eval = additive_predict(delta_W, x_eval_oh)
        m          = np.isfinite(y_eval) & np.isfinite(y_hat_eval)
        rho_ev     = float(spearmanr(y_eval[m], y_hat_eval[m])[0]) if m.sum() >= 3 else np.nan

        print(f"\n  [{gt_key}]  rho_eval={rho_ev:+.4f}")
        rows.append({
            "seq_idx":    seq_idx,
            "seq_len":    L,
            "gt_key":     gt_key,
            "surrogate":  "additive_weights",
            "rho_random": np.nan,
            "rho_eval":   rho_ev,
        })
    return rows


# ---------------------------------------------------------------------------
# Worker / orchestrator
# ---------------------------------------------------------------------------

def worker_main(args):
    gt_keys = ALL_GT_KEYS
    rows    = run_one_seq(
        args._worker_seq_idx, args._worker_seq,
        args.seed, args.eval_n_pool, args.eval_target_n, args.eval_mut_count, gt_keys,
    )
    out_csv = os.path.join(args.out_dir, f"seq_{args._worker_seq_idx}_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[worker] saved → {out_csv}")


def _launch_worker(seq_idx, seq, gpu_id, args):
    import subprocess
    env      = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    log_path = os.path.join(args.out_dir, f"seq_{seq_idx}_gpu{gpu_id}.log")
    cmd = [
        WORKER_PYTHON, os.path.abspath(__file__),
        "--_worker_seq_idx", str(seq_idx),
        "--_worker_seq",     seq,
        "--out_dir",         args.out_dir,
        "--seed",            str(args.seed),
        "--eval_n_pool",     str(args.eval_n_pool),
        "--eval_target_n",   str(args.eval_target_n),
        "--eval_mut_count",  str(args.eval_mut_count),
    ]
    log_fh = open(log_path, "w")
    p = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    print(f"[orchestrator] seq {seq_idx:2d} on GPU {gpu_id}  (pid={p.pid})")
    return p, log_fh


def orchestrator_main(args, sequences):
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor

    pending = []
    for seq_idx, seq in enumerate(sequences):
        csv_path = os.path.join(args.out_dir, f"seq_{seq_idx}_results.csv")
        if os.path.exists(csv_path):
            print(f"[orchestrator] seq {seq_idx} already done — skipping")
        else:
            pending.append((seq_idx, seq))

    if not pending:
        print("[orchestrator] all sequences done; aggregating.")
        return

    gpu_queue = Queue()
    for i in range(args.num_gpus):
        gpu_queue.put(i)

    def run_one(item):
        seq_idx, seq = item
        gpu_id = gpu_queue.get()
        try:
            p, log_fh = _launch_worker(seq_idx, seq, gpu_id, args)
            p.wait()
            log_fh.close()
            if p.returncode != 0:
                print(f"[orchestrator] WARNING: seq {seq_idx} failed (code {p.returncode})")
            else:
                print(f"[orchestrator] seq {seq_idx} done on GPU {gpu_id}")
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=args.num_gpus) as pool:
        list(pool.map(run_one, pending))


def build_parser():
    p = argparse.ArgumentParser()
    seq_grp = p.add_mutually_exclusive_group()
    seq_grp.add_argument("--seqs_file", type=str)
    seq_grp.add_argument("--num_seqs",  type=int, default=10)
    p.add_argument("--seq_len",      type=int, default=41)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--out_dir",      type=str, default="outputs/lib_size_saturated")
    p.add_argument("--eval_n_pool",   type=int, default=500_000)
    p.add_argument("--eval_target_n", type=int, default=5_000)
    p.add_argument("--eval_mut_count",type=int, default=4,
                   help="Mutations per sequence in the eval library (match main experiment)")
    p.add_argument("--num_gpus",      type=int, default=8)
    p.add_argument("--_worker_seq_idx", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_worker_seq",     type=str, default=None, help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args._worker_seq_idx is not None:
        worker_main(args)
        return

    if args.seqs_file:
        with open(args.seqs_file) as fh:
            sequences = [l.strip().upper().replace("T","U")
                         for l in fh if l.strip() and not l.startswith("#")]
    else:
        sequences = generate_random_seqs(args.num_seqs, args.seq_len, args.seed)

    seqs_log = os.path.join(args.out_dir, "sequences.txt")
    with open(seqs_log, "w") as fh:
        for i, s in enumerate(sequences):
            fh.write(f"# seq_{i}  len={len(s)}\n{s}\n")

    orchestrator_main(args, sequences)

    dfs = []
    for i in range(len(sequences)):
        p = os.path.join(args.out_dir, f"seq_{i}_results.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"[warn] missing seq_{i}_results.csv")
    if not dfs:
        raise RuntimeError("No per-sequence CSVs found.")

    per_seq = pd.concat(dfs, ignore_index=True)
    per_seq.to_csv(os.path.join(args.out_dir, "per_seq_results.csv"), index=False)

    agg = (
        per_seq
        .groupby(["gt_key", "surrogate"])[["rho_random", "rho_eval"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["gt_key", "surrogate",
                   "rho_random_mean", "rho_random_std",
                   "rho_eval_mean",   "rho_eval_std"]
    agg.to_csv(os.path.join(args.out_dir, "aggregate_results.csv"), index=False)
    print(f"[done] results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
