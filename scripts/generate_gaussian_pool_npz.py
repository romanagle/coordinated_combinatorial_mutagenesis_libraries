"""generate_gaussian_pool_npz.py

Generates a pool of randomly mutated sequences scored against the Gaussian
pairwise GT and saves an .npz in the same format as pool_2M.npz, so that
plot_pool_distributions.py can produce the random vs eval overlay figures.

Usage:
    python scripts/generate_gaussian_pool_npz.py \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --out outputs/lib_size_experiment/gaussian_pairwise/pool_gaussian.npz
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT,
    init_pairwise_gaussian,
    init_sigmoid_nonlin,
    additive_affinity_noWT,
    compute_gt_scores_for_library_potts,
)

NONLIN_NAME = "sigmoid"
SEED        = 0
CHUNK       = 50_000


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",      type=str, required=True)
    parser.add_argument("--mut_rate", type=int, default=4)
    parser.add_argument("--n_pool",   type=int, default=200_000)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--out",      type=str,
                        default="outputs/lib_size_experiment/gaussian_pairwise/pool_gaussian.npz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── 1. Parse WT
    oh_seq            = rna_to_one_hot(args.seq)
    no_padding_seq, _ = remove_padding(oh_seq)
    wt_onehot         = no_padding_seq
    L                 = wt_onehot.shape[0]
    print(f"[init] L={L}  mut_rate={args.mut_rate}")

    # ── 2. GT params (same seed as library_size_experiment.py)
    rng_gt = np.random.default_rng(args.seed)
    W_mut, mut_map, b0 = init_additive_noWT(
        rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0,
    )
    edges, J = init_pairwise_gaussian(rng_gt, wt_onehot, sigma=0.3, wt_rowcol_zero=True)
    print(f"[init] Gaussian pairwise: {len(edges):,} edges (all pairs)")

    # ── 3. Sigmoid params from 200K reference library
    print("[init] Computing sigmoid ref_std …")
    rng_ref   = np.random.default_rng(args.seed + 1)
    X_ref     = generate_random_library(wt_onehot, 200_000, args.mut_rate, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"[init] ref_std={nonlin_kwargs['_norm_std']:.4f}")

    # ── 4. Generate pool and score
    print(f"\n[pool] Generating N={args.n_pool:,} sequences …")
    rng_pool = np.random.default_rng(args.seed + 2)
    X_pool   = generate_random_library(wt_onehot, args.n_pool, args.mut_rate, rng_pool)

    print("[pool] Scoring …")
    gt_scores = compute_gt_scores_for_library_potts(
        X_pool,
        W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name=NONLIN_NAME, nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )
    del X_pool

    # ── 5. Save npz (keys match plot_pool_distributions.py expectations)
    np.savez(
        args.out,
        scores_additive=gt_scores["additive"].astype(np.float32),
        scores_additive_pairwise=gt_scores["additive_pairwise"].astype(np.float32),
        scores_nonlin_additive=gt_scores["nonlin_additive"].astype(np.float32),
        scores_nonlin_additive_pairwise=gt_scores["nonlin_additive_pairwise"].astype(np.float32),
    )
    print(f"[done] saved → {args.out}")
    for k, v in gt_scores.items():
        print(f"  {k}: mean={np.mean(v):.3f}  std={np.std(v):.3f}  "
              f"range [{np.min(v):.3f}, {np.max(v):.3f}]")


if __name__ == "__main__":
    main()
