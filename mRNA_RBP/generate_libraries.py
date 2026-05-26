"""
mRNA_RBP/generate_libraries.py

Unified pipeline: generate 2M-sequence pools for a single manually-specified
MrnaRbpGroundTruth sequence across N_INSTANCES random weight seeds,
MUT_RATES_PCT mutation rates, then subsample each pool to every size in LIB_SIZES.

Instance k uses MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k).
All instances share the same sequence and structural specification; only the
random weight draws differ across instances.

Output tree:
  <OUT_BASE>/
    instance_00/
      gt_params.npz                  – alpha (L,4), edges (E,2), W_mut, mut_map, J, seed
      ssm.npz                        – single-site mutagenesis scores (3L seqs)
      mut05/
        pool_2M.npz                  – nuc_ids (2M×L uint8) + 4 score arrays
        lib_200.npz  lib_2000.npz  lib_20000.npz  lib_200000.npz
        eval_uniform_<gt_key>_lib<N>.npz  (per GT key × lib size)
      mut10/ ...
      mut25/ ...
    instance_01/ ...
"""

import os
import sys
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from ground_truth import uniformize_by_histogram
from seq_utils import rna_to_one_hot


# ===========================================================================
# ── Configuration ─────────────────────────────────────────────────────────
# ===========================================================================

# Fixed WT sequence with manually specified stem-loop structure.
# Stem:  8-pair nested WC stem (positions 8-15 pair with 23-30)
# Loop:  7-nt hairpin loop (pos 16-22) containing UGCAUG Nova motif
SEQ             = 'AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA'
STEM_PAIRS      = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
MOTIF_POSITIONS = [17, 18, 19, 20, 21]

STEM_SIGMA           = 3.0          # pairwise coupling magnitude — 2× stem_additive_sigma for near-full compensatory rescue

N_INSTANCES          = 10
N_LARGE              = 2_000_000
MUT_RATES_PCT        = [5, 10, 25]
LIB_SIZES            = [200, 2_000, 20_000]
CHUNK                = 100_000
OUT_BASE             = os.path.join(_HERE, "outputs")
N_POOL_PER_RATE_EVAL = 300_000   # seqs per mut rate for type1 eval pool
TYPE2_TARGET_N       = 20_000    # total seqs in the type2 eval lib
TYPE2_MUT_RANGE      = [2, 3, 4] # mutation counts in the mixed eval lib
TYPE2_MAX_2MUT       = 6_500     # cap on 2-mut seqs (only ~7,380 unique exist for L=41)
TYPE2_POOL_PER_MUT   = 100_000   # random pool per mutation count (3-mut, 4-mut)

SEQS = [SEQ] * N_INSTANCES

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]


# ===========================================================================
# ── Helpers ───────────────────────────────────────────────────────────────
# ===========================================================================

def mut_count_for(pct: int, L: int) -> int:
    return max(1, round(pct * L / 100))


def generate_pool(wt_onehot: np.ndarray, n_seqs: int,
                  mut_count: int, rng) -> np.ndarray:
    """Return (n_seqs, L) uint8 array: each row is WT with exactly mut_count
    positions mutated to a uniformly random non-WT nucleotide."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    nuc_ids = np.empty((n_seqs, L), dtype=np.uint8)

    for start in range(0, n_seqs, CHUNK):
        nc   = min(CHUNK, n_seqs - start)
        buf  = np.tile(wt_idx[None, :], (nc, 1))

        noise = rng.random((nc, L))
        pos   = np.argpartition(noise, mut_count, axis=1)[:, :mut_count]

        wt_at    = wt_idx[pos.ravel()]
        rand_nuc = rng.integers(0, 3, size=nc * mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)

        n_idx = np.repeat(np.arange(nc), mut_count)
        buf[n_idx, pos.ravel()] = new_nucs
        nuc_ids[start:start + nc] = buf

    return nuc_ids


def score_pool(nuc_ids: np.ndarray, gt: MrnaRbpGroundTruth) -> dict:
    """Score a pool via gt.score_all() in memory-safe chunks.

    Returns dict mapping each GT_KEY to a (N,) float32 array.
    All scores ≤ 0 with WT = 0 by MrnaRbpGroundTruth construction.
    """
    N    = nuc_ids.shape[0]
    pool = {k: np.empty(N, dtype=np.float32) for k in GT_KEYS}

    for start in range(0, N, CHUNK):
        end    = min(start + CHUNK, N)
        X      = np.eye(4, dtype=np.float32)[nuc_ids[start:end]]
        scores = gt.score_all(X)
        for k in GT_KEYS:
            pool[k][start:end] = scores[k]

    return pool


def generate_ssm(wt_onehot: np.ndarray) -> np.ndarray:
    """Return all L*3 single-mutant sequences as (L*3, L) uint8 nuc_ids."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    N      = L * 3

    nuc_ids = np.tile(wt_idx[None, :], (N, 1))
    for pos in range(L):
        wt_nuc   = int(wt_idx[pos])
        alt_nucs = [n for n in range(4) if n != wt_nuc]
        for alt_idx, alt_nuc in enumerate(alt_nucs):
            row = pos * 3 + alt_idx
            nuc_ids[row, pos] = alt_nuc

    return nuc_ids


def save_npz(path: str, nuc_ids: np.ndarray, scores: dict, edges: np.ndarray):
    np.savez_compressed(
        path,
        nuc_ids=nuc_ids,
        edges=edges,
        **{f"scores_{k}": scores[k] for k in GT_KEYS},
    )


# ===========================================================================
# ── Main pipeline ─────────────────────────────────────────────────────────
# ===========================================================================

def _missing_lib_sizes(mut_dir):
    missing_rand = [n for n in LIB_SIZES
                    if not os.path.exists(os.path.join(mut_dir, f"lib_{n}.npz"))]
    return missing_rand


def generate_pairwise_lib(k: int, inst_dir: str, gt: MrnaRbpGroundTruth):
    """All 16 nucleotide combinations per stem pair (other positions at WT).

    N = 16 × len(STEM_PAIRS) sequences.  Deterministic — no randomness needed.
    Saves inst_dir/pairwise_lib.npz.
    """
    path = os.path.join(inst_dir, "pairwise_lib.npz")
    if os.path.exists(path):
        print(f"  pairwise_lib exists — skip")
        return

    wt_oh  = gt.wt_one_hot()
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)

    seqs = []
    for (si, sj) in STEM_PAIRS:
        for ni in range(4):
            for nj in range(4):
                seq      = wt_idx.copy()
                seq[si]  = ni
                seq[sj]  = nj
                seqs.append(seq)

    nuc_ids = np.stack(seqs).astype(np.uint8)
    X       = np.eye(4, dtype=np.float32)[nuc_ids]
    scores  = gt.score_all(X)
    save_npz(path, nuc_ids, scores, gt.edges)
    print(f"  pairwise_lib saved  n={len(nuc_ids)}  "
          f"nonlin_add_pair ∈ [{scores['nonlin_additive_pairwise'].min():.4f}, "
          f"{scores['nonlin_additive_pairwise'].max():.4f}]")




def generate_type1_type2(k: int, inst_dir: str, gt: MrnaRbpGroundTruth):
    """Build mixed-rate Type1 (per lib_size) and Type2 (shared 20K) eval libs.

    Draws N_POOL_PER_RATE_EVAL sequences from each mutation rate's pool_2M.npz,
    combines them into one pool, scores all 4 GT keys, then uniformizes by
    nonlin_additive_pairwise score.

    Saves:
        inst_dir/type1_lib{N}.npz   target_n = max(1, int(0.2 * N))
        inst_dir/type2.npz          target_n = TYPE2_TARGET_N
    Each file: nuc_ids, rate_labels, edges, scores_{key} for all GT_KEYS.
    """
    need = [f"type1_lib{N}" for N in LIB_SIZES
            if not os.path.exists(os.path.join(inst_dir, f"type1_lib{N}.npz"))]
    if not os.path.exists(os.path.join(inst_dir, "type2.npz")):
        need.append("type2")
    if not need:
        print(f"  type1/type2 all present — skip")
        return

    print(f"  building type1/type2 (missing: {need})")
    rng = np.random.default_rng(k * 10_000 + 500)
    all_nuc_ids, all_labels = [], []
    all_scores = {key: [] for key in GT_KEYS}

    for pct in MUT_RATES_PCT:
        pool_path = os.path.join(inst_dir, f"mut{pct:02d}", "pool_2M.npz")
        if not os.path.exists(pool_path):
            print(f"    [WARN] missing pool mut{pct} — skipped")
            continue
        d = np.load(pool_path)
        actual_n = len(d["nuc_ids"])
        n_sample = min(N_POOL_PER_RATE_EVAL, actual_n)
        idx = rng.choice(actual_n, size=n_sample, replace=False)
        all_nuc_ids.append(d["nuc_ids"][idx])
        all_labels.append(np.full(n_sample, pct, dtype=np.uint8))
        for key in GT_KEYS:
            all_scores[key].append(d[f"scores_{key}"][idx])

    if not all_nuc_ids:
        print(f"    [WARN] no pools found — aborting type1/type2")
        return

    combined_ids    = np.concatenate(all_nuc_ids, axis=0)
    combined_labels = np.concatenate(all_labels,  axis=0)
    combined_scores = {key: np.concatenate(all_scores[key]) for key in GT_KEYS}
    primary         = combined_scores["nonlin_additive_pairwise"].astype(float)

    def _save(path, keep_idx):
        np.savez_compressed(
            path,
            nuc_ids     = combined_ids[keep_idx],
            rate_labels = combined_labels[keep_idx],
            edges       = gt.edges,
            **{f"scores_{key}": combined_scores[key][keep_idx] for key in GT_KEYS},
        )

    for N in LIB_SIZES:
        path = os.path.join(inst_dir, f"type1_lib{N}.npz")
        if os.path.exists(path):
            continue
        target_n = max(1, int(0.2 * N))
        n_bins   = min(200, target_n)
        _, keep  = uniformize_by_histogram(
            primary, X=None, n_bins=n_bins, clip_lo=1, clip_hi=98,
            target_n=target_n, seed=k * 10_000 + 501 + N,
        )
        _save(path, keep)
        print(f"    saved type1_lib{N}  n={len(keep)}")

    path = os.path.join(inst_dir, "type2.npz")
    if not os.path.exists(path):
        rng2   = np.random.default_rng(k * 10_000 + 599)
        wt_oh  = gt.wt_one_hot()
        wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
        L2     = len(wt_idx)

        all_ids, all_labels = [], []

        # 2-mut: enumerate all unique 2-mutation sequences, cap at TYPE2_MAX_2MUT
        seqs2 = []
        for p1 in range(L2):
            for p2 in range(p1 + 1, L2):
                wt1, wt2 = int(wt_idx[p1]), int(wt_idx[p2])
                for n1 in range(4):
                    if n1 == wt1: continue
                    for n2 in range(4):
                        if n2 == wt2: continue
                        s = wt_idx.copy(); s[p1] = n1; s[p2] = n2
                        seqs2.append(s)
        pool2 = np.unique(np.stack(seqs2).astype(np.uint8), axis=0)
        if len(pool2) > TYPE2_MAX_2MUT:
            pool2 = pool2[rng2.choice(len(pool2), size=TYPE2_MAX_2MUT, replace=False)]
        all_ids.append(pool2)
        all_labels.append(np.full(len(pool2), 2, dtype=np.uint8))
        print(f"    type2: 2-mut pool  n={len(pool2):,}")

        # 3-mut and 4-mut: random pool, deduplicate
        for n_mut in [3, 4]:
            pool = generate_pool(wt_oh, TYPE2_POOL_PER_MUT, n_mut, rng2)
            pool = np.unique(pool, axis=0)
            all_ids.append(pool)
            all_labels.append(np.full(len(pool), n_mut, dtype=np.uint8))
            print(f"    type2: {n_mut}-mut pool  n={len(pool):,}")

        all_ids    = np.concatenate(all_ids, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        _, uniq_idx = np.unique(all_ids, axis=0, return_index=True)
        uniq_idx.sort()
        all_ids    = all_ids[uniq_idx]
        all_labels = all_labels[uniq_idx]
        print(f"    type2: {len(all_ids):,} unique sequences after global dedup")

        pool_scores = score_pool(all_ids, gt)
        primary     = pool_scores["nonlin_additive_pairwise"].astype(float)
        _, keep     = uniformize_by_histogram(
            primary, X=None, n_bins=200, clip_lo=1, clip_hi=99,
            target_n=TYPE2_TARGET_N, seed=k * 10_000 + 600,
        )
        t2_ids    = all_ids[keep]
        t2_labels = all_labels[keep]
        t2_scores = {key: pool_scores[key][keep] for key in GT_KEYS}
        np.savez_compressed(
            path,
            nuc_ids     = t2_ids,
            rate_labels = t2_labels,
            edges       = gt.edges,
            **{f"scores_{key}": t2_scores[key] for key in GT_KEYS},
        )
        s = t2_scores["nonlin_additive_pairwise"]
        vals, cnts = np.unique(t2_labels, return_counts=True)
        dist_str = "  ".join(f"{m}mut:{c}" for m, c in zip(vals, cnts))
        print(f"    saved type2  n={len(t2_ids)} (uniformized from {len(all_ids):,} unique)  "
              f"nonlin_add_pair ∈ [{s.min():.4f}, {s.max():.4f}]")
        print(f"    mutation count distribution: {dist_str}")


def run_pipeline(n_instances: int = N_INSTANCES, pool_size: int = N_LARGE):
    L = len(SEQ)
    mut_counts = {pct: mut_count_for(pct, L) for pct in MUT_RATES_PCT}
    print(f"Sequence: {SEQ}")
    print(f"Stem pairs: {STEM_PAIRS}")
    print(f"Motif positions: {MOTIF_POSITIONS}")
    print(f"Instances: {N_INSTANCES}  |  L={L}  |  Pool size: {N_LARGE:,}  |  "
          f"Subsample sizes: {LIB_SIZES}")
    print(f"Mutation counts: { {p: f'{c} ({c/L*100:.1f}%)' for p, c in mut_counts.items()} }\n")

    for k in range(n_instances):
        inst_dir = os.path.join(OUT_BASE, f"instance_{k:02d}")
        os.makedirs(inst_dir, exist_ok=True)
        t_inst = time.time()

        with open(os.path.join(inst_dir, "wt_seq.txt"), "w") as f:
            f.write(SEQ + "\n")

        # ── GT: reconstruct deterministically from seed=k ──────────────────
        gt           = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k,
                                          stem_sigma=STEM_SIGMA)
        wt_oh        = gt.wt_one_hot()
        gt_params_path = os.path.join(inst_dir, "gt_params.npz")

        if not os.path.exists(gt_params_path):
            np.savez_compressed(
                gt_params_path,
                alpha   = gt.alpha,
                edges   = gt.edges,
                W_mut   = gt._W_mut,
                mut_map = gt._mut_map,
                J       = gt._J,
                seed    = np.array([k]),
            )
            print(f"[instance {k:02d}]  saved gt_params  stem_pairs={len(STEM_PAIRS)}")
        else:
            print(f"[instance {k:02d}]  gt_params exists — skip save")

        # ── SSM ─────────────────────────────────────────────────────────────
        ssm_path = os.path.join(inst_dir, "ssm.npz")
        if not os.path.exists(ssm_path):
            ssm_ids    = generate_ssm(wt_oh)
            X_ssm      = np.eye(4, dtype=np.float32)[ssm_ids]
            ssm_scores = gt.score_all(X_ssm)
            save_npz(ssm_path, ssm_ids, ssm_scores, gt.edges)
            print(f"           ssm saved  nonlin_add_pair ∈ "
                  f"[{ssm_scores['nonlin_additive_pairwise'].min():.4f}, "
                  f"{ssm_scores['nonlin_additive_pairwise'].max():.4f}]")
        else:
            print(f"           ssm already exists — skip")

        # ── Per mutation rate ────────────────────────────────────────────────
        for r_idx, pct in enumerate(MUT_RATES_PCT):
            mc      = mut_counts[pct]
            mut_dir = os.path.join(inst_dir, f"mut{pct:02d}")
            os.makedirs(mut_dir, exist_ok=True)
            t_r = time.time()

            pool_path    = os.path.join(mut_dir, "pool_2M.npz")
            missing_rand = _missing_lib_sizes(mut_dir)

            if not missing_rand and os.path.exists(pool_path):
                print(f"  mut{pct:02d}%  all libs present — skip")
                continue

            # ── Load or generate pool
            if os.path.exists(pool_path):
                d           = np.load(pool_path)
                nuc_ids     = d["nuc_ids"]
                pool_scores = {key: d[f"scores_{key}"] for key in GT_KEYS}
                print(f"  mut{pct:02d}%  loaded pool  missing libs={missing_rand}")
            else:
                rng_lib = np.random.default_rng(k * 1000 + 200 + r_idx * 10)
                nuc_ids = generate_pool(wt_oh, pool_size, mc, rng_lib)
                nuc_ids = np.unique(nuc_ids, axis=0)   # deduplicate
                pool_scores = score_pool(nuc_ids, gt)
                save_npz(pool_path, nuc_ids, pool_scores, gt.edges)
                print(f"  mut{pct:02d}%  pool generated  unique={len(nuc_ids):,}")

            # ── Random subsamples
            if missing_rand:
                actual_pool = len(nuc_ids)
                rng_sub = np.random.default_rng(k * 1000 + 300 + r_idx * 10)
                for n_sub in LIB_SIZES:
                    n_draw = min(n_sub, actual_pool)
                    idx = rng_sub.choice(actual_pool, size=n_draw, replace=False)
                    idx.sort()
                    lib_path = os.path.join(mut_dir, f"lib_{n_sub}.npz")
                    if n_sub in missing_rand:
                        save_npz(lib_path, nuc_ids[idx[:n_draw]],
                                 {key: pool_scores[key][idx] for key in GT_KEYS}, gt.edges)

            del nuc_ids, pool_scores
            print(f"  mut{pct:02d}%  done  ({time.time()-t_r:.1f}s)")

        # ── Type1 / Type2 eval libs + pairwise structured lib
        generate_type1_type2(k, inst_dir, gt)
        generate_pairwise_lib(k, inst_dir, gt)

        print(f"  → instance {k:02d} done  ({time.time()-t_inst:.1f}s)\n")

    print("Pipeline complete.")


# ---------------------------------------------------------------------------
# Single-instance convenience function
# ---------------------------------------------------------------------------

def generate_library(gt: MrnaRbpGroundTruth, n_seqs: int,
                     mut_rate: float, seed: int = 0) -> tuple:
    """Generate a mutagenesis library for one GT instance.

    Parameters
    ----------
    gt       : MrnaRbpGroundTruth instance
    n_seqs   : number of sequences to generate
    mut_rate : fraction of L to mutate (e.g. 0.10 for 10%)
    seed     : random seed

    Returns
    -------
    x_mut : (n_seqs, L, 4) float32 one-hot
    y_mut : (n_seqs,) float32 nonlin_additive_pairwise scores
    """
    L       = len(gt.seq)
    mc      = max(1, round(mut_rate * L))
    rng     = np.random.default_rng(seed)
    nuc_ids = generate_pool(gt.wt_one_hot(), n_seqs, mc, rng)
    x_mut   = np.eye(4, dtype=np.float32)[nuc_ids]
    y_mut   = gt(x_mut)
    return x_mut, y_mut


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--n_instances", type=int, default=N_INSTANCES,
                    help="Number of GT instances to generate (default: all)")
    _p.add_argument("--pool_size", type=int, default=N_LARGE,
                    help="Pool size per (instance, mut_rate) (default: 2M)")
    _args = _p.parse_args()
    run_pipeline(n_instances=_args.n_instances, pool_size=_args.pool_size)
