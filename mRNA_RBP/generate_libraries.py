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
from typing import Optional
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.oracles import (
    MRNA_GT_KEYS,
    MRNA_ORACLE,
    RESIDUALBIND_MSI1_ORACLE,
    build_oracle,
    default_output_base,
    normalize_oracle_name,
    oracle_gt_keys,
    primary_gt_key,
)
from mRNA_RBP.ground_truth import uniformize_by_histogram
from mRNA_RBP.seq_utils import rna_to_one_hot
from mRNA_RBP.sequence_configs import (
    vts1_sequence_config,
)


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
ACTIVITY_BALANCED_TARGET_N     = 20_000
ACTIVITY_BALANCED_CANDIDATE_N  = 200_000
ACTIVITY_BALANCED_MUT_COUNTS   = [3, 5, 7, 15]

# Backward-compatible aliases for older analysis scripts/results schemas.
# New outputs should use activity_balanced.npz only; do not create type2.npz.
TYPE2_TARGET_N       = ACTIVITY_BALANCED_TARGET_N
TYPE2_MUT_RANGE      = ACTIVITY_BALANCED_MUT_COUNTS
TYPE2_POOL_PER_MUT   = ACTIVITY_BALANCED_CANDIDATE_N // len(ACTIVITY_BALANCED_MUT_COUNTS)

SEQS = [SEQ] * N_INSTANCES

GT_KEYS = MRNA_GT_KEYS


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


def sample_unique_mutants(wt_onehot: np.ndarray, mut_count: int,
                          target_n: int, rng, max_rounds: int = 12) -> np.ndarray:
    """Sample exactly target_n unique mutants when combinatorially feasible."""
    collected = None
    n_request = int(target_n)
    for _ in range(max_rounds):
        pool = generate_pool(wt_onehot, n_request, mut_count, rng)
        collected = pool if collected is None else np.concatenate([collected, pool], axis=0)
        collected = np.unique(collected, axis=0)
        if len(collected) >= target_n:
            return collected[:target_n]
        n_request = max((target_n - len(collected)) * 3, 1)
    return collected


def score_pool(nuc_ids: np.ndarray, gt: MrnaRbpGroundTruth,
               gt_keys=None) -> dict:
    """Score a pool via gt.score_all() in memory-safe chunks.

    Returns dict mapping each GT_KEY to a (N,) float32 array.
    All scores ≤ 0 with WT = 0 by MrnaRbpGroundTruth construction.
    """
    N    = nuc_ids.shape[0]
    gt_keys = gt_keys or GT_KEYS
    pool = {k: np.empty(N, dtype=np.float32) for k in gt_keys}

    for start in range(0, N, CHUNK):
        end    = min(start + CHUNK, N)
        X      = np.eye(4, dtype=np.float32)[nuc_ids[start:end]]
        scores = gt.score_all(X)
        for k in gt_keys:
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


def save_npz(path: str, nuc_ids: np.ndarray, scores: dict, edges: np.ndarray,
             gt_keys=None, extra: Optional[dict] = None):
    gt_keys = gt_keys or GT_KEYS
    extra = extra or {}
    np.savez_compressed(
        path,
        nuc_ids=nuc_ids,
        edges=edges,
        **{f"scores_{k}": scores[k] for k in gt_keys},
        **extra,
    )


# ===========================================================================
# ── Main pipeline ─────────────────────────────────────────────────────────
# ===========================================================================

def _missing_lib_sizes(mut_dir, lib_sizes=LIB_SIZES):
    missing_rand = [n for n in lib_sizes
                    if not os.path.exists(os.path.join(mut_dir, f"lib_{n}.npz"))]
    return missing_rand


def generate_pairwise_lib(k: int, inst_dir: str, gt: MrnaRbpGroundTruth,
                          gt_keys=None,
                          primary_key: str = "nonlin_additive_pairwise",
                          force: bool = False):
    gt_keys = gt_keys or GT_KEYS
    """All 16 nucleotide combinations per stem pair (other positions at WT).

    N = 16 × len(STEM_PAIRS) sequences.  Deterministic — no randomness needed.
    Saves inst_dir/pairwise_lib.npz.
    """
    path = os.path.join(inst_dir, "pairwise_lib.npz")
    if os.path.exists(path) and not force:
        print(f"  pairwise_lib exists — skip")
        return

    wt_oh  = gt.wt_one_hot()
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)

    seqs = []
    for (si, sj) in gt.stem_pairs:
        for ni in range(4):
            for nj in range(4):
                seq      = wt_idx.copy()
                seq[si]  = ni
                seq[sj]  = nj
                seqs.append(seq)

    nuc_ids = np.stack(seqs).astype(np.uint8)
    X       = np.eye(4, dtype=np.float32)[nuc_ids]
    scores  = gt.score_all(X)
    save_npz(path, nuc_ids, scores, gt.edges, gt_keys)
    print(f"  pairwise_lib saved  n={len(nuc_ids)}  "
          f"{primary_key} ∈ [{scores[primary_key].min():.4f}, "
          f"{scores[primary_key].max():.4f}]")




def _activity_balanced_paths(inst_dir: str) -> tuple:
    return (
        os.path.join(inst_dir, "activity_balanced.npz"),
        os.path.join(inst_dir, "type2.npz"),
    )


def activity_balanced_path(inst_dir: str) -> str:
    """Canonical activity-balanced library path, with read-only legacy fallback."""
    path, legacy_path = _activity_balanced_paths(inst_dir)
    return path if os.path.exists(path) else legacy_path


def generate_type1_activity_balanced(k: int, inst_dir: str, gt: MrnaRbpGroundTruth,
                                     out_base: str = OUT_BASE,
                                     gt_keys=None,
                                     primary_key: str = "nonlin_additive_pairwise",
                                     lib_sizes=None,
                                     force: bool = False):
    gt_keys = gt_keys or GT_KEYS
    lib_sizes = lib_sizes or LIB_SIZES
    """Build mixed-rate Type1 eval libs and the activity-balanced eval lib.

    Draws N_POOL_PER_RATE_EVAL sequences from each mutation rate's pool_2M.npz,
    combines them into one pool, scores all 4 GT keys, then uniformizes by
    nonlin_additive_pairwise score.

    Saves:
        inst_dir/type1_lib{N}.npz   target_n = max(1, int(0.2 * N))
        inst_dir/activity_balanced.npz  target_n = ACTIVITY_BALANCED_TARGET_N
    Each file: nuc_ids, rate_labels, edges, scores_{key} for all GT_KEYS.

    Activity-balanced initialization:
      1. Draw unique exact-mutant candidates at 3, 5, 7, and 15 mutations.
      2. Use ACTIVITY_BALANCED_CANDIDATE_N total candidates split as evenly
         as possible across those mutation counts.
      3. Deduplicate globally.
      4. Score with the selected oracle's primary score.
      5. Histogram-uniformize in score space with 200 equal-width bins,
         percentile clipping [1, 99], and seed k*10_000 + 600.
      6. Cap at ACTIVITY_BALANCED_TARGET_N sequences. The final count can be
         lower than the cap if nonempty score bins are sparse.
    """
    activity_path, _legacy_type2_path = _activity_balanced_paths(inst_dir)
    need = [f"type1_lib{N}" for N in lib_sizes
            if force or not os.path.exists(os.path.join(inst_dir, f"type1_lib{N}.npz"))]
    if force or not os.path.exists(activity_path):
        need.append("activity_balanced")
    if not need:
        print(f"  type1/activity-balanced all present — skip")
        return

    print(f"  building type1/activity-balanced (missing: {need})")
    rng = np.random.default_rng(k * 10_000 + 500)
    all_nuc_ids, all_labels = [], []
    all_scores = {key: [] for key in gt_keys}

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
        for key in gt_keys:
            all_scores[key].append(d[f"scores_{key}"][idx])

    if not all_nuc_ids:
        print(f"    [WARN] no pools found — aborting type1/activity-balanced")
        return

    combined_ids    = np.concatenate(all_nuc_ids, axis=0)
    combined_labels = np.concatenate(all_labels,  axis=0)
    combined_scores = {key: np.concatenate(all_scores[key]) for key in gt_keys}
    primary         = combined_scores[primary_key].astype(float)

    def _save(path, keep_idx):
        np.savez_compressed(
            path,
            nuc_ids     = combined_ids[keep_idx],
            rate_labels = combined_labels[keep_idx],
            edges       = gt.edges,
            **{f"scores_{key}": combined_scores[key][keep_idx] for key in gt_keys},
        )

    for N in lib_sizes:
        path = os.path.join(inst_dir, f"type1_lib{N}.npz")
        if os.path.exists(path) and not force:
            continue
        target_n = max(1, int(0.2 * N))
        n_bins   = min(200, target_n)
        _, keep  = uniformize_by_histogram(
            primary, X=None, n_bins=n_bins, clip_lo=1, clip_hi=98,
            target_n=target_n, seed=k * 10_000 + 501 + N,
        )
        _save(path, keep)
        print(f"    saved type1_lib{N}  n={len(keep)}")

    if force or not os.path.exists(activity_path):
        rng2   = np.random.default_rng(k * 10_000 + 599)
        wt_oh  = gt.wt_one_hot()
        all_ids, all_labels = [], []

        per_count, rem = divmod(ACTIVITY_BALANCED_CANDIDATE_N,
                                len(ACTIVITY_BALANCED_MUT_COUNTS))
        for i, n_mut in enumerate(ACTIVITY_BALANCED_MUT_COUNTS):
            n_candidates = per_count + (1 if i < rem else 0)
            pool = sample_unique_mutants(wt_oh, n_mut, n_candidates, rng2)
            all_ids.append(pool)
            all_labels.append(np.full(len(pool), n_mut, dtype=np.uint8))
            print(f"    activity-balanced: {n_mut}-mut candidate pool  n={len(pool):,}")

        all_ids    = np.concatenate(all_ids, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        _, uniq_idx = np.unique(all_ids, axis=0, return_index=True)
        uniq_idx.sort()
        all_ids    = all_ids[uniq_idx]
        all_labels = all_labels[uniq_idx]
        print(f"    activity-balanced: {len(all_ids):,} unique sequences after global dedup")

        pool_scores = score_pool(all_ids, gt, gt_keys)
        primary     = pool_scores[primary_key].astype(float)
        _, keep     = uniformize_by_histogram(
            primary, X=None, n_bins=200, clip_lo=1, clip_hi=99,
            target_n=ACTIVITY_BALANCED_TARGET_N, seed=k * 10_000 + 600,
        )
        t2_ids    = all_ids[keep]
        t2_labels = all_labels[keep]
        t2_scores = {key: pool_scores[key][keep] for key in gt_keys}
        payload = dict(
            nuc_ids=t2_ids,
            rate_labels=t2_labels,
            edges=gt.edges,
            activity_balanced_init=np.array([
                "candidate_mut_counts=3,5,7,15;candidate_n=200000;"
                "histogram_uniformize_bins=200;clip_percentiles=1,99;"
                "target_n=20000;seed=k*10000+600"
            ]),
            **{f"scores_{key}": t2_scores[key] for key in gt_keys},
        )
        np.savez_compressed(activity_path, **payload)
        s = t2_scores[primary_key]
        vals, cnts = np.unique(t2_labels, return_counts=True)
        dist_str = "  ".join(f"{m}mut:{c}" for m, c in zip(vals, cnts))
        print(f"    saved activity-balanced library  n={len(t2_ids)} "
              f"(uniformized from {len(all_ids):,} unique)  "
              f"{primary_key} ∈ [{s.min():.4f}, {s.max():.4f}]")
        print(f"    mutation count distribution: {dist_str}")


def run_pipeline(n_instances: int = N_INSTANCES, pool_size: int = N_LARGE,
                 oracle_name: str = MRNA_ORACLE,
                 out_base: Optional[str] = None,
                 residualbind_dir: Optional[str] = None,
                 wt_activity: str = "high"):
    oracle_name = normalize_oracle_name(oracle_name)
    gt_keys = oracle_gt_keys(oracle_name)
    primary_key = primary_gt_key(oracle_name)
    out_base = out_base or default_output_base(_HERE, oracle_name)
    use_vts1_seq = oracle_name != MRNA_ORACLE and oracle_name != RESIDUALBIND_MSI1_ORACLE
    if use_vts1_seq:
        seq, stem_pairs, motif_positions = vts1_sequence_config(wt_activity)
    else:
        seq, stem_pairs, motif_positions = SEQ, STEM_PAIRS, MOTIF_POSITIONS
    L = len(seq)
    mut_counts = {pct: mut_count_for(pct, L) for pct in MUT_RATES_PCT}
    print(f"Sequence: {seq}")
    print(f"Stem pairs: {stem_pairs}")
    print(f"Motif positions: {motif_positions}")
    print(f"Oracle: {oracle_name}  |  primary score: {primary_key}")
    if use_vts1_seq:
        print(f"VTS1 WT activity context: {wt_activity}")
    print(f"Output base: {out_base}")
    print(f"Instances: {n_instances}  |  L={L}  |  Pool size: {pool_size:,}  |  "
          f"Subsample sizes: {LIB_SIZES}")
    print(f"Mutation counts: { {p: f'{c} ({c/L*100:.1f}%)' for p, c in mut_counts.items()} }\n")

    for k in range(n_instances):
        inst_dir = os.path.join(out_base, f"instance_{k:02d}")
        os.makedirs(inst_dir, exist_ok=True)
        t_inst = time.time()
        wt_seq_path = os.path.join(inst_dir, "wt_seq.txt")
        force_regen = False
        if os.path.exists(wt_seq_path):
            with open(wt_seq_path) as f:
                saved_seq = f.read().strip()
            force_regen = saved_seq != seq
            if force_regen:
                print(f"[instance {k:02d}]  existing wt_seq differs; regenerating native {oracle_name} libraries")

        with open(wt_seq_path, "w") as f:
            f.write(seq + "\n")

        # ── GT: reconstruct deterministically from seed=k ──────────────────
        gt = build_oracle(
            oracle_name,
            seq=seq,
            stem_pairs=stem_pairs,
            motif_positions=motif_positions,
            seed=k,
            stem_sigma=STEM_SIGMA,
            residualbind_dir=residualbind_dir,
        )
        wt_oh        = gt.wt_one_hot()
        gt_params_path = os.path.join(inst_dir, "gt_params.npz")

        if force_regen or not os.path.exists(gt_params_path):
            np.savez_compressed(
                gt_params_path,
                alpha   = getattr(gt, "alpha", np.empty((0, 4), dtype=np.float32)),
                edges   = gt.edges,
                W_mut   = getattr(gt, "_W_mut", np.empty((0, 3), dtype=np.float32)),
                mut_map = getattr(gt, "_mut_map", np.empty((0, 3), dtype=np.int32)),
                J       = getattr(gt, "_J", np.empty((0, 0, 4, 4), dtype=np.float32)),
                seed    = np.array([k]),
                oracle  = np.array([oracle_name]),
                wt_activity = np.array([wt_activity if use_vts1_seq else "default"]),
                wt_seq  = np.array(seq),
            )
            print(f"[instance {k:02d}]  saved gt_params  stem_pairs={len(stem_pairs)}")
        else:
            print(f"[instance {k:02d}]  gt_params exists — skip save")

        # ── SSM ─────────────────────────────────────────────────────────────
        ssm_path = os.path.join(inst_dir, "ssm.npz")
        if force_regen or not os.path.exists(ssm_path):
            ssm_ids    = generate_ssm(wt_oh)
            X_ssm      = np.eye(4, dtype=np.float32)[ssm_ids]
            ssm_scores = gt.score_all(X_ssm)
            wt_scores = gt.score_all(wt_oh[None, :, :])
            wt_extra = {
                f"wt_score_{key}": np.array([float(wt_scores[key][0])], dtype=np.float32)
                for key in gt_keys
            }
            save_npz(ssm_path, ssm_ids, ssm_scores, gt.edges, gt_keys, extra=wt_extra)
            print(f"           ssm saved  {primary_key} ∈ "
                  f"[{ssm_scores[primary_key].min():.4f}, "
                  f"{ssm_scores[primary_key].max():.4f}]")
        else:
            print(f"           ssm already exists — skip")

        # ── Per mutation rate ────────────────────────────────────────────────
        for r_idx, pct in enumerate(MUT_RATES_PCT):
            mc      = mut_counts[pct]
            mut_dir = os.path.join(inst_dir, f"mut{pct:02d}")
            os.makedirs(mut_dir, exist_ok=True)
            t_r = time.time()

            pool_path    = os.path.join(mut_dir, "pool_2M.npz")
            missing_rand = LIB_SIZES if force_regen else _missing_lib_sizes(mut_dir)

            if not force_regen and not missing_rand and os.path.exists(pool_path):
                print(f"  mut{pct:02d}%  all libs present — skip")
                continue

            # ── Load or generate pool
            if not force_regen and os.path.exists(pool_path):
                d           = np.load(pool_path)
                nuc_ids     = d["nuc_ids"]
                pool_scores = {key: d[f"scores_{key}"] for key in gt_keys}
                print(f"  mut{pct:02d}%  loaded pool  missing libs={missing_rand}")
            else:
                rng_lib = np.random.default_rng(k * 1000 + 200 + r_idx * 10)
                nuc_ids = generate_pool(wt_oh, pool_size, mc, rng_lib)
                nuc_ids = np.unique(nuc_ids, axis=0)   # deduplicate
                pool_scores = score_pool(nuc_ids, gt, gt_keys)
                save_npz(pool_path, nuc_ids, pool_scores, gt.edges, gt_keys)
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
                                 {key: pool_scores[key][idx] for key in gt_keys},
                                 gt.edges, gt_keys)

            del nuc_ids, pool_scores
            print(f"  mut{pct:02d}%  done  ({time.time()-t_r:.1f}s)")

        # ── Type1 / activity-balanced eval libs + pairwise structured lib
        generate_type1_activity_balanced(k, inst_dir, gt, out_base, gt_keys, primary_key, force=force_regen)
        generate_pairwise_lib(k, inst_dir, gt, gt_keys, primary_key, force=force_regen)

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
    _p.add_argument("--oracle", default=MRNA_ORACLE,
                    choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1"],
                    help="Oracle to score libraries with")
    _p.add_argument("--out_base", default=None,
                    help="Output directory. Defaults to outputs or outputs_<oracle>")
    _p.add_argument("--residualbind_dir", default=None,
                    help="Directory containing ResidualBind ensemble member*.pt checkpoints")
    _p.add_argument("--wt_activity", choices=["high", "low"], default="high",
                    help="VTS1 ResidualBind WT sequence context used for natural random-library distributions")
    _args = _p.parse_args()
    run_pipeline(n_instances=_args.n_instances, pool_size=_args.pool_size,
                 oracle_name=_args.oracle, out_base=_args.out_base,
                 residualbind_dir=_args.residualbind_dir,
                 wt_activity=_args.wt_activity)
