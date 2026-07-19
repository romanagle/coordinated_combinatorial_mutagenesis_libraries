"""
mRNA_RBP/generate_varied_mutrate_library.py

Standalone library generator (instance_00 only): one VTS1-GCUGG pool whose
sequences span mutation counts 3-10 (number of positions mutated away from
WT), rather than a fixed percentage like the mut05/mut10/mut25 pools used
elsewhere in this pipeline.

Output: mRNA_RBP/outputs/instance_00/varied_mutrate/pool.npz
    nuc_ids       (N, L) uint8
    mut_counts    (N,)   uint8   -- number of mutated positions per sequence
    edges         (E, 2) int32   -- gt.edges
    scores_<key>  (N,)   float32 for each of the 4 GT keys
"""

import itertools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    STEM_SIGMA,
    generate_pool, score_pool, GT_KEYS,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.sequence_configs import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
)


MUT_COUNTS    = list(range(3, 11))   # 3 to 10 mutated positions, inclusive
TARGET_TOTAL  = 200_000
REDUCED_TOTAL = 200_000
SEED          = 0                    # instance 0
OUT_PATH      = os.path.join(_HERE, "outputs", "instance_00", "varied_mutrate", "pool.npz")


def comb(n: int, k: int) -> int:
    """math.comb backport (this pipeline's interpreter is Python 3.7)."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def max_unique(L: int, m: int) -> int:
    """Distinct sequences reachable by mutating exactly m of L positions
    (3 alternative nucleotides per position)."""
    return comb(L, m) * (3 ** m)


def water_fill_targets(L: int, mut_counts: list, total: int) -> dict:
    """Equal per-mutation-count share of `total`, redistributing the
    shortfall from combinatorially-capped bins onto bins with headroom."""
    targets = {m: total // len(mut_counts) for m in mut_counts}
    for _ in range(len(mut_counts)):
        capped = {m: max_unique(L, m) for m in mut_counts if targets[m] > max_unique(L, m)}
        if not capped:
            break
        leftover = sum(targets[m] - cap for m, cap in capped.items())
        for m, cap in capped.items():
            targets[m] = cap
        open_bins = [m for m in mut_counts if m not in capped]
        if not open_bins:
            break
        share, rem = divmod(leftover, len(open_bins))
        for i, m in enumerate(open_bins):
            targets[m] += share + (1 if i < rem else 0)
    return targets


def enumerate_exact(wt_idx: np.ndarray, m: int) -> np.ndarray:
    """All C(L,m)*3^m sequences with exactly m mutated positions."""
    L = len(wt_idx)
    rows = []
    for combo in itertools.combinations(range(L), m):
        alt_choices = [[n for n in range(4) if n != int(wt_idx[p])] for p in combo]
        for alts in itertools.product(*alt_choices):
            seq = wt_idx.copy()
            for p, a in zip(combo, alts):
                seq[p] = a
            rows.append(seq)
    return np.stack(rows).astype(np.uint8)


def sample_unique(wt_oh: np.ndarray, m: int, target_n: int, rng,
                  max_rounds: int = 20) -> np.ndarray:
    """Random sampling + dedup until target_n unique sequences are reached
    (or repeated rounds stop yielding new ones)."""
    collected = None
    n_request = target_n
    for _ in range(max_rounds):
        pool = generate_pool(wt_oh, n_request, m, rng)
        collected = pool if collected is None else np.concatenate([collected, pool], axis=0)
        collected = np.unique(collected, axis=0)
        if len(collected) >= target_n:
            return collected[:target_n]
        n_request = max(target_n - len(collected), 1) * 2
    return collected


def build_pool(L: int, wt_oh: np.ndarray, total: int, seed: int):
    targets = water_fill_targets(L, MUT_COUNTS, total)
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
    rng = np.random.default_rng(seed * 10_000 + 9000)

    all_ids, all_counts = [], []
    for m in MUT_COUNTS:
        cap = max_unique(L, m)
        t0 = time.time()
        if targets[m] >= cap:
            seqs = enumerate_exact(wt_idx, m)
        else:
            seqs = sample_unique(wt_oh, m, targets[m], rng)
        all_ids.append(seqs)
        all_counts.append(np.full(len(seqs), m, dtype=np.uint8))
        print(f"    mut={m:2d}  target={targets[m]:>9,}  cap={cap:>14,}  "
              f"got={len(seqs):>9,}  ({time.time()-t0:.1f}s)")

    nuc_ids = np.concatenate(all_ids, axis=0)
    mut_counts = np.concatenate(all_counts, axis=0)
    return nuc_ids, mut_counts


def main(target_total: int = TARGET_TOTAL, reduced_total: int = REDUCED_TOTAL,
        out_path: str = OUT_PATH):
    L = len(VTS1_SEQ)
    gt = MrnaRbpGroundTruth(
        VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS, seed=SEED, stem_sigma=STEM_SIGMA
    )
    wt_oh = gt.wt_one_hot()

    print(f"Sequence: {VTS1_SEQ}  (L={L})")
    print(f"Mutation counts: {MUT_COUNTS[0]}-{MUT_COUNTS[-1]}")
    print(f"Target total: {target_total:,}\n")

    nuc_ids, mut_counts = build_pool(L, wt_oh, target_total, SEED)
    achieved = len(nuc_ids)
    print(f"\nAchieved {achieved:,} / {target_total:,} unique sequences "
          f"({achieved / target_total:.1%})")

    if achieved < 0.5 * target_total:
        print(f"Too much redundancy at {target_total:,} -- regenerating at "
              f"reduced target {reduced_total:,}\n")
        nuc_ids, mut_counts = build_pool(L, wt_oh, reduced_total, SEED)
        achieved = len(nuc_ids)
        print(f"\nAchieved {achieved:,} / {reduced_total:,} unique sequences")

    print("\nScoring pool...")
    scores = score_pool(nuc_ids, gt, GT_KEYS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        nuc_ids=nuc_ids,
        mut_counts=mut_counts,
        edges=gt.edges,
        **{f"scores_{k}": scores[k] for k in GT_KEYS},
    )
    print(f"\nSaved {len(nuc_ids):,} sequences -> {out_path}")

    vals, cnts = np.unique(mut_counts, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  mut={v:2d}: {c:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--target_total", type=int, default=TARGET_TOTAL)
    p.add_argument("--reduced_total", type=int, default=REDUCED_TOTAL)
    p.add_argument("--out_path", type=str, default=OUT_PATH)
    args = p.parse_args()
    main(target_total=args.target_total, reduced_total=args.reduced_total,
         out_path=args.out_path)
