"""
mRNA_RBP/generate_vts1_mut10_library.py

Standalone library generator (instance_00 only): one VTS1-GCUGG pool at a
fixed 10% mutation rate, lib_size=20,000 -- the "other surrogate" condition
compared against the varied-mutation-rate (3-10 mut) pool in
plot_coefficients_compare.py. Mirrors generate_varied_mutrate_library.py's
conventions but with a single fixed mutation count instead of water-filled
bins across counts 3-10.

Output: mRNA_RBP/outputs/instance_00/vts1_mut10/lib_20000.npz
    nuc_ids       (N, L) uint8
    edges         (E, 2) int32   -- gt.edges
    scores_<key>  (N,)   float32 for each of the 4 GT keys
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    STEM_SIGMA, mut_count_for, generate_pool, score_pool, GT_KEYS,
)
from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth

MUT_PCT  = 10
LIB_SIZE = 20_000
SEED     = 0
OUT_PATH = os.path.join(_HERE, "outputs", "instance_00", "vts1_mut10", "lib_20000.npz")


def sample_unique(wt_oh: np.ndarray, mc: int, target_n: int, rng,
                  max_rounds: int = 20) -> np.ndarray:
    collected = None
    n_request = target_n
    for _ in range(max_rounds):
        pool = generate_pool(wt_oh, n_request, mc, rng)
        collected = pool if collected is None else np.concatenate([collected, pool], axis=0)
        collected = np.unique(collected, axis=0)
        if len(collected) >= target_n:
            return collected[:target_n]
        n_request = max(target_n - len(collected), 1) * 2
    return collected


def main():
    L  = len(VTS1_SEQ)
    mc = mut_count_for(MUT_PCT, L)
    gt = MrnaRbpGroundTruth(
        VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MOTIF_POSITIONS, seed=SEED, stem_sigma=STEM_SIGMA
    )
    wt_oh = gt.wt_one_hot()

    print(f"Sequence: {VTS1_SEQ}  (L={L})")
    print(f"Mutation rate: {MUT_PCT}%  -> mut_count={mc}")
    print(f"Lib size: {LIB_SIZE:,}\n")

    rng = np.random.default_rng(SEED * 1000 + 500)
    nuc_ids = sample_unique(wt_oh, mc, LIB_SIZE, rng)
    print(f"Got {len(nuc_ids):,} / {LIB_SIZE:,} unique sequences")

    print("Scoring pool...")
    scores = score_pool(nuc_ids, gt, GT_KEYS)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        nuc_ids=nuc_ids,
        edges=gt.edges,
        **{f"scores_{k}": scores[k] for k in GT_KEYS},
    )
    print(f"\nSaved {len(nuc_ids):,} sequences -> {OUT_PATH}")


if __name__ == "__main__":
    main()
