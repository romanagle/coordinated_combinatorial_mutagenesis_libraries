"""
mRNA_RBP/scripts/pipeline/generate_varied_mutrate_library.py

Layer-2 "deep squid" pool generator (instance_00 only): one pool per real
ResidualBind oracle (MSI1, VTS1, HuR, QKI, ...) whose sequences span
mutation counts 3-10 (number of positions mutated away from WT), rather
than a fixed percentage like the mut05/mut10/mut25 pools used elsewhere in
this pipeline. This wide mutation-count spread is what makes the resulting
MAVE-NN surrogate (trained in train_surrogate_varied_mutrate.py) a
reasonable general-purpose stand-in ("deep squid") for the real oracle.

Scored with the live oracle. Sequence/stem-pairs/motif are resolved
generically via oracles.sequence_config_for_oracle -- MSI1 uses the fixed
synthetic sequence; VTS1/HuR/QKI use their natural-probe high/low WT
configs (--wt_activity), never a hardcoded MSI1-vs-VTS1 binary.

Output: <out_base>/instance_00/varied_mutrate/pool.npz
    nuc_ids       (N, L) uint8
    mut_counts    (N,)   uint8   -- number of mutated positions per sequence
    edges         (E, 2) int32   -- oracle.edges
    scores_<key>  (N,)   float32 -- oracle's primary score key

Usage (squid env -- needs a working torch install; see LD_LIBRARY_PATH note
in CLAUDE.md/oracles.py for the nvidia-pip-libs-before-system-cuda fix):
    python mRNA_RBP/scripts/pipeline/generate_varied_mutrate_library.py --oracle residualbind_ensemble
    python mRNA_RBP/scripts/pipeline/generate_varied_mutrate_library.py --oracle vts1_residualbind
"""

import itertools
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.scripts.pipeline.generate_libraries import (
    STEM_SIGMA,
    generate_pool, score_pool,
)
from mRNA_RBP.src.oracles import (
    build_oracle, default_output_base, normalize_oracle_name, primary_gt_key,
    oracle_uses_wt_activity, sequence_config_for_oracle,
)


MUT_COUNTS    = list(range(3, 11))   # 3 to 10 mutated positions, inclusive
TARGET_TOTAL  = 200_000
REDUCED_TOTAL = 200_000
SEED          = 0                    # instance 0


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
              f"got={len(seqs):>9,}  ({time.time()-t0:.1f}s)", flush=True)

    nuc_ids = np.concatenate(all_ids, axis=0)
    mut_counts = np.concatenate(all_counts, axis=0)
    return nuc_ids, mut_counts


def main(oracle_name: str, target_total: int = TARGET_TOTAL,
        reduced_total: int = REDUCED_TOTAL,
        out_base: str = None, out_path: str = None,
        residualbind_dir: str = None, wt_activity: str = "high"):
    oracle_name = normalize_oracle_name(oracle_name)
    seq, stem_pairs, motif_positions = sequence_config_for_oracle(oracle_name, wt_activity)
    score_key    = primary_gt_key(oracle_name)

    out_base = out_base or default_output_base(_HERE, oracle_name, wt_activity)
    out_path = out_path or os.path.join(out_base, "instance_00", "varied_mutrate", "pool.npz")

    L = len(seq)
    print(f"Oracle: {oracle_name}  |  score key: {score_key}")
    if oracle_uses_wt_activity(oracle_name):
        print(f"WT activity context: {wt_activity}")
    print(f"Sequence: {seq}  (L={L})")
    print(f"Mutation counts: {MUT_COUNTS[0]}-{MUT_COUNTS[-1]}")
    print(f"Target total: {target_total:,}\n", flush=True)

    print("[load] building oracle...", flush=True)
    oracle = build_oracle(
        oracle_name, seq=seq, stem_pairs=stem_pairs, motif_positions=motif_positions,
        seed=SEED, stem_sigma=STEM_SIGMA, residualbind_dir=residualbind_dir,
        wt_activity=wt_activity,
    )
    wt_oh = oracle.wt_one_hot()

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

    print("\nScoring pool...", flush=True)
    scores = score_pool(nuc_ids, oracle, [score_key])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        nuc_ids=nuc_ids,
        mut_counts=mut_counts,
        edges=oracle.edges,
        wt_seq=np.array(seq),
        **{f"scores_{score_key}": scores[score_key]},
    )
    print(f"\nSaved {len(nuc_ids):,} sequences -> {out_path}")

    vals, cnts = np.unique(mut_counts, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  mut={v:2d}: {c:,}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", required=True,
                    help="residualbind_ensemble/msi1 or vts1_residualbind/vts1")
    p.add_argument("--target_total", type=int, default=TARGET_TOTAL)
    p.add_argument("--reduced_total", type=int, default=REDUCED_TOTAL)
    p.add_argument("--out_base", default=None)
    p.add_argument("--out_path", default=None)
    p.add_argument("--residualbind_dir", default=None)
    p.add_argument("--wt_activity", choices=["high", "low"], default="high",
                    help="Natural-probe WT sequence context (VTS1/HuR/QKI)")
    args = p.parse_args()
    main(oracle_name=args.oracle, target_total=args.target_total,
         reduced_total=args.reduced_total, out_base=args.out_base,
         out_path=args.out_path, residualbind_dir=args.residualbind_dir,
         wt_activity=args.wt_activity)
