"""Shared sequence/structure definitions for RBP-specific mRNA experiments."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

try:
    from typing import Literal
except ImportError:  # Python 3.7 in the SQUID/MAVE-NN environment
    from typing_extensions import Literal

import numpy as np


NUCS = ("A", "C", "G", "U")
NUC_TO_IDX = {n: i for i, n in enumerate(NUCS)}

MSI1_SEQ = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS = [(8, 30), (9, 29), (10, 28), (11, 27), (12, 26), (13, 25), (14, 24), (15, 23)]
MSI1_MOTIF_POSITIONS = [17, 18, 19, 20, 21]

# VTS1 natural probes used to build the ResidualBind random-library region-class
# figures in outputs/ground_truth_collections/ResidualBind oracle VTS1/figures.
#
# High-WT figure:
#   rand_lib_dist_vts1_oracle_region_classes.png
# Source cache:
#   vts1_natural_random_library_scores.npz
# Raw VTS1 ResidualBind ensemble WT score: 4.545361.
VTS1_HIGH_ACTIVITY_SEQ = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
VTS1_HIGH_ACTIVITY_DOTBRACKET = "..............(((....)))((.....))........"
VTS1_HIGH_ACTIVITY_STEM_PAIRS = [(16, 21), (15, 22), (14, 23), (25, 31), (24, 32)]
VTS1_HIGH_ACTIVITY_MOTIF_POSITIONS = [20, 21, 22, 23, 24]

# Low-WT figure:
#   rand_lib_dist_vts1_oracle_region_classes_low_wt.png
# Source cache:
#   vts1_natural_random_library_scores_low_wt.npz
# Raw VTS1 ResidualBind ensemble WT score: 0.103703.
VTS1_LOW_ACTIVITY_SEQ = "AAAAGAUGGCUAUGCGACCCGCUGGAACUAGUAAGUGAAAA"
VTS1_LOW_ACTIVITY_DOTBRACKET = ".........(((.(((...))))))................"
VTS1_LOW_ACTIVITY_STEM_PAIRS = [(9, 24), (10, 23), (11, 22), (13, 21), (14, 20), (15, 19)]
VTS1_LOW_ACTIVITY_MOTIF_POSITIONS = [20, 21, 22, 23, 24]

# Backward-compatible aliases: existing VTS1 pipeline runs used the high-WT
# natural probe unless explicitly staging the low-WT distribution cache.
VTS1_SEQ = VTS1_HIGH_ACTIVITY_SEQ
VTS1_STEM_PAIRS = VTS1_HIGH_ACTIVITY_STEM_PAIRS
VTS1_MOTIF_POSITIONS = VTS1_HIGH_ACTIVITY_MOTIF_POSITIONS


def vts1_sequence_config(
    wt_activity: Literal["high", "low"] = "high",
) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Return the remembered VTS1 ResidualBind WT sequence/structure config."""
    if wt_activity == "high":
        return (
            VTS1_HIGH_ACTIVITY_SEQ,
            list(VTS1_HIGH_ACTIVITY_STEM_PAIRS),
            list(VTS1_HIGH_ACTIVITY_MOTIF_POSITIONS),
        )
    if wt_activity == "low":
        return (
            VTS1_LOW_ACTIVITY_SEQ,
            list(VTS1_LOW_ACTIVITY_STEM_PAIRS),
            list(VTS1_LOW_ACTIVITY_MOTIF_POSITIONS),
        )
    raise ValueError(f"Unsupported VTS1 wt_activity {wt_activity!r}; expected 'high' or 'low'")


def ids_to_seq(nuc_ids: Sequence[int]) -> str:
    return "".join(NUCS[int(i)] for i in nuc_ids)


def wt_ids(seq: str) -> np.ndarray:
    return np.array([NUC_TO_IDX[c] for c in seq.upper()], dtype=np.uint8)


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    num = 1
    for i in range(k):
        num = num * (n - i) // (i + 1)
    return num


def max_exact_mutants(length: int, mut_count: int) -> int:
    return comb(length, mut_count) * (3 ** mut_count)


def enumerate_exact_mutants(seq: str, mut_count: int) -> np.ndarray:
    wt = wt_ids(seq)
    rows = []
    for positions in itertools.combinations(range(len(wt)), mut_count):
        alt_choices = [[n for n in range(4) if n != int(wt[p])] for p in positions]
        for alts in itertools.product(*alt_choices):
            row = wt.copy()
            for p, a in zip(positions, alts):
                row[p] = a
            rows.append(row)
    if not rows:
        return wt[None, :].copy()
    return np.stack(rows).astype(np.uint8)


def sample_exact_mutants(
    seq: str,
    mut_count: int,
    target_n: int,
    seed: int,
    max_rounds: int = 20,
) -> np.ndarray:
    """Return up to target_n unique sequences with exactly mut_count mutations."""
    cap = max_exact_mutants(len(seq), mut_count)
    if cap <= target_n:
        return enumerate_exact_mutants(seq, mut_count)

    rng = np.random.default_rng(seed)
    wt = wt_ids(seq)
    collected = None
    request_n = target_n
    for _ in range(max_rounds):
        rows = np.tile(wt[None, :], (request_n, 1))
        noise = rng.random((request_n, len(wt)))
        positions = np.argpartition(noise, mut_count, axis=1)[:, :mut_count]
        wt_at = wt[positions.ravel()]
        new_nucs = rng.integers(0, 3, size=request_n * mut_count, dtype=np.uint8)
        new_nucs = np.where(new_nucs >= wt_at, new_nucs + 1, new_nucs).astype(np.uint8)
        rows[np.repeat(np.arange(request_n), mut_count), positions.ravel()] = new_nucs

        collected = rows if collected is None else np.concatenate([collected, rows], axis=0)
        collected = np.unique(collected, axis=0)
        if len(collected) >= target_n:
            return collected[:target_n].astype(np.uint8)
        request_n = max(target_n - len(collected), 1) * 2
    return collected.astype(np.uint8)


def generate_pairwise_nuc_ids(seq: str, stem_pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    wt = wt_ids(seq)
    rows = []
    for i, j in stem_pairs:
        for a in range(4):
            for b in range(4):
                row = wt.copy()
                row[i] = a
                row[j] = b
                rows.append(row)
    if not rows:
        return np.empty((0, len(wt)), dtype=np.uint8)
    return np.stack(rows).astype(np.uint8)


def generate_ssm_nuc_ids(seq: str) -> np.ndarray:
    wt = wt_ids(seq)
    rows = []
    for pos in range(len(wt)):
        for nuc in range(4):
            if nuc == wt[pos]:
                continue
            row = wt.copy()
            row[pos] = nuc
            rows.append(row)
    return np.stack(rows).astype(np.uint8)


def generate_type3_nuc_ids(seq: str, n_3mut: int = 4, seed: int = 42):
    wt = wt_ids(seq)
    rows, labels = [], []
    for i in range(len(wt)):
        for a in range(4):
            if a == wt[i]:
                continue
            row = wt.copy()
            row[i] = a
            rows.append(row)
            labels.append(1)
    for i in range(len(wt)):
        for j in range(i + 1, len(wt)):
            for a in range(4):
                if a == wt[i]:
                    continue
                for b in range(4):
                    if b == wt[j]:
                        continue
                    row = wt.copy()
                    row[i] = a
                    row[j] = b
                    rows.append(row)
                    labels.append(2)
    rng = np.random.default_rng(seed)
    for _ in range(n_3mut):
        row = wt.copy()
        positions = rng.choice(len(wt), size=3, replace=False)
        for pos in positions:
            alt = int(rng.integers(0, 3))
            if alt >= int(wt[pos]):
                alt += 1
            row[pos] = alt
        rows.append(row)
        labels.append(3)
    return np.stack(rows).astype(np.uint8), np.array(labels, dtype=np.int32)
