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
# High-WT sequence: RNAcompete-2013 RNCMPT00111 test probe 107676. Selected
# because its GCUGG motif is fully unpaired in its own real RNAfold MFE
# structure (verified 2026-08-31: `echo AAAAAAGACGAGAGCGACACCGGCUGGCCCGACGG
# AAAAAA | RNAfold` -> dot-bracket below, base pairs (19,34)/(20,33)/(21,32),
# zero overlap with motif positions 22-26). Raw VTS1 ResidualBind ensemble
# WT score: 8.825594.
#
# Replaces an earlier high-WT choice
# (AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA) whose declared stem pairs were
# an unsourced hand-entered constant that, independently re-folded with
# RNAfold, both failed to reproduce (5 declared pairs vs. 10 in the real MFE
# fold) and overlapped the motif at 4 of its 5 positions -- the opposite of
# the intended non-overlapping design and lower raw activity (4.545361).
VTS1_HIGH_ACTIVITY_SEQ = "AAAAAAGACGAGAGCGACACCGGCUGGCCCGACGGAAAAAA"
VTS1_HIGH_ACTIVITY_DOTBRACKET = "...................(((..........)))......"
VTS1_HIGH_ACTIVITY_STEM_PAIRS = [(19, 34), (20, 33), (21, 32)]
VTS1_HIGH_ACTIVITY_MOTIF_POSITIONS = [22, 23, 24, 25, 26]

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


# HuR (ELAVL1, RNCMPT00112) and QKI (RNCMPT00047) natural probes, selected the
# same way as MSI1/VTS1 by rbp/scripts/select_natural_wt.py: highest/lowest
# measured RNAcompete-2013 intensity among probes containing exactly one
# occurrence of the RBP's core motif, within mean +/- 2 s.d. of the finite
# train/valid/test distribution. Motif choice: HuR = "AUUUA" (canonical ARE
# pentamer core); QKI = "ACUAAC" (QRE core half-site) -- both are the
# well-established literature/RNAcompete-derived cores, not independently
# re-derived from this dataset. Stem pairs are user-supplied RNAFold
# dot-bracket structures, parsed to zero-based pairs.
#
# HuR high  db: ((((..(((...................)))..))))....
# HuR low   db: ....(((((........)).)))((((.....)))).....
# QKI high  db: ...(((......)))(((..........)))..........
# QKI low   db: (((((.....)))))..........................
#
# Note: in HUR_LOW, the AUUUA motif (34-38) partially overlaps stem pairs
# (24,34) and (23,35) -- i.e. RNAFold predicts part of the ARE motif is
# base-paired (structurally occluded) in this probe, plausibly explaining why
# it was selected as the *low*-activity construct (HuR binds single-stranded
# AU-rich elements).

HUR_HIGH_ACTIVITY_SEQ = "AAGGGGUACACAUCAACGACAAUUUAGCGUAAACUUUGUAA"
HUR_HIGH_ACTIVITY_MOTIF_POSITIONS = [21, 22, 23, 24, 25]
HUR_HIGH_ACTIVITY_STEM_PAIRS = [(0, 36), (1, 35), (2, 34), (3, 33), (6, 30), (7, 29), (8, 28)]

HUR_LOW_ACTIVITY_SEQ = "AAAAGACAGGAACUGGGCUCGUCAUAGGAACGCUAUUUAAA"
HUR_LOW_ACTIVITY_MOTIF_POSITIONS = [34, 35, 36, 37, 38]
HUR_LOW_ACTIVITY_STEM_PAIRS = [(4, 22), (5, 21), (6, 20), (7, 18), (8, 17), (23, 35), (24, 34), (25, 33), (26, 32)]

QKI_HIGH_ACTIVITY_SEQ = "AAAAGAGACUAAUCUGCUACUAACCGACAGCUGACAUAAAA"
QKI_HIGH_ACTIVITY_MOTIF_POSITIONS = [18, 19, 20, 21, 22, 23]
QKI_HIGH_ACTIVITY_STEM_PAIRS = [(3, 14), (4, 13), (5, 12), (15, 30), (16, 29), (17, 28)]

QKI_LOW_ACTIVITY_SEQ = "AAAAGAAGGCCUUUUGAACUGCGCUCCACUAACUGAUGAAA"
QKI_LOW_ACTIVITY_MOTIF_POSITIONS = [27, 28, 29, 30, 31, 32]
QKI_LOW_ACTIVITY_STEM_PAIRS = [(0, 14), (1, 13), (2, 12), (3, 11), (4, 10)]


def hur_sequence_config(
    wt_activity: Literal["high", "low"] = "high",
) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Return the HuR (RNCMPT00112) natural WT sequence/structure config."""
    if wt_activity == "high":
        return (HUR_HIGH_ACTIVITY_SEQ, list(HUR_HIGH_ACTIVITY_STEM_PAIRS), list(HUR_HIGH_ACTIVITY_MOTIF_POSITIONS))
    if wt_activity == "low":
        return (HUR_LOW_ACTIVITY_SEQ, list(HUR_LOW_ACTIVITY_STEM_PAIRS), list(HUR_LOW_ACTIVITY_MOTIF_POSITIONS))
    raise ValueError(f"Unsupported HuR wt_activity {wt_activity!r}; expected 'high' or 'low'")


def qki_sequence_config(
    wt_activity: Literal["high", "low"] = "high",
) -> tuple[str, list[tuple[int, int]], list[int]]:
    """Return the QKI (RNCMPT00047) natural WT sequence/structure config."""
    if wt_activity == "high":
        return (QKI_HIGH_ACTIVITY_SEQ, list(QKI_HIGH_ACTIVITY_STEM_PAIRS), list(QKI_HIGH_ACTIVITY_MOTIF_POSITIONS))
    if wt_activity == "low":
        return (QKI_LOW_ACTIVITY_SEQ, list(QKI_LOW_ACTIVITY_STEM_PAIRS), list(QKI_LOW_ACTIVITY_MOTIF_POSITIONS))
    raise ValueError(f"Unsupported QKI wt_activity {wt_activity!r}; expected 'high' or 'low'")


# Twister ribozyme (Kobori & Yokobayashi 2016, Angew. Chem. Int. Ed. 55, 10354;
# PMC5113685) -- Osa-1-4 construct, doped region positions 7-54 (48 nt, 0-based
# array index = paper position - 7). Real deep-sequencing dataset: WT + all 144
# single + 10,152 double mutants (see parse_twister_data.py). Unlike
# MSI1/VTS1/HuR/QKI there is no live black-box oracle to query arbitrary
# sequences -- "deep squid" here (train_twister_deepsquid.py) is trained
# directly on this real data and *is* the only scorable oracle.
#
# Stem pairs: derived from the user-supplied dot-bracket structure
# "((((((..((((.......((((........)))).....))))....))))))" (1-based
# positions 1-54; positions 1-6 are constant flanking sequence, outside the
# doped region, so only P2 and P4 -- both fully inside 7-54 -- are
# representable as intra-array pairs. Plus one pseudoknot pair stated
# explicitly in the main text ("the Watson-Crick base pair C14-G30", T2).
# Cross-checked: WT nt at position 21/34 is C/G (matches Figure 1A's labeled
# "21 C-G 34"); WT nt at position 14/30 is C/G (matches the main-text quote).
# T1/T2 beyond that one pair, and P1 (half outside the doped region), are not
# declared -- the pairwise gpmap still learns the full unrestricted (L,L,4,4)
# coupling tensor regardless; these edges only control highlighting in the
# coefficient/stem-block comparison plots.
TWISTER_SEQ = "AACACUGCCAAUGCCGGUCCCAAGCCCGGAUAAAAGUGGAGGGGGCGG"
TWISTER_STEM_PAIRS = [
    (2, 37), (3, 36), (4, 35), (5, 34),      # P2  (paper pos 9-12 / 41-44)
    (13, 28), (14, 27), (15, 26), (16, 25),  # P4  (paper pos 20-23 / 32-35)
    (7, 23),                                 # T2 pseudoknot (paper pos 14, 30)
]
TWISTER_MOTIF_POSITIONS: list[int] = []  # no RBP motif -- this is a ribozyme


def twister_sequence_config() -> tuple[str, list[tuple[int, int]], list[int]]:
    """Return the Twister ribozyme WT sequence/structure config."""
    return TWISTER_SEQ, list(TWISTER_STEM_PAIRS), list(TWISTER_MOTIF_POSITIONS)


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


def generate_type3_nuc_ids(seq: str):
    """Return the saturated additive-plus-pairwise evaluation library.

    The library contains every non-WT single substitution and every pair of
    non-WT substitutions.  It intentionally contains no variants with three
    or more mutations: those probe higher-order sequence effects rather than
    saturated additive or pairwise coverage.
    """
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
    return np.stack(rows).astype(np.uint8), np.array(labels, dtype=np.int32)
