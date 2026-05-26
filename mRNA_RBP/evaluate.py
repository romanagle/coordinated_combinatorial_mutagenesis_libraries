"""
mRNA_RBP/evaluate.py

Structured evaluation datasets mirroring demo_additive_pairwise (3).ipynb:

  make_additive_dataset  – k-mutant seqs restricted to non-stem positions
  make_pairwise_dataset  – all 16 nucleotide combos per declared stem pair
  make_ssm_deltas        – (4, L) single-site delta weight matrix from GT
  predict_ssm            – additive SSM prediction for one-hot sequences
"""

import numpy as np


def make_additive_dataset(wt_oh, gt, ks=(2, 3, 5), n_per_k=500, seed=0):
    """k-mutant sequences with mutations drawn only from non-stem positions.

    Because the GT has no pairwise interaction outside stem positions, activity
    at non-stem sites is a pure sum of additive terms.  A surrogate that
    correctly learned additive structure should track GT well at all k; deviation
    at higher k reveals spurious pairwise contamination in non-stem regions.

    Parameters
    ----------
    wt_oh   : (L, 4) float32 one-hot WT sequence
    gt      : MrnaRbpGroundTruth instance
    ks      : tuple of mutation counts to test
    n_per_k : sequences per k value
    seed    : random seed

    Returns
    -------
    x        : (N, L, 4) float32
    y_gt     : (N,) float32  nonlin_additive_pairwise GT scores
    k_labels : (N,) int      mutation count per sequence
    """
    L        = wt_oh.shape[0]
    stem_set = set(p for pair in gt.stem_pairs for p in pair)
    non_stem = [i for i in range(L) if i not in stem_set]
    rng      = np.random.default_rng(seed)
    wt_idx   = np.argmax(wt_oh, axis=1)

    seqs, labels = [], []
    for k in ks:
        k_eff = min(k, len(non_stem))
        for _ in range(n_per_k):
            seq = wt_idx.copy()
            pos = rng.choice(non_stem, size=k_eff, replace=False)
            for p in pos:
                alts   = [n for n in range(4) if n != int(wt_idx[p])]
                seq[p] = int(rng.choice(alts))
            seqs.append(seq)
            labels.append(k)

    nuc_ids  = np.stack(seqs).astype(np.uint8)
    x        = np.eye(4, dtype=np.float32)[nuc_ids]
    scores   = gt.score_all(x)
    y_gt     = scores["nonlin_additive_pairwise"]
    return x, y_gt, np.array(labels, dtype=int)


def make_pairwise_dataset(wt_oh, gt):
    """All 16 nucleotide combinations for each declared stem pair.

    For each stem pair (i, j), enumerate all 4×4 combinations while holding
    every other position at WT.  The 4×4 heatmaps reveal the full epistatic
    landscape: the surrogate should reproduce the GT pattern (low activity for
    non-canonical pairs, partial recovery for compensatory mutations).

    Parameters
    ----------
    wt_oh : (L, 4) float32 one-hot WT sequence
    gt    : MrnaRbpGroundTruth instance

    Returns
    -------
    x           : (N, L, 4) float32   N = 16 × len(stem_pairs)
    y_gt        : (N,) float32
    pair_labels : (N,) int   index into gt.stem_pairs
    nuc_combos  : list of (nuc_i, nuc_j) tuples  length N
    """
    L      = wt_oh.shape[0]
    wt_idx = np.argmax(wt_oh, axis=1)

    seqs, plabels, combos = [], [], []
    for pidx, (si, sj) in enumerate(gt.stem_pairs):
        for ni in range(4):
            for nj in range(4):
                seq     = wt_idx.copy()
                seq[si] = ni
                seq[sj] = nj
                seqs.append(seq)
                plabels.append(pidx)
                combos.append((ni, nj))

    nuc_ids     = np.stack(seqs).astype(np.uint8)
    x           = np.eye(4, dtype=np.float32)[nuc_ids]
    scores      = gt.score_all(x)
    y_gt        = scores["nonlin_additive_pairwise"]
    return x, y_gt, np.array(plabels, dtype=int), combos


def make_ssm_deltas(wt_oh, gt):
    """Build (4, L) additive delta weight matrix from single-site GT scores.

    Each cell [nuc, pos] = GT score of the sequence with position pos mutated
    to nucleotide nuc (all other positions WT).  WT nucleotide at each position
    has delta = 0 by GT construction.

    Returns
    -------
    delta_W : (4, L) float64
    """
    L      = wt_oh.shape[0]
    wt_idx = np.argmax(wt_oh, axis=1)

    seqs, positions, nucs = [], [], []
    for pos in range(L):
        for nuc in range(4):
            seq     = wt_idx.copy()
            seq[pos] = nuc
            seqs.append(seq)
            positions.append(pos)
            nucs.append(nuc)

    nuc_ids = np.stack(seqs).astype(np.uint8)
    x       = np.eye(4, dtype=np.float32)[nuc_ids]
    scores  = gt.score_all(x)["nonlin_additive_pairwise"]

    delta_W = np.zeros((4, L), dtype=np.float64)
    for idx, (pos, nuc) in enumerate(zip(positions, nucs)):
        delta_W[nuc, pos] = float(scores[idx])

    return delta_W


def predict_ssm(x, delta_ssm, wt_activity=0.0):
    """Additive SSM prediction for (N, L, 4) one-hot sequences.

    Approximates activity as a sum of per-position single-mutant effects,
    ignoring pairwise interactions.  Serves as a baseline that a pairwise
    surrogate should outperform on epistatic sequences.

    Parameters
    ----------
    x           : (N, L, 4) float32 one-hot
    delta_ssm   : (4, L) float64 from make_ssm_deltas
    wt_activity : float  WT baseline (0.0 by GT convention)

    Returns
    -------
    y_hat : (N,) float64
    """
    return np.einsum("nla,al->n", x.astype(np.float64), delta_ssm) + wt_activity
