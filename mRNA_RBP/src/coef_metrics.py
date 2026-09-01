"""mRNA_RBP/src/coef_metrics.py

Mean cosine similarity between a ground-truth/oracle coefficient view and a
trained surrogate's coefficients:

- additive (alpha, shape (L, 4)): cosine similarity computed per sequence
  position (the 4-dim row), averaged over positions where both rows are
  nonzero. Positions with an all-zero row on either side (e.g. stem
  positions in the synthetic GT, which have zero additive weight by
  construction) are excluded rather than contributing an undefined 0/0.
- pairwise (beta, one {(i,j): (4,4)} block per stem pair): cosine similarity
  computed per stem pair on the flattened 4x4 block, averaged over stem
  pairs where both blocks are nonzero.

Cosine similarity is scale-invariant (unaffected by any positive rescaling
of either matrix) but NOT sign-invariant, and MAVE-NN's GE nonlinearity has
a real gauge freedom that a raw sign flip can trigger spuriously: for
y = a*sigma(b*phi + c) + d with phi = alpha.x (+ pairwise term), the
substitution (alpha, J, b) -> (-alpha, -J, -b) reproduces identical
predictions -- the optimizer's choice of sign is arbitrary, not biological.
``compute_coefficient_similarity`` also reports a *sign-corrected* variant:
one global sign (applied jointly to both alpha and beta, since the gauge
flip negates the whole phi at once) chosen to maximize agreement, so a
surrogate that recovered the right structure up to this legitimate flip
isn't scored as if it learned the opposite direction.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _cosine(u: np.ndarray, v: np.ndarray) -> Optional[float]:
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return None
    return float(np.dot(u, v) / (nu * nv))


def mean_cosine_similarity_additive(alpha_gt: np.ndarray, alpha_sur: np.ndarray) -> dict:
    alpha_gt = np.asarray(alpha_gt, dtype=np.float64)
    alpha_sur = np.asarray(alpha_sur, dtype=np.float64)
    if alpha_gt.shape != alpha_sur.shape:
        raise ValueError(f"alpha shape mismatch: {alpha_gt.shape} vs {alpha_sur.shape}")
    sims = [c for c in (_cosine(alpha_gt[i], alpha_sur[i]) for i in range(alpha_gt.shape[0]))
            if c is not None]
    return {
        "mean_cosine_similarity_additive": float(np.mean(sims)) if sims else None,
        "n_positions_used": len(sims),
        "n_positions_total": int(alpha_gt.shape[0]),
    }


def mean_cosine_similarity_pairwise(beta_gt: dict, beta_sur: dict, stem_pairs: list) -> dict:
    sims = []
    for (i, j) in stem_pairs:
        key = (int(i), int(j))
        if key not in beta_gt or key not in beta_sur:
            continue
        c = _cosine(beta_gt[key], beta_sur[key])
        if c is not None:
            sims.append(c)
    return {
        "mean_cosine_similarity_pairwise": float(np.mean(sims)) if sims else None,
        "n_pairs_used": len(sims),
        "n_pairs_total": len(stem_pairs),
    }


def compute_coefficient_similarity(alpha_gt: np.ndarray, alpha_sur: np.ndarray,
                                   beta_gt: dict, beta_sur: dict, stem_pairs: list) -> dict:
    out = {}
    out.update(mean_cosine_similarity_additive(alpha_gt, alpha_sur))
    out.update(mean_cosine_similarity_pairwise(beta_gt, beta_sur, stem_pairs))

    additive_raw = out["mean_cosine_similarity_additive"]
    pairwise_raw = out["mean_cosine_similarity_pairwise"]
    # Decide the global sign by pooling every individual position/pair-level
    # cosine term (not by averaging the two matrix-type means unweighted) --
    # equivalent to weighting each matrix type by how many independent
    # comparisons back it (n_positions_used vs n_pairs_used). Unweighted
    # would let e.g. a 9-pair pairwise average outvote a 48-position
    # additive average; weighting by count means the sign decision reflects
    # which orientation more of the *underlying* comparisons actually agree
    # with, not just which matrix type happened to have the bigger mean.
    weighted_sum = 0.0
    if additive_raw is not None:
        weighted_sum += additive_raw * out["n_positions_used"]
    if pairwise_raw is not None:
        weighted_sum += pairwise_raw * out["n_pairs_used"]
    sign = 1 if weighted_sum >= 0 else -1

    out["sign_correction"] = sign
    out["mean_cosine_similarity_additive_sign_corrected"] = (
        sign * additive_raw if additive_raw is not None else None
    )
    out["mean_cosine_similarity_pairwise_sign_corrected"] = (
        sign * pairwise_raw if pairwise_raw is not None else None
    )
    return out
