"""
ground_truth.py – Ground-truth energy model functions.

Covers:
  - Additive / pairwise / Potts affinity computations (noWT parameterisation)
  - Global nonlinearity wrappers
  - Score computation over libraries
  - Library uniformisation and curated eval-lib generation
"""

import os
import numpy as np
from typing import Optional

from seq_utils import rna_to_one_hot, onehot_to_seq, write_eval_library_txt


# ---------------------------------------------------------------------------
# Nonlinearity helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1, temperature: float = 1.0) -> np.ndarray:
    t = max(float(temperature), 1e-12)
    z = x / t
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _mixture_of_tanh(
    s: np.ndarray,
    alphas: np.ndarray,              # (K,)
    betas: np.ndarray,               # (K,)
    mix_logits: np.ndarray,          # (K,)
    temperature: float = 1.0,
    mix_weights: Optional[np.ndarray] = None,  # (K,) raw signed weights; overrides softmax
) -> np.ndarray:
    """g(s) = sum_k w_k * tanh(alpha_k * s + beta_k).  Returns (N,1).

    If mix_weights is provided it is used directly (allowing negative values and
    therefore a non-monotone mixture).  Otherwise w_k = softmax(mix_logits /
    temperature), which is always positive and therefore always monotone.
    """
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    elif s.ndim == 2 and s.shape[1] != 1:
        raise ValueError(f"s must be (N,) or (N,1); got {s.shape}")

    alphas     = np.asarray(alphas,     dtype=float).reshape(1, -1)
    betas      = np.asarray(betas,      dtype=float).reshape(1, -1)
    mix_logits = np.asarray(mix_logits, dtype=float).reshape(-1)

    if mix_weights is not None:
        pi = np.asarray(mix_weights, dtype=float).reshape(1, -1)
    else:
        pi = _softmax(mix_logits, axis=-1, temperature=temperature).reshape(1, -1)

    comps = np.tanh(s * alphas + betas)   # (N,K)
    return (comps * pi).sum(axis=-1, keepdims=True)   # (N,1)


# ---------------------------------------------------------------------------
# L1 prox
# ---------------------------------------------------------------------------

def init_mix_tanh_nonlin(rng, n_components=50):
    """Sample a random n_components-mixture-of-tanh nonlinearity.

    Matches the overparameterised GE nonlinearity used in the paper:
        g(s) = sum_k w_k * tanh(alpha_k * s + beta_k)
    where w_k = softmax(mix_logits).

    Alpha range is kept moderate ([0.3, 1.5]) so that components are not
    excessively saturating for unit-std inputs — avoids collapsing most of
    the library to a single value near ±1.  Betas are drawn uniformly over
    [-4, 4] to stagger inflection points across the full input range so
    different components activate at different energy levels.

    Returns a dict suitable for nonlin_name='mix_tanh'.
    """
    mix_alphas = rng.uniform(0.5, 2.0,  size=n_components)
    # Restrict betas to [-2, 2] so every component is in the "active" input range
    # for unit-std energies — components with |beta| >> 2 are saturated for nearly
    # all inputs and contribute only a constant, killing output variance.
    mix_betas  = rng.uniform(-2.0, 2.0, size=n_components)
    mix_logits = rng.normal(0.0, 1.0,   size=n_components)  # kept for API compat

    # Asymmetric signed weights: positive components normalised to sum +0.75,
    # negative components to sum -0.25.  The asymmetry prevents full cancellation
    # (which collapsed std to ~0.02 with symmetric L1-normalisation) while keeping
    # negative components for non-monotone structure.
    raw_weights = rng.normal(0.0, 1.0, size=n_components)
    pos_mask    = raw_weights >= 0
    neg_mask    = ~pos_mask
    mix_weights = np.zeros(n_components)
    pos_sum = raw_weights[pos_mask].sum()
    neg_sum = np.abs(raw_weights[neg_mask].sum())
    if pos_sum > 0:
        mix_weights[pos_mask] = raw_weights[pos_mask] / pos_sum * 0.75
    if neg_sum > 0:
        mix_weights[neg_mask] = raw_weights[neg_mask] / neg_sum * 0.25  # negative

    return dict(
        mix_alphas=mix_alphas.tolist(),
        mix_betas=mix_betas.tolist(),
        mix_logits=mix_logits.tolist(),
        mix_weights=mix_weights.tolist(),   # overrides softmax in _mixture_of_tanh
        mix_temperature=1.0,
    )


def init_tanh_nonlin(rng, alpha_loc=1.0, alpha_scale=0.5, beta_scale=1.0, out_bias_scale=0.5):
    """Sample random tanh nonlinearity coefficients.

    Randomising these prevents the nonlinearity from always squashing values
    near zero: tanh_beta shifts the inflection point away from 0, and
    out_bias adds a vertical offset so the output range is not centred at 0.

    Returns a dict suitable for use as nonlin_kwargs with nonlin_name='tanh'.
    """
    tanh_alpha = float(np.abs(rng.normal(alpha_loc, alpha_scale))) + 0.1  # always positive
    tanh_beta  = float(rng.normal(0.0, beta_scale))
    out_bias   = float(rng.normal(0.0, out_bias_scale))
    return dict(tanh_alpha=tanh_alpha, tanh_beta=tanh_beta, out_scale=1.0, out_bias=out_bias)


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Prox operator for L1: argmin_z 0.5||z-x||^2 + lam||z||_1."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


# ---------------------------------------------------------------------------
# Additive (noWT) model
# ---------------------------------------------------------------------------

def init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.0, bias=0.0):
    """Initialise additive weights for non-WT nucleotides.

    Returns: W_mut (L,3), mut_index_map (L,3), bias (float).
    """
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)  # (L,)

    mut_index_map = np.zeros((L, 3), dtype=int)
    for i in range(L):
        non_wt = [n for n in range(4) if n != wt_idx[i]]
        mut_index_map[i] = non_wt

    W_mut = rng.normal(0.0, sigma, size=(L, 3)).astype(np.float32)
    if l1_w > 0:
        W_mut = soft_threshold(W_mut, l1_w)

    return W_mut, mut_index_map, float(bias)


def additive_affinity_noWT(x_mut, W_mut, mut_index_map, b=0.0):
    """
    x_mut: (N,L,4), W_mut: (L,3), mut_index_map: (L,3)  →  (N,1)
    """
    N, L, _ = x_mut.shape
    s = np.zeros(N, dtype=np.float32)
    for k in range(3):
        nuc_idx = mut_index_map[:, k]   # (L,)
        Xk = x_mut[np.arange(N)[:, None], np.arange(L)[None, :], nuc_idx[None, :]]
        s += np.einsum("nl,l->n", Xk, W_mut[:, k])
    return (s + b).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Adjacent pairwise (noWT) model
# ---------------------------------------------------------------------------

def init_pairwise_adjacent_noWT(rng, wt_onehot, mut_map, sigma_P=0.2, l1_P=0.0):
    """Returns P_mut: (L-1,3,3) for adjacent pairs, excluding WT at each position."""
    L = wt_onehot.shape[0]
    P_mut = rng.normal(0.0, sigma_P, size=(L - 1, 3, 3)).astype(np.float32)
    if l1_P > 0:
        P_mut = soft_threshold(P_mut, l1_P)
    return P_mut


def pairwise_adjacent_noWT(x_mut, P_mut, mut_map, b=0.0):
    """
    x_mut: (N,L,4), P_mut: (L-1,3,3), mut_map: (L,3)  →  (N,1)
    """
    N, L, _ = x_mut.shape
    s = np.zeros(N, dtype=np.float32)
    for i in range(L - 1):
        a_idx = mut_map[i]       # (3,)
        b_idx = mut_map[i + 1]   # (3,)
        Xi = x_mut[:, i, a_idx]           # (N,3)
        Xj = x_mut[:, i + 1, b_idx]      # (N,3)
        s += np.einsum("na,nb,ab->n", Xi, Xj, P_mut[i])
    return (s + b).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Potts model
# ---------------------------------------------------------------------------

def init_pairwise_potts_optionA(
    rng,
    wt_onehot: np.ndarray,       # (L,4)
    p_edge: float = 0.9,
    df: float = 2.0,
    lambda_J: float = 3,
    p_rescue: float = 0.10,
    wt_rowcol_zero: bool = True,
):
    """Sample sparse Potts couplings.

    Returns:
      edges: (M,2) int array of interacting pairs (i<j)
      J:     (L,L,4,4) float32 dense tensor
    """
    wt_onehot = np.asarray(wt_onehot, dtype=np.float32)
    assert wt_onehot.ndim == 2 and wt_onehot.shape[1] == 4
    L = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)

    edges = []
    for i in range(L):
        for j in range(i + 1, L):
            if rng.random() < p_edge:
                edges.append((i, j))
    edges = np.array(edges, dtype=np.int32)

    J = np.zeros((L, L, 4, 4), dtype=np.float32)
    for (i, j) in edges:
        M = np.abs(rng.standard_t(df, size=(4, 4))).astype(np.float32) * float(lambda_J)
        if p_rescue > 0:
            mask = (rng.random((4, 4)) < p_rescue)
            M[mask] *= -1.0
        if wt_rowcol_zero:
            wi, wj = int(wt_idx[i]), int(wt_idx[j])
            M[wi, :] = 0.0
            M[:, wj] = 0.0
            M[wi, wj] = 0.0
        J[i, j, :, :] = M

    return edges, J


def pairwise_potts_energy(
    x_mut: np.ndarray,  # (N,L,4)
    edges: np.ndarray,  # (M,2)
    J: np.ndarray,      # (L,L,4,4)
    b: float = 0.0,
) -> np.ndarray:
    """sum_{(i,j) in edges} J[i,j, x_i, x_j]  →  (N,1)."""
    x_mut = np.asarray(x_mut, dtype=np.float32)
    assert x_mut.ndim == 3 and x_mut.shape[2] == 4
    N = x_mut.shape[0]
    x_idx = np.argmax(x_mut, axis=2).astype(np.int32)   # (N,L)

    s = np.zeros(N, dtype=np.float32)
    for (i, j) in edges:
        s += J[i, j, x_idx[:, i], x_idx[:, j]]
    return (s + float(b)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Global nonlinearity wrappers  (legacy / may reference undefined additive_affinity)
# ---------------------------------------------------------------------------

def apply_global_nonlin(s, nonlin_name, nonlin_kwargs):
    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    out_scale = float(nonlin_kwargs.get("out_scale", 1.0))
    out_bias  = float(nonlin_kwargs.get("out_bias",  0.0))

    if nonlin_name == "tanh":
        alpha = float(nonlin_kwargs.get("tanh_alpha", 1.0))
        beta  = float(nonlin_kwargs.get("tanh_beta",  0.0))
        y = np.tanh(alpha * s + beta)

    elif nonlin_name == "softmax_gate":
        gate_logits = np.asarray(
            nonlin_kwargs.get("gate_logits", np.array([0.0, 0.0])), dtype=float
        ).reshape(-1)
        T     = float(nonlin_kwargs.get("gate_temperature", 1.0))
        pi    = _softmax(gate_logits, axis=-1, temperature=T)
        alpha = float(nonlin_kwargs.get("tanh_alpha", 1.0))
        beta  = float(nonlin_kwargs.get("tanh_beta",  0.0))
        y = pi[0] * s + pi[1] * np.tanh(alpha * s + beta)

    elif nonlin_name == "mix_tanh":
        _mw = nonlin_kwargs.get("mix_weights")
        y = _mixture_of_tanh(
            s,
            np.asarray(nonlin_kwargs["mix_alphas"], dtype=float).reshape(-1),
            np.asarray(nonlin_kwargs["mix_betas"],  dtype=float).reshape(-1),
            np.asarray(nonlin_kwargs["mix_logits"], dtype=float).reshape(-1),
            temperature=float(nonlin_kwargs.get("mix_temperature", 1.0)),
            mix_weights=np.asarray(_mw, dtype=float).reshape(-1) if _mw is not None else None,
        )
    else:
        raise ValueError("nonlin_name must be 'tanh', 'softmax_gate', or 'mix_tanh'")

    return (out_scale * y + out_bias).reshape(-1, 1)


def global_nonlinearity_additive_affinity(
    x_mut: np.ndarray,
    w: np.ndarray,
    b: float = 0.0,
    *,
    nonlin: str = "tanh",
    out_scale: float = 1.0,
    out_bias: float = 0.0,
    tanh_alpha: float = 1.0,
    tanh_beta: float = 0.0,
    gate_logits: Optional[np.ndarray] = None,
    gate_temperature: float = 1.0,
    mix_alphas: Optional[np.ndarray] = None,
    mix_betas: Optional[np.ndarray] = None,
    mix_logits: Optional[np.ndarray] = None,
    mix_temperature: float = 1.0,
) -> np.ndarray:
    s = additive_affinity(x_mut, w, b=b)  # type: ignore[name-defined]  # legacy ref

    if nonlin == "tanh":
        y = np.tanh(float(tanh_alpha) * s + float(tanh_beta))
    elif nonlin == "softmax_gate":
        if gate_logits is None:
            gate_logits = np.array([0.0, 0.0], dtype=float)
        gate_logits = np.asarray(gate_logits, dtype=float).reshape(-1)
        if gate_logits.shape[0] != 2:
            raise ValueError("gate_logits must have shape (2,) for softmax_gate.")
        pi = _softmax(gate_logits, axis=-1, temperature=gate_temperature)
        y = pi[0] * s + pi[1] * np.tanh(float(tanh_alpha) * s + float(tanh_beta))
    elif nonlin == "mix_tanh":
        if mix_alphas is None or mix_betas is None or mix_logits is None:
            raise ValueError("mix_alphas, mix_betas, mix_logits required for nonlin='mix_tanh'.")
        y = _mixture_of_tanh(
            s,
            np.asarray(mix_alphas, dtype=float).reshape(-1),
            np.asarray(mix_betas,  dtype=float).reshape(-1),
            np.asarray(mix_logits, dtype=float).reshape(-1),
            temperature=mix_temperature,
        )
    else:
        raise ValueError("nonlin must be one of: 'tanh', 'softmax_gate', 'mix_tanh'.")

    return (float(out_scale) * y + float(out_bias)).reshape(-1, 1)


def global_nonlinearity_additive_pairwise_affinity(
    x_mut: np.ndarray,
    w: np.ndarray,
    P: np.ndarray,
    b: float = 0.0,
    *,
    nonlin: str = "tanh",
    out_scale: float = 1.0,
    out_bias: float = 0.0,
    tanh_alpha: float = 1.0,
    tanh_beta: float = 0.0,
    gate_logits: Optional[np.ndarray] = None,
    gate_temperature: float = 1.0,
    mix_alphas: Optional[np.ndarray] = None,
    mix_betas: Optional[np.ndarray] = None,
    mix_logits: Optional[np.ndarray] = None,
    mix_temperature: float = 1.0,
) -> np.ndarray:
    s = additive_pairwise_affinity(x_mut, w, P, b=b)  # type: ignore[name-defined]  # legacy ref

    if nonlin == "tanh":
        y = np.tanh(float(tanh_alpha) * s + float(tanh_beta))
    elif nonlin == "softmax_gate":
        if gate_logits is None:
            gate_logits = np.array([0.0, 0.0], dtype=float)
        gate_logits = np.asarray(gate_logits, dtype=float).reshape(-1)
        if gate_logits.shape[0] != 2:
            raise ValueError("gate_logits must have shape (2,) for softmax_gate.")
        pi = _softmax(gate_logits, axis=-1, temperature=gate_temperature)
        y = pi[0] * s + pi[1] * np.tanh(float(tanh_alpha) * s + float(tanh_beta))
    elif nonlin == "mix_tanh":
        if mix_alphas is None or mix_betas is None or mix_logits is None:
            raise ValueError("mix_alphas, mix_betas, mix_logits required for nonlin='mix_tanh'.")
        y = _mixture_of_tanh(
            s,
            np.asarray(mix_alphas, dtype=float).reshape(-1),
            np.asarray(mix_betas,  dtype=float).reshape(-1),
            np.asarray(mix_logits, dtype=float).reshape(-1),
            temperature=mix_temperature,
        )
    else:
        raise ValueError("nonlin must be one of: 'tanh', 'softmax_gate', 'mix_tanh'.")

    return (float(out_scale) * y + float(out_bias)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Score computation over libraries
# ---------------------------------------------------------------------------

def compute_gt_scores_for_library(
    X: np.ndarray,
    *,
    W_mut: np.ndarray,
    P_mut: np.ndarray,
    edges: np.ndarray,
    J: np.ndarray,
    mut_map: np.ndarray,
    b0: float,
    nonlin_name: str,
    nonlin_kwargs: dict,
):
    """Returns dict with 4 GT keys, each → (N,) float array."""
    s_add  = additive_affinity_noWT(X, W_mut, mut_map, b=b0)
    s_pair = pairwise_potts_energy(X, edges, J, b=0.0)
    s_addpair = s_add + s_pair

    y_add             = s_add.reshape(-1)
    y_addpair         = s_addpair.reshape(-1)
    # Normalise each energy stream to unit std before the nonlinearity so
    # that tanh/mix_tanh parameters have a consistent interpretation
    # regardless of whether the pairwise term inflates the energy scale.
    s_add_n     = s_add     / (np.std(s_add)     + 1e-8)
    s_addpair_n = s_addpair / (np.std(s_addpair) + 1e-8)
    y_nonlin_add      = apply_global_nonlin(s_add_n,     nonlin_name, nonlin_kwargs).reshape(-1)
    y_nonlin_addpair  = apply_global_nonlin(s_addpair_n, nonlin_name, nonlin_kwargs).reshape(-1)

    return {
        "additive":               y_add,
        "additive_pairwise":      y_addpair,
        "nonlin_additive":        y_nonlin_add,
        "nonlin_additive_pairwise": y_nonlin_addpair,
    }


def compute_gt_scores_for_library_potts(
    X: np.ndarray,
    *,
    W_mut: np.ndarray,
    mut_map: np.ndarray,
    b0: float,
    nonlin_name: str,
    nonlin_kwargs: dict,
    P_mut: np.ndarray = None,
    edges: np.ndarray = None,
    J: np.ndarray = None,
):
    """Returns dict with 4 GT keys, each → (N,) float array.

    Accepts either Potts (edges/J) or adjacent (P_mut) pairwise term.
    """
    s_add = additive_affinity_noWT(X, W_mut, mut_map, b=b0)

    if edges is not None and J is not None:
        s_pair = pairwise_potts_energy(X, edges, J, b=0.0)
    elif P_mut is not None:
        s_pair = pairwise_adjacent_noWT(X, P_mut, mut_map, b=0.0)
    else:
        s_pair = np.zeros_like(s_add)

    s_addpair = s_add + s_pair

    y_add             = s_add.reshape(-1)
    y_addpair         = s_addpair.reshape(-1)
    s_add_n     = s_add     / (np.std(s_add)     + 1e-8)
    s_addpair_n = s_addpair / (np.std(s_addpair) + 1e-8)
    y_nonlin_add      = apply_global_nonlin(s_add_n,     nonlin_name, nonlin_kwargs).reshape(-1)
    y_nonlin_addpair  = apply_global_nonlin(s_addpair_n, nonlin_name, nonlin_kwargs).reshape(-1)

    return {
        "additive":               y_add,
        "additive_pairwise":      y_addpair,
        "nonlin_additive":        y_nonlin_add,
        "nonlin_additive_pairwise": y_nonlin_addpair,
    }


# ---------------------------------------------------------------------------
# Library uniformisation
# ---------------------------------------------------------------------------

def uniformize_by_histogram(scores, X=None, n_bins=200, clip_hi=98, seed=42, target_n=None):
    """Equalise counts per score bin and optionally cap the total output.

    Uses rank-based equal-count binning: sequences are sorted by score and
    split into n_bins groups of equal size.  This guarantees every bin is
    non-empty and avoids the np.unique / duplicate-edge collapse that occurs
    with percentile-based edges when the score distribution is heavily
    saturated (many sequences at the same float value).

    Parameters
    ----------
    scores   : (N,) array of GT scores
    X        : (N, L, A) one-hot sequences, or None.
               When None, returns (scores_uniform, keep_indices) instead of
               (scores_uniform, X_uniform) so callers can reconstruct X cheaply.
    n_bins   : number of rank-based bins
    clip_hi  : upper percentile clip (removes extreme-high outliers)
    target_n : if set, cap per_bin so total output ≈ target_n.

    Returns
    -------
    (scores_uniform, X_uniform)    when X is provided
    (scores_uniform, keep_indices) when X is None
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float).reshape(-1)

    finite   = np.isfinite(scores)
    orig_idx = np.where(finite)[0]
    scores   = scores[orig_idx]
    if X is not None:
        X = np.asarray(X)[orig_idx]

    hi = np.nanpercentile(scores, clip_hi)

    # Work only on sequences within the clip range
    in_range = scores <= hi
    range_pos = np.where(in_range)[0]           # positions within filtered scores
    scores_r  = scores[range_pos]
    n_r       = len(scores_r)

    if n_r == 0:
        raise ValueError("No sequences within clip range; can't uniformize.")

    n_bins_eff = min(int(n_bins), n_r)

    # Rank-based binning: sort by score, assign equal-count bins by rank.
    # This is immune to duplicate score values (saturation) — every bin gets
    # floor(n_r / n_bins_eff) sequences regardless of score distribution shape.
    sort_order           = np.argsort(scores_r, kind="stable")
    bin_ids_sorted       = np.floor(np.arange(n_r) * n_bins_eff / n_r).astype(int)
    bin_ids_sorted       = np.clip(bin_ids_sorted, 0, n_bins_eff - 1)
    bin_ids              = np.empty(n_r, dtype=int)
    bin_ids[sort_order]  = bin_ids_sorted

    counts  = np.bincount(bin_ids, minlength=n_bins_eff)
    per_bin = int(counts[counts > 0].min())
    if target_n is not None:
        n_nonempty = int(np.sum(counts > 0))
        per_bin = min(per_bin, max(1, int(target_n) // max(n_nonempty, 1)))

    keep_r = []   # positions within range_pos
    for b in range(n_bins_eff):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue
        keep_r.append(idx if idx.size <= per_bin
                       else rng.choice(idx, size=per_bin, replace=False))

    keep_r     = np.concatenate(keep_r) if keep_r else np.array([], dtype=int)
    keep       = range_pos[keep_r]           # positions within filtered scores array
    scores_out = scores[keep]
    if X is not None:
        return scores_out, X[keep]
    return scores_out, orig_idx[keep]        # map back to pre-finite-filter indices
