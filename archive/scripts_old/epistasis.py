"""
epistasis.py – In-silico mutagenesis and epistasis tensor computation.

All functions are pure (no global state).  The `predictor` callable is
always passed explicitly as an argument.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Batched predictor helper
# ---------------------------------------------------------------------------

def _batch_predictor(predictor, X, batch_size=4096):
    """X: (N,L,4) → (N,) scores.  predictor(X, iteration, plot=False) → (aux, scores)."""
    N   = X.shape[0]
    out = np.empty(N, dtype=np.float32)
    i   = 0
    while i < N:
        j = min(N, i + batch_size)
        _, s = predictor(X[i:j], None, plot=False)
        out[i:j] = np.asarray(s).reshape(-1).astype(np.float32)
        i = j
    return out


# ---------------------------------------------------------------------------
# Double-mutant tensor
# ---------------------------------------------------------------------------

def ism_double_tensor(input_seq, predictor, batch_size=4096,
                      skip_wt=False, include_diag=False, return_raw=False):
    """Build full double-mutant Δ tensor.

    Args:
      input_seq:   (L,4) one-hot
      predictor:   callable(X, iter, plot=False) → (aux, scores)
      skip_wt:     if True, cells where a==WT_i or b==WT_j are NaN
      include_diag: include i==j pairs
      return_raw:  also return raw score tensor

    Returns (delta_pairs, [raw_pairs,] baseline, wt_idx):
      delta_pairs: (4,4,L,L) Δ = score(mut) − score(WT)
      raw_pairs:   (4,4,L,L) raw scores  (only when return_raw=True)
      baseline:    float score(WT)
      wt_idx:      (L,) int WT base per position
    """
    x      = np.asarray(input_seq, dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] == 4, "Expecting (L,4) one-hot"
    L      = x.shape[0]
    wt_idx = x.argmax(axis=1)

    _, wt_score = predictor(x[None, ...], None, plot=False)
    baseline    = float(np.asarray(wt_score).reshape(-1)[0])

    raw_pairs   = np.full((4, 4, L, L), np.nan, dtype=np.float32)
    delta_pairs = np.full_like(raw_pairs, np.nan)

    combos, meta = [], []
    max_chunk    = max(batch_size, 4096)

    for i in range(L):
        for j in range(L):
            if (i == j) and not include_diag:
                continue
            for a in range(4):
                if skip_wt and a == wt_idx[i]:
                    continue
                for b in range(4):
                    if skip_wt and b == wt_idx[j]:
                        continue
                    X = x.copy()
                    X[i, :] = 0.0; X[i, a] = 1.0
                    X[j, :] = 0.0; X[j, b] = 1.0
                    combos.append(X); meta.append((a, b, i, j))

                    if len(combos) >= max_chunk:
                        S = _batch_predictor(predictor, np.stack(combos), batch_size)
                        for val, (aa, bb, ii, jj) in zip(S, meta):
                            raw_pairs[aa, bb, ii, jj] = float(val)
                        combos.clear(); meta.clear()

    if combos:
        S = _batch_predictor(predictor, np.stack(combos), batch_size)
        for val, (aa, bb, ii, jj) in zip(S, meta):
            raw_pairs[aa, bb, ii, jj] = float(val)

    mask = ~np.isnan(raw_pairs)
    delta_pairs[mask] = raw_pairs[mask] - baseline

    if return_raw:
        return delta_pairs, raw_pairs, baseline, wt_idx
    return delta_pairs, baseline, wt_idx


# ---------------------------------------------------------------------------
# Epistasis tensor / map
# ---------------------------------------------------------------------------

def max_with_sign(e):
    """Signed max over nucleotide axes.

    e: (A, B, L, L)  →  (L, L) with the signed value e[a*, b*, i, j]
    where (a*,b*) = argmax over |e[:, :, i, j]|.
    """
    A, B, L, _ = e.shape
    abs_e  = np.abs(e)
    flat   = abs_e.reshape(A * B, L, L)
    all_nan = np.all(np.isnan(flat), axis=0)

    safe   = np.where(np.isnan(flat), -np.inf, flat)
    idx    = np.argmax(safe, axis=0)
    a_idx, b_idx = np.divmod(idx, B)

    I   = np.arange(L)[:, None]
    J   = np.arange(L)[None, :]
    out = e[a_idx, b_idx, I, J]
    out[all_nan] = np.nan
    return out


def epistasis_tensor(raw_pairs, singles_raw, F0, mask, outputdir):
    """Compute epistasis tensor e_{ij}^{(a,b)} = F_ij - F_i - F_j + F0.

    raw_pairs:   (4,4,L,L)
    singles_raw: (4,L)
    F0:          float (WT score)
    mask:        (L,L) bool mask for zeroing
    outputdir:   path for saving masked npy

    Returns: (L,L) max-sign epistasis map
    """
    A0, A1, L0, L1 = raw_pairs.shape
    assert A0 == 4 and A1 == 4 and L0 == L1 == singles_raw.shape[1]

    e  = raw_pairs.copy()
    e -= singles_raw[:, None, :, None]    # subtract F(x_i^a)
    e -= singles_raw[None, :, None, :]    # subtract F(x_j^b)
    e += F0

    maxwithsign = max_with_sign(e)
    masked_copy = maxwithsign.copy()
    masked_copy[mask] = 0
    masked_copy = np.nan_to_num(masked_copy, nan=0.0)

    import os
    os.makedirs(outputdir, exist_ok=True)
    np.save(f"{outputdir}/epistasis_masked.npy", masked_copy)

    return maxwithsign


def epistasis_map(e_tensor, wt_idx, agg="maxabs", skip_wt=True):
    """Aggregate the full epistasis tensor to a (L,L) map.

    e_tensor: (4,4,L,L)
    wt_idx:   (L,)
    agg:      'maxabs' | 'mean' | 'max' | 'min'
    """
    A, _, L, _ = e_tensor.shape
    mask = np.ones_like(e_tensor, dtype=bool)

    if skip_wt:
        ai_ok = (np.arange(A)[:, None] != wt_idx[None, :])[:, None, :, None]
        bj_ok = (np.arange(A)[None, :] != wt_idx[:, None]).T[None, :, None, :]
        mask &= ai_ok
        mask &= bj_ok

    e_masked = np.where(mask, e_tensor, np.nan)

    if   agg == "maxabs": E = np.nanmax(np.abs(e_masked), axis=(0, 1))
    elif agg == "mean":   E = np.nanmean(e_masked,         axis=(0, 1))
    elif agg == "max":    E = np.nanmax(e_masked,          axis=(0, 1))
    elif agg == "min":    E = np.nanmin(e_masked,          axis=(0, 1))
    else:
        raise ValueError("agg must be 'maxabs' | 'mean' | 'max' | 'min'")

    np.fill_diagonal(E, 0.0)
    E = 0.5 * (E + E.T)
    return E
