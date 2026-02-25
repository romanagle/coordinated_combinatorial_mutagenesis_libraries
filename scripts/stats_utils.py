"""
stats_utils.py – Lightweight array / statistics helpers.

All functions are pure (no global state, no I/O).
"""

import numpy as np
from scipy.stats import gaussian_kde, spearmanr


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _finite_1d(x):
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def _flatten_runs(arr_list):
    """List of 1-D arrays (one per run) → single concatenated finite array."""
    if not arr_list:
        return np.array([], dtype=float)
    chunks = [_finite_1d(a) for a in arr_list if a is not None and _finite_1d(a).size]
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def _flatten_list_of_arrays(list_of_arrays):
    if not list_of_arrays:
        return np.array([], dtype=float)
    chunks = [_finite_1d(a) for a in list_of_arrays if a is not None and _finite_1d(a).size]
    return np.concatenate(chunks) if chunks else np.array([], dtype=float)


def _finite_flat(arr_list):
    """Alias used in coefficient-store helpers."""
    if not arr_list:
        return np.array([])
    chunks = []
    for a in arr_list:
        if a is None:
            continue
        x = np.asarray(a).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size:
            chunks.append(x)
    return np.concatenate(chunks) if chunks else np.array([])


def _as_list_of_arrays(x):
    """Normalise x to a list of 1-D finite arrays."""
    if x is None:
        return []
    if isinstance(x, list):
        return [np.asarray(a).ravel() for a in x if a is not None and np.asarray(a).size]
    if isinstance(x, np.ndarray):
        return [x.ravel()] if x.size else []
    return []


# ---------------------------------------------------------------------------
# Bin / scale helpers
# ---------------------------------------------------------------------------

def _fd_bins(x, clamp=(25, 80)):
    """Freedman–Diaconis bin count, clamped."""
    x = _finite_1d(x)
    n = x.size
    if n < 2:
        return clamp[0]
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return clamp[0]
    h = 2.0 * iqr * (n ** (-1 / 3))
    if h <= 0:
        return clamp[0]
    nb = int(np.ceil((x.max() - x.min()) / h))
    return max(clamp[0], min(clamp[1], nb))


def _robust_z(x, eps=1e-9):
    x   = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    s   = 1.4826 * mad
    if not np.isfinite(s) or s < eps:
        s = np.nanstd(x)
    if not np.isfinite(s) or s < eps:
        s = 1.0
    return (x - med) / s


def _transform(x, mode):
    """Transform a 1-D array.  mode: None | 'logabs' | 'robust_z'."""
    if mode is None:
        return x
    if mode == "logabs":
        return np.log10(np.abs(x) + 1e-8)
    if mode == "robust_z":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        s   = 1.4826 * mad if mad > 0 else np.nanstd(x)
        if not np.isfinite(s) or s == 0:
            s = 1.0
        return (x - med) / s
    raise ValueError("transform must be None | 'logabs' | 'robust_z'")


# ---------------------------------------------------------------------------
# KDE / log helpers
# ---------------------------------------------------------------------------

def _logabs(x, eps=1e-8):
    x = _finite_1d(x)
    return np.log10(np.abs(x) + eps)


def _kde_line(x, grid, bw_scale=1.25):
    kde = gaussian_kde(x)
    kde.set_bandwidth(bw_method=kde.factor * float(bw_scale))
    return kde(grid)


def _kde_curve(x, grid, bw_scale=1.3):
    kde = gaussian_kde(x)
    kde.set_bandwidth(bw_method=kde.factor * float(bw_scale))
    return kde(grid)


# ---------------------------------------------------------------------------
# Potts / coefficient extraction
# ---------------------------------------------------------------------------

def extract_active_potts_J(J, edges, drop_zeros=True):
    """Return 1-D vector of active Potts coupling entries from selected edges."""
    vals = []
    for (i, j) in edges:
        block = np.asarray(J[i, j, :, :], dtype=float).reshape(-1)
        if drop_zeros:
            block = block[block != 0.0]
        if block.size:
            vals.append(block)
    if not vals:
        return np.array([], dtype=float)
    return _finite_1d(np.concatenate(vals))


def get_coef_vec(coef_store, cfg, which, gt_key):
    arr_list = coef_store.get(cfg, {}).get(which, {}).get(gt_key, [])
    if not arr_list:
        return np.array([], dtype=float)
    x = np.concatenate(
        [np.asarray(a).reshape(-1) for a in arr_list if a is not None], axis=0
    )
    return x[np.isfinite(x)]


# ---------------------------------------------------------------------------
# Spearman helper
# ---------------------------------------------------------------------------

def spearman_on_testdf(model, test_df):
    """Compute Spearman correlation on a MAVE-NN test dataframe."""
    cols = list(test_df.columns)

    if "x" in cols:
        X = np.asarray(test_df["x"])
    elif "X" in cols:
        X = np.asarray(test_df["X"])
    else:
        raise KeyError(f"no sequence col in test_df. cols={cols}")

    y_col = "y" if "y" in cols else next(
        (c for c in cols if c.startswith("y")), None
    )
    if y_col is None:
        raise KeyError(f"no y col in test_df. cols={cols}")
    y = np.asarray(test_df[y_col], dtype=float).reshape(-1)

    yhat = np.asarray(model.x_to_yhat(X), dtype=float).reshape(-1)

    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3 or np.nanstd(y[m]) == 0 or np.nanstd(yhat[m]) == 0:
        return np.nan

    rho, _ = spearmanr(yhat[m], y[m])
    return float(rho)
