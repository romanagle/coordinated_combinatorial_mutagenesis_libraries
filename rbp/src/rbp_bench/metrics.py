"""Shared evaluation metric: Pearson correlation (the standard RNAcompete metric)."""

from __future__ import annotations

import numpy as np


def pearson(y_true, y_pred) -> float:
    """Pearson correlation between two 1-D arrays; NaN for degenerate (zero-variance) inputs."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])
