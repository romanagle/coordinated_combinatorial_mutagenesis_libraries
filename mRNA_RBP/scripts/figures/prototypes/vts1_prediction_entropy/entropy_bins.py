"""Pure binning and entropy logic for the throwaway VTS1 prototype."""

from __future__ import annotations

import numpy as np


def prediction_quantile_bins(predictions: np.ndarray, n_bins: int = 30) -> list[np.ndarray]:
    """Return stable, ascending, nearly equal-count prediction bins."""
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("predictions must be a finite one-dimensional array")
    if n_bins < 1 or n_bins > len(predictions):
        raise ValueError("n_bins must be between 1 and the number of predictions")
    return list(np.array_split(np.argsort(predictions, kind="stable"), n_bins))


def shannon_entropy_by_bin(
    sequences: np.ndarray,
    bins: list[np.ndarray],
    alphabet_size: int = 4,
) -> np.ndarray:
    """Compute nucleotide Shannon entropy (bits) at every bin and position."""
    sequences = np.asarray(sequences)
    entropy = np.empty((len(bins), sequences.shape[1]), dtype=float)
    for row, indices in enumerate(bins):
        frequencies = np.stack(
            [(sequences[indices] == nucleotide).mean(axis=0) for nucleotide in range(alphabet_size)]
        )
        terms = np.zeros_like(frequencies)
        np.log2(frequencies, out=terms, where=frequencies > 0)
        entropy[row] = -(frequencies * terms).sum(axis=0)
    return entropy
