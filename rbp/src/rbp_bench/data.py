"""RNAcompete-2009 data loading for AlphaGenome-encoder fine-tuning.

The ``rnacompete2009.h5`` file is the one used by ResidualBind. Its layout is *nested per
RBP* (see ``residualbind/helper.py``)::

    /{RBP}/X_train  (N, 9, L)  float32   channels 0-3 = one-hot A,C,G,U; 4-8 = RNAplfold structure
    /{RBP}/Y_train  (N, 1)     float32   binding intensity
    /{RBP}/X_valid  ...        /{RBP}/Y_valid
    /{RBP}/X_test   ...        /{RBP}/Y_test

We keep only the 4 sequence channels (ResidualBind's ``ss_type="seq"``); AlphaGenome takes
4-channel input. Two facts make this trivial to feed to AlphaGenome:

* AlphaGenome's one-hot convention is also ``A, C, G, T`` (and it treats ``U``/``N`` as
  all-zeros). ResidualBind stores ``U`` in the ``T`` slot, so the *stored one-hot is already
  compatible* -- we transpose ``(N, 4, L) -> (N, L, 4)`` and feed it as-is. We never round-trip
  through a string (that would zero out every ``U``).
* Probes are ~39 nt. The encoder's ``Pool1d`` uses SAME padding with ceil division, so any length
  is accepted; we center-pad to a fixed ``sequence_length`` (default 128) only for clean batching.

Target normalisation mirrors ResidualBind (``residualbind/helper.py``): ``log_norm`` or
``clip_norm`` followed by z-scoring. Parameters are fit on the *train* split and reused on
valid/test so metrics are comparable; the fitted params are returned for denormalisation at eval.
"""

from __future__ import annotations

from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# RNAcompete 2009 RBP experiments (top-level h5 groups), per residualbind/demo.ipynb.
RNACOMPETE_2009_RBPS = (
    "VTS1",
    "Fusip",
    "HuR",
    "PTB",
    "RBM4",
    "SF2",
    "SLM2",
    "U1A",
    "YB1",
)

Normalization = Literal["log_norm", "clip_norm", "none"]


def normalize_targets(
    data: np.ndarray,
    normalization: Normalization,
    params: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Normalise binding intensities, optionally with pre-fit ``params``.

    Formulas copied from ``residualbind/helper.py:normalize_data`` so the AlphaGenome models are
    trained/evaluated on exactly the same target distribution as ResidualBind.

    * ``clip_norm``: clip at 4*std, then z-score. ``params = [mu, sigma]``.
    * ``log_norm``:  ``log(y - MIN + 1)``, then z-score. ``params = [MIN, mu, sigma]``.
    * ``none``: identity. ``params = []``.

    When ``params`` is provided it is reused (fit on train) instead of being recomputed.
    """
    data = np.asarray(data, dtype=np.float64).copy()

    if normalization == "none":
        return data.astype(np.float32), []

    if normalization == "clip_norm":
        if params is None:
            std = np.std(data)
            data[data > std * 4] = std * 4
            mu, sigma = float(np.mean(data)), float(np.std(data))
            params = [mu, sigma]
        else:
            mu, sigma = params
            # Re-apply the same clip threshold derived from this split's std is *not* used;
            # we only re-apply the train mu/sigma so valid/test land on the train scale.
        data_norm = (data - mu) / sigma
        return data_norm.astype(np.float32), [float(mu), float(sigma)]

    if normalization == "log_norm":
        if params is None:
            MIN = float(np.min(data))
            logged = np.log(data - MIN + 1.0)
            mu, sigma = float(np.mean(logged)), float(np.std(logged))
            params = [MIN, mu, sigma]
        else:
            MIN, mu, sigma = params
            logged = np.log(data - MIN + 1.0)
        data_norm = (logged - mu) / sigma
        return data_norm.astype(np.float32), [float(MIN), float(mu), float(sigma)]

    raise ValueError(f"Unknown normalization: {normalization!r}")


def _center_pad(onehot: np.ndarray, sequence_length: int | None) -> np.ndarray:
    """Center-pad a ``(L, 4)`` one-hot array with zeros to ``sequence_length``.

    Center padding (rather than right padding) keeps symmetric flanks around the probe, matching
    ResidualBind's ``generate_*`` preprocessing. Sequences longer than the target are not expected
    for RNAcompete; if encountered we center-crop.
    """
    if sequence_length is None:
        return onehot.astype(np.float32, copy=False)
    length = onehot.shape[0]
    if length == sequence_length:
        return onehot.astype(np.float32, copy=False)
    if length > sequence_length:
        start = (length - sequence_length) // 2
        return onehot[start : start + sequence_length].astype(np.float32, copy=False)
    padded = np.zeros((sequence_length, 4), dtype=np.float32)
    offset = (sequence_length - length) // 2
    padded[offset : offset + length] = onehot.astype(np.float32, copy=False)
    return padded


class RNACompeteDataset(Dataset):
    """One RBP / one split of the RNAcompete-2009 dataset.

    Args:
        h5_path: Path to ``rnacompete2009.h5``.
        rbp: RBP experiment name (top-level h5 group), e.g. ``"VTS1"``.
        split: ``"train"``, ``"valid"`` or ``"test"``.
        sequence_length: Fixed length to center-pad probes to (default 128 -> one 128 bp encoder
            bin). Use ``None`` to keep the native length (only safe with ``batch_size=1`` when
            lengths differ).
        normalization: ``"log_norm"`` (default), ``"clip_norm"`` or ``"none"``.
        norm_params: Pre-fit normalisation params (fit on train). Required for ``valid``/``test``
            so they use the train scale; if ``None`` they are fit on this split.
    """

    def __init__(
        self,
        h5_path: str,
        rbp: str,
        split: Literal["train", "valid", "test"],
        *,
        sequence_length: int | None = 128,
        normalization: Normalization = "log_norm",
        norm_params: list[float] | None = None,
    ) -> None:
        self.rbp = rbp
        self.split = split
        self.sequence_length = sequence_length

        with h5py.File(h5_path, "r") as f:
            x = np.asarray(f[f"/{rbp}/X_{split}"], dtype=np.float32)  # (N, C, L), C in {4, 9}
            y = np.asarray(f[f"/{rbp}/Y_{split}"], dtype=np.float32)  # (N,) or (N, 1)

        # The processed RNAcompete-2009 h5 stores C=9 channels: 4 nucleotide one-hot (A,C,G,U)
        # followed by 5 RNAplfold secondary-structure profiles. AlphaGenome takes 4-channel input,
        # so keep only the sequence channels -- equivalent to ResidualBind's ss_type="seq".
        if x.shape[1] < 4:
            raise ValueError(f"Expected >=4 channels in X_{split}, got shape {x.shape}")
        x = x[:, :4, :]
        x = np.transpose(x, (0, 2, 1))  # (N, L, 4) -- channel order already A,C,G,U(=T)
        y = np.squeeze(y)
        if y.ndim == 0:
            y = y[None]

        # Drop probes with NaN targets (unassayed), mirroring ResidualBind.
        keep = ~np.isnan(y)
        x, y = x[keep], y[keep]

        y_norm, params = normalize_targets(y, normalization, norm_params)

        self.inputs = x  # (N, L, 4) float32
        self.targets = y_norm.astype(np.float32)  # (N,)
        self.norm_params = params

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        onehot = _center_pad(self.inputs[index], self.sequence_length)
        return (
            torch.from_numpy(onehot),  # (sequence_length, 4)
            torch.tensor(self.targets[index], dtype=torch.float32),  # scalar
        )


def build_loaders(
    h5_path: str,
    rbp: str,
    *,
    sequence_length: int | None = 128,
    normalization: Normalization = "log_norm",
    batch_size: int = 100,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[float]]:
    """Build train/val/test loaders for one RBP, returning the train-fit ``norm_params``.

    The valid and test datasets reuse the train split's normalisation params so all three land on
    the same scale and Pearson is computed against comparable targets.
    """
    train_ds = RNACompeteDataset(
        h5_path, rbp, "train",
        sequence_length=sequence_length, normalization=normalization,
    )
    params = train_ds.norm_params
    valid_ds = RNACompeteDataset(
        h5_path, rbp, "valid",
        sequence_length=sequence_length, normalization=normalization, norm_params=params,
    )
    test_ds = RNACompeteDataset(
        h5_path, rbp, "test",
        sequence_length=sequence_length, normalization=normalization, norm_params=params,
    )

    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, valid_loader, test_loader, params
