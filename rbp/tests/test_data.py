"""Tests for the model-agnostic data pipeline (no AlphaGenome weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.data import RNACompeteDataset, build_loaders, normalize_targets  # noqa: E402

SEQ_LEN = 39  # native RNAcompete-2009 probe length


def _make_synthetic_h5(path: Path, rbp: str = "VTS1", n: int = 64) -> None:
    """Tiny nested-per-RBP h5 mirroring the real schema: X (N,9,L), Y (N,1).

    Channels 0-3 are one-hot sequence; 4-8 are dummy structure profiles (must be dropped).
    """
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        grp = f.create_group(rbp)
        for split, count in (("train", n), ("valid", n // 2), ("test", n // 2)):
            x = np.zeros((count, 9, SEQ_LEN), dtype=np.float32)
            bases = rng.integers(0, 4, size=(count, SEQ_LEN))
            for i in range(count):
                x[i, bases[i], np.arange(SEQ_LEN)] = 1.0
            x[:, 4:, :] = rng.random((count, 5, SEQ_LEN)).astype(np.float32)  # junk structure ch
            y = rng.normal(5.0, 2.0, size=(count, 1)).astype(np.float32)
            if split == "train":
                y[0] = np.nan  # exercise NaN filtering
            grp.create_dataset(f"X_{split}", data=x)
            grp.create_dataset(f"Y_{split}", data=y)


@pytest.fixture()
def h5_file(tmp_path: Path) -> Path:
    p = tmp_path / "rnacompete2009.h5"
    _make_synthetic_h5(p)
    return p


def test_dataset_shapes_nan_filter_and_channel_slice(h5_file: Path):
    ds = RNACompeteDataset(h5_file, "VTS1", "train", sequence_length=128)
    assert len(ds) == 63  # one NaN target filtered out
    x, y = ds[0]
    assert x.shape == (128, 4)  # 9 channels sliced to 4, center-padded to 128
    assert x.dtype == torch.float32
    assert x.sum().item() == pytest.approx(SEQ_LEN)  # structure channels dropped, one-hot intact
    assert not torch.isnan(y)


def test_center_pad_native_length(h5_file: Path):
    ds = RNACompeteDataset(h5_file, "VTS1", "train", sequence_length=None)
    x, _ = ds[0]
    assert x.shape == (SEQ_LEN, 4)


def test_normalization_round_trip():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    norm, params = normalize_targets(y, "log_norm")
    assert norm.mean() == pytest.approx(0.0, abs=1e-5)
    assert norm.std() == pytest.approx(1.0, abs=1e-5)
    norm2, params2 = normalize_targets(y, "log_norm", params)
    assert params2 == params
    np.testing.assert_allclose(norm, norm2, atol=1e-6)


def test_build_loaders_shares_train_params(h5_file: Path):
    train, valid, test, params = build_loaders(h5_file, "VTS1", batch_size=16, num_workers=0)
    assert len(params) == 3
    assert valid.dataset.norm_params == params
    assert test.dataset.norm_params == params
    xb, yb = next(iter(train))
    assert xb.shape == (16, 128, 4)
    assert yb.shape == (16,)
