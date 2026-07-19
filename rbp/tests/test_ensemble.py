"""Ensemble: mean-of-members forward, end-to-end training, and inference reload."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.ensemble_train import train_ensemble_rbp  # noqa: E402
from rbp_bench.models import EnsembleModel, load_ensemble  # noqa: E402

SEQ_LEN = 39


class _Const(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0],), float(self.value))


def test_ensemble_averages_members():
    ens = EnsembleModel([_Const(1.0), _Const(2.0), _Const(3.0)])
    out = ens(torch.zeros(5, SEQ_LEN, 4))
    assert out.shape == (5,)
    assert torch.allclose(out, torch.full((5,), 2.0))  # mean of 1,2,3


def _make_h5(path: Path, rbp: str = "VTS1", n: int = 48) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group(rbp)
        for split, count in (("train", n), ("valid", n // 2), ("test", n // 2)):
            x = np.zeros((count, 9, SEQ_LEN), dtype=np.float32)
            bases = rng.integers(0, 4, size=(count, SEQ_LEN))
            for i in range(count):
                x[i, bases[i], np.arange(SEQ_LEN)] = 1.0
            g.create_dataset(f"X_{split}", data=x)
            g.create_dataset(f"Y_{split}", data=rng.normal(0, 1, (count, 1)).astype(np.float32))


def _cfg():
    return {
        "model": "residualbind",
        "data": {"sequence_length": None, "normalization": "log_norm",
                 "batch_size": 16, "num_workers": 0},
        "model_args": {},
        "stages": [{"trainable": "all", "lr": 1e-3, "num_epochs": 1, "patience": 2}],
    }


def test_train_ensemble_saves_members_and_metrics(tmp_path):
    h5 = tmp_path / "d.h5"
    _make_h5(h5)
    out = tmp_path / "rb_ens3"
    result = train_ensemble_rbp("residualbind", "VTS1", _cfg(), str(h5), torch.device("cpu"),
                                out, n_members=3, base_seed=43)

    assert result["n_members"] == 3
    assert len(result["member_val_pearsons"]) == 3
    assert "best_val_pearson" in result and "mean_member_val_pearson" in result
    for i in range(3):
        assert (out / "VTS1" / f"member{i}.pt").exists()
    saved = json.loads((out / "VTS1" / "metrics.json").read_text())
    assert saved == result


def test_load_ensemble_roundtrip(tmp_path):
    h5 = tmp_path / "d.h5"
    _make_h5(h5)
    out = tmp_path / "rb_ens2"
    train_ensemble_rbp("residualbind", "VTS1", _cfg(), str(h5), torch.device("cpu"),
                       out, n_members=2, base_seed=43)
    paths = [out / "VTS1" / f"member{i}.pt" for i in range(2)]
    ens = load_ensemble(paths, "residualbind", {}, torch.device("cpu"),
                        sample_input=torch.zeros(2, SEQ_LEN, 4))
    ens.eval()
    out_pred = ens(torch.zeros(4, SEQ_LEN, 4))
    assert out_pred.shape == (4,)
