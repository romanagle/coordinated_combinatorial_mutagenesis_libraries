"""Trainer behavior: best-val reporting and the optional test evaluation (ResidualBind, no weights)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.trainer import train_rbp  # noqa: E402

SEQ_LEN = 39


def _make_h5(path: Path, rbp: str = "VTS1", n: int = 48) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group(rbp)
        for split, count in (("train", n), ("valid", n // 2), ("test", n // 2)):
            x = np.zeros((count, 9, SEQ_LEN), dtype=np.float32)
            bases = rng.integers(0, 4, size=(count, SEQ_LEN))
            for i in range(count):
                x[i, bases[i], np.arange(SEQ_LEN)] = 1.0
            y = rng.normal(0.0, 1.0, size=(count, 1)).astype(np.float32)
            g.create_dataset(f"X_{split}", data=x)
            g.create_dataset(f"Y_{split}", data=y)


@pytest.fixture()
def cfg() -> dict:
    return {
        "model": "residualbind",
        "data": {"sequence_length": None, "normalization": "log_norm",
                 "batch_size": 16, "num_workers": 0},
        "model_args": {},
        "stages": [{"trainable": "all", "lr": 1e-3, "num_epochs": 1, "patience": 2}],
        "seed": 43,
    }


def test_reports_best_val_and_skips_test_by_default(tmp_path, cfg):
    h5 = tmp_path / "d.h5"
    _make_h5(h5)
    out = tmp_path / "out"
    result = train_rbp("residualbind", "VTS1", cfg, str(h5), "cpu", out, eval_test=False)

    assert "best_val_pearson" in result          # best val always reported
    assert "test_pearson" not in result          # test skipped by default
    assert "test_mse" not in result
    # per-stage breakdown (single stage for residualbind), no test in the stage entry
    assert len(result["stages"]) == 1
    assert result["best_stage"] == 1
    stage = result["stages"][0]
    assert stage["trainable"] == "all" and "best_val_pearson" in stage
    assert "test_pearson" not in stage
    # per-stage checkpoint + best.pt are saved
    assert (out / "VTS1" / "stage1_all.pt").exists()
    assert (out / "VTS1" / "best.pt").exists()
    # persisted metrics.json matches the returned dict
    saved = json.loads((out / "VTS1" / "metrics.json").read_text())
    assert saved == result


def test_eval_test_adds_test_metrics(tmp_path, cfg):
    h5 = tmp_path / "d.h5"
    _make_h5(h5)
    out = tmp_path / "out"
    result = train_rbp("residualbind", "VTS1", cfg, str(h5), "cpu", out, eval_test=True)

    assert "best_val_pearson" in result
    assert "test_pearson" in result and "test_mse" in result
    # test metrics also recorded per stage
    assert "test_pearson" in result["stages"][0]
