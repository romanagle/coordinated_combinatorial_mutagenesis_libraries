"""Plotting: matrix assembly, ordering, NaN handling, and that a figure file is written."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.plotting import _matrix, performance_heatmap  # noqa: E402


def test_matrix_handles_missing_rbps():
    summaries = {
        "modelA": {"VTS1": 0.7, "HuR": 0.5},
        "modelB": {"VTS1": 0.4},  # missing HuR -> NaN
    }
    M, rbps, labels = _matrix(summaries)
    assert labels == ["modelA", "modelB"]
    assert set(rbps) == {"VTS1", "HuR"}
    # the missing (modelB, HuR) cell is NaN
    j = labels.index("modelB")
    i = rbps.index("HuR")
    assert np.isnan(M[i, j])


def test_heatmap_writes_file(tmp_path):
    summaries = {
        "residualbind": {"VTS1": 0.71, "HuR": 0.55, "PTB": 0.60},
        "agft_bin4": {"VTS1": 0.68, "HuR": 0.52},          # missing PTB
        "agft_trunk": {"VTS1": 0.40, "HuR": 0.30, "PTB": 0.35},
    }
    fig, ax = performance_heatmap(summaries, metric_label="best val Pearson r", split="valid")
    out = tmp_path / "valid_heatmap.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
    # columns ordered best-mean first -> residualbind leftmost, agft_trunk rightmost
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels[0] == "residualbind"
    assert xticklabels[-1] == "agft_trunk"


def test_single_vs_ensemble_bar_writes_file(tmp_path):
    from rbp_bench.plotting import single_vs_ensemble_bar
    per_rbp = {
        "VTS1": {"members": [0.70, 0.72, 0.69, 0.71], "ensemble": 0.74},
        "HuR": {"members": [0.80, 0.82, 0.81], "ensemble": 0.84},
    }
    fig, ax = single_vs_ensemble_bar(per_rbp, split="test")
    out = tmp_path / "bar.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
