"""Tests for models, the registry, and the report table (no AlphaGenome weights needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rbp_bench.models import REGISTRY, get_builder  # noqa: E402
from rbp_bench.models.alphagenome_ft import RBPHead, stage_channels  # noqa: E402
from rbp_bench.models.residualbind import ResidualBindTorch  # noqa: E402
from rbp_bench.report import format_table  # noqa: E402

SEQ_LEN = 39


def test_registry_has_models():
    assert set(REGISTRY) == {"residualbind", "alphagenome_ft"}
    assert callable(get_builder("residualbind"))


def test_residualbind_forward_and_backward():
    model = ResidualBindTorch(num_outputs=1)
    x = torch.randn(8, SEQ_LEN, 4)
    out = model(x)
    assert out.shape == (8,)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_residualbind_set_stage_default():
    model = ResidualBindTorch()
    model.set_stage("all")  # default stage works
    assert all(p.requires_grad for p in model.parameters())


def test_rbphead_trunk_layout():
    head = RBPHead(encoder_dim=stage_channels(None), pooling_type="flatten")  # 1536
    assert head(torch.randn(8, 1, 1536)).shape == (8,)


def test_rbphead_finer_intermediate_ncl():
    ch = stage_channels(8)  # 1152
    head = RBPHead(encoder_dim=ch, pooling_type="flatten")
    assert head(torch.randn(8, ch, 5)).shape == (8,)  # NCL auto-transposed


def test_rbphead_mean_multi_output():
    head = RBPHead(encoder_dim=1536, pooling_type="mean", num_outputs=3)
    assert head(torch.randn(4, 5, 1536)).shape == (4, 3)


def test_report_table():
    summaries = {
        "residualbind": {"VTS1": 0.71, "HuR": 0.55},
        "agft_modeA": {"VTS1": 0.40, "HuR": float("nan")},
    }
    table = format_table(summaries)
    assert "VTS1" in table and "residualbind" in table and "agft_modeA" in table
    assert "0.7100" in table and "—" in table  # NaN rendered as dash
    assert "**mean**" in table
