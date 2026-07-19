"""ResidualBind baseline, ported to PyTorch.

A faithful *architectural* port of ``residualbind/residualbind.py`` (the regression model used for
RNAcompete-2009; its commented-out "layer 2" block is omitted, matching the original). Trained from
scratch, so results are equivalent-but-not-bit-identical to the published TF artifact -- the point
is an apples-to-apples baseline that runs in the same torch/CUDA env and on the same
``RNACompeteDataset`` (identical 4-channel slice, log_norm targets, splits) as the other models.

Keras -> PyTorch notes: Keras Conv1D is NLC ``(N, L, C)`` while ``nn.Conv1d`` is NCL ``(N, C, L)``,
so we transpose the input once. Keras ``BatchNormalization`` (axis=-1) over channels == ``BatchNorm1d``
in NCL layout. ``padding='same'`` is supported for stride-1 (incl. dilated) convs.

Original architecture (input ``(N, 39, 4)``)::

    Conv1d(96, k=11, same, no-bias) -> BN -> ReLU -> Dropout(0.1)
    dilated residual block (k=3):
        Conv1d(96, k=3, same, no-bias, dil=1) -> BN
        for d in (2, 4, 8): ReLU -> Dropout(0.1) -> Conv1d(96, k=3, same, no-bias, dil=d) -> BN
        add(input, branch) -> ReLU
    AvgPool1d(10) -> Dropout(0.2)                # 39 -> 3
    Flatten -> Linear(256, no-bias) -> BN -> ReLU -> Dropout(0.5)
    Linear(num_outputs)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BenchModel


class _DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilations: tuple[int, ...] = (2, 4, 8), dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = dropout
        self.first_conv = nn.Conv1d(channels, channels, kernel_size, padding="same", bias=False)
        self.first_bn = nn.BatchNorm1d(channels)
        self.branch = nn.ModuleList(
            nn.ModuleDict({
                "conv": nn.Conv1d(channels, channels, kernel_size,
                                  padding="same", dilation=d, bias=False),
                "bn": nn.BatchNorm1d(channels),
            })
            for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.first_bn(self.first_conv(x))
        for layer in self.branch:
            out = F.relu(out)
            out = F.dropout(out, self.dropout, self.training)
            out = layer["bn"](layer["conv"](out))
        return F.relu(out + residual)


class ResidualBindTorch(BenchModel):
    """PyTorch port of the ResidualBind regression model. Input ``(B, L, 4)`` -> ``(B,)``."""

    def __init__(self, num_outputs: int = 1, conv_filters: int = 96, pool_size: int = 10) -> None:
        super().__init__()
        self.num_outputs = int(num_outputs)
        self.stem_conv = nn.Conv1d(4, conv_filters, kernel_size=11, padding="same", bias=False)
        self.stem_bn = nn.BatchNorm1d(conv_filters)
        self.resblock = _DilatedResidualBlock(conv_filters, kernel_size=3, dilations=(2, 4, 8))
        self.pool = nn.AvgPool1d(kernel_size=pool_size)  # 'valid': stride == kernel_size
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(256, bias=False)  # infers conv_filters * pooled_len
        self.fc_bn = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, self.num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, L, 4) -> (B, 4, L)
        x = F.dropout(F.relu(self.stem_bn(self.stem_conv(x))), 0.1, self.training)
        x = self.resblock(x)
        x = F.dropout(self.pool(x), 0.2, self.training)
        x = self.flatten(x)
        x = F.dropout(F.relu(self.fc_bn(self.fc(x))), 0.5, self.training)
        out = self.out(x)
        return out.squeeze(-1) if self.num_outputs == 1 else out


def build(model_args: dict, device) -> ResidualBindTorch:
    """Registry builder. ``model_args`` may set ``num_outputs``/``conv_filters``/``pool_size``."""
    model = ResidualBindTorch(
        num_outputs=model_args.get("num_outputs", 1),
        conv_filters=model_args.get("conv_filters", 96),
        pool_size=model_args.get("pool_size", 10),
    )
    return model.to(device)
