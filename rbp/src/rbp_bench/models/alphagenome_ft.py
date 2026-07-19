"""AlphaGenome-encoder fine-tuning model for RBP binding.

Wraps the pretrained AlphaGenome encoder (from the installed ``alphagenome_encoder_ft`` /
``alphagenome_pytorch`` packages) with a small regression head, as a ``BenchModel`` with
``forward(x:(B,L,4)) -> (B,)``. Nothing here modifies those installed packages.

Short-sequence handling (RNAcompete probes are ~39 nt; the encoder downsamples 128x):

* ``feature_stage=None`` -- final 128 bp trunk ``(B, 1, 1536)`` (a ~39 nt probe padded to <=128
  pools to a single bin). [mode A]
* ``feature_stage=k`` (e.g. ``8``) -- a finer encoder intermediate ``bin_size_{k}`` ``(B, C_k, S/k)``,
  preserving within-probe positions. Reached from outside via ``inner.encoder(seq)`` which returns
  ``(trunk, intermediates)``. [mode B]

Two training stages via :meth:`set_stage`: ``"head"`` (encoder frozen) then ``"all"`` (encoder
unfrozen at a low LR).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import load_trunk, remove_all_heads
from alphagenome_encoder_ft.model import AlphaGenomeEncoderModel

from .base import BenchModel

# Channels produced at each encoder intermediate ``bin_size_{k}``; trunk (128 bp) is 1536.
STAGE_CHANNELS: dict[int, int] = {1: 768, 2: 896, 4: 1024, 8: 1152, 16: 1280, 32: 1408, 64: 1536}
TRUNK_CHANNELS = 1536

PoolingType = Literal["flatten", "mean", "sum", "max"]


def stage_channels(feature_stage: int | None) -> int:
    if feature_stage is None:
        return TRUNK_CHANNELS
    if feature_stage not in STAGE_CHANNELS:
        raise ValueError(f"feature_stage must be one of {sorted(STAGE_CHANNELS)} or None")
    return STAGE_CHANNELS[feature_stage]


def encode_features(inner: AlphaGenomeEncoderModel, sequences: torch.Tensor,
                    feature_stage: int | None) -> torch.Tensor:
    """Encoder features for ``sequences`` ``(B, L, 4)`` at the requested resolution.

    ``feature_stage is None`` -> 128 bp trunk ``(B, S//128, 1536)`` (NLC); else the
    ``bin_size_{feature_stage}`` intermediate ``(B, C, S//stage)`` (NCL). Both flow through the
    encoder so gradients reach it (stage-2 unfreeze works).
    """
    if feature_stage is None:
        return inner.encode(sequences)
    _trunk, intermediates = inner.encoder(sequences)
    return intermediates[f"bin_size_{feature_stage}"]


# --------------------------------------------------------------------------- head


def _as_nlc(x: torch.Tensor, encoder_dim: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected rank-3 encoder output, got {tuple(x.shape)}")
    if x.shape[-1] == encoder_dim:
        return x
    if x.shape[1] == encoder_dim:
        return x.transpose(1, 2)
    raise ValueError(f"Neither dim of {tuple(x.shape)} matches encoder_dim={encoder_dim}")


class RBPHead(nn.Module):
    """LayerNorm -> position pooling -> MLP -> regression, with a configurable ``encoder_dim``.

    Mirrors ``alphagenome_encoder_ft.heads.MPRAHead`` pooling but takes an explicit ``encoder_dim``
    so it sits on the 1536-ch trunk *or* a finer intermediate (e.g. 1152-ch ``bin_size_8``).
    Accepts NLC ``(B, L, C)`` or NCL ``(B, C, S)`` (auto-transposed).
    """

    def __init__(self, encoder_dim: int, *, pooling_type: PoolingType = "flatten",
                 hidden_sizes: int | Sequence[int] = (1024,), dropout: float | None = 0.1,
                 activation: Literal["relu", "gelu"] = "relu", num_outputs: int = 1) -> None:
        super().__init__()
        if pooling_type not in ("flatten", "mean", "sum", "max"):
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
        if dropout is not None and not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.encoder_dim = int(encoder_dim)
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.activation = activation
        self.num_outputs = int(num_outputs)
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        self.norm = nn.LayerNorm(self.encoder_dim)
        self.hidden_layers = nn.ModuleList()
        in_features: int | None = None
        for hidden in self.hidden_sizes:
            self.hidden_layers.append(
                nn.LazyLinear(hidden) if in_features is None else nn.Linear(in_features, hidden)
            )
            in_features = hidden
        assert in_features is not None
        self.output_layer = nn.Linear(in_features, self.num_outputs)

    def _activation(self) -> nn.Module:
        return nn.GELU() if self.activation == "gelu" else nn.ReLU()

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "flatten":
            return x.flatten(start_dim=1)
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        if self.pooling_type == "sum":
            return x.sum(dim=1)
        return x.max(dim=1).values

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        x = self.norm(_as_nlc(encoder_output, self.encoder_dim))
        x = self._pool(x)
        for linear in self.hidden_layers:
            x = linear(x)
            if self.dropout is not None:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self._activation()(x)
        out = self.output_layer(x)
        return out.squeeze(-1) if self.num_outputs == 1 else out


# --------------------------------------------------------------------------- model


class AlphaGenomeFTModel(BenchModel):
    """AlphaGenome encoder + ``RBPHead``, exposing the two-stage freeze/unfreeze schedule."""

    def __init__(self, inner: AlphaGenomeEncoderModel, head: RBPHead,
                 feature_stage: int | None) -> None:
        super().__init__()
        self.inner = inner
        self.head = head
        self.feature_stage = feature_stage
        self._encoder_frozen = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(encode_features(self.inner, x, self.feature_stage))

    def set_stage(self, stage: str) -> None:
        if stage == "head":
            self.inner.set_encoder_trainable(False)
            self._encoder_frozen = True
        elif stage == "all":
            self.inner.set_encoder_trainable(True)
            self._encoder_frozen = False
        else:
            raise ValueError(f"AlphaGenomeFTModel stages are 'head'/'all', got {stage!r}")
        for p in self.head.parameters():
            p.requires_grad = True
        self.train(self.training)  # re-apply frozen-encoder eval mode (see train())

    def train(self, mode: bool = True):
        """Keep a frozen encoder in eval mode so its BatchNorm stats / dropout stay fixed.

        Without this, ``model.train()`` each epoch would put the frozen encoder back in train mode
        and its running stats would drift -- making the "frozen" features non-deterministic and the
        head-only stage checkpoint inexact.
        """
        super().train(mode)
        if self._encoder_frozen:
            self.inner.encoder.eval()
        return self

    def checkpoint_payload(self, trainable: str) -> dict:
        """Minimal, loadable weights per stage (mirrors the package's save modes).

        ``head`` stage: head only (encoder == the pretrained weights). ``all`` stage: encoder + head
        (the unused transformer/decoder in the backbone are never saved).
        """
        head_sd = self.head.state_dict()
        if trainable == "head":
            return {"format": "agft-head", "feature_stage": self.feature_stage,
                    "head_state_dict": head_sd}
        return {"format": "agft-encoder+head", "feature_stage": self.feature_stage,
                "encoder_state_dict": self.inner.encoder.state_dict(),
                "head_state_dict": head_sd}


def build(model_args: dict, device) -> AlphaGenomeFTModel:
    """Registry builder. ``model_args`` needs ``pretrained_weights``; optional ``feature_stage``,
    ``pooling_type``, ``hidden_sizes``, ``dropout``, ``activation``, ``num_outputs``."""
    weights = model_args.get("pretrained_weights")
    if not weights:
        raise ValueError("alphagenome_ft requires model_args.pretrained_weights "
                         "(set it in the config or pass --pretrained_weights)")
    feature_stage = model_args.get("feature_stage")

    backbone = AlphaGenome()
    backbone = load_trunk(backbone, str(weights), exclude_heads=True)
    backbone = remove_all_heads(backbone)
    inner = AlphaGenomeEncoderModel(backbone, nn.Identity())  # head lives on the wrapper

    head = RBPHead(
        encoder_dim=stage_channels(feature_stage),
        pooling_type=model_args.get("pooling_type", "flatten"),
        hidden_sizes=tuple(model_args.get("hidden_sizes", (1024,))),
        dropout=model_args.get("dropout", 0.1),
        activation=model_args.get("activation", "relu"),
        num_outputs=model_args.get("num_outputs", 1),
    )
    model = AlphaGenomeFTModel(inner, head, feature_stage)
    model.set_stage("head")  # start frozen
    return model.to(device)
