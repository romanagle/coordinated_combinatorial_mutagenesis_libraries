"""Ensemble of independently-trained members; prediction = mean of member outputs.

For robustness we train N members of the same base model with different seeds (different init +
data order) and average their predictions at inference. Members are trained *independently* (not
jointly) so they stay diverse -- the averaging is what reduces variance.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .base import BenchModel


class EnsembleModel(BenchModel):
    """Wrap trained members; ``forward(x)`` returns the mean of their outputs."""

    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        if len(members) == 0:
            raise ValueError("EnsembleModel needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.members], dim=0).mean(dim=0)


def load_ensemble(member_paths, base_model: str, model_args: dict, device,
                  sample_input: torch.Tensor) -> EnsembleModel:
    """Rebuild an ensemble for inference from saved member checkpoints.

    ``sample_input`` (a ``(B, L, 4)`` batch) is used to materialise any lazy params before loading.
    Members must be ``"full"``-format checkpoints (the default for small models like ResidualBind).
    """
    from . import get_builder  # local import avoids an import cycle

    members = []
    for path in member_paths:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        if ckpt.get("format") != "full":
            raise ValueError(
                f"load_ensemble expects 'full' member checkpoints, got {ckpt.get('format')!r}"
            )
        m = get_builder(base_model)(model_args, device)
        with torch.no_grad():
            m(sample_input.to(device))  # materialise lazy params before load_state_dict
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        members.append(m)
    return EnsembleModel(members).to(device)
