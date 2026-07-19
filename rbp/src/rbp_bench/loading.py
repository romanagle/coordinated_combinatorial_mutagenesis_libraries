"""Rebuild a trained model from a saved checkpoint for evaluation/inference (no retraining).

Handles the checkpoint formats written by ``trainer``/``ensemble_train`` / the model
``checkpoint_payload`` methods:

* ``"full"`` -- whole ``state_dict`` (ResidualBind, ensemble members).
* ``"agft-encoder+head"`` -- AlphaGenome encoder + head (the fine-tuned ``all`` stage).
* ``"agft-head"`` -- head only; encoder == the pretrained weights (the frozen ``head`` stage).
"""

from __future__ import annotations

import torch

from .models import get_builder


def load_model(ckpt: dict, base_model: str, model_args: dict, device, sample_input: torch.Tensor):
    """Rebuild ``base_model`` and load weights from ``ckpt``. ``sample_input`` ``(B,L,4)`` materialises
    lazy params before loading. Returns the model in eval mode."""
    model = get_builder(base_model)(model_args, device)
    with torch.no_grad():
        model(sample_input.to(device))  # materialise LazyLinear etc.

    fmt = ckpt.get("format")
    if fmt == "full":
        model.load_state_dict(ckpt["model_state_dict"])
    elif fmt == "agft-encoder+head":
        model.inner.encoder.load_state_dict(ckpt["encoder_state_dict"])
        model.head.load_state_dict(ckpt["head_state_dict"])
    elif fmt == "agft-head":
        model.head.load_state_dict(ckpt["head_state_dict"])
    else:
        raise ValueError(f"Unknown checkpoint format {fmt!r}")

    model.eval()
    return model
