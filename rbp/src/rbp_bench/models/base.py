"""Common interface every benchmark model implements.

A benchmark model is a plain ``nn.Module`` whose ``forward(x)`` maps a batch of one-hot probes
``(B, L, 4)`` to predictions ``(B,)`` (or ``(B, num_outputs)``). Multi-stage training (e.g. freeze
then unfreeze) is expressed through :meth:`set_stage`, which the trainer calls before each stage;
the trainer then optimises whatever parameters have ``requires_grad=True``. The default is a single
stage with everything trainable, so simple models need no override.
"""

from __future__ import annotations

import torch.nn as nn


class BenchModel(nn.Module):
    """Base class for benchmark models. ``forward(x:(B,L,4)) -> (B,)``."""

    def set_stage(self, stage: str) -> None:
        """Configure trainable params for a named stage. Default: train everything."""
        if stage not in ("all",):
            # Models with real multi-stage schedules override this; the default only knows "all".
            raise ValueError(
                f"{type(self).__name__} does not define stage {stage!r} (only 'all')."
            )
        for p in self.parameters():
            p.requires_grad = True

    def checkpoint_payload(self, trainable: str) -> dict:
        """Weights to persist for a stage. Default: the full ``state_dict`` (fine for small models).

        Large pretrained models should override to save only the trained/used parts (see
        ``AlphaGenomeFTModel``) and tag the dict with a ``format`` so a loader knows how to restore.
        """
        return {"format": "full", "model_state_dict": self.state_dict()}
