"""Model registry: name -> builder ``build(model_args: dict, device) -> BenchModel``.

Add a method by dropping a module here that exposes ``build(...)`` and registering it below.
"""

from __future__ import annotations

from . import residualbind
from .base import BenchModel
from .ensemble import EnsembleModel, load_ensemble

# name -> build callable
REGISTRY = {
    "residualbind": residualbind.build,
}

try:
    from . import alphagenome_ft
except ModuleNotFoundError:
    alphagenome_ft = None
else:
    REGISTRY["alphagenome_ft"] = alphagenome_ft.build


def get_builder(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["REGISTRY", "get_builder", "BenchModel", "EnsembleModel", "load_ensemble",
           "residualbind", "alphagenome_ft"]
