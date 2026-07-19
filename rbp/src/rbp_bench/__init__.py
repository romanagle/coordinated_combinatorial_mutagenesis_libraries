"""rbp-benchmark: model RNAcompete RBP binding and benchmark multiple methods.

Standalone suite (outside the vendored ``residualbind`` / ``alphagenome-encoder-ft`` repos) that
consumes them as installed libraries. Shared data pipeline + metric + training harness; models live
in ``rbp_bench.models`` behind a registry, so adding a method is a single new module + config.
"""

from .data import RNACOMPETE_2009_RBPS, RNACompeteDataset, build_loaders, normalize_targets
from .ensemble_train import run_ensemble, train_ensemble_rbp
from .metrics import pearson
from .models import REGISTRY, BenchModel, EnsembleModel, get_builder, load_ensemble
from .trainer import evaluate, run_benchmark, run_stage, train_rbp

__all__ = [
    "RNACOMPETE_2009_RBPS",
    "RNACompeteDataset",
    "build_loaders",
    "normalize_targets",
    "pearson",
    "REGISTRY",
    "BenchModel",
    "EnsembleModel",
    "load_ensemble",
    "get_builder",
    "evaluate",
    "run_stage",
    "train_rbp",
    "run_benchmark",
    "run_ensemble",
    "train_ensemble_rbp",
]
