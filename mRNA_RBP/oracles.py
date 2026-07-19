"""Oracle implementations for the mRNA_RBP downstream pipeline.

The pipeline historically used :class:`MrnaRbpGroundTruth` directly.  This
module keeps that default intact while adding a ResidualBind ensemble oracle
with the same small interface: ``wt_one_hot()``, ``score_all(x)``, ``__call__``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent

from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.seq_utils import rna_to_one_hot


MRNA_ORACLE = "mrna_ground_truth"
RESIDUALBIND_MSI1_ORACLE = "residualbind_ensemble"
RESIDUALBIND_VTS1_ORACLE = "vts1_residualbind"
SURROGATE_ORACLE = "surrogate_varied_mutrate"

# Back-compat aliases: many scripts import these two names directly.
RESIDUALBIND_ORACLE = RESIDUALBIND_MSI1_ORACLE
VTS1_ORACLE = RESIDUALBIND_VTS1_ORACLE

MRNA_GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]
RESIDUALBIND_GT_KEYS = ["residualbind_ensemble"]
VTS1_GT_KEYS = ["vts1_residualbind"]
SURROGATE_SCORE_KEY = "surrogate_residualbind_nonlin_pairwise"
SURROGATE_GT_KEYS = [SURROGATE_SCORE_KEY]

ORACLE_GT_KEYS = {
    MRNA_ORACLE: MRNA_GT_KEYS,
    RESIDUALBIND_MSI1_ORACLE: RESIDUALBIND_GT_KEYS,
    RESIDUALBIND_VTS1_ORACLE: VTS1_GT_KEYS,
    SURROGATE_ORACLE: SURROGATE_GT_KEYS,
}
PRIMARY_GT_KEY = {
    MRNA_ORACLE: "nonlin_additive_pairwise",
    RESIDUALBIND_MSI1_ORACLE: "residualbind_ensemble",
    RESIDUALBIND_VTS1_ORACLE: "vts1_residualbind",
    SURROGATE_ORACLE: SURROGATE_SCORE_KEY,
}


def normalize_oracle_name(name: str) -> str:
    """Map a user-facing oracle name to its canonical internal identity.

    RESIDUALBIND_MSI1_ORACLE and RESIDUALBIND_VTS1_ORACLE are two distinct,
    independently-trained ResidualBind ensembles integrated under
    outputs/ground_truth_collections/ (MSI1 and VTS1/RNCMPT00111 are separate)
    -- never conflate one for the other.
    The bare "residualbind"/"residualbind_ensemble" aliases historically meant
    MSI1 specifically (kept for back-compat with existing cached filenames);
    use "residualbind_msi1" or "residualbind_vts1" to be unambiguous.
    """
    aliases = {
        "mrna": MRNA_ORACLE,
        "mrnagroundtruthrbp": MRNA_ORACLE,
        "mrna_ground_truth": MRNA_ORACLE,
        "residualbind": RESIDUALBIND_MSI1_ORACLE,
        "residualbind_ensemble": RESIDUALBIND_MSI1_ORACLE,
        "ensembled_residualbind": RESIDUALBIND_MSI1_ORACLE,
        "msi1": RESIDUALBIND_MSI1_ORACLE,
        "residualbind_msi1": RESIDUALBIND_MSI1_ORACLE,
        "vts1": RESIDUALBIND_VTS1_ORACLE,
        "vts1_residualbind": RESIDUALBIND_VTS1_ORACLE,
        "residualbind_vts1": RESIDUALBIND_VTS1_ORACLE,
        "rncmpt00111": RESIDUALBIND_VTS1_ORACLE,
        "surrogate": SURROGATE_ORACLE,
        "surrogate_varied_mutrate": SURROGATE_ORACLE,
        "varied_mutrate_surrogate": SURROGATE_ORACLE,
    }
    key = str(name).strip().lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            f"Unknown oracle {name!r}. Choose one of: {', '.join(sorted(set(aliases.values())))}"
        )
    return aliases[key]


def default_output_base(base_dir: str, oracle_name: str) -> str:
    oracle_name = normalize_oracle_name(oracle_name)
    if oracle_name == MRNA_ORACLE:
        return os.path.join(base_dir, "outputs")
    return os.path.join(base_dir, f"outputs_{oracle_name}")


def oracle_gt_keys(oracle_name: str) -> list[str]:
    return list(ORACLE_GT_KEYS[normalize_oracle_name(oracle_name)])


def primary_gt_key(oracle_name: str) -> str:
    return PRIMARY_GT_KEY[normalize_oracle_name(oracle_name)]


def build_oracle(
    oracle_name: str,
    *,
    seq: str,
    stem_pairs: list[tuple[int, int]],
    motif_positions: list[int],
    seed: int,
    stem_sigma: float,
    residualbind_dir: Optional[str] = None,
    residualbind_batch_size: int = 4096,
    surrogate_model_dir: Optional[str] = None,
):
    oracle_name = normalize_oracle_name(oracle_name)
    if oracle_name == MRNA_ORACLE:
        return MrnaRbpGroundTruth(
            seq, stem_pairs, motif_positions, seed=seed, stem_sigma=stem_sigma
        )
    if oracle_name == SURROGATE_ORACLE:
        return SurrogateMAVENNOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            model_dir=surrogate_model_dir,
        )
    if oracle_name == RESIDUALBIND_VTS1_ORACLE:
        return ResidualBindEnsembleOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            checkpoint_dir=residualbind_dir or _DEFAULT_VTS1_CHECKPOINT_DIR,
            score_key="vts1_residualbind",
            batch_size=residualbind_batch_size,
        )
    if oracle_name == RESIDUALBIND_MSI1_ORACLE:
        return ResidualBindEnsembleOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            checkpoint_dir=residualbind_dir or _DEFAULT_MSI1_CHECKPOINT_DIR,
            score_key="residualbind_ensemble",
            batch_size=residualbind_batch_size,
        )
    raise ValueError(f"build_oracle: unhandled oracle_name {oracle_name!r}")


class ResidualBindEnsembleOracle:
    """ResidualBind ensemble inference wrapper anchored to WT score = 0.

    Every ResidualBind oracle in this pipeline must be a multi-member
    rbp_bench ensemble (never a single model -- a single VTS1 Keras model
    only reached Pearson r=0.21 on its own real RNAcompete test set, vs.
    r=0.88 for the MSI1 ensemble; ensembling suppresses per-model noise that
    a lone checkpoint can't). ``checkpoint_dir`` must contain ``member*.pt``
    files trained the same way as the integrated MSI1 ResidualBind checkpoints.

    ``checkpoint_dir`` is required -- MSI1 and VTS1 are two distinct,
    independently-trained ensembles, so there is no safe implicit default.
    Use :func:`build_oracle` with an explicit oracle name instead of
    instantiating this class directly with ``checkpoint_dir=None``.
    """

    def __init__(
        self,
        *,
        seq: str,
        stem_pairs: Optional[list[tuple[int, int]]] = None,
        checkpoint_dir: str,
        score_key: str = "residualbind_ensemble",
        batch_size: int = 4096,
        device: Optional[str] = None,
    ) -> None:
        self.seq = seq
        self.stem_pairs = list(stem_pairs or [])
        self._edges = np.asarray(self.stem_pairs, dtype=np.int32)
        self._wt_oh = rna_to_one_hot(seq).astype(np.float32)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.score_key = score_key
        self.batch_size = int(batch_size)
        self.device_name = device
        self._model = None
        self._torch = None
        self._device = None
        self._wt_score = None

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    def wt_one_hot(self) -> np.ndarray:
        return self._wt_oh.copy()

    def wt_activity(self) -> float:
        return 0.0

    def _member_paths(self) -> list[Path]:
        paths = list(self.checkpoint_dir.glob("member*.pt"))
        if not paths:
            raise FileNotFoundError(f"No member*.pt checkpoints found in {self.checkpoint_dir}")
        return sorted(paths, key=lambda p: int(re.search(r"member(\d+)", p.name).group(1)))

    def _load(self) -> None:
        if self._model is not None:
            return
        repo = Path(__file__).resolve().parents[1]
        rbp_src = repo / "rbp" / "src"
        if str(rbp_src) not in sys.path:
            sys.path.insert(0, str(rbp_src))
        try:
            import torch
            from rbp_bench.models import load_ensemble
        except ImportError as exc:
            raise ImportError(
                "ResidualBind oracle requires the rbp benchmark Python environment "
                "with torch installed. Run from that environment or install rbp dependencies."
            ) from exc

        device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        sample = torch.zeros((2, self._wt_oh.shape[0], 4), dtype=torch.float32)
        self._model = load_ensemble(self._member_paths(), "residualbind", {}, device, sample)
        self._model.eval()
        self._torch = torch
        self._device = device
        self._wt_score = float(self._predict_raw(self._wt_oh[None, :, :])[0])

    def _predict_raw(self, x: np.ndarray) -> np.ndarray:
        self._load()
        torch = self._torch
        outs = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(
                    np.asarray(x[start:start + self.batch_size], dtype=np.float32)
                ).to(self._device)
                outs.append(self._model(xb).detach().cpu().numpy().reshape(-1))
        return np.concatenate(outs).astype(np.float32)

    def score_all(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != self._wt_oh.shape:
            raise ValueError(
                f"Expected x shape (N, {self._wt_oh.shape[0]}, 4), got {x.shape}"
            )
        y = self._predict_raw(x) - float(self._wt_score)
        return {self.score_key: y.astype(np.float32)}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.score_all(x)[self.score_key]

    def __repr__(self) -> str:
        return (
            f"ResidualBindEnsembleOracle(L={len(self.seq)}, "
            f"members_dir={self.checkpoint_dir}, wt_activity=0.000)"
        )


_DEFAULT_MSI1_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "ground_truth_collections"
    / "ResidualBind oracle MSI1" / "cached_libraries"
    / "integrated_residualbind" / "results" / "msi1"
)
_DEFAULT_VTS1_CHECKPOINT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "ground_truth_collections"
    / "ResidualBind oracle VTS1" / "cached_libraries"
    / "integrated_residualbind" / "results" / "vts1"
)


_DEFAULT_SURROGATE_MODEL_DIR = (
    Path(__file__).resolve().parent / "outputs" / "ground_truth_collections"
    / "ResidualBind oracle MSI1" / "pipeline_workspace"
    / "outputs_residualbind_ensemble"
    / "surrogate_models" / "varied_mutrate_nonlin_pairwise"
)


class SurrogateMAVENNOracle:
    """Wraps a trained MAVE-NN surrogate (saved via train_surrogate_varied_mutrate.py
    -- a nonlinear additive+pairwise model distilled from ResidualBind-ensemble
    labels on the varied-mutation-rate library) behind the same small oracle
    interface as MrnaRbpGroundTruth / ResidualBindEnsembleOracle: wt_one_hot(),
    score_all(x), __call__(x). Anchored to WT score = 0, same convention."""

    def __init__(
        self,
        *,
        seq: str,
        stem_pairs: Optional[list[tuple[int, int]]] = None,
        model_dir: Optional[str] = None,
        gpmap: str = "pairwise",
    ) -> None:
        self.seq = seq
        self.stem_pairs = list(stem_pairs or [])
        self._edges = np.asarray(self.stem_pairs, dtype=np.int32)
        self._wt_oh = rna_to_one_hot(seq).astype(np.float32)
        self.model_dir = Path(model_dir or _DEFAULT_SURROGATE_MODEL_DIR)
        self._gpmap = gpmap
        meta = np.load(self.model_dir / "meta.npz", allow_pickle=True)
        self._y_mean = float(meta["y_mean"])
        self._y_std = float(meta["y_std"])
        self.score_key = str(meta["score_key"][0])
        self._model = None
        self._wt_score = None

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    def wt_one_hot(self) -> np.ndarray:
        return self._wt_oh.copy()

    def wt_activity(self) -> float:
        return 0.0

    def _load(self) -> None:
        if self._model is not None:
            return
        import mavenn
        self._model = mavenn.load(str(self.model_dir / f"mavenn_model_{self._gpmap}"))
        self._wt_score = float(self._predict_raw(self._wt_oh[None, :, :])[0])

    @staticmethod
    def _to_str(x: np.ndarray) -> np.ndarray:
        nucs = np.array(list("ACGU"))
        idx = np.argmax(x, axis=2).astype(np.uint8)
        return np.array(["".join(nucs[row]) for row in idx], dtype=object)

    def _predict_raw(self, x: np.ndarray, chunk: int = 4096) -> np.ndarray:
        self._load()
        x_str = self._to_str(x)
        parts = []
        for start in range(0, len(x_str), chunk):
            yhat = np.asarray(
                self._model.x_to_yhat(x_str[start:start + chunk]), dtype=float
            ).ravel()
            parts.append(yhat * self._y_std + self._y_mean)
        return np.concatenate(parts).astype(np.float32)

    def score_all(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != self._wt_oh.shape:
            raise ValueError(
                f"Expected x shape (N, {self._wt_oh.shape[0]}, 4), got {x.shape}"
            )
        self._load()
        y = self._predict_raw(x) - float(self._wt_score)
        return {self.score_key: y.astype(np.float32)}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.score_all(x)[self.score_key]

    def __repr__(self) -> str:
        return (
            f"SurrogateMAVENNOracle(L={len(self.seq)}, model_dir={self.model_dir}, "
            f"gpmap={self._gpmap}, wt_activity=0.000)"
        )
