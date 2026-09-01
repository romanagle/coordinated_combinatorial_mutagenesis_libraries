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

_HERE = Path(__file__).resolve().parent.parent
_REPO_ROOT = _HERE.parent

from mRNA_RBP.src.gt_init import MrnaRbpGroundTruth, MrnaNegativeControlGroundTruth
from mRNA_RBP.src.seq_utils import rna_to_one_hot


MRNA_ORACLE = "mrna_ground_truth"
MRNA_NEGATIVE_CONTROL_ORACLE = "mrna_negative_control"
RESIDUALBIND_MSI1_ORACLE = "residualbind_ensemble"
RESIDUALBIND_VTS1_ORACLE = "vts1_residualbind"
RESIDUALBIND_HUR_ORACLE = "hur_residualbind"
RESIDUALBIND_QKI_ORACLE = "qki_residualbind"
SURROGATE_ORACLE = "surrogate_varied_mutrate"
TWISTER_ORACLE = "twister_ribozyme"
DEEPSQUID_HUR_ORACLE = "deepsquid_hur"
DEEPSQUID_VTS1_ORACLE = "deepsquid_vts1"

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
HUR_GT_KEYS = ["hur_residualbind"]
QKI_GT_KEYS = ["qki_residualbind"]
SURROGATE_SCORE_KEY = "surrogate_residualbind_nonlin_pairwise"
SURROGATE_GT_KEYS = [SURROGATE_SCORE_KEY]
# Twister has no live black-box oracle -- deep squid (trained directly on the
# real RA measurements) is the only score key, not a distillation of one.
TWISTER_SCORE_KEY = "deepsquid_twister"
TWISTER_GT_KEYS = [TWISTER_SCORE_KEY]

# Deep squid treated as its own first-class oracle (not a distillation
# comparison bolted on afterward): the standard pipeline (generate_libraries,
# lib_size_spearman, etc.) runs against it unmodified, the same way it runs
# against the real ResidualBind ensemble. One score key per real oracle it
# was distilled from -- see deepsquid_score_key -- so deepsquid_hur is never
# conflated with deepsquid_vts1/deepsquid_msi1.
DEEPSQUID_HUR_GT_KEYS = ["deepsquid_hur"]
DEEPSQUID_VTS1_GT_KEYS = ["deepsquid_vts1"]

ORACLE_GT_KEYS = {
    MRNA_ORACLE: MRNA_GT_KEYS,
    MRNA_NEGATIVE_CONTROL_ORACLE: MRNA_GT_KEYS,
    RESIDUALBIND_MSI1_ORACLE: RESIDUALBIND_GT_KEYS,
    RESIDUALBIND_VTS1_ORACLE: VTS1_GT_KEYS,
    RESIDUALBIND_HUR_ORACLE: HUR_GT_KEYS,
    RESIDUALBIND_QKI_ORACLE: QKI_GT_KEYS,
    SURROGATE_ORACLE: SURROGATE_GT_KEYS,
    TWISTER_ORACLE: TWISTER_GT_KEYS,
    DEEPSQUID_HUR_ORACLE: DEEPSQUID_HUR_GT_KEYS,
    DEEPSQUID_VTS1_ORACLE: DEEPSQUID_VTS1_GT_KEYS,
}
PRIMARY_GT_KEY = {
    MRNA_ORACLE: "nonlin_additive_pairwise",
    MRNA_NEGATIVE_CONTROL_ORACLE: "nonlin_additive_pairwise",
    RESIDUALBIND_MSI1_ORACLE: "residualbind_ensemble",
    RESIDUALBIND_VTS1_ORACLE: "vts1_residualbind",
    RESIDUALBIND_HUR_ORACLE: "hur_residualbind",
    RESIDUALBIND_QKI_ORACLE: "qki_residualbind",
    SURROGATE_ORACLE: SURROGATE_SCORE_KEY,
    TWISTER_ORACLE: TWISTER_SCORE_KEY,
    DEEPSQUID_HUR_ORACLE: "deepsquid_hur",
    DEEPSQUID_VTS1_ORACLE: "deepsquid_vts1",
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
        "mrna_negative_control": MRNA_NEGATIVE_CONTROL_ORACLE,
        "negative_control": MRNA_NEGATIVE_CONTROL_ORACLE,
        "negctrl": MRNA_NEGATIVE_CONTROL_ORACLE,
        "mrna_negctrl": MRNA_NEGATIVE_CONTROL_ORACLE,
        "residualbind": RESIDUALBIND_MSI1_ORACLE,
        "residualbind_ensemble": RESIDUALBIND_MSI1_ORACLE,
        "ensembled_residualbind": RESIDUALBIND_MSI1_ORACLE,
        "msi1": RESIDUALBIND_MSI1_ORACLE,
        "residualbind_msi1": RESIDUALBIND_MSI1_ORACLE,
        "vts1": RESIDUALBIND_VTS1_ORACLE,
        "vts1_residualbind": RESIDUALBIND_VTS1_ORACLE,
        "residualbind_vts1": RESIDUALBIND_VTS1_ORACLE,
        "rncmpt00111": RESIDUALBIND_VTS1_ORACLE,
        "hur": RESIDUALBIND_HUR_ORACLE,
        "elavl1": RESIDUALBIND_HUR_ORACLE,
        "hur_residualbind": RESIDUALBIND_HUR_ORACLE,
        "residualbind_hur": RESIDUALBIND_HUR_ORACLE,
        "rncmpt00112": RESIDUALBIND_HUR_ORACLE,
        "qki": RESIDUALBIND_QKI_ORACLE,
        "qki_residualbind": RESIDUALBIND_QKI_ORACLE,
        "residualbind_qki": RESIDUALBIND_QKI_ORACLE,
        "rncmpt00047": RESIDUALBIND_QKI_ORACLE,
        "surrogate": SURROGATE_ORACLE,
        "surrogate_varied_mutrate": SURROGATE_ORACLE,
        "varied_mutrate_surrogate": SURROGATE_ORACLE,
        "twister": TWISTER_ORACLE,
        "twister_ribozyme": TWISTER_ORACLE,
        "kobori": TWISTER_ORACLE,
        "kobori_yokobayashi": TWISTER_ORACLE,
        "deepsquid_hur": DEEPSQUID_HUR_ORACLE,
        "deepsquid_hur_oracle": DEEPSQUID_HUR_ORACLE,
        "hur_deepsquid": DEEPSQUID_HUR_ORACLE,
        "deep_squid_hur": DEEPSQUID_HUR_ORACLE,
        "deepsquid_vts1": DEEPSQUID_VTS1_ORACLE,
        "deepsquid_vts1_oracle": DEEPSQUID_VTS1_ORACLE,
        "vts1_deepsquid": DEEPSQUID_VTS1_ORACLE,
        "deep_squid_vts1": DEEPSQUID_VTS1_ORACLE,
    }
    key = str(name).strip().lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            f"Unknown oracle {name!r}. Choose one of: {', '.join(sorted(set(aliases.values())))}"
        )
    return aliases[key]


def default_output_base(base_dir: str, oracle_name: str, wt_activity: Optional[str] = None) -> str:
    """Default output root for an oracle's pipeline artifacts.

    Biological runs live under ``runs/<oracle-family>/<target>/<wt>`` so the
    manuscript-facing ``outputs/`` tree remains reserved for curated evidence.
    """
    oracle_name = normalize_oracle_name(oracle_name)
    if oracle_name == MRNA_ORACLE:
        return os.path.join(base_dir, "outputs")
    if oracle_name == TWISTER_ORACLE:
        return os.path.join(base_dir, "outputs_twister_ribozyme")
    run_parts = {
        DEEPSQUID_HUR_ORACLE: ("deepsquid", "hur"),
        DEEPSQUID_VTS1_ORACLE: ("deepsquid", "vts1"),
        RESIDUALBIND_HUR_ORACLE: ("residualbind", "hur"),
        RESIDUALBIND_VTS1_ORACLE: ("residualbind", "vts1"),
    }
    if oracle_name in run_parts:
        family, target = run_parts[oracle_name]
        activity = wt_activity or "high"
        return os.path.join(base_dir, "runs", family, target, activity)
    suffix = f"_{wt_activity}" if wt_activity and oracle_uses_wt_activity(oracle_name) else ""
    return os.path.join(base_dir, "runs", f"{oracle_name}{suffix}")


def oracle_gt_keys(oracle_name: str) -> list[str]:
    return list(ORACLE_GT_KEYS[normalize_oracle_name(oracle_name)])


def primary_gt_key(oracle_name: str) -> str:
    return PRIMARY_GT_KEY[normalize_oracle_name(oracle_name)]


def oracle_uses_wt_activity(oracle_name: str) -> bool:
    """Whether this oracle has a high/low natural-probe WT variant (vs. one fixed WT)."""
    return normalize_oracle_name(oracle_name) in (
        RESIDUALBIND_VTS1_ORACLE, RESIDUALBIND_HUR_ORACLE, RESIDUALBIND_QKI_ORACLE,
        DEEPSQUID_HUR_ORACLE, DEEPSQUID_VTS1_ORACLE,
    )


ORACLE_SHORT_NAME = {
    RESIDUALBIND_MSI1_ORACLE: "msi1",
    RESIDUALBIND_VTS1_ORACLE: "vts1",
    RESIDUALBIND_HUR_ORACLE: "hur",
    RESIDUALBIND_QKI_ORACLE: "qki",
    TWISTER_ORACLE: "twister",
    DEEPSQUID_HUR_ORACLE: "deepsquid_hur",
    DEEPSQUID_VTS1_ORACLE: "deepsquid_vts1",
}


def deepsquid_score_key(oracle_name: str) -> str:
    """Score key for the layer-2 'deep squid' surrogate distilled from a real
    ResidualBind ensemble oracle (see train_surrogate_varied_mutrate.py).
    Registered per-oracle (never a shared implicit default) so e.g.
    deepsquid_hur/deepsquid_qki are never conflated with deepsquid_msi1/vts1.
    """
    name = normalize_oracle_name(oracle_name)
    if name not in ORACLE_SHORT_NAME:
        raise ValueError(
            f"deepsquid_score_key: {oracle_name!r} has no registered short name "
            f"(only real ResidualBind ensemble oracles have a deep-squid surrogate)"
        )
    return f"deepsquid_{ORACLE_SHORT_NAME[name]}"


def sequence_config_for_oracle(oracle_name: str, wt_activity: str = "high"):
    """Return ``(seq, stem_pairs, motif_positions)`` for a normalized oracle name.

    ``mrna``/``residualbind_msi1`` use the fixed canonical MSI1-style sequence
    (no high/low natural-probe variant). ``vts1``/``hur``/``qki`` use their
    natural-probe high/low configs from ``sequence_configs.py``. Any other
    oracle (e.g. the generic varied-mutrate surrogate oracle) falls back to
    the VTS1 config, matching prior behavior.
    """
    from mRNA_RBP.src.sequence_configs import (
        MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_MOTIF_POSITIONS,
        vts1_sequence_config, hur_sequence_config, qki_sequence_config,
        twister_sequence_config,
    )

    name = normalize_oracle_name(oracle_name)
    if name in (MRNA_ORACLE, RESIDUALBIND_MSI1_ORACLE):
        return MSI1_SEQ, list(MSI1_STEM_PAIRS), list(MSI1_MOTIF_POSITIONS)
    if name == MRNA_NEGATIVE_CONTROL_ORACLE:
        # Motif-only control: same sequence and privileged motif as the
        # structured GT, but no declared stem or pairwise interactions.
        return MSI1_SEQ, [], list(MSI1_MOTIF_POSITIONS)
    if name in (RESIDUALBIND_HUR_ORACLE, DEEPSQUID_HUR_ORACLE):
        return hur_sequence_config(wt_activity)
    if name == RESIDUALBIND_QKI_ORACLE:
        return qki_sequence_config(wt_activity)
    if name == TWISTER_ORACLE:
        return twister_sequence_config()
    return vts1_sequence_config(wt_activity)


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
    wt_activity: str = "high",
):
    oracle_name = normalize_oracle_name(oracle_name)
    if oracle_name == MRNA_ORACLE:
        return MrnaRbpGroundTruth(
            seq, stem_pairs, motif_positions, seed=seed, stem_sigma=stem_sigma
        )
    if oracle_name == MRNA_NEGATIVE_CONTROL_ORACLE:
        return MrnaNegativeControlGroundTruth(
            seq, motif_positions=motif_positions, bg_sigma=0.10, seed=seed
        )
    if oracle_name == DEEPSQUID_HUR_ORACLE:
        return SurrogateMAVENNOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            model_dir=surrogate_model_dir or _default_deepsquid_hur_model_dir(wt_activity),
        )
    if oracle_name == DEEPSQUID_VTS1_ORACLE:
        return SurrogateMAVENNOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            model_dir=surrogate_model_dir or _default_deepsquid_vts1_model_dir(wt_activity),
        )
    if oracle_name == SURROGATE_ORACLE:
        return SurrogateMAVENNOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            model_dir=surrogate_model_dir,
        )
    if oracle_name == TWISTER_ORACLE:
        return SurrogateMAVENNOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            model_dir=surrogate_model_dir or _DEFAULT_TWISTER_MODEL_DIR,
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
    if oracle_name == RESIDUALBIND_HUR_ORACLE:
        return ResidualBindEnsembleOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            checkpoint_dir=residualbind_dir or _DEFAULT_HUR_CHECKPOINT_DIR,
            score_key="hur_residualbind",
            batch_size=residualbind_batch_size,
        )
    if oracle_name == RESIDUALBIND_QKI_ORACLE:
        return ResidualBindEnsembleOracle(
            seq=seq,
            stem_pairs=stem_pairs,
            checkpoint_dir=residualbind_dir or _DEFAULT_QKI_CHECKPOINT_DIR,
            score_key="qki_residualbind",
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
        repo = _REPO_ROOT
        rbp_src = repo / "rbp" / "src"
        if str(rbp_src) not in sys.path:
            sys.path.insert(0, str(rbp_src))
        try:
            import typing
            if not hasattr(typing, "Literal"):
                import typing_extensions
                typing.Literal = typing_extensions.Literal
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

    def raw_score(self, x: np.ndarray) -> np.ndarray:
        """Raw (non-WT-anchored) ResidualBind ensemble prediction, on the
        model's native output scale -- unlike score_all(), not shifted so
        WT=0. Meaningful only for real ResidualBind-backed oracles."""
        x = np.asarray(x, dtype=np.float32)
        return self._predict_raw(x)

    @property
    def wt_raw_score(self) -> float:
        """The WT sequence's raw (non-anchored) ResidualBind prediction."""
        self._load()
        return float(self._wt_score)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.score_all(x)[self.score_key]

    def __repr__(self) -> str:
        return (
            f"ResidualBindEnsembleOracle(L={len(self.seq)}, "
            f"members_dir={self.checkpoint_dir}, wt_activity=0.000)"
        )


_DEFAULT_MSI1_CHECKPOINT_DIR = (
    _HERE / "outputs" / "ground_truth_collections"
    / "ResidualBind oracle MSI1" / "libraries_used_for_figures"
    / "integrated_residualbind" / "results" / "msi1"
)
_DEFAULT_VTS1_CHECKPOINT_DIR = (
    _REPO_ROOT / "rbp" / "results" / "vts1"
)

# HuR/QKI checkpoints are still being trained under rbp/results/ -- these
# will move under outputs/ground_truth_collections/ (mirroring MSI1/VTS1)
# once the curated collection is assembled; update then.
_DEFAULT_HUR_CHECKPOINT_DIR = (
    _REPO_ROOT / "rbp" / "results" / "hur"
)
_DEFAULT_QKI_CHECKPOINT_DIR = (
    _REPO_ROOT / "rbp" / "results" / "qki"
)


_DEFAULT_SURROGATE_MODEL_DIR = (
    _HERE / "outputs" / "ground_truth_collections"
    / "ResidualBind oracle MSI1" / "libraries_used_for_figures"
    / "surrogate_models" / "varied_mutrate_nonlin_pairwise"
)

_DEFAULT_TWISTER_MODEL_DIR = (
    _HERE / "outputs_twister_ribozyme"
    / "surrogate_models" / "deepsquid_twister"
)


def _default_deepsquid_hur_model_dir(wt_activity: str) -> Path:
    """Deep squid for HuR is trained per wt_activity (HuR has two distinct
    native sequences/structures, high and low), so unlike
    _DEFAULT_TWISTER_MODEL_DIR there is no single fixed default -- reads the
    canonical model distilled from the corresponding ResidualBind run."""
    return (
        _HERE
        / "models" / "residualbind" / "hur" / wt_activity
        / "varied_mutrate_nonlin_pairwise"
    )


def _default_deepsquid_vts1_model_dir(wt_activity: str) -> Path:
    """Deep squid for VTS1 is trained per wt_activity (VTS1 has two distinct
    native sequences/structures, high and low), mirroring
    _default_deepsquid_hur_model_dir and reads the canonical retained model."""
    return (
        _HERE
        / "models" / "residualbind" / "vts1" / wt_activity
        / "varied_mutrate_nonlin_pairwise"
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
