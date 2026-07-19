"""
Synthetic mRNA-RBP random-library score distributions split by mutated region.

Produces exactly two notebook figures:
  mRNA_RBP/outputs/notebook_plots/rand_lib_dist_synthetic_high_wt.png
  mRNA_RBP/outputs/notebook_plots/rand_lib_dist_synthetic_low_wt.png

The random libraries come from the cached synthetic instance_00 mut05/mut10/mut25
libraries. The two WT states use the same synthetic sequence and coefficients,
but different sigmoid operating points to represent high-activity and
low-activity WT contexts.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mRNA_RBP.generate_libraries import OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS
from mRNA_RBP.gt_init import MrnaRbpGroundTruth


GT_KEY = "nonlin_additive_pairwise"
INSTANCE = 0
MUT_RATES = [(5, "5%"), (10, "10%"), (25, "25%")]
BINS = 60

OUT_DIR = REPO / "mRNA_RBP" / "outputs" / "notebook_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WT_CONFIGS = {
    "high": {
        "c": 1.5,
        "out": OUT_DIR / "rand_lib_dist_synthetic_high_wt.png",
        "label": "Synthetic high-WT activity",
    },
    "low": {
        "c": -1.5,
        "out": OUT_DIR / "rand_lib_dist_synthetic_low_wt.png",
        "label": "Synthetic low-WT activity",
    },
}

CATEGORY_ORDER = ("stem_motif_intact", "motif_hit", "neither")
CATEGORY_LABELS = {
    "stem_motif_intact": "Stem-hit, motif intact",
    "motif_hit": "Motif-hit",
    "neither": "Neither stem nor motif",
}
CATEGORY_COLORS = {
    "stem_motif_intact": "#4C72B0",
    "motif_hit": "#DD8452",
    "neither": "#55A868",
}


def _mut_count_for(pct: int) -> int:
    return max(1, round(pct * len(SEQ) / 100))


def _load_nuc_ids(pct: int) -> np.ndarray:
    path = Path(OUT_BASE) / f"instance_{INSTANCE:02d}" / f"mut{pct:02d}" / "lib_20000.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["nuc_ids"].astype(np.uint8)


def _score(gt: MrnaRbpGroundTruth, nuc_ids: np.ndarray) -> np.ndarray:
    x = np.eye(4, dtype=np.float32)[nuc_ids]
    return gt.score_all(x)[GT_KEY].astype(np.float64)


def _classify(nuc_ids: np.ndarray, wt_ids: np.ndarray) -> dict[str, np.ndarray]:
    stem_positions = np.array(sorted({p for pair in STEM_PAIRS for p in pair}), dtype=np.int32)
    motif_positions = np.array(MOTIF_POSITIONS, dtype=np.int32)
    stem_hit = (nuc_ids[:, stem_positions] != wt_ids[stem_positions]).any(axis=1)
    motif_hit = (nuc_ids[:, motif_positions] != wt_ids[motif_positions]).any(axis=1)
    return {
        "stem_motif_intact": stem_hit & ~motif_hit,
        "motif_hit": motif_hit,
        "neither": ~(stem_hit | motif_hit),
    }


def _shared_xlim(groups: list[np.ndarray]) -> tuple[float, float]:
    finite = [np.asarray(g, dtype=float) for g in groups if len(g)]
    lo = min(np.percentile(g, 0.2) for g in finite)
    hi = max(np.percentile(g, 99.8) for g in finite)
    pad = max((hi - lo) * 0.03, 1e-6)
    return lo - pad, hi + pad


def _build_plot_data(kind: str) -> tuple[dict, list[np.ndarray]]:
    cfg = WT_CONFIGS[kind]
    gt = MrnaRbpGroundTruth(
        SEQ,
        STEM_PAIRS,
        MOTIF_POSITIONS,
        seed=INSTANCE,
        c=cfg["c"],
    )
    wt_ids = np.argmax(gt.wt_one_hot(), axis=1).astype(np.uint8)

    panel_groups = []
    per_rate = {}
    for pct, label in MUT_RATES:
        nuc_ids = _load_nuc_ids(pct)
        scores = _score(gt, nuc_ids)
        masks = _classify(nuc_ids, wt_ids)
        per_rate[label] = {cat: scores[mask] for cat, mask in masks.items()}
        panel_groups.extend(per_rate[label].values())
        counts = ", ".join(
            f"{CATEGORY_LABELS[cat]} n={len(per_rate[label][cat]):,}"
            for cat in CATEGORY_ORDER
        )
        print(f"{kind} {label}: {counts}")
    return per_rate, panel_groups


def _plot(kind: str, per_rate: dict, xlim: tuple[float, float]) -> Path:
    cfg = WT_CONFIGS[kind]
    out_path = cfg["out"]
    bins = np.linspace(xlim[0], xlim[1], BINS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), sharex=True, sharey=False)
    fig.subplots_adjust(wspace=0.28)

    for ax, (pct, label) in zip(axes, MUT_RATES):
        groups = per_rate[label]
        legend_handles = []
        legend_labels = []
        for cat in CATEGORY_ORDER:
            scores = groups[cat]
            if len(scores) == 0:
                continue
            mean = float(np.mean(scores))
            ax.hist(
                scores,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.38,
                linewidth=1.4,
                color=CATEGORY_COLORS[cat],
            )
            ax.axvline(mean, color=CATEGORY_COLORS[cat], linestyle="--", linewidth=1.4)
            legend_handles.append(
                (
                    Patch(facecolor=CATEGORY_COLORS[cat], alpha=0.38),
                    Line2D([0], [0], color=CATEGORY_COLORS[cat], linestyle="--", linewidth=1.4),
                )
            )
            legend_labels.append(f"{CATEGORY_LABELS[cat]}  n={len(scores):,}, mean={mean:.3f}")

        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.5)
        legend_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.5))
        legend_labels.append("WT=0")
        ax.set_title(f"{cfg['label']} | {label} random lib ({_mut_count_for(pct)} mutations)", fontsize=9.5)
        ax.set_xlabel("Synthetic GT score relative to WT")
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(legend_handles, legend_labels, fontsize=7.4, framealpha=0.85, loc="upper right")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt", choices=["high", "low", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requested_kinds = ("high", "low") if args.wt == "both" else (args.wt,)
    # Always compute both WT states for a common x-axis, even when regenerating
    # only one requested plot.
    data = {}
    all_groups = []
    for kind in ("high", "low"):
        per_rate, groups = _build_plot_data(kind)
        data[kind] = per_rate
        all_groups.extend(groups)
    xlim = _shared_xlim(all_groups)
    print(f"Shared high/low WT x-axis: [{xlim[0]:.4f}, {xlim[1]:.4f}]")

    for kind in requested_kinds:
        out_path = WT_CONFIGS[kind]["out"]
        if out_path.exists() and not args.force:
            print(f"Using cached {kind} synthetic random-library plot -> {out_path}")
            continue
        _plot(kind, data[kind], xlim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
