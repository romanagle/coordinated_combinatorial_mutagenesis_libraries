"""
ResidualBind VTS1 random-library score distributions split by mutated region.

Produces the two canonical VTS1 ResidualBind figures consumed by
scripts/reproduce_gt_pipeline.py:

  mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle VTS1/figures/
    rand_lib_dist_vts1_oracle_region_classes.png
    rand_lib_dist_vts1_oracle_region_classes_low_wt.png

The random libraries and ResidualBind scores are loaded from the natural VTS1
score caches archived with this ground-truth collection.
"""

from __future__ import annotations

import argparse
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

from mRNA_RBP.plots.provenance import stamp_figure  # noqa: E402

COLLECTION_DIR = (
    REPO
    / "mRNA_RBP"
    / "outputs"
    / "ground_truth_collections"
    / "ResidualBind oracle VTS1"
)
CACHE_DIR = COLLECTION_DIR / "libraries_used_for_figures" / "random_region_score_cache"
OUT_DIR = COLLECTION_DIR / "figures"

WT_CONFIGS = {
    "high": {
        "cache": CACHE_DIR / "vts1_natural_random_library_scores.npz",
        "out": OUT_DIR / "rand_lib_dist_vts1_oracle_region_classes.png",
        "label": "ResidualBind VTS1 high-WT activity",
    },
    "low": {
        "cache": CACHE_DIR / "vts1_natural_random_library_scores_low_wt.npz",
        "out": OUT_DIR / "rand_lib_dist_vts1_oracle_region_classes_low_wt.png",
        "label": "ResidualBind VTS1 low-WT activity",
    },
}

MUT_RATES = [(5, "5%"), (10, "10%"), (25, "25%")]
BINS = 60

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
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def _wt_ids(seq: str) -> np.ndarray:
    return np.array([NUC_TO_IDX[c] for c in seq.upper()], dtype=np.uint8)


def _classify(nuc_ids: np.ndarray, wt_ids: np.ndarray, stem_pairs, motif_positions) -> dict[str, np.ndarray]:
    stem_positions = np.array(sorted({int(p) for pair in stem_pairs for p in pair}), dtype=np.int32)
    motif_positions = np.asarray(motif_positions, dtype=np.int32)
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


def _load_plot_data(kind: str) -> tuple[dict, list[np.ndarray]]:
    cfg = WT_CONFIGS[kind]
    if not cfg["cache"].exists():
        raise FileNotFoundError(cfg["cache"])

    z = np.load(cfg["cache"])
    wt_seq = str(z["wt_seq"].item())
    wt_ids = _wt_ids(wt_seq)
    stem_pairs = z["stem_pairs"].astype(np.int32)
    motif_positions = z["motif_positions"].astype(np.int32)

    per_rate = {}
    panel_groups = []
    for pct, label in MUT_RATES:
        prefix = f"rand{pct:02d}"
        nuc_ids = z[f"{prefix}_nids"].astype(np.uint8)
        scores = z[f"{prefix}_delta_scores"].astype(np.float64)
        masks = _classify(nuc_ids, wt_ids, stem_pairs, motif_positions)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        ax.set_title(f"{cfg['label']} | {label} random lib", fontsize=9.5)
        ax.set_xlabel("ResidualBind VTS1 score relative to WT")
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(legend_handles, legend_labels, fontsize=7.4, framealpha=0.85, loc="upper right")

    stamp_figure(fig, library_status="cached", source_paths=[cfg["cache"]])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt", choices=["high", "low", "both"], default="both")
    args = parser.parse_args()

    requested = ("high", "low") if args.wt == "both" else (args.wt,)
    data = {}
    all_groups = []
    for kind in ("high", "low"):
        per_rate, groups = _load_plot_data(kind)
        data[kind] = per_rate
        all_groups.extend(groups)
    xlim = _shared_xlim(all_groups)
    print(f"Shared high/low WT x-axis: [{xlim[0]:.4f}, {xlim[1]:.4f}]")

    for kind in requested:
        _plot(kind, data[kind], xlim)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
