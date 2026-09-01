"""
mRNA_RBP/scripts/figures/core/plot_residualbind_vts1_rand_region_distributions.py

Random-library ResidualBind score distributions split by mutated region
(stem-hit / motif-hit / neither), for any oracle with a natural-probe WT
(VTS1, HuR, ...) or a single fixed WT (MSI1).

Reads the cache built by generate_random_region_score_cache.py:
  <out_base>/<short>_natural_random_library_scores.npz          (high WT)
  <out_base>/<short>_natural_random_library_scores_low_wt.npz   (low WT, if any)

For oracles with a high/low natural-probe WT variant, both are loaded
together (even though each lives in a separate out_base) so the two panels
share one x-axis -- this is why it's run once per oracle after both
wt_activity conditions have finished, not per-condition like the other plot
stages.

Output (oracle-agnostic naming, matches the pre-existing VTS1/MSI1 files):
  rand_lib_dist_<short>_oracle_region_classes.png          (high / single WT)
  rand_lib_dist_<short>_oracle_region_classes_low_wt.png   (low WT, if any)

If the cache also has raw (non-WT-anchored) ResidualBind scores --
``rand{pct}_raw_scores`` / ``wt_raw_score``, written by
generate_random_region_score_cache.py for real ResidualBind-backed oracles
(MSI1/VTS1/HuR/QKI) -- a companion figure is produced in raw-score units:
  rand_lib_dist_<short>_oracle_region_classes_raw.png
  rand_lib_dist_<short>_oracle_region_classes_raw_low_wt.png
Skipped for oracles with no raw ResidualBind score (synthetic GT, Twister/
deep-squid) or for older caches predating this field.
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

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.src.oracles import (
    MRNA_ORACLE, ORACLE_SHORT_NAME, default_output_base,
    normalize_oracle_name, oracle_uses_wt_activity,
)
from mRNA_RBP.scripts.figures.core.provenance import stamp_figure

OUT_DIR_DEFAULT = os.path.join(_HERE, "outputs", "notebook_plots")

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


def _classify(nuc_ids: np.ndarray, wt_ids: np.ndarray, stem_pairs, motif_positions) -> dict:
    stem_positions = np.array(sorted({int(p) for pair in stem_pairs for p in pair}), dtype=np.int32)
    motif_positions = np.asarray(motif_positions, dtype=np.int32)
    stem_hit = (nuc_ids[:, stem_positions] != wt_ids[stem_positions]).any(axis=1)
    motif_hit = (nuc_ids[:, motif_positions] != wt_ids[motif_positions]).any(axis=1)
    return {
        "stem_motif_intact": stem_hit & ~motif_hit,
        "motif_hit": motif_hit,
        "neither": ~(stem_hit | motif_hit),
    }


def _shared_xlim(groups: list) -> tuple:
    finite = [np.asarray(g, dtype=float) for g in groups if len(g)]
    lo = min(np.percentile(g, 0.2) for g in finite)
    hi = max(np.percentile(g, 99.8) for g in finite)
    pad = max((hi - lo) * 0.03, 1e-6)
    return lo - pad, hi + pad


def _load_plot_data(cache_path: str, label: str, score_field: str = "delta_scores") -> tuple:
    """Returns (per_rate, panel_groups, anchor, raw_source) or
    (None, None, None, None) if score_field isn't present in this cache (e.g.
    an older cache with no raw-score fields, or an oracle with no raw
    ResidualBind score). raw_source is the real oracle name the raw anchor
    was borrowed from (deep-squid surrogates only, see
    generate_random_region_score_cache.py), else None."""
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(cache_path)

    z = np.load(cache_path)
    if score_field == "raw_scores" and "wt_raw_score" not in z:
        return None, None, None, None

    wt_seq = str(z["wt_seq"].item()) if z["wt_seq"].ndim == 0 else str(z["wt_seq"])
    wt_ids = _wt_ids(wt_seq)
    stem_pairs = z["stem_pairs"].astype(np.int32)
    motif_positions = z["motif_positions"].astype(np.int32)
    anchor = float(z["wt_raw_score"]) if score_field == "raw_scores" else 0.0
    raw_source = str(z["wt_raw_score_source_oracle"][0]) if "wt_raw_score_source_oracle" in z else None

    per_rate = {}
    panel_groups = []
    for pct, rate_label in MUT_RATES:
        prefix = f"rand{pct:02d}"
        nuc_ids = z[f"{prefix}_nids"].astype(np.uint8)
        scores = z[f"{prefix}_{score_field}"].astype(np.float64)
        masks = _classify(nuc_ids, wt_ids, stem_pairs, motif_positions)
        per_rate[rate_label] = {cat: scores[mask] for cat, mask in masks.items()}
        panel_groups.extend(per_rate[rate_label].values())
        counts = ", ".join(
            f"{CATEGORY_LABELS[cat]} n={len(per_rate[rate_label][cat]):,}"
            for cat in CATEGORY_ORDER
        )
        print(f"{label} {rate_label} ({score_field}): {counts}")
    return per_rate, panel_groups, anchor, raw_source


def _plot(label: str, title: str, per_rate: dict, xlim: tuple, out_path: Path,
          cache_path: str, anchor: float = 0.0, raw: bool = False,
          raw_source: str = None, save_svg: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(xlim[0], xlim[1], BINS + 1)
    xlabel = f"{label} raw score" if raw else f"{label} score relative to WT"
    if raw and raw_source:
        xlabel += f" (from {raw_source})"
    anchor_label = f"WT={anchor:.3f}" if raw else "WT=0"

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), sharex=True, sharey=False)
    fig.subplots_adjust(wspace=0.28)

    for ax, (pct, rate_label) in zip(axes, MUT_RATES):
        groups = per_rate[rate_label]
        legend_handles = []
        legend_labels = []
        for cat in CATEGORY_ORDER:
            scores = groups[cat]
            if len(scores) == 0:
                continue
            mean = float(np.mean(scores))
            ax.hist(
                scores, bins=bins, density=True, histtype="stepfilled",
                alpha=0.38, linewidth=1.4, color=CATEGORY_COLORS[cat],
            )
            ax.axvline(mean, color=CATEGORY_COLORS[cat], linestyle="--", linewidth=1.4)
            legend_handles.append(
                (Patch(facecolor=CATEGORY_COLORS[cat], alpha=0.38),
                 Line2D([0], [0], color=CATEGORY_COLORS[cat], linestyle="--", linewidth=1.4))
            )
            legend_labels.append(f"{CATEGORY_LABELS[cat]}  n={len(scores):,}, mean={mean:.3f}")

        ax.axvline(anchor, color="black", linestyle=":", linewidth=1.5)
        legend_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.5))
        legend_labels.append(anchor_label)
        ax.set_title(f"{title} | {rate_label} random lib", fontsize=9.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(legend_handles, legend_labels, fontsize=7.4, framealpha=0.85, loc="upper right")

    stamp_figure(fig, library_status="cached", source_paths=[cache_path])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if save_svg:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default="vts1")
    parser.add_argument("--wt", choices=["high", "low", "both"], default="both")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--svg", action="store_true")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    if oracle_name == MRNA_ORACLE:
        print("Synthetic GT has no ResidualBind natural-probe cache -- nothing to do.")
        return 0

    short = ORACLE_SHORT_NAME.get(oracle_name, oracle_name)
    out_dir = Path(args.out_dir) if args.out_dir else Path(OUT_DIR_DEFAULT)
    if short.startswith("deepsquid_"):
        label = f"deepSQUID {short.split('_', 1)[-1].upper()}"
    elif "residualbind" not in short:
        label = f"{short.upper()} ResidualBind"
    else:
        label = f"{short} ResidualBind"

    has_wt_variant = oracle_uses_wt_activity(oracle_name)
    kinds = ("high", "low") if has_wt_variant else ("high",)
    requested = ("high", "low") if (args.wt == "both" and has_wt_variant) else \
                (args.wt,) if args.wt != "both" else ("high",)

    cache_paths, out_paths, raw_out_paths, titles = {}, {}, {}, {}
    for kind in kinds:
        out_base = default_output_base(_HERE, oracle_name, kind)
        cache_suffix = "" if kind == "high" else "_low_wt"
        cache_paths[kind] = os.path.join(out_base, f"{short}_natural_random_library_scores{cache_suffix}.npz")
        out_name = f"rand_lib_dist_{short}_oracle_region_classes{cache_suffix}.png"
        out_paths[kind] = out_dir / out_name
        raw_out_paths[kind] = out_dir / f"rand_lib_dist_{short}_oracle_region_classes_raw{cache_suffix}.png"
        titles[kind] = f"{label} {kind}-WT activity" if has_wt_variant else f"{label}"

    data = {}
    all_groups = []
    for kind in kinds:
        per_rate, groups, _anchor, _src = _load_plot_data(cache_paths[kind], f"{short} {kind}")
        data[kind] = per_rate
        all_groups.extend(groups)
    xlim = _shared_xlim(all_groups)
    print(f"Shared x-axis across {kinds}: [{xlim[0]:.4f}, {xlim[1]:.4f}]")

    raw_data, raw_anchors, raw_sources = {}, {}, {}
    raw_groups = []
    for kind in kinds:
        per_rate, groups, anchor, raw_source = _load_plot_data(
            cache_paths[kind], f"{short} {kind}", score_field="raw_scores")
        if per_rate is None:
            print(f"  [skip raw] {cache_paths[kind]} has no raw ResidualBind scores cached")
            continue
        raw_data[kind] = per_rate
        raw_anchors[kind] = anchor
        raw_sources[kind] = raw_source
        raw_groups.extend(groups)
    raw_xlim = _shared_xlim(raw_groups) if raw_groups else None
    if raw_xlim:
        print(f"Shared raw-score x-axis across {list(raw_data)}: [{raw_xlim[0]:.4f}, {raw_xlim[1]:.4f}]")

    for kind in requested:
        _plot(label, titles[kind], data[kind], xlim, out_paths[kind],
              cache_paths[kind], save_svg=args.svg)
        if kind in raw_data:
            _plot(label, titles[kind], raw_data[kind], raw_xlim, raw_out_paths[kind],
                  cache_paths[kind], anchor=raw_anchors[kind], raw=True,
                  raw_source=raw_sources[kind], save_svg=args.svg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
