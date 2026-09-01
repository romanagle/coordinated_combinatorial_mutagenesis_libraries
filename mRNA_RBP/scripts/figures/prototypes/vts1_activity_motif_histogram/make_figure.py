"""Plot the VTS1-high activity density colored by motif mutation burden."""

from __future__ import annotations

import argparse
import csv
from math import ceil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIBRARY = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz"
)
WT = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
WT_MOTIF = "GCUGG"
MOTIF_LABEL = "GCNGG"
MOTIF_START = WT.index(WT_MOTIF)
MOTIF_OFFSETS = np.asarray([0, 1, 3, 4])  # The central U is unconstrained (N).
ALPHABET = np.asarray(list("ACGU"))
MOTIF_CMAP = LinearSegmentedColormap.from_list(
    "motif_mutations", ["#f1f7ec", "#f3c969", "#df7b3f", "#8f2d56"]
)


def summarize(
    scores: np.ndarray, motif_mutations: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-width edges, density, mean motif mutations, and bin counts."""
    counts, edges = np.histogram(scores, bins=n_bins)
    widths = np.diff(edges)
    density = counts / (len(scores) * widths)
    bin_index = np.searchsorted(edges, scores, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    motif_sum = np.bincount(bin_index, weights=motif_mutations, minlength=n_bins)
    mean_motif_mutations = np.divide(
        motif_sum,
        counts,
        out=np.full(n_bins, np.nan, dtype=float),
        where=counts > 0,
    )
    return edges, density, mean_motif_mutations, counts


def write_summary(
    path: Path,
    edges: np.ndarray,
    density: np.ndarray,
    mean_motif_mutations: np.ndarray,
    counts: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "activity_bin_left",
                "activity_bin_right",
                "activity_bin_midpoint",
                "n_sequences",
                "density",
                "mean_mutated_motif_positions",
            ]
        )
        for i in range(len(counts)):
            writer.writerow(
                [
                    edges[i],
                    edges[i + 1],
                    (edges[i] + edges[i + 1]) / 2,
                    counts[i],
                    density[i],
                    mean_motif_mutations[i],
                ]
            )


def make_figure(
    edges: np.ndarray,
    density: np.ndarray,
    mean_motif_mutations: np.ndarray,
    out_path: Path,
) -> None:
    color_max = max(1, ceil(np.nanmax(mean_motif_mutations)))
    norm = Normalize(vmin=0, vmax=color_max)
    fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="white")
    ax.bar(
        edges[:-1],
        density,
        width=np.diff(edges),
        align="edge",
        color=MOTIF_CMAP(norm(mean_motif_mutations)),
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("ResidualBind activity score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "VTS1 high-WT activity distribution",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.text(
        0,
        1.015,
        "Bar color shows mean mutations at the four constrained GCNGG positions",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=MOTIF_CMAP)
    colorbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.025, fraction=0.055)
    colorbar.set_label("Mean GCNGG positions mutated", fontsize=10)
    colorbar.set_ticks(np.arange(color_max + 1))

    fig.subplots_adjust(left=0.11, right=0.9, bottom=0.14, top=0.82)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_activity_motif_histogram" / "outputs")
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    with np.load(args.library) as library:
        sequences = ALPHABET[library["nuc_ids"].astype(np.uint8)]
        scores = library["scores_deepsquid_vts1"].astype(float)

    wt = np.asarray(list(WT))
    motif_positions = MOTIF_START + MOTIF_OFFSETS
    motif_mutations = np.sum(
        sequences[:, motif_positions] != wt[motif_positions], axis=1
    ).astype(float)
    edges, density, mean_motif_mutations, counts = summarize(
        scores, motif_mutations, args.bins
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "vts1_activity_motif_histogram.png"
    summary_path = args.out_dir / "vts1_activity_motif_histogram.csv"
    make_figure(edges, density, mean_motif_mutations, figure_path)
    write_summary(summary_path, edges, density, mean_motif_mutations, counts)

    print(f"Saved {figure_path}")
    print(f"Saved {summary_path}")
    print(f"Sequences: {len(scores):,}")
    print(
        f"Mean {MOTIF_LABEL} positions mutated: {np.mean(motif_mutations):.3f} "
        f"of {len(MOTIF_OFFSETS)} constrained positions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
