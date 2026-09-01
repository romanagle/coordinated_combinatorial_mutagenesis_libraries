"""Map VTS1-high mutations across oracle-activity quantile bins."""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIBRARY = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz"
)
WT = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
MOTIF = "GCUGG"
MOTIF_START = WT.index(MOTIF)
ALPHABET = np.asarray(list("ACGU"))
MUTATION_CMAP = LinearSegmentedColormap.from_list(
    "mutation_frequency", ["#f7fbff", "#6baed6", "#08306b"]
)


def expected_motif_intact(mutation_counts: np.ndarray) -> float:
    """Probability of missing all motif positions under uniform exact-n mutation."""
    length = len(WT)
    motif_length = len(MOTIF)
    probabilities = [
        comb(length - motif_length, int(n)) / comb(length, int(n))
        if n <= length - motif_length
        else 0.0
        for n in mutation_counts
    ]
    return float(np.mean(probabilities))


def summarize(
    sequences: np.ndarray, scores: np.ndarray, mutation_counts: np.ndarray, n_bins: int
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    bins = list(np.array_split(order, n_bins))
    wt = np.asarray(list(WT))
    mutation_frequency = np.asarray(
        [np.mean(sequences[index] != wt, axis=0) for index in bins]
    )
    motif_slice = slice(MOTIF_START, MOTIF_START + len(MOTIF))
    motif_intact = np.asarray(
        [np.mean(np.all(sequences[index, motif_slice] == wt[motif_slice], axis=1)) for index in bins]
    )
    motif_null = np.asarray([expected_motif_intact(mutation_counts[index]) for index in bins])
    activity_medians = np.asarray([np.median(scores[index]) for index in bins])
    mean_mutations = np.asarray([np.mean(mutation_counts[index]) for index in bins])
    return bins, mutation_frequency, motif_intact, motif_null, activity_medians, mean_mutations


def write_summary(
    path: Path,
    bins: list[np.ndarray],
    activity_medians: np.ndarray,
    motif_intact: np.ndarray,
    motif_null: np.ndarray,
    mean_mutations: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "activity_bin",
                "n_sequences",
                "median_oracle_activity",
                "motif_intact_fraction",
                "mutation_count_matched_null_fraction",
                "mean_mutations",
            ]
        )
        for row, index in enumerate(bins):
            writer.writerow(
                [
                    row + 1,
                    len(index),
                    activity_medians[row],
                    motif_intact[row],
                    motif_null[row],
                    mean_mutations[row],
                ]
            )


def make_figure(
    mutation_frequency: np.ndarray,
    motif_intact: np.ndarray,
    motif_null: np.ndarray,
    activity_medians: np.ndarray,
    out_path: Path,
) -> None:
    n_bins = len(activity_medians)
    rows = np.arange(n_bins)
    fig = plt.figure(figsize=(12.4, 7.4), facecolor="white")
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(4.5, 1.55),
        left=0.08,
        right=0.96,
        top=0.79,
        bottom=0.19,
        wspace=0.12,
    )
    heat_ax = fig.add_subplot(grid[0, 0])
    motif_ax = fig.add_subplot(grid[0, 1], sharey=heat_ax)

    image = heat_ax.imshow(
        100 * mutation_frequency,
        cmap=MUTATION_CMAP,
        vmin=0,
        vmax=40,
        aspect="auto",
        interpolation="nearest",
    )
    heat_ax.axvspan(MOTIF_START - 0.5, MOTIF_START + len(MOTIF) - 0.5, color="#f28e2b", alpha=0.19)
    heat_ax.vlines(
        [MOTIF_START - 0.5, MOTIF_START + len(MOTIF) - 0.5],
        -0.5,
        n_bins - 0.5,
        colors="#d55e00",
        linewidth=1.4,
    )
    heat_ax.text(
        MOTIF_START + (len(MOTIF) - 1) / 2,
        -1.25,
        "GCUGG",
        color="#a94700",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    heat_ax.set_xlabel("Position in the 41-nt VTS1 sequence", fontsize=12)
    heat_ax.set_ylabel("Actual activity bin (low → high)", fontsize=12)
    heat_ax.set_xticks(np.arange(0, len(WT), 5), labels=np.arange(1, len(WT) + 1, 5))
    heat_ax.set_yticks([0, 4, 9, 14, 19], labels=[1, 5, 10, 15, 20])

    color_ax = fig.add_axes([0.29, 0.835, 0.31, 0.022])
    colorbar = fig.colorbar(image, cax=color_ax, orientation="horizontal", ticks=[0, 20, 40])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Sequences mutated at position (%)", fontsize=10, labelpad=3)

    motif_ax.plot(100 * motif_intact, rows, "o-", color="#d55e00", lw=2.2, ms=4.8, label="Observed")
    motif_ax.plot(100 * motif_null, rows, "--", color="#777777", lw=1.8, label="Random-position null")
    motif_ax.set_xlim(0, 100)
    motif_ax.set_xticks([0, 50, 100])
    motif_ax.set_xlabel("Full GCUGG motif intact (%)", fontsize=11)
    motif_ax.tick_params(axis="y", left=False, labelleft=False)
    motif_ax.spines[["top", "right", "left"]].set_visible(False)
    motif_ax.grid(axis="x", color="#dddddd", lw=0.8)
    motif_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)

    fig.suptitle(
        "Where mutations occur across the VTS1 activity-balanced library",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.52,
        0.075,
        "20 equal-count bins of actual deepSQUID VTS1 activity (n=1,000 each). "
        "The null accounts for each bin's mixture of 3-, 5-, 7-, and 15-mutation sequences.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.text(
        0.52,
        0.045,
        f"Median activity spans {activity_medians[0]:.2f} to {activity_medians[-1]:.2f}; "
        "orange shading marks WT positions 21–25.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_activity_mutation_map" / "outputs")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()

    with np.load(args.library) as library:
        sequences = ALPHABET[library["nuc_ids"].astype(np.uint8)]
        scores = library["scores_deepsquid_vts1"].astype(float)
        mutation_counts = library["rate_labels"].astype(int)

    if sequences.shape[1] != len(WT):
        raise ValueError(f"Expected {len(WT)}-nt sequences, got {sequences.shape[1]}")
    bins, mutation_frequency, motif_intact, motif_null, medians, mean_mutations = summarize(
        sequences, scores, mutation_counts, args.bins
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "vts1_activity_mutation_map.png"
    summary_path = args.out_dir / "vts1_activity_motif_summary.csv"
    make_figure(mutation_frequency, motif_intact, motif_null, medians, figure_path)
    write_summary(summary_path, bins, medians, motif_intact, motif_null, mean_mutations)

    motif_slice = slice(MOTIF_START, MOTIF_START + len(MOTIF))
    overall_intact = np.mean(
        np.all(sequences[:, motif_slice] == np.asarray(list(WT))[motif_slice], axis=1)
    )
    print(f"Saved {figure_path}")
    print(f"Saved {summary_path}")
    print(f"Overall motif intact: {100 * overall_intact:.2f}%")
    print(
        "Lowest/middle/highest bin motif intact: "
        f"{100 * motif_intact[0]:.1f}% / {100 * motif_intact[len(motif_intact)//2]:.1f}% / "
        f"{100 * motif_intact[-1]:.1f}%"
    )
    print(
        "Matched null for same bins: "
        f"{100 * motif_null[0]:.1f}% / {100 * motif_null[len(motif_null)//2]:.1f}% / "
        f"{100 * motif_null[-1]:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
