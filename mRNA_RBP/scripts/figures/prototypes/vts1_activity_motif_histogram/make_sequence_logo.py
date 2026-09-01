"""Build a sequence logo for all positive-activity VTS1 sequences."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIBRARY = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz"
)
WT = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
WT_MOTIF = "GCUGG"
MOTIF_START = WT.index(WT_MOTIF)
MOTIF_OFFSETS = np.asarray([0, 1, 3, 4])
ALPHABET = np.asarray(list("ACGU"))
COLORS = {"A": "#2ca02c", "C": "#377eb8", "G": "#f29e2e", "U": "#d62728"}


def frequency_matrix(sequences: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            nucleotide: np.mean(sequences == nucleotide, axis=0)
            for nucleotide in ALPHABET
        }
    )


def make_logo(frequencies: pd.DataFrame, n_sequences: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 4.8), facecolor="white")
    logomaker.Logo(
        frequencies,
        ax=ax,
        color_scheme=COLORS,
        stack_order="small_on_top",
        fade_probabilities=False,
    )
    for offset in MOTIF_OFFSETS:
        position = MOTIF_START + int(offset)
        ax.axvspan(position - 0.5, position + 0.5, color="#f29e2e", alpha=0.1, zorder=-1)
    central_n = MOTIF_START + 2
    ax.axvspan(central_n - 0.5, central_n + 0.5, color="#888888", alpha=0.1, zorder=-1)

    ax.set_xlim(-0.6, len(WT) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Nucleotide frequency", fontsize=11)
    ax.set_xlabel("Position in the 41-nt VTS1 sequence", fontsize=11)
    ticks = np.arange(0, len(WT), 5)
    ax.set_xticks(ticks, labels=ticks + 1)
    ax.set_title(
        "Positive-activity VTS1 sequences",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0,
        1.025,
        f"ResidualBind activity score > 0; n = {n_sequences:,}. Orange marks constrained GCNGG positions; gray marks N.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555555",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.79)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_frequencies(path: Path, frequencies: pd.DataFrame) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["position", "wt_nucleotide", *ALPHABET])
        for position, row in frequencies.iterrows():
            writer.writerow(
                [position + 1, WT[position], *(row[nucleotide] for nucleotide in ALPHABET)]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_activity_motif_histogram" / "outputs")
    args = parser.parse_args()

    with np.load(args.library) as library:
        sequences = ALPHABET[library["nuc_ids"].astype(np.uint8)]
        scores = library["scores_deepsquid_vts1"].astype(float)

    selected = sequences[scores > 0]
    frequencies = frequency_matrix(selected)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.out_dir / "vts1_positive_activity_sequence_logo.png"
    frequency_path = args.out_dir / "vts1_positive_activity_nucleotide_frequencies.csv"
    make_logo(frequencies, len(selected), figure_path)
    write_frequencies(frequency_path, frequencies)
    print(f"Saved {figure_path}")
    print(f"Saved {frequency_path}")
    print(f"Selected sequences: {len(selected):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
