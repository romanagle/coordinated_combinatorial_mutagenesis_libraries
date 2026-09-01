"""THROWAWAY PROTOTYPE: high-WT entropy across prediction quantile bins.

Question: When the activity-balanced VTS1-high library is ordered by final
prediction from the 10%-mutation, 20K nonlinear additive-plus-pairwise
surrogate, where along the 41-nt sequence does diversity change across 30
equal-count prediction bins?
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from entropy_bins import prediction_quantile_bins, shannon_entropy_by_bin


ROOT = Path(__file__).resolve().parents[5]
TARGETS = {
    "vts1-high": {
        "label": "VTS1-high",
        "stem": "vts1_high",
        "library": ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz",
        "predictions": ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/scatter_by_mutcount_predictions.npz",
    },
    "hur-high": {
        "label": "HuR-high",
        "stem": "hur_high",
        "library": ROOT / "mRNA_RBP/runs/deepsquid/hur/high/instance_00/activity_balanced.npz",
        "predictions": ROOT / "mRNA_RBP/runs/deepsquid/hur/high/scatter_by_mutcount_predictions.npz",
    },
}
COLORS = LinearSegmentedColormap.from_list(
    "entropy_reference",
    ["#fff4e8", "#f46d43", "#d7194a", "#762a83", "#071326"],
)


def load_inputs(
    library_path: Path, predictions_path: Path, mutation_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(library_path) as library, np.load(predictions_path, allow_pickle=True) as predictions:
        sequences = library["nuc_ids"].astype(np.uint8)
        predicted = predictions[f"yhat_activity_{mutation_rate}"].astype(float)
        prediction_counts = predictions[f"rate_labels_{mutation_rate}"].astype(int)
        library_counts = library["rate_labels"].astype(int)
        model_name = str(predictions["model_name"])

    if sequences.shape != (20_000, 41):
        raise ValueError(f"Expected a (20000, 41) VTS1-high library, got {sequences.shape}")
    if len(predicted) != len(sequences) or not np.array_equal(library_counts, prediction_counts):
        raise ValueError("Library rows do not align with cached prediction rows")
    if model_name != "nonlinear additive + pairwise":
        raise ValueError(f"Unexpected cached model: {model_name}")
    return sequences, predicted


def write_assignments(path: Path, bins: list[np.ndarray], predictions: np.ndarray) -> None:
    bin_for_row = np.empty(len(predictions), dtype=int)
    for bin_number, indices in enumerate(bins, start=1):
        bin_for_row[indices] = bin_number
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence_row", "prediction_bin", "surrogate_prediction"])
        writer.writerows(
            (row, int(bin_for_row[row]), float(predictions[row])) for row in range(len(predictions))
        )


def make_figure(
    entropy: np.ndarray,
    medians: np.ndarray,
    label: str,
    mutation_rate: int,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(10.6, 7.0), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(12, 2.15), left=0.09, right=0.96, top=0.83, bottom=0.18, wspace=0.04
    )
    heat_ax = fig.add_subplot(grid[0, 0])
    pred_ax = fig.add_subplot(grid[0, 1], sharey=heat_ax)

    image = heat_ax.imshow(entropy, cmap=COLORS, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    heat_ax.set_xlabel("Position (nt)", fontsize=14)
    heat_ax.set_ylabel("Prediction bin", fontsize=14)
    heat_ax.set_xticks(np.arange(0, 41, 5), labels=np.arange(1, 42, 5))
    heat_ax.set_yticks([0, 9, 19, 29], labels=[1, 10, 20, 30])
    heat_ax.tick_params(labelsize=10)

    color_ax = fig.add_axes([0.39, 0.87, 0.34, 0.022])
    colorbar = fig.colorbar(image, cax=color_ax, orientation="horizontal", ticks=[0, 1, 2])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label("Shannon entropy (bits)", fontsize=11, labelpad=4)

    rows = np.arange(30)
    pred_ax.barh(rows, medians, height=0.78, color="#287db2", edgecolor="none")
    pred_ax.axvline(0.0, color="#e31a1c", lw=2.2, label="WT = 0")
    pred_ax.set_xlabel("Median prediction", fontsize=13)
    pred_ax.tick_params(axis="y", left=False, labelleft=False)
    pred_ax.tick_params(axis="x", labelsize=9)
    pred_ax.spines[["top", "right", "left"]].set_visible(False)
    pred_ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=9)

    fig.suptitle(
        f"{label} sequence diversity across surrogate prediction bins",
        x=0.5,
        y=0.975,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.055,
        "Activity-balanced evaluation (n=20,000) · 30 equal-count bins (n=666–667) · "
        f"{mutation_rate}% mutation, 20K random-trained nonlinear additive + pairwise surrogate",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=TARGETS, default="vts1-high")
    parser.add_argument("--mutation-rate", type=int, choices=(5, 10, 25), default=10)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_prediction_entropy" / "outputs")
    args = parser.parse_args()

    target = TARGETS[args.target]
    library_path = args.library or target["library"]
    predictions_path = args.predictions or target["predictions"]
    sequences, predicted = load_inputs(library_path, predictions_path, args.mutation_rate)
    bins = prediction_quantile_bins(predicted, n_bins=30)
    entropy = shannon_entropy_by_bin(sequences, bins)
    medians = np.asarray([np.median(predicted[indices]) for indices in bins])
    sizes = np.asarray([len(indices) for indices in bins])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"mut{args.mutation_rate:02d}"
    figure_path = args.out_dir / f"{target['stem']}_{suffix}_prediction_bin_entropy.png"
    assignments_path = args.out_dir / f"{target['stem']}_{suffix}_prediction_bin_assignments.csv"
    make_figure(entropy, medians, str(target["label"]), args.mutation_rate, figure_path)
    write_assignments(assignments_path, bins, predicted)

    print(f"Saved {figure_path}")
    print(f"Saved {assignments_path}")
    print(f"Bin sizes: {sizes.min()}–{sizes.max()}; median predictions: {medians[0]:.3f}–{medians[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
