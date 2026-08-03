"""THROWAWAY PROTOTYPE: render three mutation-rate figure concepts."""

from pathlib import Path
import json
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
import numpy as np


ROOT = HERE.parents[1]
MODEL = "nonlinear additive + pairwise"
LIBRARY_SIZE = "20000"
RATES = [5, 10, 25]
DATASETS = [
    (
        "Synthetic GT",
        ROOT / "outputs/ground_truth_collections/Synthetic GT/libraries_used_for_figures/cross_mutrate_results.json",
        "nonlin_additive_pairwise",
    ),
    ("deepSQUID VTS1\nhigh-WT", ROOT / "outputs_deepsquid_vts1_high/cross_mutrate_results.json", "deepsquid_vts1"),
    ("deepSQUID HuR\nhigh-WT", ROOT / "outputs_deepsquid_hur_high/cross_mutrate_results.json", "deepsquid_hur"),
]


def load_matrix(path, landscape):
    payload = json.loads(path.read_text())["cross"][landscape][MODEL]
    matrix = np.zeros((3, 3), dtype=float)
    for row, train_rate in enumerate(RATES):
        record = payload[str(train_rate)][LIBRARY_SIZE]
        matrix[row, row] = np.mean(record["same_rate"])
        for col, test_rate in enumerate(RATES):
            if col != row:
                matrix[row, col] = np.mean(record["cross"][str(test_rate)])
    return matrix


MATRICES = [(label, load_matrix(path, key)) for label, path, key in DATASETS]
NORM = Normalize(0, 1)
CMAP = plt.colormaps["viridis"]


def stamp(fig, title):
    fig.suptitle(title, x=0.06, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.06, 0.925, f"Spearman ρ · {MODEL} · library size 20,000", fontsize=10, color="#4b5563")
    fig.text(0.995, 0.01, "THROWAWAY PROTOTYPE", ha="right", fontsize=7, color="#9ca3af")


def annotate_matrix(ax, matrix, emphasize=False):
    for row in range(3):
        for col in range(3):
            color = "white" if matrix[row, col] < 0.60 else "#111827"
            weight = "bold" if emphasize and row > col else "normal"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=color, fontweight=weight, fontsize=10)


def variant_a():
    """Conventional matched heatmaps; shaded triangle makes direction explicit."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), constrained_layout=False)
    stamp(fig, "A · Matched matrices with a directional triangle")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.18, top=0.80, wspace=0.30)
    for idx, (ax, (label, matrix)) in enumerate(zip(axes, MATRICES)):
        im = ax.imshow(matrix, cmap=CMAP, norm=NORM)
        annotate_matrix(ax, matrix, emphasize=True)
        # Row > column means train rate is higher than test rate.
        ax.add_patch(Polygon([(-0.5, 0.5), (-0.5, 2.5), (1.5, 2.5)], closed=True, fill=False, edgecolor="#f97316", lw=3))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(3), [f"{r}%" for r in RATES])
        ax.set_yticks(range(3), [f"{r}%" for r in RATES])
        ax.set_xlabel("Test mutation rate")
        if idx == 0:
            ax.set_ylabel("Training mutation rate")
        else:
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([0.93, 0.23, 0.018, 0.52])
    fig.colorbar(im, cax=cax, label="Spearman ρ")
    fig.text(0.09, 0.075, "Orange outline: training mutation rate > test mutation rate", color="#c2410c", fontsize=10, fontweight="bold")
    fig.savefig(HERE / "variant_a_directional_matrices.png", dpi=200, facecolor="white")
    plt.close(fig)


def variant_b():
    """Lead with the three directional comparisons, leaving matrices behind."""
    comparisons = [(10, 5), (25, 5), (25, 10)]
    colors = ["#2563eb", "#ef4444", "#10b981"]
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    stamp(fig, "B · Direction-first comparison")
    fig.subplots_adjust(left=0.23, right=0.95, bottom=0.20, top=0.72)
    ybase = np.arange(3)[::-1]
    offsets = [0.22, 0, -0.22]
    for d_idx, ((label, matrix), color, off) in enumerate(zip(MATRICES, colors, offsets)):
        values = []
        for train, test in comparisons:
            values.append(matrix[RATES.index(train), RATES.index(test)])
        for y, value in zip(ybase + off, values):
            ax.plot([0, value], [y, y], color=color, alpha=0.20, lw=6, solid_capstyle="round")
            ax.scatter(value, y, s=110, color=color, edgecolor="white", linewidth=1.5, zorder=3, label=label.replace("\n", " ") if y == ybase[0] + off else None)
            ax.text(value + 0.018, y, f"{value:.2f}", va="center", fontsize=9, color=color, fontweight="bold")
    ax.set_yticks(ybase, ["Train 10% → test 5%", "Train 25% → test 5%", "Train 25% → test 10%"])
    ax.set_xlim(0, 1.07)
    ax.set_xlabel("Spearman ρ")
    ax.grid(axis="x", color="#e5e7eb")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.text(0.23, 0.065, "Only high-train-rate → low-test-rate cells are shown; full 3×3 matrices are omitted.", fontsize=9, color="#4b5563")
    fig.savefig(HERE / "variant_b_direction_first.png", dpi=200, facecolor="white")
    plt.close(fig)


def variant_c():
    """Compact matrix cards plus an honest reserved slot for mixed-rate training."""
    fig = plt.figure(figsize=(12, 6.5))
    stamp(fig, "C · Compact cards with a reserved mixed-rate row")
    grid = fig.add_gridspec(2, 3, left=0.07, right=0.95, bottom=0.12, top=0.80, height_ratios=[3, 1], hspace=0.45, wspace=0.32)
    for idx, (label, matrix) in enumerate(MATRICES):
        ax = fig.add_subplot(grid[0, idx])
        ax.imshow(matrix, cmap=CMAP, norm=NORM)
        annotate_matrix(ax, matrix)
        for row in range(3):
            for col in range(3):
                if row > col:
                    ax.add_patch(Rectangle((col - 0.46, row - 0.46), 0.92, 0.92, fill=False, edgecolor="#fb923c", lw=2))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(3), [f"{r}%" for r in RATES])
        ax.set_yticks(range(3), [f"{r}%" for r in RATES] if idx == 0 else [])
        ax.set_xlabel("Test rate")
        if idx == 0:
            ax.set_ylabel("Train rate")
        reserve = fig.add_subplot(grid[1, idx])
        reserve.set_facecolor("#f3f4f6")
        reserve.text(0.5, 0.62, "Mixed-rate training", ha="center", va="center", fontweight="bold", color="#6b7280")
        reserve.text(0.5, 0.30, "not available in current results", ha="center", va="center", fontsize=9, color="#9ca3af")
        reserve.set_xticks([]); reserve.set_yticks([])
        for spine in reserve.spines.values():
            spine.set_color("#d1d5db"); spine.set_linestyle("--")
    fig.text(0.07, 0.055, "Orange boxes: training mutation rate > test mutation rate · Gray row: layout reservation only; no values shown", fontsize=9, color="#4b5563")
    fig.savefig(HERE / "variant_c_mixed_rate_reservation.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    variant_a()
    variant_b()
    variant_c()
    print(f"Wrote three throwaway prototypes to {HERE}")
