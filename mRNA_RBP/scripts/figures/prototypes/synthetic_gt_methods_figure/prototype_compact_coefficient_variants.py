#!/usr/bin/env python3
"""THROWAWAY PROTOTYPE: three compact coefficient-map variants using real GT data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_methods_figure"
ROOT = HERE.parents[1]
OUTPUT = HERE / "outputs" / "prototype_compact_coefficient_variants.png"
SOURCES = [
    ("Positive control", ROOT / "outputs" / "instance_00" / "gt_params.npz"),
    (
        "Negative control",
        ROOT / "runs" / "synthetic_negative_control" / "instance_00" / "gt_params.npz",
    ),
]
VARIANTS = [
    "A — Additive strip + interaction arcs",
    "B — Additive strip + sparse edge map",
    "C — Additive strip + interaction tiles",
]
NUCLEOTIDE_ORDER = [1, 2, 3, 0]  # cached A,C,G,U -> displayed C,G,U,A
NUCLEOTIDE_LABELS = ["C", "G", "U", "A"]
POSITIVE_STEMS = set(range(8, 16)) | set(range(23, 31))
POSITIVE_MOTIF = set(range(17, 22))


def load(path: Path) -> dict:
    with np.load(path) as data:
        alpha = data["alpha"].astype(float)
        edges = [(int(i), int(j)) for i, j in data["edges"]]
        tensor = data["J"].astype(float)
    beta = [tensor[i, j] for i, j in edges]
    return {"alpha": alpha, "edges": edges, "beta": beta}


def strongest(matrix: np.ndarray) -> float:
    flat = matrix.ravel()
    return float(flat[np.argmax(np.abs(flat))])


def additive_strip(ax, data: dict, positive: bool) -> None:
    alpha = data["alpha"].T[NUCLEOTIDE_ORDER]
    vmax = max(float(np.max(np.abs(alpha))), 1e-6)
    ax.imshow(alpha, cmap="RdBu", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
              aspect="equal", interpolation="nearest")
    if positive:
        for positions, color in ((POSITIVE_STEMS, "#1E8449"), (POSITIVE_MOTIF, "#85C1E9")):
            for start, end in contiguous(positions):
                ax.add_patch(Rectangle((start - .5, -.5), end - start + 1, 4,
                                       fill=False, edgecolor=color, linewidth=1.8))
    ax.set_yticks(range(4), NUCLEOTIDE_LABELS, fontsize=6)
    ax.set_xticks(range(0, 41, 10))
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.set_xlim(-.5, 40.5)


def contiguous(positions: set[int]) -> list[tuple[int, int]]:
    values = sorted(positions)
    spans = []
    start = end = values[0]
    for value in values[1:]:
        if value == end + 1:
            end = value
        else:
            spans.append((start, end))
            start = end = value
    spans.append((start, end))
    return spans


def draw_arcs(ax, data: dict) -> None:
    values = np.array([strongest(matrix) for matrix in data["beta"]])
    vmax = max(float(np.max(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for (i, j), value in zip(data["edges"], values):
        distance = j - i
        arc = FancyArrowPatch(
            (i, 0), (j, 0), arrowstyle="-",
            connectionstyle=f"arc3,rad={-min(.75, .18 + distance / 55):.3f}",
            color=plt.cm.RdBu(norm(value)), linewidth=1.5 + 3.5 * abs(value) / vmax,
        )
        ax.add_patch(arc)
        ax.scatter([i, j], [0, 0], s=10, color="#333333", zorder=4)
    ax.axhline(0, color="#777777", linewidth=.7)
    ax.set_xlim(-1, 41)
    ax.set_ylim(-16, 1.5)
    ax.set_xticks(range(0, 41, 5))
    ax.tick_params(axis="x", labelsize=6)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set_xlabel("Sequence position", fontsize=7)


def draw_sparse_map(ax, data: dict) -> None:
    values = np.array([strongest(matrix) for matrix in data["beta"]])
    vmax = max(float(np.max(np.abs(values))), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    x = [j for i, j in data["edges"]]
    y = [i for i, j in data["edges"]]
    sizes = 45 + 190 * np.abs(values) / vmax
    ax.scatter(x, y, s=sizes, c=values, cmap="RdBu", norm=norm,
               marker="s", edgecolors="#333333", linewidths=.55)
    ax.plot([0, 40], [0, 40], color="#CCCCCC", linewidth=.7)
    ax.set_xlim(-1, 41)
    ax.set_ylim(41, -1)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, 41, 10))
    ax.set_yticks(range(0, 41, 10))
    ax.tick_params(labelsize=6)
    ax.set_xlabel("Position j", fontsize=7)
    ax.set_ylabel("Position i", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def draw_tiles(parent_ax, data: dict) -> None:
    parent_ax.axis("off")
    vmax = max(float(np.max(np.abs(matrix))) for matrix in data["beta"])
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for index, ((i, j), matrix) in enumerate(zip(data["edges"], data["beta"])):
        row, col = divmod(index, 4)
        ax = parent_ax.inset_axes([col * .25 + .025, .54 - row * .52, .20, .40])
        ax.imshow(matrix, cmap="RdBu", norm=norm, interpolation="nearest")
        ax.set_title(f"{i}–{j}", fontsize=6, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> None:
    datasets = [load(path) for _, path in SOURCES]
    fig = plt.figure(figsize=(15, 7.8))
    outer = fig.add_gridspec(2, 3, wspace=.27, hspace=.36)
    for col, title in enumerate(VARIANTS):
        fig.text((col + .5) / 3, .955, title, ha="center", va="top",
                 fontsize=11, fontweight="bold")
    for row, ((row_label, _), data) in enumerate(zip(SOURCES, datasets)):
        fig.text(.018, .70 if row == 0 else .285, row_label, rotation=90,
                 ha="center", va="center", fontsize=10, fontweight="bold")
        for col in range(3):
            inner = outer[row, col].subgridspec(2, 1, height_ratios=(1, 3.1), hspace=.28)
            add_ax = fig.add_subplot(inner[0])
            additive_strip(add_ax, data, positive=(row == 0))
            detail_ax = fig.add_subplot(inner[1])
            if col == 0:
                draw_arcs(detail_ax, data)
            elif col == 1:
                draw_sparse_map(detail_ax, data)
            else:
                draw_tiles(detail_ax, data)
    fig.suptitle("PROTOTYPE — compact Synthetic GT coefficient displays",
                 fontsize=15, fontweight="bold", y=.995)
    fig.text(.5, .012,
             "Real instance-00 coefficients; each control uses its own symmetric color scale",
             ha="center", fontsize=8, color="#555555")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
