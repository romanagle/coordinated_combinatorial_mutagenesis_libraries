#!/usr/bin/env python3
"""Populate the 2 x 3 synthetic-GT control activity-distribution figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_distribution_figure"
MRNA_RBP = HERE.parents[1]
OUTPUT = HERE / "outputs" / "synthetic_gt_activity_distributions.png"
SCORE_KEY = "scores_nonlin_additive_pairwise"
SPLIT_SEED = 20260809
COEFFICIENT_SOURCES = [
    MRNA_RBP / "outputs" / "instance_00" / "gt_params.npz",
    MRNA_RBP / "runs" / "mrna_negative_control" / "instance_00" / "gt_params.npz",
]
NUCLEOTIDE_ORDER = [1, 2, 3, 0]  # cached A,C,G,U -> displayed C,G,U,A
NUCLEOTIDE_LABELS = ["C", "G", "U", "A"]
POSITIVE_STEMS = set(range(8, 16)) | set(range(23, 31))
POSITIVE_MOTIF = set(range(17, 22))

ROWS = [
    (
        "Positive control\n(structured Synthetic GT)",
        MRNA_RBP / "outputs" / "instance_00",
        "#4C72B0",
    ),
    (
        "Negative control\n(motif-only Synthetic GT)",
        MRNA_RBP / "runs" / "mrna_negative_control" / "instance_00",
        "#DD8452",
    ),
]

COLUMNS = [
    "Random model-development set\n(training + validation)",
    "Held-out random test set",
    "Activity-balanced evaluation",
]


def load_scores(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required library: {path}")
    with np.load(path) as data:
        if SCORE_KEY not in data.files:
            raise KeyError(f"{path} does not contain {SCORE_KEY!r}")
        return data[SCORE_KEY].astype(np.float64)


def split_random_library(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the fitter's 60/20/20 allocation, combining train + validation."""
    rng = np.random.default_rng(SPLIT_SEED)
    assignments = rng.choice(3, size=len(scores), p=(0.6, 0.2, 0.2))
    return scores[assignments != 2], scores[assignments == 2]


def load_row(instance_dir: Path) -> list[np.ndarray]:
    random_scores = load_scores(instance_dir / "mut10" / "lib_20000.npz")
    development, test = split_random_library(random_scores)
    activity_balanced = load_scores(instance_dir / "activity_balanced.npz")
    return [development, test, activity_balanced]


def load_coefficients(path: Path) -> tuple[np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing coefficient data: {path}")
    with np.load(path) as data:
        alpha = data["alpha"].astype(np.float64)
        edges = [(int(i), int(j)) for i, j in data["edges"]]
        tensor = data["J"].astype(np.float64)
    return alpha, edges, [tensor[i, j] for i, j in edges]


def contiguous_spans(positions: set[int]) -> list[tuple[int, int]]:
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


def draw_compact_coefficients(
    parent_ax: plt.Axes,
    coefficient_source: Path,
    *,
    structured: bool,
) -> None:
    alpha, edges, beta = load_coefficients(coefficient_source)
    parent_ax.axis("off")

    # The square pairwise axes below occupies the centered 53.4% of this
    # parent after Matplotlib enforces equal x/y scaling. Use that same actual
    # horizontal footprint for the additive strip.
    additive_ax = parent_ax.inset_axes([0.233, 0.80, 0.534, 0.17])
    additive_vmax = max(float(np.max(np.abs(alpha))), 1e-6)
    additive_ax.imshow(
        alpha.T[NUCLEOTIDE_ORDER],
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-additive_vmax, vcenter=0.0, vmax=additive_vmax),
        aspect="equal",
        interpolation="nearest",
    )
    if structured:
        for positions, color in ((POSITIVE_STEMS, "#1E8449"), (POSITIVE_MOTIF, "#85C1E9")):
            for start, end in contiguous_spans(positions):
                additive_ax.add_patch(
                    Rectangle(
                        (start - 0.5, -0.5),
                        end - start + 1,
                        4,
                        fill=False,
                        edgecolor=color,
                        linewidth=1.5,
                    )
                )
    additive_ax.set_yticks(range(4), NUCLEOTIDE_LABELS, fontsize=5.5)
    additive_ax.set_xticks(range(0, 41, 10))
    additive_ax.tick_params(axis="x", labelsize=5.5, length=2, pad=1)
    additive_ax.tick_params(axis="y", length=0, pad=1)

    # Summarize each 4 x 4 nucleotide-coupling block by its strongest signed
    # coefficient, then show those position-pair values in the conventional
    # upper-triangular interaction map.
    pairwise = np.zeros((alpha.shape[0], alpha.shape[0]), dtype=np.float64)
    for (i, j), matrix in zip(edges, beta):
        strongest_index = np.argmax(np.abs(matrix))
        pairwise[i, j] = matrix.flat[strongest_index]

    upper_triangle = np.triu(np.ones_like(pairwise, dtype=bool), k=1)
    pairwise_vmax = max(float(np.max(np.abs(pairwise[upper_triangle]))), 1e-6)
    pairwise_norm = TwoSlopeNorm(vmin=-pairwise_vmax, vcenter=0.0, vmax=pairwise_vmax)
    pairwise_masked = np.ma.masked_where(~upper_triangle, pairwise)
    pairwise_cmap = plt.colormaps["RdBu"].copy()
    pairwise_cmap.set_bad("white")

    # Match the additive strip's horizontal bounds while keeping the pairwise
    # map square, so sequence positions align vertically across both maps.
    pairwise_ax = parent_ax.inset_axes([0.12, 0.01, 0.76, 0.70])
    pairwise_ax.imshow(
        pairwise_masked,
        cmap=pairwise_cmap,
        norm=pairwise_norm,
        aspect="equal",
        interpolation="nearest",
    )
    pairwise_ax.plot(
        [-0.5, alpha.shape[0] - 0.5],
        [-0.5, alpha.shape[0] - 0.5],
        color="#999999",
        linewidth=0.7,
    )
    if structured:
        for i, j in edges:
            pairwise_ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#1E8449",
                    linewidth=0.8,
                )
            )
    ticks = range(0, alpha.shape[0], 10)
    pairwise_ax.set_xticks(ticks)
    pairwise_ax.set_yticks(ticks)
    pairwise_ax.tick_params(labelsize=5.5, length=2, pad=1)
    pairwise_ax.xaxis.tick_top()
    pairwise_ax.xaxis.set_label_position("top")
    pairwise_ax.yaxis.tick_right()
    pairwise_ax.yaxis.set_label_position("right")
    pairwise_ax.set_xlabel("Position j", fontsize=6.5, labelpad=2)
    pairwise_ax.set_ylabel("Position i", fontsize=6.5, labelpad=2)


def main() -> None:
    row_scores = [load_row(instance_dir) for _, instance_dir, _ in ROWS]
    all_scores = np.concatenate([scores for row in row_scores for scores in row])
    x_min = float(np.min(all_scores))
    x_pad = max(0.02, 0.035 * (0.0 - x_min))
    x_limits = (x_min - x_pad, x_pad)
    bins = np.linspace(*x_limits, 55)
    kde_x = np.linspace(*x_limits, 500)

    fig = plt.figure(figsize=(17.5, 8.5))
    grid = fig.add_gridspec(2, 4, width_ratios=(1.08, 1, 1, 1))
    axes = np.empty((2, 3), dtype=object)
    for row_index in range(2):
        for col_index in range(3):
            share_axis = axes[0, 0] if row_index or col_index else None
            axes[row_index, col_index] = fig.add_subplot(
                grid[row_index, col_index + 1],
                sharex=share_axis,
                sharey=share_axis,
            )
    coefficient_axes = [fig.add_subplot(grid[row_index, 0]) for row_index in range(2)]
    for row_index, ((row_label, _, color), panels) in enumerate(zip(ROWS, row_scores)):
        for col_index, (ax, scores) in enumerate(zip(axes[row_index], panels)):
            ax.hist(
                scores,
                bins=bins,
                density=True,
                color=color,
                alpha=0.48,
                edgecolor="white",
                linewidth=0.35,
            )
            if np.unique(scores).size > 1:
                kde = gaussian_kde(scores)
                ax.plot(kde_x, kde(kde_x), color=color, linewidth=2.0)
            ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.25)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(COLUMNS[col_index], fontsize=11, fontweight="bold", pad=10)
            if col_index == 0:
                ax.set_ylabel("Density", fontsize=10, fontweight="bold")
            if row_index == 1:
                ax.set_xlabel("WT-referenced ground-truth activity", fontsize=10)

    for row_index, (ax, coefficient_source) in enumerate(
        zip(coefficient_axes, COEFFICIENT_SOURCES)
    ):
        draw_compact_coefficients(
            ax,
            coefficient_source,
            structured=(row_index == 0),
        )
        if row_index == 0:
            ax.set_title("Ground-truth coefficients", fontsize=11, fontweight="bold", pad=10)

    axes[0, 2].text(
        0.98,
        0.96,
        "Dashed line: WT = 0",
        transform=axes[0, 2].transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#333333",
    )
    fig.suptitle(
        "Synthetic ground-truth activity distributions",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    for y, (row_label, _, _) in zip((0.69, 0.295), ROWS):
        fig.text(
            0.018,
            y,
            row_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    fig.text(
        0.5,
        0.012,
        f"Instance 00; 10% mutation random library; fixed development/test split seed {SPLIT_SEED}",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.055, 0.045, 1, 0.95), w_pad=1.15, h_pad=1.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
