"""Plot exact additive and pairwise matrices for both synthetic ground truths."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_coefficient_maps"
ALPHABET = np.asarray(list("ACGU"))
SOURCES = [
    ROOT / "mRNA_RBP/outputs/instance_00/gt_params.npz",
    ROOT / "mRNA_RBP/runs/mrna_negative_control/instance_00/gt_params.npz",
]
TITLES = [
    "Structured synthetic GT\npositive control",
    "Motif-only synthetic GT\nnegative control",
]
OUT = HERE / "both_synthetic_gt_additive_pairwise_matrices.png"
OUT_PDF = HERE / "both_synthetic_gt_additive_pairwise_matrices.pdf"


def load_maps(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        alpha = data["alpha"].astype(float)
        J = data["J"].astype(float)  # (position i, position j, nuc i, nuc j)
        edges = data["edges"].astype(int)
    length = len(alpha)
    matrix = np.full((4 * length, 4 * length), np.nan)
    for i, j in edges:
        matrix[4 * i : 4 * i + 4, 4 * j : 4 * j + 4] = J[i, j]
    return alpha, matrix, edges


def symmetric_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max()) if len(finite) else 0.0, 1e-8)


def position_axes(ax, length: int) -> None:
    positions = np.arange(0, length, 5)
    centers = 4 * positions + 1.5
    ax.set_xticks(centers, positions + 1, fontsize=8)
    ax.set_yticks(centers, positions + 1, fontsize=8)
    ax.set_xlabel("Position j", fontsize=9)
    ax.set_ylabel("Position i", fontsize=9)
    for boundary in np.arange(20, 4 * length, 20):
        ax.axhline(boundary - 0.5, color="black", lw=0.25, alpha=0.35)
        ax.axvline(boundary - 0.5, color="black", lw=0.25, alpha=0.35)


def main() -> None:
    datasets = [load_maps(path) for path in SOURCES]
    fig = plt.figure(figsize=(11.0, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 7])

    for column, ((alpha, pairwise, edges), title) in enumerate(zip(datasets, TITLES)):
        length = len(alpha)
        alpha_ax = fig.add_subplot(grid[0, column])
        pair_ax = fig.add_subplot(grid[1, column])

        alpha_vmax = symmetric_limit(alpha)
        alpha_im = alpha_ax.imshow(
            alpha.T,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-alpha_vmax, vcenter=0, vmax=alpha_vmax),
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        alpha_ax.set_title(title, fontsize=12, fontweight="bold")
        alpha_ax.set_yticks(range(4), ALPHABET, fontsize=8)
        alpha_ax.set_xticks(np.arange(0, length, 5))
        alpha_ax.tick_params(axis="x", labelbottom=False)
        alpha_ax.set_ylabel("Additive α", fontsize=9)
        alpha_cbar = fig.colorbar(alpha_im, ax=alpha_ax, fraction=0.025, pad=0.018)
        alpha_cbar.ax.tick_params(labelsize=7)

        pair_vmax = symmetric_limit(pairwise)
        pair_im = pair_ax.imshow(
            pairwise,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-pair_vmax, vcenter=0, vmax=pair_vmax),
            origin="upper",
            interpolation="nearest",
        )
        position_axes(pair_ax, length)
        pair_cbar = fig.colorbar(pair_im, ax=pair_ax, fraction=0.046, pad=0.025)
        pair_cbar.set_label("Exact pairwise coefficient J", fontsize=8)
        pair_cbar.ax.tick_params(labelsize=7)
        pair_ax.text(
            0.02,
            0.98,
            f"{len(edges)} nonzero interaction blocks",
            transform=pair_ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        if column == 0:
            pair_ax.text(-0.13, 0.5, "Pairwise matrix", transform=pair_ax.transAxes,
                         rotation=90, va="center", ha="center", fontsize=10,
                         fontweight="bold")

    fig.suptitle(
        "Exact additive and pairwise matrices for both synthetic ground truths — instance 00",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "Additive cells are square; A/C/G/U states are nested within each pairwise position block.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    HERE.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
