#!/usr/bin/env python3
"""THROWAWAY PLACEHOLDER: 2 × 3 Synthetic GT activity-distribution figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_distribution_figure"
OUTPUT = HERE / "outputs/synthetic_gt_distribution_placeholder.png"

COLUMNS = [
    "Random training\nactivity distribution",
    "Random test\nactivity distribution",
    "Uniform test\nactivity distribution",
]
ROWS = [
    ("Negative-control Synthetic GT", "Not yet generated"),
    ("Synthetic GT", "Existing distribution to insert"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for row_index, (row_label, status) in enumerate(ROWS):
        for col_index, column_label in enumerate(COLUMNS):
            ax = axes[row_index, col_index]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.add_patch(
                Rectangle(
                    (0.04, 0.05), 0.92, 0.90,
                    facecolor="#FAFAFA", edgecolor="#555555",
                    linewidth=1.8, linestyle="--",
                )
            )
            ax.text(
                0.5, 0.54, column_label,
                ha="center", va="center", fontsize=11, fontweight="bold",
            )
            ax.text(
                0.5, 0.31, status,
                ha="center", va="center", fontsize=9, color="#777777",
            )
            if col_index == 0:
                ax.text(
                    -0.09, 0.5, row_label,
                    ha="right", va="center", rotation=90,
                    fontsize=11, fontweight="bold",
                    transform=ax.transAxes,
                )

    fig.suptitle(
        "Synthetic ground-truth activity distributions",
        fontsize=15, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.01,
        "Placeholder only — no distribution shapes or values are represented",
        ha="center", color="#666666", fontsize=9,
    )
    fig.tight_layout(rect=(0.08, 0.05, 1, 0.94), w_pad=1.2, h_pad=1.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
