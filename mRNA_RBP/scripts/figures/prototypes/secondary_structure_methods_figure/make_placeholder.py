#!/usr/bin/env python3
"""THROWAWAY PLACEHOLDER: VTS1 and HuR secondary-structure renderings."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "secondary_structure_methods_figure"
OUTPUT = HERE / "outputs/vts1_hur_secondary_structure_placeholder.png"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for ax, label in zip(axes, ["VTS1 high-activity sequence", "HuR high-activity sequence"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            Rectangle(
                (0.05, 0.08), 0.90, 0.84,
                facecolor="#FAFAFA", edgecolor="#555555",
                linewidth=1.8, linestyle="--",
            )
        )
        ax.text(0.5, 0.58, label, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(
            0.5, 0.40, "Secondary-structure rendering\nto be inserted",
            ha="center", va="center", fontsize=10, color="#777777",
        )
    fig.suptitle("High-activity RNA secondary structures", fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.02, "Placeholder only", ha="center", fontsize=9, color="#666666")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92), w_pad=1.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
