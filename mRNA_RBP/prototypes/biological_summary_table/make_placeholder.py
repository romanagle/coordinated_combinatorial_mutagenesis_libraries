#!/usr/bin/env python3
"""THROWAWAY PLACEHOLDER: grouped HuR/VTS1 summary-table layout."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs/hur_vts1_summary_table_placeholder.png"

ROWS = ["SSM", "5%", "10%", "25%"]
METRICS = ["Mean ρ ± SD", "Additive\ncosine", "Pairwise\ncosine"]
KNOWN_SSM_RHO = {"HuR": "0.865", "VTS1": "0.728"}


def cell(ax, x, y, w, h, text="", face="#FFFFFF", edge="#555555", weight="normal", size=9):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, fontweight=weight)


def main():
    condition_w = 1.2
    metric_w = 1.35
    row_h = 0.78
    header_h = 0.78
    group_h = 0.62
    metrics_per_landscape = len(METRICS)
    total_w = condition_w + 2 * metrics_per_landscape * metric_w
    total_h = 4 * row_h + header_h + group_h

    fig, ax = plt.subplots(figsize=(13.0, 5.2))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    # Group headers.
    cell(ax, 0, total_h - group_h - header_h, condition_w, group_h + header_h,
         "Training\ncondition", face="#E6E6E6", weight="bold", size=10)
    cell(ax, condition_w, total_h - group_h, metrics_per_landscape * metric_w, group_h,
         "HuR", face="#DDEAF5", weight="bold", size=11)
    cell(ax, condition_w + metrics_per_landscape * metric_w, total_h - group_h, metrics_per_landscape * metric_w, group_h,
         "VTS1", face="#E6E0F2", weight="bold", size=11)

    # Metric headers.
    for landscape_index in range(2):
        start_x = condition_w + landscape_index * metrics_per_landscape * metric_w
        for metric_index, metric in enumerate(METRICS):
            cell(ax, start_x + metric_index * metric_w, total_h - group_h - header_h,
                 metric_w, header_h, metric, face="#F0F0F0", weight="bold", size=9)

    # Data placeholders.
    for row_index, row_label in enumerate(ROWS):
        y = total_h - group_h - header_h - (row_index + 1) * row_h
        cell(ax, 0, y, condition_w, row_h, row_label, face="#F7F7F7", weight="bold", size=10)
        for landscape_index, landscape in enumerate(["HuR", "VTS1"]):
            start_x = condition_w + landscape_index * metrics_per_landscape * metric_w
            for metric_index in range(metrics_per_landscape):
                text = "TBD"
                face = "#FFFFFF"
                if row_label == "SSM" and metric_index == 0:
                    text = f"{KNOWN_SSM_RHO[landscape]} ± TBD"
                if row_label == "SSM" and metric_index == 2:
                    text = "N/A"
                    face = "#D3D3D3"
                cell(ax, start_x + metric_index * metric_w, y, metric_w, row_h, text, face=face)

    fig.suptitle("Biological landscape recovery summary — layout prototype", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.02,
        "Known SSM mean ρ values shown; all other entries await metric and condition definitions",
        ha="center", fontsize=9, color="#666666",
    )
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.93))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
