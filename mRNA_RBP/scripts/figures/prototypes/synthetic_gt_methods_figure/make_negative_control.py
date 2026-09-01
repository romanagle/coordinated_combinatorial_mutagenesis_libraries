#!/usr/bin/env python3
"""Plot the instance-00 negative-control Synthetic GT coefficients."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "synthetic_gt_methods_figure"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT.parent))

from mRNA_RBP.src import viz


SOURCE = ROOT / "runs" / "synthetic_negative_control" / "instance_00" / "gt_params.npz"
OUTPUT = HERE / "outputs" / "synthetic_gt_negative_control_coefficients.png"


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing negative-control coefficients: {SOURCE}")

    with np.load(SOURCE) as data:
        alpha = data["alpha"].astype(np.float32)
        edges = [(int(i), int(j)) for i, j in data["edges"]]
        pairwise_tensor = data["J"].astype(np.float32)

    beta = {(i, j): pairwise_tensor[i, j] for i, j in edges}
    vmax = max(
        float(np.max(np.abs(alpha))),
        max(float(np.max(np.abs(matrix))) for matrix in beta.values()),
    )
    figure = viz.plot_coefficients(
        alpha,
        beta,
        len(alpha),
        stem_pairs=[],
        motif_positions=[],
        vmax=vmax,
    )
    legend = figure.axes[0].get_legend()
    if legend is not None:
        legend.remove()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
