#!/usr/bin/env python3
"""SIMULATED PLACEHOLDER: expected Synthetic GT holdout scatter patterns."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "outputs/synthetic_gt_scatter_expectation.png"
RANDOM_COLOR = "#D94B45"
UNIFORM_COLOR = "#3776B6"


def clipped(values):
    return np.clip(values, 0, 1)


def rho(x, y):
    return spearmanr(x, y).statistic


def main():
    rng = np.random.default_rng(14)

    # Negative control: both evaluation regimes closely follow the diagonal.
    neg_random_x = rng.uniform(0.05, 0.95, 130)
    neg_random_y = clipped(neg_random_x + rng.normal(0, 0.035, neg_random_x.size))
    neg_uniform_x = rng.uniform(0.03, 0.97, 130)
    neg_uniform_y = clipped(neg_uniform_x + rng.normal(0, 0.045, neg_uniform_x.size))

    # Existing Synthetic GT expectation: random holdout occupies the lower-left
    # diagonal region, while uniform holdout spans the grid without agreement.
    gt_random_x = rng.beta(1.7, 5.0, 150) * 0.58
    gt_random_y = clipped(gt_random_x + rng.normal(0, 0.025, gt_random_x.size))
    gt_uniform_x = rng.uniform(0.03, 0.97, 180)
    gt_uniform_y = rng.uniform(0.03, 0.97, 180)

    panels = [
        (
            "Negative-control Synthetic GT",
            neg_random_x, neg_random_y, neg_uniform_x, neg_uniform_y,
        ),
        (
            "Synthetic GT",
            gt_random_x, gt_random_y, gt_uniform_x, gt_uniform_y,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), sharex=True, sharey=True)
    for ax, (title, random_x, random_y, uniform_x, uniform_y) in zip(axes, panels):
        ax.scatter(
            uniform_x, uniform_y, s=18, alpha=0.48,
            color=UNIFORM_COLOR, edgecolors="none", label="Uniform holdout",
        )
        ax.scatter(
            random_x, random_y, s=18, alpha=0.58,
            color=RANDOM_COLOR, edgecolors="none", label="Random-library holdout",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", linewidth=1.3)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=34)
        ax.text(
            0.5, 1.03,
            f"Random ρ = {rho(random_x, random_y):.2f}   |   Uniform ρ = {rho(uniform_x, uniform_y):.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#DDDDDD", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("Ground-truth activity")
    axes[0].set_ylabel("Predicted activity")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Expected holdout behavior (simulated placeholder)", fontsize=14, fontweight="bold", y=1.10)
    fig.text(
        0.5, 0.01,
        "Illustrative simulated points only — values are not experimental results",
        ha="center", color="#666666", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
