"""PROTOTYPE: three views of a 41 x 4 pairwise saturation landscape."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "pairwise_saturation_landscape"
LIBRARY = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/type3.npz"
WT_PATH = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/wt_seq.txt"
ALPHABET = np.asarray(list("ACGU"))
SCORE_KEY = "scores_deepsquid_vts1"
CMAP = "viridis"


def load_landscape():
    wt = WT_PATH.read_text().strip()
    wt_ids = np.asarray([np.flatnonzero(ALPHABET == base)[0] for base in wt])
    with np.load(LIBRARY) as data:
        sequences = data["nuc_ids"]
        scores = data[SCORE_KEY].astype(float)
        orders = data["rate_labels"]

    xs, ys, values = [], [], []
    single_x, single_values = [], []
    for sequence, score, order in zip(sequences, scores, orders):
        changed = np.flatnonzero(sequence != wt_ids)
        states = [4 * pos + int(sequence[pos]) for pos in changed]
        if order == 1:
            single_x.append(states[0])
            single_values.append(score)
        else:
            xs.append(states[0])
            ys.append(states[1])
            values.append(score)
    return (
        wt,
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(values),
        np.asarray(single_x),
        np.asarray(single_values),
    )


def style_axis(ax, *, ylabel=True):
    centers = np.arange(41) * 4 + 1.5
    labels = np.arange(1, 42)
    shown = np.arange(0, 41, 5)
    ax.set_xticks(centers[shown], labels[shown], fontsize=8)
    ax.set_yticks(centers[shown], labels[shown], fontsize=8)
    ax.set_xlim(-1, 164)
    ax.set_ylim(-1, 164)
    ax.set_xlabel("First mutation: position × nucleotide (A C G U)")
    if ylabel:
        ax.set_ylabel("Second mutation: position × nucleotide (A C G U)")
    for boundary in np.arange(0, 165, 4):
        ax.axvline(boundary - 0.5, color="#d9d9d9", lw=0.18, zorder=0)
        ax.axhline(boundary - 0.5, color="#d9d9d9", lw=0.18, zorder=0)


def add_header(fig, subtitle):
    fig.suptitle("Pairwise saturated mutagenesis landscape — VTS1", y=0.985,
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.947, subtitle, ha="center", color="#555555", fontsize=9.5)


def variant_a(wt, x, y, scores, single_x, single_scores):
    norm = Normalize(scores.min(), scores.max())
    fig, ax = plt.subplots(figsize=(10.2, 9.2), facecolor="white")
    dots = ax.scatter(x, y, c=scores, cmap=CMAP, norm=norm, s=8, linewidths=0,
                      rasterized=True)
    ax.scatter(single_x, single_x, c=single_scores, cmap=CMAP, norm=norm,
               s=18, edgecolors="white", linewidths=0.25, zorder=3)
    style_axis(ax)
    ax.plot([-1, 164], [-1, 164], color="#777777", lw=0.65, zorder=1)
    ax.set_aspect("equal")
    ax.text(3, 156, "7,380 double mutants", fontsize=9, color="#444444")
    ax.text(91, 91, "123 singles\non diagonal", fontsize=8, color="#444444",
            rotation=45, ha="center")
    cbar = fig.colorbar(dots, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("Activity score")
    add_header(fig, "Each dot is one assayed sequence; blank states contain a WT nucleotide")
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.09, top=0.91)
    fig.savefig(HERE / "variant_a_upper_triangle_dots.png", dpi=220)
    plt.close(fig)


def variant_b(wt, x, y, scores, single_x, single_scores):
    matrix = np.full((164, 164), np.nan)
    matrix[y, x] = scores
    matrix[x, y] = scores
    matrix[single_x, single_x] = single_scores
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#f1f1f1")
    fig, ax = plt.subplots(figsize=(10.2, 9.2), facecolor="white")
    image = ax.imshow(matrix, origin="lower", cmap=cmap, interpolation="nearest",
                      aspect="equal")
    style_axis(ax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("Activity score")
    add_header(fig, "Symmetric lookup map; each double mutant appears twice for visual continuity")
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.09, top=0.91)
    fig.savefig(HERE / "variant_b_symmetric_heatmap.png", dpi=220)
    plt.close(fig)


def variant_c(wt, x, y, scores, single_x, single_scores):
    fig = plt.figure(figsize=(11.2, 8.7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    dots = ax.scatter(x, y, scores, c=scores, cmap=CMAP, s=4, alpha=0.82,
                      linewidths=0, rasterized=True)
    ax.scatter(single_x, single_x, single_scores, c=single_scores, cmap=CMAP,
               s=12, edgecolors="white", linewidths=0.2)
    ticks = np.arange(0, 41, 10) * 4 + 1.5
    ax.set_xticks(ticks, np.arange(1, 42, 10))
    ax.set_yticks(ticks, np.arange(1, 42, 10))
    ax.set_xlabel("First mutation position × nucleotide", labelpad=10)
    ax.set_ylabel("Second mutation position × nucleotide", labelpad=10)
    ax.set_zlabel("Activity score", labelpad=8)
    ax.view_init(elev=28, azim=-58)
    cbar = fig.colorbar(dots, ax=ax, fraction=0.025, pad=0.08, shrink=0.72)
    cbar.set_label("Activity score")
    add_header(fig, "3D alternative: activity is redundant in both height and color")
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.02, top=0.91)
    fig.savefig(HERE / "variant_c_3d_activity.png", dpi=220)
    plt.close(fig)


def main():
    landscape = load_landscape()
    variant_a(*landscape)
    variant_b(*landscape)
    variant_c(*landscape)
    print(f"Wrote three prototypes to {HERE}")


if __name__ == "__main__":
    main()
