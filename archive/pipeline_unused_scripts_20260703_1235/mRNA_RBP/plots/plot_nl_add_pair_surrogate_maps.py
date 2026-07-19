"""
Standalone SQUID-style coefficient maps for the nonlinear additive+pairwise
surrogate.

Defaults to the Type3 cache for instance 0, mut_rate=10%, lib_size=20000.
Outputs are written under outputs/notebook_plots without replacing the older
notebook figures.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle, Patch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import SEQ, STEM_PAIRS, MOTIF_POSITIONS
import mRNA_RBP.viz as viz


OUT_DIR = os.path.join(_HERE, "outputs", "notebook_plots")
TYPE3_COEF_DIR = os.path.join(_HERE, "outputs", "surrogate_coefs_type3")
NUCS = ["A", "U", "G", "C"]

MOTIF_COLOR = "#AED6F1"
STEM_COLOR = "#1E8449"
PAIR_BOX_COLOR = "#1E8449"


def _coef_path(coef_dir, instance, mut_rate, lib_size):
    return os.path.join(
        coef_dir,
        f"coefs_k{instance:02d}_mut{mut_rate:02d}_lib{lib_size}"
        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz",
    )


def _contiguous_spans(positions):
    if not positions:
        return []
    sorted_positions = sorted(positions)
    spans = []
    start = end = sorted_positions[0]
    for pos in sorted_positions[1:]:
        if pos == end + 1:
            end = pos
        else:
            spans.append((start, end))
            start = end = pos
    spans.append((start, end))
    return spans


def _save(fig, stem, save_svg=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, stem + ".png")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    print(png)
    if save_svg:
        svg = os.path.join(OUT_DIR, stem + ".svg")
        fig.savefig(svg, bbox_inches="tight")
        print(svg)
    plt.close(fig)


def plot_additive_4xl(alpha, title):
    alpha = np.asarray(alpha, dtype=np.float32)
    L = alpha.shape[0]
    disp = alpha.T[[0, 3, 2, 1], :]
    vmax = max(float(np.abs(disp).max()), 1e-6)

    fig, ax = plt.subplots(figsize=(11.0, 2.1))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(
        disp, aspect="auto", cmap=plt.cm.RdBu, norm=norm,
        origin="lower", interpolation="nearest", zorder=1,
    )

    for start, end in _contiguous_spans(MOTIF_POSITIONS):
        ax.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.2, edgecolor=MOTIF_COLOR, facecolor="none", zorder=4,
        ))
    stem_positions = sorted({p for pair in STEM_PAIRS for p in pair})
    for start, end in _contiguous_spans(stem_positions):
        ax.add_patch(Rectangle(
            (start - 0.5, -0.5), end - start + 1, 4,
            linewidth=3.2, edgecolor=STEM_COLOR, facecolor="none", zorder=4,
        ))

    step = max(1, L // 15)
    ax.set_xticks(range(0, L, step))
    ax.set_xticklabels([str(i) for i in range(0, L, step)], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(NUCS, fontsize=9)
    ax.set_xlabel("Position", fontsize=9)
    ax.set_ylabel("Nucleotide", fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(
        handles=[
            Patch(edgecolor=MOTIF_COLOR, facecolor="none", linewidth=3.2, label="Motif"),
            Patch(edgecolor=STEM_COLOR, facecolor="none", linewidth=3.2, label="Stem"),
        ],
        loc="upper right", fontsize=8, framealpha=0.9,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("Additive coefficient", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig


def plot_pairwise_4lxl(J, title):
    J = np.asarray(J, dtype=np.float32)
    L = J.shape[0]
    mat = np.full((4 * L, 4 * L), np.nan, dtype=np.float32)
    for i in range(L):
        for j in range(i + 1, L):
            mat[4 * i:4 * i + 4, 4 * j:4 * j + 4] = J[i, :, j, :]

    finite = mat[np.isfinite(mat)]
    vmax = max(float(np.abs(finite).max()) if len(finite) else 0.0, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    im = ax.imshow(
        mat, aspect="equal", cmap=cmap, norm=norm,
        origin="upper", interpolation="nearest",
    )

    grid_kw = dict(color="#CCCCCC", linewidth=0.28, zorder=3)
    for k in range(L + 1):
        ax.axhline(4 * k - 0.5, **grid_kw)
        ax.axvline(4 * k - 0.5, **grid_kw)

    for i, j in STEM_PAIRS:
        ax.add_patch(Rectangle(
            (4 * j - 0.5, 4 * i - 0.5), 4, 4,
            linewidth=1.8, edgecolor=PAIR_BOX_COLOR, facecolor="none", zorder=4,
        ))

    ax.plot(
        [-0.5, 4 * L - 0.5], [-0.5, 4 * L - 0.5],
        color="#AAAAAA", linewidth=0.8, zorder=6,
    )
    step = max(1, L // 15)
    tick_pos = list(range(0, L, step))
    tick_cell = [4 * p + 1.5 for p in tick_pos]
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(tick_cell)
    ax.set_xticklabels([str(p) for p in tick_pos], fontsize=8)
    ax.set_xlabel("Position j", fontsize=9)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticks(tick_cell)
    ax.set_yticklabels([str(p) for p in tick_pos], fontsize=8)
    ax.set_ylabel("Position i", fontsize=9)
    ax.set_xlim(-0.5, 4 * L - 0.5)
    ax.set_ylim(4 * L - 0.5, -0.5)
    ax.set_title(title, fontsize=11, pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.05)
    cbar.set_label("Pairwise coefficient", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20000)
    parser.add_argument("--coef_dir", default=TYPE3_COEF_DIR)
    parser.add_argument("--out_prefix", default="type3_nl_add_pair_surrogate")
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    coef_path = _coef_path(args.coef_dir, args.instance, args.mut_rate, args.lib_size)
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(coef_path)

    coefs = np.load(coef_path)
    alpha = coefs["alpha"]
    J = coefs["J"]
    L = len(SEQ)
    beta_full = {
        (i, j): J[i, :, j, :]
        for i in range(L)
        for j in range(i + 1, L)
    }

    label = (
        "Nonlinear additive+pairwise surrogate "
        f"(k={args.instance}, mut={args.mut_rate}%, n={args.lib_size:,})"
    )

    fig_add = plot_additive_4xl(alpha, label + " additive 4xL map")
    _save(fig_add, args.out_prefix + "_additive_4xL", save_svg=args.svg)

    fig_pair = plot_pairwise_4lxl(J, label + " pairwise 4Lx4L epistasis map")
    _save(fig_pair, args.out_prefix + "_pairwise_4Lx4L", save_svg=args.svg)

    fig_combined = viz.plot_coefficients_4Lx4L(
        alpha, beta_full, L,
        stem_pairs=STEM_PAIRS,
        motif_positions=MOTIF_POSITIONS,
        title=label + " coefficients",
    )
    _save(fig_combined, args.out_prefix + "_combined_coefficients_4Lx4L", save_svg=args.svg)


if __name__ == "__main__":
    main()
