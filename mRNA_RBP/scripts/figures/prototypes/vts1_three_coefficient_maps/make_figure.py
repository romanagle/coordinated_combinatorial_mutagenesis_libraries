"""Compare VTS1 pairwise maps from two 10%-mutation/20K fits and saturation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_three_coefficient_maps"
ALPHABET = np.asarray(list("ACGU"))
L = 41
RBP = "VTS1"
SCORE_KEY = "deepsquid_vts1"

SQUID = ROOT / (
    "mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle VTS1/"
    "libraries_used_for_figures/surrogate_coefs_high/"
    "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz"
)
DEEPSQUID = ROOT / (
    "mRNA_RBP/outputs/ground_truth_collections/deepSQUID VTS1/"
    "libraries_used_for_figures/surrogate_coefs_high/"
    "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_vts1.npz"
)
INSTANCE = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00"
OUT = HERE / "vts1_additive_pairwise_coefficient_maps_10pct_20k.png"
OUT_PDF = HERE / "vts1_additive_pairwise_coefficient_maps_10pct_20k.pdf"


def fitted_maps(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return additive alpha and upper-triangle 4L x 4L J maps."""
    with np.load(path) as data:
        alpha = data["alpha"].astype(float)
        J = data["J"].astype(float)
    matrix = J.transpose(0, 1, 2, 3).reshape(4 * L, 4 * L)
    position_i = np.repeat(np.arange(L), 4)[:, None]
    position_j = np.repeat(np.arange(L), 4)[None, :]
    matrix[position_i >= position_j] = np.nan
    return alpha, matrix


def saturation_maps() -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct additive single effects and double-mutant epistasis."""
    wt = (INSTANCE / "wt_seq.txt").read_text().strip()
    wt_ids = np.asarray([np.flatnonzero(ALPHABET == b)[0] for b in wt])
    with np.load(INSTANCE / "ssm.npz") as data:
        single_ids = data["nuc_ids"]
        single_scores = data[f"scores_{SCORE_KEY}"].astype(float)
        wt_score = float(data[f"wt_score_{SCORE_KEY}"][0])
    singles = {}
    alpha = np.zeros((L, 4), dtype=float)
    for seq, score in zip(single_ids, single_scores):
        changed = np.flatnonzero(seq != wt_ids)
        if len(changed) == 1:
            p = int(changed[0])
            singles[(p, int(seq[p]))] = score
            alpha[p, int(seq[p])] = score - wt_score

    matrix = np.full((4 * L, 4 * L), np.nan)
    with np.load(INSTANCE / "type3.npz") as data:
        sequences = data["nuc_ids"]
        scores = data[f"scores_{SCORE_KEY}"].astype(float)
        orders = data["rate_labels"]
    for seq, score, order in zip(sequences, scores, orders):
        if int(order) != 2:
            continue
        changed = np.flatnonzero(seq != wt_ids)
        if len(changed) != 2:
            continue
        i, j = map(int, changed)
        a, b = int(seq[i]), int(seq[j])
        epistasis = score - singles[(i, a)] - singles[(j, b)] + wt_score
        matrix[4 * i + a, 4 * j + b] = epistasis
    return alpha, matrix


def add_position_axes(ax):
    positions = np.arange(0, L, 5)
    centers = 4 * positions + 1.5
    ax.set_xticks(centers, positions + 1, fontsize=7)
    ax.set_yticks(centers, positions + 1, fontsize=7)
    ax.set_xlabel("Position j", fontsize=9)
    ax.set_ylabel("Position i", fontsize=9)
    for boundary in np.arange(4 * 5, 4 * L, 4 * 5):
        ax.axhline(boundary - 0.5, color="black", lw=0.22, alpha=0.35)
        ax.axvline(boundary - 0.5, color="black", lw=0.22, alpha=0.35)


def main():
    maps = [fitted_maps(SQUID), fitted_maps(DEEPSQUID), saturation_maps()]
    titles = [
        f"SQUID surrogate\nResidualBind {RBP}",
        f"deepSQUID surrogate\ndeepSQUID {RBP}",
        "Single/double saturation\n7,503 exhaustive variants",
    ]
    pair_labels = ["Latent pairwise coefficient J", "Latent pairwise coefficient J", "Activity epistasis"]

    fig = plt.figure(figsize=(15.8, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 7])
    pair_axes = []
    for column, ((alpha, matrix), title, pair_label) in enumerate(zip(maps, titles, pair_labels)):
        alpha_ax = fig.add_subplot(grid[0, column])
        ax = fig.add_subplot(grid[1, column])
        pair_axes.append(ax)

        alpha_abs = np.abs(alpha[np.isfinite(alpha)])
        alpha_vmax = float(np.quantile(alpha_abs, 0.995)) if len(alpha_abs) else 1.0
        alpha_im = alpha_ax.imshow(
            alpha.T,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-alpha_vmax, vcenter=0, vmax=alpha_vmax),
            origin="upper",
            aspect="equal",
            interpolation="nearest",
        )
        alpha_ax.set_title(title, fontsize=11, fontweight="bold")
        positions = np.arange(0, L, 5)
        alpha_ax.set_xticks(positions, positions + 1, fontsize=7)
        alpha_ax.set_yticks(range(4), ALPHABET, fontsize=8)
        alpha_ax.set_ylabel("Additive α", fontsize=9)
        alpha_ax.tick_params(axis="x", labelbottom=False)
        alpha_cbar = fig.colorbar(alpha_im, ax=alpha_ax, fraction=0.025, pad=0.018)
        alpha_cbar.ax.tick_params(labelsize=6)

        finite = np.abs(matrix[np.isfinite(matrix)])
        vmax = float(np.quantile(finite, 0.995)) if len(finite) else 1.0
        im = ax.imshow(
            matrix,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
            origin="upper",
            interpolation="nearest",
        )
        add_position_axes(ax)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(pair_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        if column == 0:
            ax.text(-0.13, 0.5, "Pairwise matrix", transform=ax.transAxes,
                    rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"{RBP}-high additive and pairwise maps — 10% mutation / 20K fits vs exhaustive saturation",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.01,
        "A/C/G/U states are nested within each position; colors are scaled independently per panel (99.5th percentile).",
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
