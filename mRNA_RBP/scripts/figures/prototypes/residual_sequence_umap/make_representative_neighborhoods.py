"""Show representative residual neighborhoods and their shared mutations."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from make_neighbor_graph import (
    DEFAULT_EMBEDDINGS,
    DEFAULT_GT,
    MUTATION_COUNTS,
    mutation_tokens,
    nearest_neighbors,
)


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "residual_sequence_umap"


def choose_neighborhoods(
    sequences: np.ndarray,
    residuals: np.ndarray,
    neighbor_indices: np.ndarray,
    wt: np.ndarray,
) -> list[tuple[str, int, np.ndarray]]:
    """Choose dense neighborhoods spanning negative, near-zero, and positive error."""
    groups = np.column_stack([np.arange(len(sequences)), neighbor_indices])
    means = residuals[groups].mean(axis=1)
    # Rank candidates by exact substitutions shared with their local group.
    density = np.asarray([
        np.mean(np.sum((sequences[group] == sequences[i]) & (sequences[i] != wt), axis=1))
        for i, group in enumerate(groups)
    ])
    quantile = max(1, len(sequences) // 5)
    order = np.argsort(means)
    candidate_sets = [
        ("lower-residual neighborhood", order[:quantile]),
        ("residual-near-zero neighborhood", np.argsort(np.abs(means))[:quantile]),
        ("higher-residual neighborhood", order[-quantile:]),
    ]

    selected: list[tuple[str, int, np.ndarray]] = []
    used: set[int] = set()
    for label, candidates in candidate_sets:
        ranked = candidates[np.argsort(density[candidates])[::-1]]
        chosen = None
        for center in ranked:
            group = groups[center]
            if not used.intersection(map(int, group)):
                chosen = int(center)
                break
        if chosen is None:
            chosen = int(ranked[0])
        group = groups[chosen]
        selected.append((label, chosen, group))
        used.update(map(int, group))
    return selected


def mutation_frequencies(group_sequences: np.ndarray, wt: np.ndarray):
    token_lists = [mutation_tokens(sequence, wt) for sequence in group_sequences]
    counts = Counter(token for tokens in token_lists for token in tokens)
    # Keep mutations carried by at least half the members; rarer changes obscure
    # the neighborhood's common sequence background, especially at 15 mutations.
    minimum = int(np.ceil(len(group_sequences) / 2))
    items = [(token, count) for token, count in counts.items() if count >= minimum]
    items.sort(key=lambda item: (-item[1], item[0]))
    return items, token_lists


def draw_local_graph(ax, group_residuals: np.ndarray, norm: TwoSlopeNorm) -> None:
    count = len(group_residuals) - 1
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    xy = np.column_stack([np.cos(angles), np.sin(angles)])
    for number, (point, residual) in enumerate(zip(xy, group_residuals[1:]), start=1):
        ax.plot([0, point[0]], [0, point[1]], color="#b0b0b0", lw=1, zorder=1)
        ax.scatter(*point, c=[residual], cmap="RdBu_r", norm=norm, s=90,
                   edgecolor="black", linewidth=0.4, zorder=2)
        ax.text(*point, str(number), ha="center", va="center", fontsize=7, zorder=3)
    ax.scatter(0, 0, c=[group_residuals[0]], cmap="RdBu_r", norm=norm,
               s=190, marker="*", edgecolor="black", linewidth=0.7, zorder=3)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--gt-params", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--neighbors", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    sequences = data["sequences"].astype(np.uint8)
    mutation_counts = data["mutation_counts"].astype(int)
    residuals = data["residuals"].astype(float)
    wt_string = str(np.load(args.gt_params)["wt_seq"])
    wt = np.asarray(["ACGU".index(base) for base in wt_string], dtype=np.uint8)
    vmax = float(np.percentile(np.abs(residuals), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mutation_count in MUTATION_COUNTS:
        mask = mutation_counts == mutation_count
        shell_sequences = sequences[mask]
        shell_residuals = residuals[mask]
        _, indices = nearest_neighbors(shell_sequences, args.neighbors)
        chosen = choose_neighborhoods(shell_sequences, shell_residuals, indices, wt)

        fig, axes = plt.subplots(
            3, 2, figsize=(13, 12),
            gridspec_kw={"width_ratios": (0.72, 1.5)},
            constrained_layout=True,
        )
        for row, (label, center, group) in enumerate(chosen):
            group_sequences = shell_sequences[group]
            group_residuals = shell_residuals[group]
            draw_local_graph(axes[row, 0], group_residuals, norm)
            axes[row, 0].set_title(
                f"{label}\nmean residual {group_residuals.mean():+.2f}", fontsize=11
            )

            items, _ = mutation_frequencies(group_sequences, wt)
            tokens = [item[0] for item in items][::-1]
            frequencies = np.asarray([item[1] / len(group) for item in items][::-1])
            core_cutoff = 2 / 3
            colors = ["#315a7d" if value >= core_cutoff else "#9ecae1" for value in frequencies]
            axes[row, 1].barh(tokens, frequencies, color=colors, edgecolor="white")
            axes[row, 1].axvline(core_cutoff, color="#315a7d", lw=1, linestyle="--")
            axes[row, 1].set_xlim(0, 1.05)
            axes[row, 1].set_xlabel("Fraction of neighborhood carrying exact mutation")
            axes[row, 1].tick_params(axis="y", labelsize=8)
            axes[row, 1].grid(axis="x", alpha=0.2)
            core = [token for token, count in items if count / len(group) >= core_cutoff]
            core_text = ", ".join(core) if core else "none"
            axes[row, 1].set_title(
                f"Shared core (≥2/3 of sequences): {core_text}",
                loc="left", fontsize=10, color="#315a7d",
            )

        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
            ax=axes[:, 0], shrink=0.8, pad=0.02,
        )
        colorbar.set_label("Prediction residual")
        fig.suptitle(
            f"Representative local neighborhoods: {mutation_count}-mutation sequences\n"
            "Star = center; numbered points = its eight closest Hamming neighbors",
            fontsize=14,
        )
        out_path = args.out_dir / f"vts1_high_{mutation_count}mut_representative_neighborhoods.png"
        fig.savefig(out_path, dpi=210, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
