"""Make adaptive nearest-neighbor views of the residual sequence embedding."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import HDBSCAN


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "residual_sequence_umap"
DEFAULT_EMBEDDINGS = HERE / "outputs/vts1_high_residual_sequence_umap.npz"
DEFAULT_GT = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/gt_params.npz"
MUTATION_COUNTS = (3, 5, 7, 15)
NUCLEOTIDES = np.asarray(list("ACGU"))


def nearest_neighbors(sequences: np.ndarray, count: int):
    count = min(count, len(sequences) - 1)
    distances, indices = NearestNeighbors(
        n_neighbors=count + 1,
        metric="hamming",
        algorithm="brute",
        n_jobs=-1,
    ).fit(sequences).kneighbors(sequences)
    return distances[:, 1:], indices[:, 1:]


def mutation_tokens(sequence: np.ndarray, wt: np.ndarray) -> list[str]:
    positions = np.flatnonzero(sequence != wt)
    return [f"{NUCLEOTIDES[wt[p]]}{p + 1}{NUCLEOTIDES[sequence[p]]}" for p in positions]


def shared_tokens(left: np.ndarray, right: np.ndarray, wt: np.ndarray) -> list[str]:
    shared = (left == right) & (left != wt)
    return [f"{NUCLEOTIDES[wt[p]]}{p + 1}{NUCLEOTIDES[left[p]]}" for p in np.flatnonzero(shared)]


def make_overview(
    sequences: np.ndarray,
    mutation_counts: np.ndarray,
    residuals: np.ndarray,
    embeddings: dict[int, np.ndarray],
    wt: np.ndarray,
    out_dir: Path,
    graph_neighbors: int,
) -> None:
    vmax = float(np.percentile(np.abs(residuals), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    cluster_rows = []
    for ax, mutation_count in zip(axes.flat, MUTATION_COUNTS):
        mask = mutation_counts == mutation_count
        shell_sequences = sequences[mask]
        shell_residuals = residuals[mask]
        embedding = embeddings[mutation_count]
        distances, indices = nearest_neighbors(shell_sequences, graph_neighbors)

        segments = []
        edge_strength = []
        for source in range(len(shell_sequences)):
            for rank, target in enumerate(indices[source]):
                if source < target:
                    segments.append([embedding[source], embedding[target]])
                    edge_strength.append(1.0 - distances[source, rank])
        if segments:
            strength = np.asarray(edge_strength)
            alpha = 0.025 + 0.15 * (strength - strength.min()) / max(np.ptp(strength), 1e-9)
            colors = np.zeros((len(segments), 4))
            colors[:, 3] = alpha
            ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.35))

        points = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=shell_residuals,
            cmap="RdBu_r", norm=norm, s=5, alpha=0.72, linewidths=0,
            rasterized=True, zorder=2,
        )
        ax.set_title(f"{mutation_count} mutations (n={len(shell_sequences):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("sequence-neighborhood layout")
        ax.set_ylabel("sequence-neighborhood layout")

        if mutation_count == 3:
            labels = HDBSCAN(
                min_cluster_size=75,
                min_samples=10,
                cluster_selection_method="eom",
                copy=True,
            ).fit_predict(embedding)
            valid_labels, sizes = np.unique(labels[labels >= 0], return_counts=True)
            central_label = int(valid_labels[np.argmax(sizes)])
            for cluster_label, size in zip(valid_labels, sizes):
                member_mask = labels == cluster_label
                center = np.median(embedding[member_mask], axis=0)
                if cluster_label == central_label:
                    dominant_token = "mixed"
                    dominant_fraction = np.nan
                else:
                    token_frequencies = []
                    for position in range(shell_sequences.shape[1]):
                        for nucleotide in range(4):
                            if nucleotide == wt[position]:
                                continue
                            frequency = float(np.mean(
                                shell_sequences[member_mask, position] == nucleotide
                            ))
                            token = (
                                f"{NUCLEOTIDES[wt[position]]}{position + 1}"
                                f"{NUCLEOTIDES[nucleotide]}"
                            )
                            token_frequencies.append((frequency, token))
                    dominant_fraction, dominant_token = max(token_frequencies)
                    ax.text(
                        center[0], center[1],
                        f"{dominant_token}\n{dominant_fraction:.0%} · n={size:,}",
                        ha="center", va="center", fontsize=7.5, fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                              "edgecolor": "#555555", "alpha": 0.92},
                        zorder=5,
                    )
                cluster_rows.append({
                    "hdbscan_label": int(cluster_label),
                    "display_label": dominant_token,
                    "n": int(size),
                    "dominant_mutation_fraction": dominant_fraction,
                    "mean_residual": float(np.mean(shell_residuals[member_mask])),
                })

        if mutation_count == 15:
            split = float(np.median(embedding[:, 0]))
            left = embedding[:, 0] < split
            right = ~left
            y_lo, y_hi = np.percentile(embedding[:, 1], [4, 96])
            ax.plot([split, split], [y_lo, y_hi], color="#333333", linestyle="--",
                    linewidth=1.0, alpha=0.8, zorder=4)
            ax.text(
                0.03, 0.97,
                f"Left half\nmean residual {np.mean(shell_residuals[left]):+.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#777777", "alpha": 0.9},
                zorder=5,
            )
            ax.text(
                0.97, 0.97,
                f"Right half\nmean residual {np.mean(shell_residuals[right]):+.2f}\n"
                "positions 21/22/24/25 enriched",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "#777777", "alpha": 0.9},
                zorder=5,
            )
            half_rows = []
            for position in range(shell_sequences.shape[1]):
                left_frequency = float(np.mean(shell_sequences[left, position] != wt[position]))
                right_frequency = float(np.mean(shell_sequences[right, position] != wt[position]))
                half_rows.append({
                    "position": position + 1,
                    "wt_nucleotide": NUCLEOTIDES[wt[position]],
                    "left_mutation_frequency": left_frequency,
                    "right_mutation_frequency": right_frequency,
                    "right_minus_left": right_frequency - left_frequency,
                })
            with (out_dir / "vts1_high_15mut_half_sequence_comparison.csv").open(
                "w", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "position", "wt_nucleotide", "left_mutation_frequency",
                    "right_mutation_frequency", "right_minus_left",
                ])
                writer.writeheader()
                writer.writerows(half_rows)

    colorbar = fig.colorbar(points, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label("Prediction residual (surrogate − ground truth)")
    fig.suptitle(
        f"Adaptive sequence-neighbor graph ({graph_neighbors} closest neighbors per sequence)\n"
        "Edges and layout use sequence Hamming distance only",
        fontsize=13,
    )
    fig.savefig(out_dir / "vts1_high_residual_neighbor_graph.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "vts1_high_3mut_cluster_labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "hdbscan_label", "display_label", "n",
            "dominant_mutation_fraction", "mean_residual",
        ])
        writer.writeheader()
        writer.writerows(cluster_rows)


def make_examples(
    sequences: np.ndarray,
    mutation_counts: np.ndarray,
    residuals: np.ndarray,
    wt: np.ndarray,
    out_dir: Path,
    example_neighbors: int,
) -> None:
    vmax = float(np.percentile(np.abs(residuals), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for mutation_count in MUTATION_COUNTS:
        mask = mutation_counts == mutation_count
        shell_sequences = sequences[mask]
        shell_residuals = residuals[mask]
        _, indices = nearest_neighbors(shell_sequences, example_neighbors)

        # Choose the center whose nearest neighbors share the most exact
        # substitutions with it. This creates an interpretable example, not a
        # claimed discrete cluster.
        shared_counts = np.empty_like(indices)
        for source in range(len(shell_sequences)):
            shared_counts[source] = [
                len(shared_tokens(shell_sequences[source], shell_sequences[target], wt))
                for target in indices[source]
            ]
        center = int(np.argmax(shared_counts.mean(axis=1)))
        neighbors = indices[center]

        fig = plt.figure(figsize=(14, 7), constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.45))
        ax = fig.add_subplot(grid[0, 0])
        text_ax = fig.add_subplot(grid[0, 1])
        text_ax.axis("off")

        angles = np.linspace(0, 2 * np.pi, len(neighbors), endpoint=False)
        xy = np.column_stack([np.cos(angles), np.sin(angles)])
        for number, (target, point) in enumerate(zip(neighbors, xy), start=1):
            tokens = shared_tokens(shell_sequences[center], shell_sequences[target], wt)
            ax.plot([0, point[0]], [0, point[1]], color="#999999", lw=0.7 + 0.3 * len(tokens), zorder=1)
            midpoint = point * 0.55
            ax.text(midpoint[0], midpoint[1], f"{len(tokens)} shared", fontsize=7,
                    ha="center", va="center", backgroundcolor="white")
            ax.scatter(*point, c=[shell_residuals[target]], cmap="RdBu_r", norm=norm,
                       s=180, edgecolor="black", linewidth=0.5, zorder=2)
            ax.text(*point, str(number), ha="center", va="center", fontsize=8, zorder=3)
        ax.scatter(0, 0, c=[shell_residuals[center]], cmap="RdBu_r", norm=norm,
                   s=300, marker="*", edgecolor="black", linewidth=0.8, zorder=3)
        ax.text(0, -0.18, "center", ha="center", va="top", fontsize=8)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("One locally dense neighborhood\n(color = prediction residual)")

        center_tokens = mutation_tokens(shell_sequences[center], wt)
        lines = [
            f"Center mutations ({mutation_count}): " + ", ".join(center_tokens),
            f"Center residual: {shell_residuals[center]:+.3f}",
            "",
            "Nearest sequences and exact mutations shared with the center:",
        ]
        for number, target in enumerate(neighbors, start=1):
            tokens = shared_tokens(shell_sequences[center], shell_sequences[target], wt)
            shared_text = ", ".join(tokens) if tokens else "none"
            lines.append(f"{number}. residual {shell_residuals[target]:+.3f} | {shared_text}")
        text_ax.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=9.5,
                     linespacing=1.5, family="monospace", wrap=True)
        fig.suptitle(
            f"{mutation_count}-mutation sequences: what ‘nearby’ means",
            fontsize=14,
        )
        fig.savefig(
            out_dir / f"vts1_high_{mutation_count}mut_neighbor_example.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--gt-params", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    parser.add_argument("--graph-neighbors", type=int, default=2)
    parser.add_argument("--example-neighbors", type=int, default=8)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    sequences = data["sequences"].astype(np.uint8)
    mutation_counts = data["mutation_counts"].astype(int)
    residuals = data["residuals"].astype(float)
    embeddings = {count: data[f"embedding_{count}"] for count in MUTATION_COUNTS}
    wt_string = str(np.load(args.gt_params)["wt_seq"])
    wt = np.asarray(["ACGU".index(base) for base in wt_string], dtype=np.uint8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    make_overview(
        sequences, mutation_counts, residuals, embeddings, wt,
        args.out_dir, args.graph_neighbors,
    )
    make_examples(sequences, mutation_counts, residuals, wt, args.out_dir, args.example_neighbors)
    print(f"Saved neighbor graph and examples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
