"""Map activity-balanced VTS1 variants by sequence and color by residual.

The embedding is computed independently within each exact mutation-count shell
using categorical Hamming distance. Residuals never enter the embedding. A
nearest-neighbor permutation test asks whether close sequences have more
similar residuals than expected by chance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


ROOT = Path(__file__).resolve().parents[5]
DEFAULT_LIBRARY = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz"
)
DEFAULT_PREDICTIONS = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/scatter_by_mutcount_predictions.npz"
)
MUTATION_COUNTS = (3, 5, 7, 15)


def neighborhood_test(
    sequences: np.ndarray,
    residuals: np.ndarray,
    *,
    n_neighbors: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Test whether Hamming neighbors have unusually similar residuals."""
    n_neighbors = min(n_neighbors, len(sequences) - 1)
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric="hamming",
        algorithm="brute",
        n_jobs=-1,
    )
    distances, indices = model.fit(sequences).kneighbors(sequences)
    # Column zero is the sequence itself.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    observed = float(np.mean(np.abs(residuals[:, None] - residuals[indices])))
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(residuals)
        null[i] = np.mean(np.abs(shuffled[:, None] - shuffled[indices]))

    # Small differences indicate stronger local similarity.
    p_value = float((1 + np.count_nonzero(null <= observed)) / (n_permutations + 1))
    return {
        "n": len(sequences),
        "neighbors": n_neighbors,
        "median_neighbor_hamming": float(np.median(distances)),
        "median_neighbor_mismatches": float(np.median(distances) * sequences.shape[1]),
        "observed_mean_abs_residual_difference": observed,
        "null_mean_abs_residual_difference": float(np.mean(null)),
        "observed_over_null": float(observed / np.mean(null)),
        "permutation_p_value": p_value,
        "n_permutations": n_permutations,
    }


def load_inputs(library_path: Path, predictions_path: Path):
    library = np.load(library_path)
    predictions = np.load(predictions_path)

    sequences = library["nuc_ids"].astype(np.uint8)
    mutation_counts = library["rate_labels"].astype(int)
    truth = predictions["y_activity_10"].astype(float)
    predicted = predictions["yhat_activity_10"].astype(float)
    prediction_counts = predictions["rate_labels_10"].astype(int)

    if len(sequences) != len(truth):
        raise ValueError("Library and prediction artifacts have different row counts")
    if not np.array_equal(mutation_counts, prediction_counts):
        raise ValueError("Library and prediction mutation-count rows are misaligned")
    if str(predictions["model_name"]) != "nonlinear additive + pairwise":
        raise ValueError(f"Unexpected surrogate: {predictions['model_name']}")

    return sequences, mutation_counts, predicted - truth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "residual_sequence_umap" / "outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--test-neighbors", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()

    sequences, mutation_counts, residuals = load_inputs(args.library, args.predictions)
    finite = np.isfinite(residuals)
    if not finite.all():
        sequences, mutation_counts, residuals = (
            sequences[finite],
            mutation_counts[finite],
            residuals[finite],
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    vmax = float(np.percentile(np.abs(residuals), 99))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    results = []
    embeddings = {}
    for ax, mutation_count in zip(axes.flat, MUTATION_COUNTS):
        mask = mutation_counts == mutation_count
        shell_sequences = sequences[mask]
        shell_residuals = residuals[mask]
        if not len(shell_sequences):
            raise ValueError(f"No {mutation_count}-mutation sequences found")

        embedding = UMAP(
            n_neighbors=min(args.umap_neighbors, len(shell_sequences) - 1),
            min_dist=0.1,
            n_components=2,
            metric="hamming",
            random_state=args.seed,
            transform_seed=args.seed,
            low_memory=True,
        ).fit_transform(shell_sequences)
        embeddings[mutation_count] = embedding

        result = neighborhood_test(
            shell_sequences,
            shell_residuals,
            n_neighbors=args.test_neighbors,
            n_permutations=args.permutations,
            rng=rng,
        )
        result["mutation_count"] = mutation_count
        results.append(result)

        points = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=shell_residuals,
            cmap="RdBu_r",
            norm=norm,
            s=5,
            alpha=0.7,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"{mutation_count} mutations (n={len(shell_sequences):,})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.text(
            0.02,
            0.02,
            f"neighbor/null = {result['observed_over_null']:.3f}\n"
            f"permutation p = {result['permutation_p_value']:.3g}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    colorbar = fig.colorbar(points, ax=axes, shrink=0.85, pad=0.02)
    colorbar.set_label("Prediction residual (surrogate − ground truth)")
    fig.suptitle(
        "High-WT VTS1 activity-balanced sequences\n"
        "Sequence-only Hamming UMAP; 20K, 10%-mutation nonlinear pairwise surrogate",
        fontsize=13,
    )
    figure_path = args.out_dir / "vts1_high_residual_sequence_umap.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = args.out_dir / "vts1_high_residual_neighborhood_test.csv"
    fieldnames = [
        "mutation_count",
        "n",
        "neighbors",
        "median_neighbor_hamming",
        "median_neighbor_mismatches",
        "observed_mean_abs_residual_difference",
        "null_mean_abs_residual_difference",
        "observed_over_null",
        "permutation_p_value",
        "n_permutations",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    np.savez_compressed(
        args.out_dir / "vts1_high_residual_sequence_umap.npz",
        sequences=sequences,
        mutation_counts=mutation_counts,
        residuals=residuals,
        **{f"embedding_{count}": embedding for count, embedding in embeddings.items()},
    )
    print(f"Saved {figure_path}")
    print(f"Saved {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
