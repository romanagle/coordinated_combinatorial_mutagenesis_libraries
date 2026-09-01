"""Counterfactual attribution for high-scoring VTS1 motif-disrupted sequences.

For every sequence with a disrupted constrained GCNGG position in the
rightmost activity histogram bin, revert each observed mutation to WT and
measure how much the deepSQUID score falls. Positive values therefore mark
mutations that support the high-activity ("rescue") phenotype in that exact
sequence background.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
DEFAULT_LIBRARY = (
    ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/activity_balanced.npz"
)
DEFAULT_MODEL = (
    ROOT
    / "mRNA_RBP/models/residualbind/vts1/high"
    / "varied_mutrate_nonlin_pairwise"
)
DEFAULT_OUT = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "vts1_activity_motif_histogram" / "outputs" / "rescue_attribution"
WT = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
ALPHABET = np.asarray(list("ACGU"))
MOTIF_POSITIONS = np.asarray([20, 21, 23, 24])  # zero-based constrained positions


def rightmost_disrupted_indices(
    nuc_ids: np.ndarray, scores: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return rightmost-bin indices with a disrupted constrained motif."""
    _, edges = np.histogram(scores, bins=n_bins)
    bins = np.searchsorted(edges, scores, side="right") - 1
    bins = np.clip(bins, 0, n_bins - 1)
    wt_ids = np.asarray(["ACGU".index(base) for base in WT], dtype=np.uint8)
    disrupted = np.any(
        nuc_ids[:, MOTIF_POSITIONS] != wt_ids[MOTIF_POSITIONS], axis=1
    )
    indices = np.flatnonzero((bins == n_bins - 1) & disrupted)
    return indices[np.argsort(scores[indices])[::-1]], edges


def make_counterfactuals(
    nuc_ids: np.ndarray, selected: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Create all single observed-mutation-to-WT reversion sequences."""
    wt_ids = np.asarray(["ACGU".index(base) for base in WT], dtype=np.uint8)
    variants: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for sequence_index in selected:
        for position in np.flatnonzero(nuc_ids[sequence_index] != wt_ids):
            reverted = nuc_ids[sequence_index].copy()
            reverted[position] = wt_ids[position]
            variants.append(reverted)
            keys.append((int(sequence_index), int(position)))
    return np.asarray(variants, dtype=np.uint8), keys


def make_pair_counterfactuals(
    nuc_ids: np.ndarray, selected: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Create all double observed-mutation-to-WT reversion sequences."""
    wt_ids = np.asarray(["ACGU".index(base) for base in WT], dtype=np.uint8)
    variants: list[np.ndarray] = []
    keys: list[tuple[int, int, int]] = []
    for sequence_index in selected:
        positions = np.flatnonzero(nuc_ids[sequence_index] != wt_ids)
        for a, position_i in enumerate(positions):
            for position_j in positions[a + 1 :]:
                reverted = nuc_ids[sequence_index].copy()
                reverted[[position_i, position_j]] = wt_ids[[position_i, position_j]]
                variants.append(reverted)
                keys.append((int(sequence_index), int(position_i), int(position_j)))
    return np.asarray(variants, dtype=np.uint8), keys


def score_ids(oracle, nuc_ids: np.ndarray) -> np.ndarray:
    one_hot = np.eye(4, dtype=np.float32)[nuc_ids]
    return np.asarray(oracle(one_hot), dtype=float)


def write_single_csv(
    path: Path,
    nuc_ids: np.ndarray,
    observed_scores: np.ndarray,
    selected: np.ndarray,
    keys: list[tuple[int, int]],
    reverted_scores: np.ndarray,
) -> dict[tuple[int, int], float]:
    sequences = np.asarray(["".join(ALPHABET[row]) for row in nuc_ids])
    lookup: dict[tuple[int, int], float] = {}
    selected_rank = {int(index): rank + 1 for rank, index in enumerate(selected)}
    wt_ids = np.asarray(["ACGU".index(base) for base in WT], dtype=np.uint8)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "right_bin_score_rank",
                "library_index",
                "sequence",
                "observed_score",
                "position_1based",
                "wt_base",
                "mutant_base",
                "in_constrained_motif",
                "reverted_score",
                "rescue_contribution",
            ]
        )
        for (sequence_index, position), reverted_score in zip(keys, reverted_scores):
            contribution = float(observed_scores[sequence_index] - reverted_score)
            lookup[(sequence_index, position)] = contribution
            writer.writerow(
                [
                    selected_rank[sequence_index],
                    sequence_index,
                    sequences[sequence_index],
                    observed_scores[sequence_index],
                    position + 1,
                    ALPHABET[wt_ids[position]],
                    ALPHABET[nuc_ids[sequence_index, position]],
                    position in MOTIF_POSITIONS,
                    reverted_score,
                    contribution,
                ]
            )
    return lookup


def write_pair_csv(
    path: Path,
    observed_scores: np.ndarray,
    keys: list[tuple[int, int, int]],
    double_reverted_scores: np.ndarray,
    single_lookup: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((len(WT), len(WT)), dtype=float)
    counts = np.zeros((len(WT), len(WT)), dtype=int)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "library_index",
                "position_i_1based",
                "position_j_1based",
                "interaction",
            ]
        )
        for (sequence_index, i, j), double_score in zip(keys, double_reverted_scores):
            score_i = observed_scores[sequence_index] - single_lookup[(sequence_index, i)]
            score_j = observed_scores[sequence_index] - single_lookup[(sequence_index, j)]
            interaction = float(
                observed_scores[sequence_index] - score_i - score_j + double_score
            )
            writer.writerow([sequence_index, i + 1, j + 1, interaction])
            sums[i, j] += interaction
            sums[j, i] += interaction
            counts[i, j] += 1
            counts[j, i] += 1
    means = np.divide(
        sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
    )
    return means, counts


def plot_single_map(
    path: Path,
    selected: np.ndarray,
    scores: np.ndarray,
    lookup: dict[tuple[int, int], float],
) -> None:
    values = np.full((len(selected), len(WT)), np.nan)
    for row, sequence_index in enumerate(selected):
        for position in range(len(WT)):
            if (int(sequence_index), position) in lookup:
                values[row, position] = lookup[(int(sequence_index), position)]
    bound = max(0.05, float(np.nanpercentile(np.abs(values), 98)))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
    )
    for position in MOTIF_POSITIONS:
        ax.axvline(position, color="#e28e2c", linewidth=1.2)
    ax.set_xticks(np.arange(len(WT)))
    ax.set_xticklabels([f"{i + 1}\n{base}" for i, base in enumerate(WT)], fontsize=7)
    tick_rows = np.linspace(0, len(selected) - 1, min(12, len(selected)), dtype=int)
    ax.set_yticks(tick_rows)
    ax.set_yticklabels([f"{scores[selected[i]]:.3f}" for i in tick_rows])
    ax.set_xlabel("Position and WT nucleotide (orange lines: constrained motif positions)")
    ax.set_ylabel("Motif-disrupted sequences, ordered by activity score")
    ax.set_title("VTS1 rescue attribution: effect of reverting each observed mutation to WT")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Observed score − score after WT reversion")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_summary(
    path: Path,
    lookup: dict[tuple[int, int], float],
) -> None:
    by_position = [[] for _ in WT]
    for (_, position), value in lookup.items():
        by_position[position].append(value)
    counts = np.asarray([len(values) for values in by_position])
    means = np.asarray([np.mean(values) if values else np.nan for values in by_position])
    sem = np.asarray(
        [
            np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
            for values in by_position
        ]
    )
    colors = ["#e28e2c" if i in MOTIF_POSITIONS else "#4c78a8" for i in range(len(WT))]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(np.arange(len(WT)), means, yerr=sem, color=colors, capsize=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(WT)))
    ax.set_xticklabels([f"{i + 1}\n{base}" for i, base in enumerate(WT)], fontsize=7)
    ax.set_ylabel("Mean rescue contribution ± SEM")
    ax.set_xlabel("Position and WT nucleotide")
    ax.set_title("Cohort-average contribution of observed mutations")
    for i, count in enumerate(counts):
        if count:
            ax.text(i, means[i], f"n={count}", rotation=90, fontsize=6, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_cohort_csv(
    path: Path, lookup: dict[tuple[int, int], float]
) -> None:
    by_position = [[] for _ in WT]
    for (_, position), value in lookup.items():
        by_position[position].append(value)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "position_1based",
                "wt_base",
                "in_constrained_motif",
                "n_observed_mutations",
                "mean_rescue_contribution",
                "median_rescue_contribution",
                "sem_rescue_contribution",
                "fraction_positive",
            ]
        )
        for position, values in enumerate(by_position):
            values_array = np.asarray(values, dtype=float)
            n = len(values_array)
            writer.writerow(
                [
                    position + 1,
                    WT[position],
                    position in MOTIF_POSITIONS,
                    n,
                    np.mean(values_array) if n else np.nan,
                    np.median(values_array) if n else np.nan,
                    np.std(values_array, ddof=1) / np.sqrt(n) if n > 1 else 0,
                    np.mean(values_array > 0) if n else np.nan,
                ]
            )


def plot_pair_map(path: Path, means: np.ndarray, counts: np.ndarray) -> None:
    bound = max(0.05, float(np.nanpercentile(np.abs(means), 98)))
    cmap = plt.get_cmap("PuOr_r").copy()
    cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(
        means,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        interpolation="nearest",
    )
    motif_ticks = MOTIF_POSITIONS
    ax.set_xticks(motif_ticks)
    ax.set_yticks(motif_ticks)
    ax.set_xticklabels(motif_ticks + 1)
    ax.set_yticklabels(motif_ticks + 1)
    ax.set_xlabel("Position j (labels mark constrained motif positions)")
    ax.set_ylabel("Position i (labels mark constrained motif positions)")
    ax.set_title("Mean pairwise rescue interaction across observed backgrounds")
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Double-reversion interaction")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    from mRNA_RBP.src.oracles import SurrogateMAVENNOracle

    with np.load(args.library) as library:
        nuc_ids = library["nuc_ids"].astype(np.uint8)
        saved_scores = library["scores_deepsquid_vts1"].astype(float)
    selected, edges = rightmost_disrupted_indices(nuc_ids, saved_scores, args.bins)
    oracle = SurrogateMAVENNOracle(seq=WT, model_dir=args.model_dir)

    rescored = score_ids(oracle, nuc_ids[selected])
    max_error = float(np.max(np.abs(rescored - saved_scores[selected])))
    if max_error > 1e-4:
        raise RuntimeError(
            f"Loaded model does not reproduce saved scores (max error={max_error:.6g})"
        )

    single_variants, single_keys = make_counterfactuals(nuc_ids, selected)
    single_scores = score_ids(oracle, single_variants)
    pair_variants, pair_keys = make_pair_counterfactuals(nuc_ids, selected)
    pair_scores = score_ids(oracle, pair_variants)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    single_lookup = write_single_csv(
        args.out_dir / "single_reversion_attributions.csv",
        nuc_ids,
        saved_scores,
        selected,
        single_keys,
        single_scores,
    )
    pair_means, pair_counts = write_pair_csv(
        args.out_dir / "pairwise_reversion_interactions.csv",
        saved_scores,
        pair_keys,
        pair_scores,
        single_lookup,
    )
    plot_single_map(
        args.out_dir / "sequence_rescue_attribution_map.png",
        selected,
        saved_scores,
        single_lookup,
    )
    plot_cohort_summary(args.out_dir / "cohort_rescue_attribution.png", single_lookup)
    write_cohort_csv(args.out_dir / "cohort_rescue_attribution.csv", single_lookup)
    plot_pair_map(
        args.out_dir / "pairwise_rescue_interaction_map.png", pair_means, pair_counts
    )

    strongest = sorted(single_lookup.items(), key=lambda item: item[1], reverse=True)[:10]
    print(
        f"Rightmost bin: [{edges[-2]:.6f}, {edges[-1]:.6f}]; "
        f"motif-disrupted sequences: {len(selected)}"
    )
    print(f"Model reproduction max absolute error: {max_error:.3g}")
    print(f"Single reversions: {len(single_keys)}; double reversions: {len(pair_keys)}")
    print("Strongest positive rescue contributions:")
    for (sequence_index, position), contribution in strongest:
        print(
            f"  index={sequence_index} position={position + 1} "
            f"{ALPHABET[nuc_ids[sequence_index, position]]} "
            f"contribution={contribution:+.6f}"
        )
    print(f"Saved outputs -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
