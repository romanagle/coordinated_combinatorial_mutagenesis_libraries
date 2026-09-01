"""Build and plot the saturated-plus-four-triples deepSQUID VTS1 prototype."""

from pathlib import Path
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "saturated_nonlinearity_library"
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from mRNA_RBP.src.oracles import build_oracle, sequence_config_for_oracle
from mRNA_RBP.src.sequence_configs import generate_type3_nuc_ids, wt_ids


ORACLE = "deepsquid_vts1"
WT_ACTIVITY = "high"
SOURCE = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00"
OUTPUTS = HERE / "outputs"
SCORE_KEY = "deepsquid_vts1"
TRIPLE_COUNT = 4
TRIPLE_SEED = 42


def add_triples(seq, nuc_ids, labels):
    """Append the four triple mutants from the original seed-42 recipe."""
    wt = wt_ids(seq)
    rng = random.Random(TRIPLE_SEED)
    seen = {tuple(row) for row in nuc_ids}
    triples = []
    while len(triples) < TRIPLE_COUNT:
        row = wt.copy()
        for pos in rng.sample(range(len(wt)), 3):
            choices = [nuc for nuc in range(4) if nuc != int(wt[pos])]
            row[pos] = rng.choice(choices)
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            triples.append(row)
    return (
        np.concatenate([nuc_ids, np.stack(triples).astype(np.uint8)]),
        np.concatenate([labels, np.full(TRIPLE_COUNT, 3, dtype=np.int32)]),
    )


def build_library():
    seq, stem_pairs, motif_positions = sequence_config_for_oracle(
        ORACLE, WT_ACTIVITY
    )
    nuc_ids, labels = generate_type3_nuc_ids(seq)
    nuc_ids, labels = add_triples(seq, nuc_ids, labels)
    oracle = build_oracle(
        ORACLE,
        seq=seq,
        stem_pairs=stem_pairs,
        motif_positions=motif_positions,
        seed=0,
        stem_sigma=3.0,
        wt_activity=WT_ACTIVITY,
    )
    scores = oracle.score_all(np.eye(4, dtype=np.float32)[nuc_ids])[SCORE_KEY]
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / "saturated_nonlinearity.npz"
    np.savez_compressed(
        path,
        nuc_ids=nuc_ids,
        rate_labels=labels,
        **{f"scores_{SCORE_KEY}": scores},
    )
    return path, scores, labels


def load_scores(path):
    return np.load(path)[f"scores_{SCORE_KEY}"].astype(float)


def stacked_hist(ax, scores, labels, title):
    lo, hi = np.percentile(scores, [0.2, 99.8])
    pad = (hi - lo) * 0.03
    bins = np.linspace(lo - pad, hi + pad, 61)
    widths = np.diff(bins)
    bottom = np.zeros(len(widths))
    orders = sorted(set(labels.tolist()))
    colors = cm.get_cmap("viridis", len(orders) + 2)
    for index, order in enumerate(orders):
        mask = labels == order
        counts, _ = np.histogram(scores[mask], bins=bins)
        density = counts / (len(scores) * widths)
        ax.bar(
            bins[:-1], density, width=widths, bottom=bottom,
            color=colors(index + 1), align="edge", alpha=.92,
            label=f"{order}-mut (n={mask.sum():,})",
        )
        bottom += density
    ax.axvline(scores.mean(), color="black", ls="--", lw=1,
               label=f"mean={scores.mean():.3f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("deepSQUID VTS1 score")
    ax.legend(fontsize=8)


def make_figure(scores_new, labels_new):
    activity_path = SOURCE / "activity_balanced.npz"
    if not activity_path.exists():
        activity_path = SOURCE / "type2.npz"
    inputs = [
        (load_scores(activity_path), np.load(activity_path)["rate_labels"],
         "Activity-balanced"),
        (load_scores(SOURCE / "pairwise_lib.npz"), None, "Targeted pairwise"),
        (load_scores(SOURCE / "type3.npz"), np.load(SOURCE / "type3.npz")["rate_labels"],
         "Saturated singles + doubles"),
        (scores_new, labels_new, "Saturated + 4 triple probes"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, (scores, labels, title) in zip(axes, inputs):
        if labels is None:
            ax.hist(scores, bins=60, density=True, color="#4C72B0", alpha=.85)
            ax.axvline(scores.mean(), color="black", ls="--", lw=1,
                       label=f"mean={scores.mean():.3f}")
            ax.set_title(f"{title} (n={len(scores):,})", fontsize=10)
            ax.set_xlabel("deepSQUID VTS1 score")
            ax.legend(fontsize=8)
        else:
            stacked_hist(ax, scores, labels.astype(int),
                         f"{title} (n={len(scores):,})")
    axes[0].set_ylabel("Density")
    fig.subplots_adjust(wspace=.32)
    path = OUTPUTS / "library_distributions_four_way.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    stacked_hist(ax, scores_new, labels_new,
                 f"Saturated + 4 triple probes (n={len(scores_new):,})")
    ax.set_ylabel("Density")
    fig.savefig(OUTPUTS / "saturated_nonlinearity_distribution.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    library_path, scores, labels = build_library()
    figure_path = make_figure(scores, labels)
    values, counts = np.unique(labels, return_counts=True)
    print(f"Saved {library_path}")
    print(f"Saved {figure_path}")
    print(dict(zip(values.tolist(), counts.tolist())))
