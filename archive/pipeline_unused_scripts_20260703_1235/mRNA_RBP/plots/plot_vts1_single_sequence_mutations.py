"""
Score one VTS1 sequence as random mutations are cumulatively added from 0 to N.

Run from the `residbind` conda environment:

    /home/nagle/miniconda3/envs/residbind/bin/python \
        mRNA_RBP/plot_vts1_single_sequence_mutations.py

Output:
    mRNA_RBP/outputs_vts1_residualbind/instance_00/single_trajectory/
        trajectory.npz      -- nuc_ids, mut_counts, scores_vts1_residualbind
    mRNA_RBP/outputs/notebook_plots/rbp_fitness_vs_nmut_vts1_single.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent.parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from mRNA_RBP.generate_libraries import STEM_SIGMA
from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
)
from mRNA_RBP.oracles import VTS1_ORACLE, build_oracle


def build_cumulative_trajectory(wt_oh: np.ndarray, max_mut: int, rng: np.random.Generator):
    """Return (max_mut+1, L) uint8 nuc_ids where row k has k random mutations from WT."""
    L = wt_oh.shape[0]
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
    positions = rng.permutation(L)

    nuc_ids = np.tile(wt_idx, (max_mut + 1, 1))
    mut_counts = np.arange(max_mut + 1, dtype=np.uint8)

    for k in range(1, max_mut + 1):
        pos = positions[k - 1]
        wt_nuc = int(wt_idx[pos])
        alt = [n for n in range(4) if n != wt_nuc]
        nuc_ids[k, pos] = rng.choice(alt)
        # carry the mutation forward into all subsequent rows
        nuc_ids[k:, pos] = nuc_ids[k, pos]

    return nuc_ids, mut_counts


def plot(counts: np.ndarray, scores: np.ndarray, out_path: str):
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6.9, 3.9))

    color = "#c44e52"
    ax.plot(counts, scores, "o-", color=color, lw=2.5, ms=7, zorder=3)

    ax.axhline(0.0, color="black", linestyle="--", lw=1.3, zorder=1, label="WT (0.00)")
    ax.legend(loc="lower left", fontsize=10, frameon=True, edgecolor="gray")

    ax.set_xlabel("Number of mutations from WT", fontsize=13)
    ax.set_ylabel("ResidualBind score", fontsize=13)
    ax.set_title(
        "VTS1 single-sequence mutation trajectory\n(RNCMPT00111 / VTS1)",
        fontsize=13,
    )
    ax.set_xticks(counts[::2])
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_mut", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: outputs_vts1_residualbind/instance_00/single_trajectory",
    )
    ap.add_argument(
        "--plot_path",
        default=str(_HERE / "outputs" / "notebook_plots" / "rbp_fitness_vs_nmut_vts1_single.png"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir or (_HERE / "outputs_vts1_residualbind" / "instance_00" / "single_trajectory"))
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = build_oracle(
        VTS1_ORACLE,
        seq=VTS1_SEQ,
        stem_pairs=VTS1_STEM_PAIRS,
        motif_positions=VTS1_MOTIF_POSITIONS,
        seed=args.seed,
        stem_sigma=STEM_SIGMA,
    )
    wt_oh = oracle.wt_one_hot()
    rng = np.random.default_rng(args.seed)

    nuc_ids, mut_counts = build_cumulative_trajectory(wt_oh, args.max_mut, rng)
    x = np.eye(4, dtype=np.float32)[nuc_ids]
    scores = oracle.score_all(x)[oracle.score_key]

    np.savez_compressed(
        out_dir / "trajectory.npz",
        nuc_ids=nuc_ids,
        mut_counts=mut_counts,
        scores_vts1_residualbind=scores,
    )
    print(f"Saved trajectory: {out_dir / 'trajectory.npz'}")
    for k, s in zip(mut_counts, scores):
        print(f"  mut={k:2d}  score={s:+.4f}")

    plot(mut_counts, scores, args.plot_path)


if __name__ == "__main__":
    main()
