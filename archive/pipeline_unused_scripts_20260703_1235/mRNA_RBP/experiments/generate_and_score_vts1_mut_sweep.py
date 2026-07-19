"""
Generate 20k random VTS1 sequences at each mutation count, score them with the
real VTS1 ResidualBind oracle (RNCMPT00111), and plot mean score vs mutation count.

Run from the `residbind` conda environment:

    /home/nagle/miniconda3/envs/residbind/bin/python \
        mRNA_RBP/generate_and_score_vts1_mut_sweep.py

Outputs:
    mRNA_RBP/outputs_vts1_residualbind/instance_00/mut_sweep/
        mut_{mc:02d}.npz            -- nuc_ids, mut_counts, scores_vts1_residualbind
        summary.npz                 -- counts, means, stds, ns
    mRNA_RBP/outputs/notebook_plots/rbp_fitness_vs_nmut_vts1.png
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

from mRNA_RBP.generate_libraries import STEM_SIGMA, generate_pool
from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
    sample_unique,
)
from mRNA_RBP.oracles import VTS1_ORACLE, build_oracle


def max_unique(L: int, m: int) -> int:
    """C(L, m) * 3^m distinct sequences with exactly m mutations."""
    if m < 0 or m > L:
        return 0
    k = min(m, L - m)
    num = 1
    for i in range(k):
        num = num * (L - i) // (i + 1)
    return num * (3 ** m)


def generate_for_count(
    wt_oh: np.ndarray,
    mc: int,
    target_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n, L) uint8 nuc_ids with exactly mc mutations."""
    if mc == 0:
        return np.argmax(wt_oh, axis=1).astype(np.uint8)[None, :]

    L = wt_oh.shape[0]
    cap = max_unique(L, mc)
    if cap <= target_n:
        # enumerate everything (cheap for small mc)
        from itertools import combinations, product

        wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
        rows = []
        for combo in combinations(range(L), mc):
            alt_choices = [[n for n in range(4) if n != int(wt_idx[p])] for p in combo]
            for alts in product(*alt_choices):
                seq = wt_idx.copy()
                for p, a in zip(combo, alts):
                    seq[p] = a
                rows.append(seq)
        return np.stack(rows).astype(np.uint8)

    return sample_unique(wt_oh, mc, target_n, rng)


def score_nuc_ids_raw(oracle, nuc_ids: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Score (n, L) nuc_ids with raw (non-WT-anchored) ensemble predictions."""
    scores = []
    for start in range(0, len(nuc_ids), batch_size):
        ids = nuc_ids[start : start + batch_size]
        x = np.eye(4, dtype=np.float32)[ids]
        scores.append(oracle._predict_raw(x).astype(np.float32))
    return np.concatenate(scores)


def plot(
    counts: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    wt_score: float,
    out_path: str,
):
    """Reproduce the rbp_fitness_vs_nmut style with raw (unanchored) scores."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(6.9, 3.9))

    color = "#c44e52"
    ax.plot(counts, means, "o-", color=color, lw=2.5, ms=7, zorder=3)
    ax.fill_between(counts, means - stds, means + stds, color=color, alpha=0.18, zorder=2)

    ax.axhline(wt_score, color="black", linestyle="--", lw=1.3, zorder=1,
               label=f"WT ({wt_score:.2f})")
    ax.legend(loc="lower left", fontsize=10, frameon=True, edgecolor="gray")

    ax.set_xlabel("Number of mutations from WT", fontsize=13)
    ax.set_ylabel("Mean ResidualBind score (raw)", fontsize=13)
    ax.set_title(
        "RBP binding (RNCMPT00111 / VTS1)\nmean oracle score vs mutation count",
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
    ap.add_argument("--n_per_count", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=str, default=None,
                    help="WT sequence (must be 41 nt). Default: VTS1_SEQ from generate_varied_mutrate_library.")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Directory for scored per-count libraries. Default: outputs_vts1_residualbind/instance_00/mut_sweep",
    )
    ap.add_argument(
        "--plot_path",
        default=str(_HERE / "outputs" / "notebook_plots" / "rbp_fitness_vs_nmut_vts1.png"),
    )
    args = ap.parse_args()

    seq = args.seq or VTS1_SEQ
    if len(seq) != 41:
        raise ValueError(f"--seq must be 41 nt, got {len(seq)}")

    out_dir = Path(args.out_dir or (_HERE / "outputs_vts1_residualbind" / "instance_00" / "mut_sweep"))
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = build_oracle(
        VTS1_ORACLE,
        seq=seq,
        stem_pairs=VTS1_STEM_PAIRS,
        motif_positions=VTS1_MOTIF_POSITIONS,
        seed=args.seed,
        stem_sigma=STEM_SIGMA,
    )
    oracle._load()
    wt_oh = oracle.wt_one_hot()
    wt_score = float(oracle._predict_raw(wt_oh[None, :, :])[0])
    print(f"WT raw ensemble score: {wt_score:.4f}")
    rng = np.random.default_rng(args.seed)

    counts, means, stds, ns = [], [], [], []

    for mc in range(args.max_mut + 1):
        print(f"\n[mut={mc:2d}] generating sequences...", flush=True)
        nuc_ids = generate_for_count(wt_oh, mc, args.n_per_count, rng)
        print(f"  generated {len(nuc_ids):,} unique sequences", flush=True)

        print(f"  scoring with {oracle} (raw)...", flush=True)
        scores = score_nuc_ids_raw(oracle, nuc_ids)

        np.savez_compressed(
            out_dir / f"mut_{mc:02d}.npz",
            nuc_ids=nuc_ids,
            mut_counts=np.full(len(nuc_ids), mc, dtype=np.uint8),
            scores_vts1_residualbind=scores,
        )

        counts.append(mc)
        means.append(float(scores.mean()))
        stds.append(float(scores.std()))
        ns.append(len(nuc_ids))
        print(
            f"  -> mean={means[-1]:+.4f}  std={stds[-1]:.4f}  n={ns[-1]}",
            flush=True,
        )

    np.savez_compressed(
        out_dir / "summary.npz",
        counts=np.array(counts, dtype=np.int32),
        means=np.array(means, dtype=np.float32),
        stds=np.array(stds, dtype=np.float32),
        ns=np.array(ns, dtype=np.int32),
        wt_score=np.float32(wt_score),
    )
    print(f"\nSaved summary: {out_dir / 'summary.npz'}")

    plot(np.array(counts), np.array(means), np.array(stds), wt_score, args.plot_path)


if __name__ == "__main__":
    main()
