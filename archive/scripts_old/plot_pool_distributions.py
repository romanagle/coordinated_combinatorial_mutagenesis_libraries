"""
Plots from a saved pool_2M.npz file:
  1. pool_2M_distributions.png      — raw 2M pool score distributions
  2. pool_2M_random_vs_eval.png     — random sample (20k) vs uniform eval library (20k)

Usage:
    python scripts/plot_pool_distributions.py \
        --npz RNCMPT00111_graphs/gt_eval_libs/run_0/pool_2M.npz \
        --out RNCMPT00111_graphs/gt_eval_libs/run_0/pool_2M_distributions.png \
        --experiment RNCMPT00111 \
        --subdir vts1high \
        --run 0 \
        --num_muts 10 \
        --seq_len 41
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

PANEL_LABELS = {
    "scores_additive":                 "Additive",
    "scores_additive_pairwise":        "Additive + Pairwise",
    "scores_nonlin_additive":          "Nonlin Additive",
    "scores_nonlin_additive_pairwise": "Nonlin Additive + Pairwise",
}
PANEL_ORDER = list(PANEL_LABELS.keys())


def _uniformize_score_space(scores, n_bins=200, clip_lo=1, clip_hi=98, target_n=20000, seed=42):
    """Score-space equal-width uniformization (same logic as ground_truth.py)."""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float).ravel()
    lo = np.nanpercentile(scores, clip_lo)
    hi = np.nanpercentile(scores, clip_hi)
    in_range = (scores >= lo) & (scores <= hi)
    s_r = scores[in_range]
    edges = np.linspace(lo, hi, n_bins + 1)
    bin_ids = np.clip(np.digitize(s_r, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_ids, minlength=n_bins)
    per_bin = min(int(counts[counts > 0].min()),
                  target_n // max(int(np.sum(counts > 0)), 1))
    keep = []
    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if idx.size:
            keep.append(rng.choice(idx, size=min(per_bin, idx.size), replace=False))
    return s_r[np.concatenate(keep)]


def plot_pool_distributions(npz_path, out_path, experiment, subdir, run_idx, num_muts, seq_len, bins=300):
    d = np.load(npz_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key in zip(axes, PANEL_ORDER):
        scores = d[key].astype(float)
        mean_v = np.mean(scores)
        std_v  = np.std(scores)
        max_v  = np.max(scores)

        ax.hist(scores, bins=bins, density=True, color="steelblue", edgecolor="none")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
        ax.set_title(PANEL_LABELS[key], fontsize=13, fontweight="bold")
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.text(
            0.98, 0.97,
            f"mean={mean_v:.3f}  std={std_v:.3f}  max={max_v:.4f}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
        )

    title = (
        f"2M Pool (mut_rate={num_muts}/{seq_len}): GT Score Distributions\n"
        f"{experiment} · {subdir} · run_{run_idx}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def plot_random_vs_eval(npz_path, out_path, experiment, subdir, run_idx,
                        num_muts, seq_len, random_n=20000, bins=100):
    """2×2 figure: random sample vs uniform eval library, one panel per GT key."""
    d = np.load(npz_path)
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key in zip(axes, PANEL_ORDER):
        scores = d[key].astype(float)

        # Random sample — mimics the random training library distribution
        rand_idx = rng.choice(len(scores), size=min(random_n, len(scores)), replace=False)
        y_rand = scores[rand_idx]

        # Uniform eval library via score-space bins
        y_eval = _uniformize_score_space(scores, target_n=random_n)

        # Shared bin edges covering both distributions
        lo = min(y_rand.min(), y_eval.min())
        hi = max(y_rand.max(), y_eval.max())
        bin_edges = np.linspace(lo, hi, bins + 1)

        ax.hist(y_rand, bins=bin_edges, density=True,
                color="C0", alpha=0.55, label=f"Random (N={len(y_rand):,})")
        ax.hist(y_eval, bins=bin_edges, density=True,
                color="C1", alpha=0.55, label=f"Eval uniform (N={len(y_eval):,})")

        ax.set_title(PANEL_LABELS[key], fontsize=12, fontweight="bold")
        ax.set_xlabel("GT Score", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Random library vs Uniform eval library\n"
        f"{experiment} · {subdir} · run_{run_idx}  (mut_rate={num_muts}/{seq_len})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",        required=True,  help="Path to pool_2M.npz")
    parser.add_argument("--out",        required=True,  help="Output PNG path (pool distributions)")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--subdir",     default="vts1high")
    parser.add_argument("--run",        type=int, default=0)
    parser.add_argument("--num_muts",   type=int, default=10)
    parser.add_argument("--seq_len",    type=int, default=41)
    parser.add_argument("--bins",       type=int, default=300)
    args = parser.parse_args()

    plot_pool_distributions(
        npz_path=args.npz,
        out_path=args.out,
        experiment=args.experiment,
        subdir=args.subdir,
        run_idx=args.run,
        num_muts=args.num_muts,
        seq_len=args.seq_len,
        bins=args.bins,
    )

    overlay_out = args.out.replace("pool_2M_distributions", "pool_2M_random_vs_eval")
    plot_random_vs_eval(
        npz_path=args.npz,
        out_path=overlay_out,
        experiment=args.experiment,
        subdir=args.subdir,
        run_idx=args.run,
        num_muts=args.num_muts,
        seq_len=args.seq_len,
    )
