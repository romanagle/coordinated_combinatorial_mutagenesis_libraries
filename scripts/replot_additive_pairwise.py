"""replot_additive_pairwise.py

Produces two variants of lib_size_spearman_additive_pairwise.png:

  1. _truncated.png  — x-axis cut off at 20,000 (instability removed)
  2. _expected.png   — all 5 sizes shown; the 200k instability points for
                       nonlinear additive+pairwise are grayed/hollow, and a
                       dotted line shows the expected plateau (~1.0).

Usage:
    python scripts/replot_additive_pairwise.py \
        --csv  outputs/lib_size_experiment/lib_size_results.csv \
        --out  outputs/lib_size_experiment
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_KEY  = "additive_pairwise"
TITLE   = "Additive + Pairwise (Potts)"

SURROGATES = [
    ("additive",                       "#1f77b4"),
    ("additive + pairwise",            "#ff7f0e"),
    ("nonlinear additive",             "#2ca02c"),
    ("nonlinear additive + pairwise",  "#d62728"),
]

# Display labels matching the original figure
LABELS = {
    "additive":                       "additive",
    "additive + pairwise":            "additive + pairwise",
    "nonlinear additive":             "nonlinear additive",
    "nonlinear additive + pairwise":  "nonlinear additive + pairwise",
}

ALL_SIZES = [200, 2_000, 20_000, 200_000, 2_000_000]


def _base_ax(ax, sizes):
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_title(TITLE, fontsize=12)
    ax.set_ylim(-0.1, 1.05)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)


def plot_truncated(df, out_dir):
    """Only show lib sizes up to 20,000."""
    sizes = [200, 2_000, 20_000]
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = df[(df["gt_key"] == GT_KEY) & (df["library_size"].isin(sizes))]

    for cfg, color in SURROGATES:
        row = sub[sub["surrogate"] == cfg].sort_values("library_size")
        ax.semilogx(row["library_size"], row["rho_random"],
                    marker="o", color=color, linestyle="-",
                    linewidth=1.8, markersize=5,
                    label=f"{LABELS[cfg]} (random)")
        ax.semilogx(row["library_size"], row["rho_eval"],
                    marker="s", color=color, linestyle="--",
                    linewidth=1.8, markersize=5,
                    label=f"{LABELS[cfg]} (eval)")

    _base_ax(ax, sizes)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = f"{out_dir}/lib_size_spearman_additive_pairwise_truncated.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def plot_expected(df, out_dir):
    """All 5 sizes; nonlinear pairwise 200k instability grayed, dotted
    expected line drawn from 20k to 2M for those two curves."""
    sizes = ALL_SIZES
    sub = df[df["gt_key"] == GT_KEY].copy()

    fig, ax = plt.subplots(figsize=(7, 5))

    for cfg, color in SURROGATES:
        row = sub[sub["surrogate"] == cfg].sort_values("library_size")
        rnd = row["rho_random"].values
        evl = row["rho_eval"].values
        szs = row["library_size"].values

        is_nl_pairwise = (cfg == "nonlinear additive + pairwise")

        if not is_nl_pairwise:
            # Normal solid/dashed lines
            ax.semilogx(szs, rnd, marker="o", color=color, linestyle="-",
                        linewidth=1.8, markersize=5, label=f"{LABELS[cfg]} (random)")
            ax.semilogx(szs, evl, marker="s", color=color, linestyle="--",
                        linewidth=1.8, markersize=5, label=f"{LABELS[cfg]} (eval)")
        else:
            # ── Draw solid/dashed up to 20k  (indices 0,1,2)
            ax.semilogx(szs[:3], rnd[:3], marker="o", color=color, linestyle="-",
                        linewidth=1.8, markersize=5, label=f"{LABELS[cfg]} (random)")
            ax.semilogx(szs[:3], evl[:3], marker="s", color=color, linestyle="--",
                        linewidth=1.8, markersize=5, label=f"{LABELS[cfg]} (eval)")

            # ── Faded/hollow markers at 200k (index 3) to show instability
            ax.semilogx([szs[3]], [rnd[3]], marker="o", color=color,
                        linestyle="none", markersize=7, alpha=0.35,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.5)
            ax.semilogx([szs[3]], [evl[3]], marker="s", color=color,
                        linestyle="none", markersize=7, alpha=0.35,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.5)

            # ── Solid/dashed again at 2M (index 4)
            ax.semilogx([szs[4]], [rnd[4]], marker="o", color=color,
                        linestyle="none", markersize=5)
            ax.semilogx([szs[4]], [evl[4]], marker="s", color=color,
                        linestyle="none", markersize=5)

            # ── Dotted "expected" line: 20k → 2M (skipping instability dip)
            ax.semilogx([szs[2], szs[4]], [rnd[2], rnd[4]],
                        color=color, linestyle=":", linewidth=1.5, alpha=0.7,
                        zorder=1)
            ax.semilogx([szs[2], szs[4]], [evl[2], evl[4]],
                        color=color, linestyle=":", linewidth=1.5, alpha=0.7,
                        zorder=1)

            # ── Vertical annotation arrow at 200k
            ax.annotate("training\ninstability",
                        xy=(szs[3], rnd[3]),
                        xytext=(szs[3] * 1.6, rnd[3] + 0.18),
                        fontsize=7, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=0.8, connectionstyle="arc3,rad=0.2"),
                        ha="left")

    _base_ax(ax, sizes)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = f"{out_dir}/lib_size_spearman_additive_pairwise_expected.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",
                        default="outputs/lib_size_experiment/lib_size_results.csv")
    parser.add_argument("--out",
                        default="outputs/lib_size_experiment")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plot_truncated(df, args.out)
    plot_expected(df, args.out)


if __name__ == "__main__":
    main()
