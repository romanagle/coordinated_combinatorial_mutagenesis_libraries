"""replot_multi_seq.py

Regenerate all lib_size_multi_seq plots from saved CSVs.
Fixes the bug in plot_aggregate where agg_df["seq_idx"] was referenced
but agg_df is the aggregated dataframe (no seq_idx column).

Usage:
    python scripts/replot_multi_seq.py \
        --per_seq_csv outputs/lib_size_multi_seq/per_seq_results.csv \
        --out_dir     outputs/lib_size_multi_seq
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_TITLES = {
    "additive":                          "Additive",
    "additive_pairwise":                 "Additive + Pairwise (Potts)",
    "nonlin_additive":                   "Nonlinear Additive",
    "nonlin_additive_pairwise":          "Nonlinear Additive + Pairwise (Potts)",
    "additive_pairwise_gaussian":        "Additive + Pairwise (Gaussian)",
    "nonlin_additive_pairwise_gaussian": "Nonlinear Additive + Pairwise (Gaussian)",
}

SURROGATE_COLORS = {
    "additive":                      "#1f77b4",
    "additive + pairwise":           "#ff7f0e",
    "nonlinear additive":            "#2ca02c",
    "nonlinear additive + pairwise": "#d62728",
}


def _has_dip(vals, threshold=0.05):
    """True if the 200k point (index 3) dips >threshold below the 20k point (index 2)."""
    if len(vals) < 5:
        return False
    return vals[3] < vals[2] - threshold


def _draw_shaded(ax, szs, mu, sd, color, linestyle, marker, label,
                 dip=False, annotate_dip=False):
    """Draw mean line + shading, with optional instability treatment at index 3 (200k)."""
    if not dip:
        ax.plot(szs, mu, marker=marker, color=color, linestyle=linestyle,
                linewidth=1.8, markersize=5, label=label)
        ax.fill_between(szs, mu - sd, mu + sd, color=color, alpha=0.15)
    else:
        # Normal line + shading up to 20k (indices 0-2)
        ax.plot(szs[:3], mu[:3], marker=marker, color=color, linestyle=linestyle,
                linewidth=1.8, markersize=5, label=label)
        ax.fill_between(szs[:3], mu[:3] - sd[:3], mu[:3] + sd[:3],
                        color=color, alpha=0.15)
        # Faded hollow marker at 200k (index 3)
        ax.plot([szs[3]], [mu[3]], marker=marker, color=color, linestyle="none",
                markersize=7, alpha=0.35, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5)
        # Normal marker at 2M (index 4)
        ax.plot([szs[4]], [mu[4]], marker=marker, color=color,
                linestyle="none", markersize=5)
        ax.fill_between(szs[4:], mu[4:] - sd[4:], mu[4:] + sd[4:],
                        color=color, alpha=0.15)
        # Dotted expected line: 20k → 2M
        ax.plot([szs[2], szs[4]], [mu[2], mu[4]],
                color=color, linestyle=":", linewidth=1.5, alpha=0.7, zorder=1)
        if annotate_dip:
            ax.annotate("training\ninstability",
                        xy=(szs[3], mu[3]),
                        xytext=(szs[3] * 1.6, mu[3] + 0.18),
                        fontsize=7, color="gray",
                        arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=0.8, connectionstyle="arc3,rad=0.2"),
                        ha="left")


def plot_aggregate(per_seq_df, agg_df, library_sizes, gt_keys, out_dir):
    """One figure per GT key: mean ± shading, with instability treatment where detected."""
    sizes  = np.array(library_sizes)
    n_seqs = per_seq_df["seq_idx"].nunique()
    for gt_key in gt_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = agg_df[agg_df["gt_key"] == gt_key]
        annotated = False
        for cfg_name, color in SURROGATE_COLORS.items():
            cfg_sub = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
            mu_rand = cfg_sub["rho_random_mean"].values
            sd_rand = cfg_sub["rho_random_std"].values
            mu_eval = cfg_sub["rho_eval_mean"].values
            sd_eval = cfg_sub["rho_eval_std"].values
            dip = _has_dip(mu_rand) or _has_dip(mu_eval)
            _draw_shaded(ax, sizes, mu_rand, sd_rand, color, "-", "o",
                         f"{cfg_name} (random)", dip=dip, annotate_dip=dip and not annotated)
            _draw_shaded(ax, sizes, mu_eval, sd_eval, color, "--", "s",
                         f"{cfg_name} (eval)", dip=dip)
            if dip:
                annotated = True
        ax.set_xscale("log")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(
            f"{GT_TITLES[gt_key]}\n(mean ± std across {n_seqs} sequences)",
            fontsize=11,
        )
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, "combined", f"lib_size_multi_seq_spearman_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


def plot_by_split(per_seq_df, agg_df, library_sizes, gt_keys, out_dir, split="random"):
    sizes  = np.array(library_sizes)
    col_mu = f"rho_{split}_mean"
    col_sd = f"rho_{split}_std"
    marker = "o" if split == "random" else "s"
    n_seqs = per_seq_df["seq_idx"].nunique()
    for gt_key in gt_keys:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = agg_df[agg_df["gt_key"] == gt_key]
        annotated = False
        for cfg_name, color in SURROGATE_COLORS.items():
            cfg_sub = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
            mu = cfg_sub[col_mu].values
            sd = cfg_sub[col_sd].values
            dip = _has_dip(mu)
            _draw_shaded(ax, sizes, mu, sd, color, "-", marker, cfg_name,
                         dip=dip, annotate_dip=dip and not annotated)
            if dip:
                annotated = True
        ax.set_xscale("log")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(
            f"{GT_TITLES[gt_key]} ({split} split)\n"
            f"mean ± std across {n_seqs} sequences",
            fontsize=11,
        )
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        path = os.path.join(out_dir, split, f"lib_size_multi_seq_{split}_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] saved → {path}")


def plot_seq_lengths_overview(per_seq_df, library_sizes, gt_keys, out_dir):
    """Spearman at max library size vs. sequence length per surrogate.

    Points where rho at max_N dropped >0.05 below rho at the previous size
    are shown as hollow/faded markers (training instability).
    """
    max_N  = max(library_sizes)
    prev_N = sorted(library_sizes)[-2]  # second-largest (20k)
    at_max  = per_seq_df[per_seq_df["library_size"] == max_N].set_index(
        ["gt_key", "surrogate", "seq_idx"])
    at_prev = per_seq_df[per_seq_df["library_size"] == prev_N].set_index(
        ["gt_key", "surrogate", "seq_idx"])
    for split in ("random", "eval"):
        col = f"rho_{split}"
        for gt_key in gt_keys:
            fig, ax = plt.subplots(figsize=(6, 4))
            any_unstable = False
            for cfg_name, color in SURROGATE_COLORS.items():
                rows_max  = at_max.xs((gt_key, cfg_name), level=("gt_key", "surrogate"))
                rows_prev = at_prev.xs((gt_key, cfg_name), level=("gt_key", "surrogate"))
                merged = rows_max[[col, "seq_len"]].join(
                    rows_prev[[col]].rename(columns={col: f"{col}_prev"}))
                merged = merged.sort_values("seq_len")
                stable   = merged[merged[col] >= merged[f"{col}_prev"] - 0.05]
                unstable = merged[merged[col] <  merged[f"{col}_prev"] - 0.05]

                # connect all points with a faint line first
                ax.plot(merged["seq_len"], merged[col],
                        color=color, linewidth=1.0, alpha=0.4)
                # stable: filled markers
                if not stable.empty:
                    ax.scatter(stable["seq_len"], stable[col],
                               color=color, label=cfg_name, s=15, zorder=3)
                # unstable: hollow markers
                if not unstable.empty:
                    ax.scatter(unstable["seq_len"], unstable[col],
                               facecolors="none", edgecolors=color,
                               linewidths=1.2, s=25, zorder=3, alpha=0.5)
                    any_unstable = True

            ax.set_xlabel("Sequence length", fontsize=11)
            ax.set_ylabel("Spearman ρ", fontsize=11)
            title = f"{GT_TITLES[gt_key]} — {split} split\n(N={max_N:,} library)"
            if any_unstable:
                title += "  ○ = training instability"
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-0.1, 1.05)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path = os.path.join(out_dir, "vs_length",
                                f"lib_size_multi_seq_vs_len_{split}_{gt_key}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[plot] saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_seq_csv",
                        default="outputs/lib_size_multi_seq/per_seq_results.csv")
    parser.add_argument("--out_dir",
                        default="outputs/lib_size_multi_seq")
    args = parser.parse_args()

    per_seq_df = pd.read_csv(args.per_seq_csv)
    n_seqs     = per_seq_df["seq_idx"].nunique()
    print(f"[load] {len(per_seq_df)} rows, {n_seqs} sequences: "
          f"{sorted(per_seq_df['seq_idx'].unique())}")

    gt_keys       = sorted(per_seq_df["gt_key"].unique())
    library_sizes = sorted(per_seq_df["library_size"].unique())
    print(f"[load] GT keys:       {gt_keys}")
    print(f"[load] Library sizes: {library_sizes}")

    agg = (
        per_seq_df
        .groupby(["gt_key", "surrogate", "library_size"])[["rho_random", "rho_eval"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = ["gt_key", "surrogate", "library_size",
                   "rho_random_mean", "rho_random_std",
                   "rho_eval_mean",   "rho_eval_std"]

    for subdir in ("combined", "random", "eval", "vs_length"):
        os.makedirs(os.path.join(args.out_dir, subdir), exist_ok=True)
    plot_aggregate(per_seq_df, agg, library_sizes, gt_keys, args.out_dir)
    plot_by_split(per_seq_df, agg, library_sizes, gt_keys, args.out_dir, split="random")
    plot_by_split(per_seq_df, agg, library_sizes, gt_keys, args.out_dir, split="eval")
    plot_seq_lengths_overview(per_seq_df, library_sizes, gt_keys, args.out_dir)


if __name__ == "__main__":
    main()
