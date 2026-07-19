"""plot_lib_size_random.py

Reads lib_size_results.csv and saves one figure per GT key showing
only the random test-split Spearman curves (4 surrogate lines each).

Usage:
    python scripts/plot_lib_size_random.py \
        --csv outputs/lib_size_experiment/lib_size_results.csv \
        --out_dir outputs/lib_size_experiment
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]
GT_TITLES = {
    "additive":                 "Additive",
    "additive_pairwise":        "Additive + Pairwise",
    "nonlin_additive":          "Nonlinear Additive",
    "nonlin_additive_pairwise": "Nonlinear Additive + Pairwise",
}
SURROGATE_ORDER  = ["additive", "pairwise", "additive_GE", "pairwise_GE"]
SURROGATE_COLORS = {
    "additive":    "#1f77b4",
    "pairwise":    "#ff7f0e",
    "additive_GE": "#2ca02c",
    "pairwise_GE": "#d62728",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     type=str,
                        default="outputs/lib_size_experiment/lib_size_results.csv")
    parser.add_argument("--out_dir", type=str,
                        default="outputs/lib_size_experiment")
    args = parser.parse_args()

    df    = pd.read_csv(args.csv)
    sizes = sorted(df["library_size"].unique())

    for gt_key in GT_KEYS:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = df[df["gt_key"] == gt_key]

        for cfg_name in SURROGATE_ORDER:
            color = SURROGATE_COLORS[cfg_name]
            row   = sub[sub["surrogate"] == cfg_name].sort_values("library_size")
            ax.semilogx(
                row["library_size"], row["rho_random"],
                marker="o", color=color, linestyle="-",
                linewidth=1.8, markersize=5, label=cfg_name,
            )

        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Library size", fontsize=11)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(GT_TITLES[gt_key], fontsize=12)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{s:,}" for s in sizes], rotation=35, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.25)

        fig.tight_layout()
        path = f"{args.out_dir}/lib_size_spearman_random_{gt_key}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved → {path}")


if __name__ == "__main__":
    main()
