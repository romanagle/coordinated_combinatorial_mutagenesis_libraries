"""lib_size_vs_mut_rate.py

Produces two sets of 4 graphs (one per GT key):
  - eval/   : rho_eval on y-axis, saturated dashed line also uses rho_eval
  - random/ : rho_random on y-axis, saturated dashed line also uses rho_random

Each graph has one colored line per mutation rate (averaged across sequences
and surrogate configs) plus a horizontal dashed line for the saturated
single-mutant library.

Usage:
    python scripts/lib_size_vs_mut_rate.py \
        --inputs "10%:outputs/lib_size_mut10pct" \
                 "25%:outputs/lib_size_mut25pct" \
                 "30%:outputs/lib_size_mut30pct" \
        --saturated_dir outputs/lib_size_saturated \
        --out_dir outputs/lib_size_vs_mut_rate
"""

import os
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

LINE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def load_mut_rate_data(input_specs):
    dfs = []
    for spec in input_specs:
        label, src_dir = spec.split(":", 1)
        csv_path = os.path.join(src_dir, "aggregate_results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        df["mut_rate"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_saturated_data(saturated_dir):
    """Returns dict: gt_key -> {rho_random_mean, rho_eval_mean} averaged over surrogates."""
    csv_path = os.path.join(saturated_dir, "aggregate_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df  = pd.read_csv(csv_path)
    # average across surrogate configs (NaNs are skipped by nanmean)
    out = {}
    for gt_key in GT_KEYS:
        sub = df[df["gt_key"] == gt_key]
        out[gt_key] = {
            "rho_random": np.nanmean(sub["rho_random_mean"].values),
            "rho_eval":   np.nanmean(sub["rho_eval_mean"].values),
        }
    return out


def make_graphs(df, saturated, library_sizes, mut_rates, out_dir, split):
    """
    split: "random" or "eval"
    Produces 4 PNGs in out_dir/
    """
    col_mu = f"rho_{split}_mean"
    col_sd = f"rho_{split}_std"
    os.makedirs(out_dir, exist_ok=True)

    for gt_key in GT_KEYS:
        fig, ax = plt.subplots(figsize=(6, 4))

        # Colored lines — one per mutation rate
        for i, mr in enumerate(mut_rates):
            color = LINE_COLORS[i % len(LINE_COLORS)]
            sub = (
                df[(df["gt_key"] == gt_key) & (df["mut_rate"] == mr)]
                .groupby("library_size")[col_mu]
                .agg(["mean", "std"])
                .reindex(library_sizes)
                .reset_index()
            )
            mu = sub["mean"].values
            sd = sub["std"].values
            ax.semilogx(library_sizes, mu, marker="o", color=color,
                        linewidth=2.0, markersize=6, label=mr)
            ax.fill_between(library_sizes, mu - sd, mu + sd,
                            color=color, alpha=0.18)

        # Dashed line — saturated single-mutant library (always rho_eval:
        # evaluated on 5000-sequence held-out library, not MAVE-NN's tiny
        # internal test split which is unreliable with only ~12 sequences)
        if saturated and gt_key in saturated:
            sat_val = saturated[gt_key]["rho_eval"]
            if np.isfinite(sat_val):
                ax.axhline(sat_val, color="black", linestyle="--",
                           linewidth=1.5, label="saturated")

        ax.set_xlabel("Library size", fontsize=12)
        ax.set_ylabel("Spearman ρ", fontsize=12)
        ax.set_title(GT_TITLES[gt_key], fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(library_sizes)
        ax.set_xticklabels([f"{int(s):,}" for s in library_sizes],
                           rotation=30, ha="right", fontsize=9)
        ax.legend(title="Mutation rate", fontsize=10)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()

        path = os.path.join(out_dir, f"lib_size_vs_mut_rate_{gt_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] → {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs",        nargs="+", required=True, metavar="LABEL:DIR")
    p.add_argument("--saturated_dir", type=str,  default=None)
    p.add_argument("--out_dir",       type=str,  default="outputs/lib_size_vs_mut_rate")
    args = p.parse_args()

    df        = load_mut_rate_data(args.inputs)
    saturated = load_saturated_data(args.saturated_dir) if args.saturated_dir else None

    mut_rates     = list(dict.fromkeys(df["mut_rate"]))
    library_sizes = np.array(sorted(df["library_size"].unique()))

    make_graphs(df, saturated, library_sizes, mut_rates,
                out_dir=os.path.join(args.out_dir, "eval"),   split="eval")
    make_graphs(df, saturated, library_sizes, mut_rates,
                out_dir=os.path.join(args.out_dir, "random"), split="random")

    print(f"\n[done] graphs saved to {args.out_dir}/eval/ and {args.out_dir}/random/")


if __name__ == "__main__":
    main()
