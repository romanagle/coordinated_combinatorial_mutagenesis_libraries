"""mRNA_RBP/make_preliminary_spearman_type3.py

Copy of make_preliminary_spearman.py that adds the type3 (exhaustive SSM +
all-pairs + random triples) eval split. Reads from the separate
lib_size_spearman_results_type3.json cache (mut_rates 10/25 only). Does not
touch the original script, JSON, or preliminary_spearman.png/.svg.

One panel per surrogate config (or a single panel with --config).
Within each panel:
  - Color  = mutation rate (10% / 25% — 5% excluded, see CLAUDE.md)
  - Style  = evaluation split (rand / type2 / pairwise / type3)

Usage:
    # Full 4-panel figure
    python mRNA_RBP/make_preliminary_spearman_type3.py

    # Single-panel for one surrogate config
    python mRNA_RBP/make_preliminary_spearman_type3.py \
        --config "nonlinear additive + pairwise"
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURROGATE_CONFIGS = [
    "additive",
    "additive + pairwise",
    "nonlinear additive",
    "nonlinear additive + pairwise",
]

SURROGATE_LABELS = {
    "additive":                        "Additive",
    "additive + pairwise":             "Additive+Pairwise",
    "nonlinear additive":              "Nonlinear Additive",
    "nonlinear additive + pairwise":   "Nonlin Add+Pair",
}

LIB_SIZES      = [200, 2_000, 20_000]
LIB_TICK_LABELS = ["200", "2K", "20K"]
MUT_RATES      = [10, 25]   # 5% excluded — see lib_size_spearman.py / CLAUDE.md

MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}
MUT_LABELS = {5: "5% mut", 10: "10% mut", 25: "25% mut"}

# eval-split → (linestyle, marker, legend label)
SPLIT_STYLES = {
    "rand":     ("-",  "o", "rand"),
    "type2":    ("-",  "P", "activity-balanced library"),
    "pairwise": ("--", "*", "pairwise"),
    "type3":    ("-.", "s", "exhaustive eval library"),
}


def _get(cache, gt_key, cfg_name, pct, n_lib, split, instance):
    try:
        vals = cache["surrogate"][gt_key][cfg_name][str(pct)][str(n_lib)][split]
        if instance < len(vals):
            v = float(vals[instance])
            return v if np.isfinite(v) else np.nan
    except (KeyError, IndexError, TypeError):
        pass
    return np.nan


def plot_panel(ax, cache, gt_key, cfg_name, instance, mut_rates=None):
    if mut_rates is None:
        mut_rates = MUT_RATES
    x = np.array(LIB_SIZES, dtype=float)

    for pct in mut_rates:
        color = MUT_COLORS[pct]
        for split, (ls, marker, _) in SPLIT_STYLES.items():
            y = np.array([_get(cache, gt_key, cfg_name, pct, n, split, instance)
                          for n in LIB_SIZES])
            mask = np.isfinite(y)
            if mask.any():
                ax.plot(x[mask], y[mask],
                        color=color, linestyle=ls, marker=marker,
                        markersize=4, linewidth=1.4, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xticks(LIB_SIZES)
    ax.set_xticklabels(LIB_TICK_LABELS, fontsize=9)
    ax.set_ylim(-0.25, 1.05)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.set_xlabel("Training lib size", fontsize=9)
    ax.set_ylabel("Spearman ρ", fontsize=9)



def build_legend_handles(mut_rates=None):
    if mut_rates is None:
        mut_rates = MUT_RATES
    h = []
    for _, (ls, marker, label) in SPLIT_STYLES.items():
        h.append(mlines.Line2D([], [], color="gray", linestyle=ls,
                               marker=marker, markersize=4, linewidth=1.4,
                               label=label))
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default=os.path.join(_HERE, "outputs", "lib_size_spearman_results_type3.json"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "outputs", "notebook_plots", "preliminary_spearman_type3.png"),
    )
    parser.add_argument("--gt_key",   default="nonlin_additive_pairwise")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument(
        "--config", default=None,
        help="Single surrogate config to plot (single-panel mode). "
             "E.g. 'nonlinear additive + pairwise'",
    )
    parser.add_argument(
        "--mut_rates", nargs="+", type=int, default=[10],
        help="Mutation rates to include (default: 10)",
    )
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    with open(args.json) as f:
        cache = json.load(f)

    configs = [args.config] if args.config else SURROGATE_CONFIGS
    n = len(configs)

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, cfg in zip(axes, configs):
        plot_panel(ax, cache, args.gt_key, cfg, args.instance, args.mut_rates)

    for ax in axes[1:]:
        ax.set_ylabel("")

    handles = build_legend_handles(mut_rates=args.mut_rates)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=8, bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    msg = args.out
    if args.svg:
        svg_out = os.path.splitext(args.out)[0] + ".svg"
        fig.savefig(svg_out, bbox_inches="tight")
        msg = f"{args.out} / {os.path.basename(svg_out)}"
    plt.close(fig)
    print(f"[saved] {msg}")


if __name__ == "__main__":
    main()
