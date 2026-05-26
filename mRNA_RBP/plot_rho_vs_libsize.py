"""
mRNA_RBP/plot_rho_vs_libsize.py

Reads lib_size_spearman_results.json and plots Spearman ρ vs training library
size for the best surrogate (nonlinear additive + pairwise), averaging across
all 10 instances.

Eval splits shown: random holdout, type2, pairwise.
One panel per mutation rate (5%, 10%, 25%).

Output: outputs/notebook_plots/rho_vs_libsize.png / .svg
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import MUT_RATES_PCT, LIB_SIZES

JSON_PATH = os.path.join(_HERE, "outputs", "lib_size_spearman_results.json")
OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots")
CFG_NAME  = "nonlinear additive + pairwise"
GT_KEY    = "nonlin_additive_pairwise"

SPLITS = {
    "rand":     ("-",  "o", "#4C72B0", "random holdout"),
    "type2":    ("-",  "P", "#DD8452", "type2 (2/3/4-mut uniform)"),
    "pairwise": (":",  "D", "#8172B2", "pairwise (structured)"),
}

MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) >= 2 else 0.0)


def main():
    with open(JSON_PATH) as f:
        cache = json.load(f)

    sur = cache.get("surrogate", {}).get(GT_KEY, {}).get(CFG_NAME, {})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    x = np.array(LIB_SIZES, dtype=float)

    for ax, pct in zip(axes, MUT_RATES_PCT):
        spct = str(pct)
        for split, (ls, marker, color, label) in SPLITS.items():
            means, stds = [], []
            for n_lib in LIB_SIZES:
                vals = sur.get(spct, {}).get(str(n_lib), {}).get(split, [])
                m, s = _stats(vals)
                means.append(m)
                stds.append(s)
            means = np.array(means)
            stds  = np.array(stds)
            valid = np.isfinite(means)
            if not valid.any():
                continue
            ax.plot(x[valid], means[valid], color=color, ls=ls,
                    marker=marker, markersize=5, linewidth=1.8,
                    label=label)
            ax.fill_between(x[valid],
                            (means - stds)[valid],
                            (means + stds)[valid],
                            color=color, alpha=0.12)

        ax.set_xscale("log")
        ax.set_xticks(LIB_SIZES)
        ax.set_xticklabels(["200", "2K", "20K"], fontsize=9)
        ax.set_xlabel("Training library size", fontsize=10)
        ax.set_title(f"mut rate = {pct}%", fontsize=10)
        ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.set_ylim(-0.2, 1.08)
        ax.grid(True, axis="y", ls="--", alpha=0.25)

    axes[0].set_ylabel("Spearman ρ  (mean ± std, 10 instances)", fontsize=10)

    # Single shared legend below
    handles = [
        mlines.Line2D([], [], color=c, ls=ls, marker=m, markersize=5,
                      linewidth=1.8, label=lbl)
        for _, (ls, m, c, lbl) in SPLITS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(SPLITS),
               fontsize=9, bbox_to_anchor=(0.5, -0.08), framealpha=0.9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "rho_vs_libsize.png")
    out_svg = os.path.join(OUT_DIR, "rho_vs_libsize.svg")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
