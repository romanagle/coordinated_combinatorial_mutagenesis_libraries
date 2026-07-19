"""
mRNA_RBP/plot_rho_vs_libsize_type3.py

Copy of plot_rho_vs_libsize.py that adds the type3 (exhaustive SSM + all-pairs
+ random triples) eval split. Reads from the separate
lib_size_spearman_results_type3.json cache (mut_rates 10/25 only — see
lib_size_spearman.py / CLAUDE.md for why 5% is excluded). Does not touch the
original script, JSON, or rho_vs_libsize.png/.svg.

Eval splits shown: random holdout, type2, pairwise, type3.
Single panel at mutation rate = 10% (same convention as the original).

Output: outputs/notebook_plots/rho_vs_libsize_type3.png / .svg
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import LIB_SIZES

JSON_PATH = os.path.join(_HERE, "outputs", "lib_size_spearman_results_type3.json")
OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots")
CFG_NAME  = "nonlinear additive + pairwise"
GT_KEY    = "nonlin_additive_pairwise"

SPLITS = {
    "rand":     ("-",  "o", "#4C72B0", "random holdout"),
    "type2":    ("-",  "P", "#DD8452", "uniform eval (type 2)"),
    "pairwise": (":",  "D", "#8172B2", "targeted pairwise"),
    "type3":    ("-.", "s", "#55A868", "saturated additive + pw"),
}

MUT_RATE_PLOT = 10   # single mutation rate to display

MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) >= 2 else 0.0)


def main(save_svg: bool = False):
    with open(JSON_PATH) as f:
        cache = json.load(f)

    sur = cache.get("surrogate", {}).get(GT_KEY, {}).get(CFG_NAME, {})

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

    x = np.array(LIB_SIZES, dtype=float)
    spct = str(MUT_RATE_PLOT)

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
    ax.set_title(f"Nonlinear additive + pairwise  |  mut rate = {MUT_RATE_PLOT}%", fontsize=10)
    ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.set_ylim(-0.2, 1.08)
    ax.set_ylabel("Spearman ρ", fontsize=10)
    ax.grid(True, axis="y", ls="--", alpha=0.25)

    # Legend inside axes (lower right)
    handles = [
        mlines.Line2D([], [], color=c, ls=ls, marker=m, markersize=5,
                      linewidth=1.8, label=lbl)
        for _, (ls, m, c, lbl) in SPLITS.items()
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, "rho_vs_libsize_type3.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if save_svg:
        out_svg = os.path.join(OUT_DIR, "rho_vs_libsize_type3.svg")
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    main(save_svg=args.svg)
