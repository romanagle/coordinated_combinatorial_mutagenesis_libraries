"""mRNA_RBP/plot_lib_size_spearman.py

Reads lib_size_spearman_results.json and produces 8 PNGs:
  - 4 surrogate configs × 2 evaluation splits (random / eval)

Each PNG has library size on the log-scaled x-axis and Spearman ρ on the y-axis.
Lines represent 5%, 10%, 25% mutation rates (mean ± std across 10 instances).
A black dashed line marks the saturated mutagenesis baseline (mean ± shaded band).

Usage:
    python mRNA_RBP/plot_lib_size_spearman.py \
        --json  mRNA_RBP/outputs/lib_size_spearman_results.json \
        --out   mRNA_RBP/outputs/plots/lib_size_spearman/
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mRNA_RBP.generate_libraries import MUT_RATES_PCT, LIB_SIZES

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURROGATE_CONFIGS = [
    "additive",
    "additive + pairwise",
    "nonlinear additive",
    "nonlinear additive + pairwise",
]

SURROGATE_LABELS = {
    "additive":                    "Additive",
    "additive + pairwise":         "Additive + Pairwise",
    "nonlinear additive":          "Nonlinear Additive",
    "nonlinear additive + pairwise": "Nonlinear Additive + Pairwise",
}

MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}
MUT_LABELS = {5: "5% mut", 10: "10% mut", 25: "25% mut"}


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) >= 2 else 0.0)


GT_KEY = "nonlin_additive_pairwise"


def plot_one(cache: dict, cfg_name: str, split: str, out_path: str, save_svg: bool = False):
    """
    split: "rand", "type2", or "pairwise"
    """
    surrogate = (cache.get("surrogate", {})
                      .get(GT_KEY, {})
                      .get(cfg_name, {}))
    sat_key   = "type2" if split != "rand" else None
    sat_vals  = []
    for inst in cache.get("saturated", {}).values():
        entry = inst.get(GT_KEY, {}) if isinstance(inst, dict) else {}
        v = entry.get("type2", float("nan")) if sat_key else float("nan")
        if np.isfinite(v):
            sat_vals.append(v)
    sat_mean  = float(np.mean(sat_vals)) if sat_vals else np.nan
    sat_std   = float(np.std(sat_vals, ddof=1)) if len(sat_vals) >= 2 else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = np.array(LIB_SIZES, dtype=float)

    for pct in MUT_RATES_PCT:
        spct   = str(pct)
        means, stds = [], []
        for n_lib in LIB_SIZES:
            sn   = str(n_lib)
            vals = surrogate.get(spct, {}).get(sn, {}).get(split, [])
            m, s = _stats(vals)
            means.append(m)
            stds.append(s)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)
        color = MUT_COLORS[pct]

        valid = np.isfinite(means)
        if valid.any():
            ax.plot(x_vals[valid], means[valid],
                    color=color, linewidth=1.8, marker="o", markersize=4,
                    label=MUT_LABELS[pct])
            ax.fill_between(x_vals[valid],
                            (means - stds)[valid], (means + stds)[valid],
                            color=color, alpha=0.15)

    # Saturated mutagenesis dashed line
    if np.isfinite(sat_mean):
        ax.axhline(sat_mean, color="black", linewidth=1.6, linestyle="--",
                   label=f"Saturated mut. (mean={sat_mean:.3f})", zorder=5)
        ax.axhspan(sat_mean - sat_std, sat_mean + sat_std,
                   color="black", alpha=0.08, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(LIB_SIZES)
    ax.set_xticklabels(["200", "2K", "20K"], fontsize=10)
    ax.set_xlabel("Library size", fontsize=11)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle=":", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    split_label = "Random holdout" if split == "rand" else "Eval library"
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    msg = out_path
    if save_svg:
        fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight")
        msg = f"{out_path} / .svg"
    plt.close(fig)
    print(f"  [saved] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default=os.path.join(_HERE, "outputs", "lib_size_spearman_results.json"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_HERE, "outputs", "plots", "lib_size_spearman"),
    )
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        raise FileNotFoundError(f"Results JSON not found: {args.json}")

    with open(args.json) as f:
        cache = json.load(f)
    print(f"[loaded] {args.json}")

    for split in ("rand", "type2", "pairwise"):
        split_dir = os.path.join(args.out, split)
        for cfg_name in SURROGATE_CONFIGS:
            slug     = cfg_name.replace(" + ", "_plus_").replace(" ", "_")
            out_path = os.path.join(split_dir, f"{slug}.png")
            plot_one(cache, cfg_name, split, out_path, save_svg=args.svg)

    print("\n[done]")


if __name__ == "__main__":
    main()
