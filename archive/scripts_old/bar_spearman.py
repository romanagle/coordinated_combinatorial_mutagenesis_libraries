"""bar_spearman.py

Single-panel bar chart of Spearman ρ (mean ± SEM across N sequences),
loading from pre-saved rho_results.json and rb_rho_results.json.

Usage:
    python scripts/bar_spearman.py \
        --save_dir outputs/postabrcms/vts1high/spearman_violin_data \
        --out_path outputs/postabrcms/vts1high/spearman_bar_run0.png \
        --experiment RNCMPT00111
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]

GT_LABELS = {
    "additive":                 "Additive GT",
    "additive_pairwise":        "Add+Pair GT",
    "nonlin_additive":          "Nonlin Add GT",
    "nonlin_additive_pairwise": "Nonlin Add+Pair GT",
    "rb_surrogate":             "RB Surrogate GT",
}

CFG_KEYS = ["additive", "pairwise", "additive_GE", "pairwise_GE"]

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Add GE",
    "pairwise_GE": "Pair GE",
}

CFG_COLORS = {
    "additive":    "#4C72B0",
    "pairwise":    "#DD8452",
    "additive_GE": "#55A868",
    "pairwise_GE": "#C44E52",
}

ALL_GT_KEYS = GT_KEYS + ["rb_surrogate"]

BAR_W    = 0.09   # width of each individual bar
PAIR_GAP = 0.02   # gap between rand/eval bar within one surrogate config
CFG_GAP  = 0.07   # gap between surrogate config pairs within a GT group
GT_GAP   = 0.30   # gap between GT type groups


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    mean = float(np.nanmean(a))
    sem  = float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) >= 2 else 0.0
    return mean, sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",   default="outputs/postabrcms/vts1high/spearman_violin_data")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--out_path",   default="outputs/postabrcms/vts1high/spearman_bar_run0.png")
    args = parser.parse_args()

    with open(os.path.join(args.save_dir, "rho_results.json")) as f:
        rho = json.load(f)
    with open(os.path.join(args.save_dir, "rb_rho_results.json")) as f:
        rb_rho = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))

    xtick_positions = []
    xtick_labels    = []
    cfg_xtick_positions = {k: [] for k in CFG_KEYS}

    x_cursor = 0.0
    for gt_key in ALL_GT_KEYS:
        group_start = x_cursor
        for cfg_name in CFG_KEYS:
            if gt_key == "rb_surrogate":
                rand_vals = rb_rho[cfg_name]["rand"]
                eval_vals = rb_rho[cfg_name]["eval"]
            else:
                rand_vals = rho[gt_key][cfg_name]["rand"]
                eval_vals = rho[gt_key][cfg_name]["eval"]

            rand_mean, rand_sem = _stats(rand_vals)
            eval_mean, eval_sem = _stats(eval_vals)
            color = CFG_COLORS[cfg_name]
            ekw   = {"linewidth": 1.2, "capthick": 1.2, "zorder": 5}

            rand_x = x_cursor
            if np.isfinite(rand_mean):
                ax.bar(rand_x, rand_mean, width=BAR_W, color=color, alpha=0.75,
                       yerr=rand_sem, capsize=3, error_kw=ekw, zorder=3)

            eval_x = x_cursor + BAR_W + PAIR_GAP
            if np.isfinite(eval_mean):
                ax.bar(eval_x, eval_mean, width=BAR_W, color=color, alpha=0.75,
                       hatch="///", edgecolor="white", yerr=eval_sem,
                       capsize=3, error_kw=ekw, zorder=3)

            cfg_xtick_positions[cfg_name].append(rand_x + BAR_W / 2)
            x_cursor += 2 * BAR_W + PAIR_GAP + CFG_GAP

        group_end = x_cursor - CFG_GAP
        xtick_positions.append((group_start + group_end) / 2)
        xtick_labels.append(GT_LABELS[gt_key])
        x_cursor += GT_GAP

    # GT group labels on x-axis
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=12, fontweight="bold")

    # Secondary x-axis tick marks showing surrogate config labels within each group
    minor_positions = []
    minor_labels    = []
    for gt_idx, gt_key in enumerate(ALL_GT_KEYS):
        for cfg_idx, cfg_name in enumerate(CFG_KEYS):
            pos = cfg_xtick_positions[cfg_name][gt_idx] + BAR_W / 2 + PAIR_GAP / 2
            minor_positions.append(pos)
            minor_labels.append(CFG_LABELS[cfg_name])

    ax.set_xticks(minor_positions, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=7, color="dimgray", rotation=45, ha="right")
    ax.tick_params(axis="x", which="minor", length=0, pad=1)

    # Vertical separators between GT groups
    sep_x = 0.0
    for i, gt_key in enumerate(ALL_GT_KEYS):
        n_cfgs = len(CFG_KEYS)
        group_width = n_cfgs * (2 * BAR_W + PAIR_GAP + CFG_GAP) - CFG_GAP
        sep_x += group_width
        if i < len(ALL_GT_KEYS) - 1:
            ax.axvline(sep_x + GT_GAP / 2, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.4, zorder=1)
        sep_x += GT_GAP

    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(-0.05, 1.18)
    ax.set_xlim(-0.15, x_cursor - GT_GAP + 0.15)
    ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    cfg_patches = [Patch(facecolor=CFG_COLORS[c], alpha=0.75, label=CFG_LABELS[c])
                   for c in CFG_KEYS]
    rand_patch  = Patch(facecolor="gray", alpha=0.75, label="Random (MAVE test split)")
    eval_patch  = Patch(facecolor="gray", alpha=0.75, hatch="///",
                        edgecolor="white", label="Eval (curated library)")

    ax.legend(handles=cfg_patches + [rand_patch, eval_patch],
              loc="lower right", fontsize=9, ncol=3, framealpha=0.9)

    n_seqs = len(rho["additive"]["additive"]["rand"])
    ax.set_title(
        f"{args.experiment} — Spearman ρ  (N={n_seqs} random sequences, L=41)  |  Mean ± SEM",
        fontsize=13, pad=10,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fig.savefig(args.out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] bar chart → {args.out_path}")


if __name__ == "__main__":
    main()
