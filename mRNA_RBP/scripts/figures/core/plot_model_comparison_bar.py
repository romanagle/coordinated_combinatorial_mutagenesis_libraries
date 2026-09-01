"""mRNA_RBP/plot_model_comparison_bar.py

Bar chart of Spearman ρ by surrogate model type, showing how each model
generalizes to different evaluation library types (activity-balanced vs Pairwise).

X-axis: SSM | Additive | Additive+Pairwise | Nonlinear Additive | Nonlin Add+Pairwise
Two bars per model:
  - Blue  = Activity-balanced library
  - Orange = Pairwise eval library

Results are averaged across all available instances (mean ± SEM).
Requires that lib_size_spearman_results.json has been updated with the new
saturated format (run mRNA_RBP/scripts/pipeline/lib_size_spearman.py to populate/migrate).

Usage:
    python mRNA_RBP/plot_model_comparison_bar.py

    python mRNA_RBP/plot_model_comparison_bar.py \
        --gt_key nonlin_additive_pairwise \
        --mut_rate 10 --lib_size 20000 \
        --out mRNA_RBP/outputs/notebook_plots/model_comparison_bar.png
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SURROGATE_CONFIGS = [
    "additive",
    "additive + pairwise",
    "nonlinear additive",
    "nonlinear additive + pairwise",
]

X_LABELS = {
    "ssm":                              "SSM",
    "additive":                         "Additive",
    "additive + pairwise":              "Additive\n+Pairwise",
    "nonlinear additive":               "Nonlinear\nAdditive",
    "nonlinear additive + pairwise":    "Nonlin Add\n+Pairwise",
}

GT_DISPLAY = {
    "additive":                 "Additive",
    "additive_pairwise":        "Additive+Pairwise",
    "nonlin_additive":          "Nonlinear Additive",
    "nonlin_additive_pairwise": "Nonlin Additive+Pairwise",
    "residualbind_ensemble":    "ResidualBind ensemble (MSI1)",
    "vts1_residualbind":        "ResidualBind ensemble (VTS1)",
}

COLOR_T2 = "#4C72B0"
COLOR_PW = "#DD8452"
BAR_W    = 0.30
BAR_GAP  = 0.04
GROUP_GAP = 0.25


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    mean = float(np.nanmean(a))
    sem  = float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) >= 2 else 0.0
    return mean, sem


def _get_ssm(cache, gt_key):
    """Mean ± SEM of SSM ρ across instances for one GT key and both splits.

    SSM pairwise is nan when all predictions are constant (stem positions have
    no additive component in this model, so SSM has zero signal on pairwise seqs).
    We substitute 0.0 to show explicitly that SSM achieves no correlation.
    """
    t2_vals, pw_vals = [], []
    for inst_data in cache.get("saturated", {}).values():
        entry = inst_data.get(gt_key, {})
        if isinstance(entry, dict):
            t2_vals.append(entry.get("type2", float("nan")))
            pw_vals.append(entry.get("pairwise", float("nan")))
        else:
            # old flat-float format — treat as type2 only
            t2_vals.append(float(entry))
            pw_vals.append(float("nan"))
    t2 = _stats(t2_vals)
    pw_mean, pw_sem = _stats(pw_vals)
    # Constant SSM predictions on pairwise lib → Spearman undefined → treat as 0
    if not np.isfinite(pw_mean):
        pw_mean, pw_sem = 0.0, 0.0
    return t2, (pw_mean, pw_sem)


def _get_surrogate(cache, gt_key, cfg_name, pct, n_lib):
    """Mean ± SEM of surrogate ρ across instances for type2 and pairwise splits."""
    entry = (cache.get("surrogate", {})
                  .get(gt_key, {})
                  .get(cfg_name, {})
                  .get(str(pct), {})
                  .get(str(n_lib), {}))
    t2 = _stats(entry.get("type2",    []))
    pw = _stats(entry.get("pairwise", []))
    return t2, pw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=os.path.join(
        _HERE, "outputs", "lib_size_spearman_results.json"))
    parser.add_argument("--out", default=os.path.join(
        _HERE, "outputs", "notebook_plots", "model_comparison_bar.png"))
    parser.add_argument("--gt_key",   default="nonlin_additive_pairwise",
                        choices=["additive", "additive_pairwise",
                                 "nonlin_additive", "nonlin_additive_pairwise",
                                 "residualbind_ensemble", "vts1_residualbind",
                                 "hur_residualbind", "qki_residualbind",
                                 "deepsquid_msi1", "deepsquid_vts1",
                                 "deepsquid_hur", "deepsquid_qki",
                                 "deepsquid_twister"])
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20_000)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    with open(args.json) as f:
        cache = json.load(f)

    models = ["ssm"] + SURROGATE_CONFIGS
    t2_means, t2_sems = [], []
    pw_means, pw_sems = [], []

    for m in models:
        if m == "ssm":
            (tm, ts), (pm, ps) = _get_ssm(cache, args.gt_key)
        else:
            (tm, ts), (pm, ps) = _get_surrogate(
                cache, args.gt_key, m, args.mut_rate, args.lib_size)
        t2_means.append(tm); t2_sems.append(ts)
        pw_means.append(pm); pw_sems.append(ps)

    n = len(models)
    x_centers = np.arange(n, dtype=float) * (2 * BAR_W + BAR_GAP + GROUP_GAP)
    x_t2 = x_centers - (BAR_W + BAR_GAP) / 2
    x_pw = x_centers + (BAR_W + BAR_GAP) / 2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ekw = {"linewidth": 1.2, "capthick": 1.2, "zorder": 5}

    for i in range(n):
        if np.isfinite(t2_means[i]):
            ax.bar(x_t2[i], t2_means[i], width=BAR_W,
                   color=COLOR_T2, alpha=0.85,
                   yerr=t2_sems[i], capsize=4, error_kw=ekw, zorder=3)
            ax.text(x_t2[i], t2_means[i] + (t2_sems[i] or 0) + 0.015,
                    f"{t2_means[i]:.2f}", ha="center", va="bottom",
                    fontsize=8, color="black")

        if np.isfinite(pw_means[i]):
            ax.bar(x_pw[i], pw_means[i], width=BAR_W,
                   color=COLOR_PW, alpha=0.85,
                   yerr=pw_sems[i], capsize=4, error_kw=ekw, zorder=3)
            ax.text(x_pw[i], pw_means[i] + (pw_sems[i] or 0) + 0.015,
                    f"{pw_means[i]:.2f}", ha="center", va="bottom",
                    fontsize=8, color="black")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([X_LABELS[m] for m in models], fontsize=10)
    ax.set_ylabel("Spearman ρ", fontsize=11)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    ymin = min((v for v in pw_means + t2_means if np.isfinite(v)), default=-0.2)
    ymax = max((v for v in pw_means + t2_means if np.isfinite(v)), default=1.0)
    ax.set_ylim(min(ymin - 0.15, -0.25), ymax + 0.18)

    legend_handles = [
        Patch(facecolor=COLOR_T2, alpha=0.85, label="Activity-balanced library"),
        Patch(facecolor=COLOR_PW, alpha=0.85, label="Pairwise eval library"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right", framealpha=0.9)

    gt_label  = GT_DISPLAY.get(args.gt_key, args.gt_key)
    lib_label = {200: "200", 2_000: "2K", 20_000: "20K", 200_000: "200K"}.get(
        args.lib_size, str(args.lib_size))
    # Count finite surrogate instances for the first config as the canonical N
    first_cfg = SURROGATE_CONFIGS[0]
    sur_vals = (cache.get("surrogate", {})
                     .get(args.gt_key, {})
                     .get(first_cfg, {})
                     .get(str(args.mut_rate), {})
                     .get(str(args.lib_size), {})
                     .get("type2", []))
    n_sur = int(np.isfinite(np.asarray(sur_vals, dtype=float)).sum())
    ax.set_title(
        f"Surrogate model performance by eval library\n"
        f"GT: {gt_label}  |  training lib: {args.mut_rate}% mut rate, {lib_label} seqs"
        + (f"  |  N={n_sur} instance{'s' if n_sur != 1 else ''}"),
        fontsize=11, pad=8,
    )

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
