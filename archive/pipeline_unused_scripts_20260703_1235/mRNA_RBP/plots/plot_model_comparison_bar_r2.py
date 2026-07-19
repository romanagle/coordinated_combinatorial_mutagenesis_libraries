"""mRNA_RBP/plot_model_comparison_bar_r2.py

Copy of plot_model_comparison_bar.py with Pearson R² on the y-axis instead of
Spearman ρ.

R² is computed from stored surrogate coefficients (alpha, J) evaluated on the
on-disk eval libraries.  For linear surrogates the linear prediction is exact;
for GE-nonlinear surrogates this is the latent GP-map R² (GE params not saved).

Outputs:
  outputs/notebook_plots/model_comparison_bar_r2.png
  outputs/notebook_plots/model_comparison_bar_r2.svg

Usage:
    python mRNA_RBP/plot_model_comparison_bar_r2.py
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))

from mRNA_RBP.generate_libraries import (
    SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA, activity_balanced_path,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_ssm_deltas, predict_ssm

INST_DIR  = os.path.join(_HERE, "outputs")
COEF_DIR  = os.path.join(_HERE, "outputs", "surrogate_coefs")

SURROGATE_CONFIGS = [
    "additive",
    "additive + pairwise",
    "nonlinear additive",
    "nonlinear additive + pairwise",
]

# Filename-safe cfg name (mirrors lib_size_spearman.py)
_SAFE = {
    "additive":                        "additive",
    "additive + pairwise":             "additive_p_pairwise",
    "nonlinear additive":              "nonlinear_additive",
    "nonlinear additive + pairwise":   "nonlinear_additive_p_pairwise",
}

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
}

COLOR_T2 = "#4C72B0"
COLOR_PW = "#DD8452"
BAR_W    = 0.30
BAR_GAP  = 0.04
GROUP_GAP = 0.25


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1] ** 2)


def _oh(nuc_ids):
    """(N, L) uint8 → (N, L, 4) float32 one-hot."""
    return np.eye(4, dtype=np.float32)[nuc_ids.astype(int)]


def _surrogate_predict(x_oh, alpha, J, y_mean, y_std, stem_pairs):
    """Linear prediction from stored alpha + J coefficients."""
    pred = np.einsum("nla,la->n", x_oh.astype(np.float64), alpha.astype(np.float64))
    if J is not None:
        J_d = J.astype(np.float64)
        for ni in range(len(x_oh)):
            xi = x_oh[ni]
            for (ei, ej) in stem_pairs:
                pred[ni] += float(xi[ei] @ J_d[ei, :, ej, :] @ xi[ej])
    return pred * y_std + y_mean


def _load_coef(k, pct, n_lib, cfg_name, gt_key):
    safe = _SAFE[cfg_name]
    path = os.path.join(
        COEF_DIR,
        f"coefs_k{k:02d}_mut{pct:02d}_lib{n_lib}_{safe}_{gt_key}.npz",
    )
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    alpha  = d["alpha"]
    J      = d["J"] if "J" in d else None
    y_mean = float(d["y_mean"]) if "y_mean" in d else 0.0
    y_std  = float(d["y_std"])  if "y_std"  in d else 1.0
    return alpha, J, y_mean, y_std


def _r2_surrogate(k, pct, n_lib, cfg_name, gt_key):
    """Pearson R2 for surrogate on activity-balanced and pairwise eval libs."""
    coef = _load_coef(k, pct, n_lib, cfg_name, gt_key)
    if coef is None:
        return float("nan"), float("nan")
    alpha, J, y_mean, y_std = coef

    inst_dir = os.path.join(INST_DIR, f"instance_{k:02d}")
    t2_path  = activity_balanced_path(inst_dir)
    pw_path  = os.path.join(inst_dir, "pairwise_lib.npz")

    r2_t2 = float("nan")
    if os.path.isfile(t2_path):
        ev    = np.load(t2_path)
        x_oh  = _oh(ev["nuc_ids"])
        y_gt  = ev[f"scores_{gt_key}"].astype(float)
        y_hat = _surrogate_predict(x_oh, alpha, J, y_mean, y_std, STEM_PAIRS)
        r2_t2 = _r2(y_gt, y_hat)

    r2_pw = float("nan")
    if os.path.isfile(pw_path):
        ev    = np.load(pw_path)
        x_oh  = _oh(ev["nuc_ids"])
        y_gt  = ev[f"scores_{gt_key}"].astype(float)
        y_hat = _surrogate_predict(x_oh, alpha, J, y_mean, y_std, STEM_PAIRS)
        r2_pw = _r2(y_gt, y_hat)

    return r2_t2, r2_pw


def _r2_ssm(k, gt_key):
    """Pearson R2 for SSM predictor on activity-balanced and pairwise eval libs."""
    inst_dir = os.path.join(INST_DIR, f"instance_{k:02d}")
    t2_path  = activity_balanced_path(inst_dir)
    pw_path  = os.path.join(inst_dir, "pairwise_lib.npz")

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k,
                               stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()
    delta = make_ssm_deltas(wt_oh, gt)

    r2_t2 = float("nan")
    if os.path.isfile(t2_path):
        ev    = np.load(t2_path)
        x_oh  = _oh(ev["nuc_ids"])
        y_gt  = ev[f"scores_{gt_key}"].astype(float)
        y_hat = predict_ssm(x_oh, delta)
        r2_t2 = _r2(y_gt, y_hat)

    r2_pw = float("nan")
    if os.path.isfile(pw_path):
        ev    = np.load(pw_path)
        x_oh  = _oh(ev["nuc_ids"])
        y_gt  = ev[f"scores_{gt_key}"].astype(float)
        y_hat = predict_ssm(x_oh, delta)
        r2_pw = _r2(y_gt, y_hat)

    return r2_t2, r2_pw


def _stats(vals):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return float("nan"), 0.0
    mean = float(a.mean())
    sem  = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) >= 2 else 0.0
    return mean, sem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_key",   default="nonlin_additive_pairwise",
                        choices=["additive", "additive_pairwise",
                                 "nonlin_additive", "nonlin_additive_pairwise"])
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20_000)
    parser.add_argument("--out", default=os.path.join(
        _HERE, "outputs", "notebook_plots", "model_comparison_bar_r2.png"))
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    instances = sorted(
        int(d.replace("instance_", ""))
        for d in os.listdir(INST_DIR)
        if d.startswith("instance_") and os.path.isdir(os.path.join(INST_DIR, d))
    )
    print(f"Found instances: {instances}")

    models = ["ssm"] + SURROGATE_CONFIGS
    t2_vals_all = {m: [] for m in models}
    pw_vals_all = {m: [] for m in models}

    for k in instances:
        r2_t2, r2_pw = _r2_ssm(k, args.gt_key)
        t2_vals_all["ssm"].append(r2_t2)
        pw_vals_all["ssm"].append(r2_pw)
        for cfg in SURROGATE_CONFIGS:
            r2_t2, r2_pw = _r2_surrogate(k, args.mut_rate, args.lib_size, cfg, args.gt_key)
            t2_vals_all[cfg].append(r2_t2)
            pw_vals_all[cfg].append(r2_pw)
        print(f"  instance {k:02d} done")

    t2_means, t2_sems, pw_means, pw_sems = [], [], [], []
    for m in models:
        tm, ts = _stats(t2_vals_all[m])
        pm, ps = _stats(pw_vals_all[m])
        t2_means.append(tm); t2_sems.append(ts)
        pw_means.append(pm); pw_sems.append(ps)
        print(f"  {m:35s}  activity-balanced R²={tm:.3f}  pairwise R²={pm:.3f}")

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
    ax.set_ylabel("Pearson R²", fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(facecolor=COLOR_T2, alpha=0.85, label="Activity-balanced library"),
        Patch(facecolor=COLOR_PW, alpha=0.85, label="Pairwise eval library"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right", framealpha=0.9)

    gt_label  = GT_DISPLAY.get(args.gt_key, args.gt_key)
    lib_label = {200: "200", 2_000: "2K", 20_000: "20K"}.get(args.lib_size, str(args.lib_size))
    n_inst    = len(instances)
    ax.set_title(
        f"Surrogate model performance by eval library (Pearson R²)\n"
        f"GT: {gt_label}  |  training lib: {args.mut_rate}% mut rate, {lib_label} seqs"
        f"  |  N={n_inst} instance{'s' if n_inst != 1 else ''}",
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
