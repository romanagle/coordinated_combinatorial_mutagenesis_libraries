"""mRNA_RBP/bar_spearman_mrna.py

Re-runs the pairwise GE surrogate (same config as plot_surrogate_scatter.py)
across all 10 instances × 3 mut_rates × 3 lib_sizes × 4 GT keys, saves
per-instance Spearman ρ to a JSON, then generates one bar chart PNG per GT key.

Resumable: already-computed entries are skipped on re-run.

Usage:
    python mRNA_RBP/bar_spearman_mrna.py \
        --out_json  mRNA_RBP/outputs/rho_results.json \
        --out_dir   mRNA_RBP/outputs/plots/
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "residualbind"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

import squid.surrogate_zoo
from mRNA_RBP.generate_libraries import (
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES,
)

# ---------------------------------------------------------------------------
# Config — matches plot_surrogate_scatter.py exactly
# ---------------------------------------------------------------------------

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]

GT_LABELS = {
    "additive":                 "Additive GT",
    "additive_pairwise":        "Additive+Pair GT",
    "nonlin_additive":          "Nonlin Add GT",
    "nonlin_additive_pairwise": "Nonlin Add+Pair GT",
}

SURROGATE_CFG = {
    "gpmap":           "pairwise",
    "linearity":       "nonlinear",
    "regression_type": "GE",
    "noise":           "Gaussian",
    "noise_order":     2,
    "reg_strength":    0.0,
    "hidden_nodes":    50,
}

NUCS    = ["A", "C", "G", "U"]
_NUCS_A = np.array(list("ACGU"))

BLUE   = "#4C72B0"
ORANGE = "#DD8452"

MUT_LABELS = {10: "10%", 25: "25%", 50: "50%"}
LIB_LABELS = {200: "200", 2_000: "2K", 20_000: "20K"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nuc_ids_to_str(nuc_ids: np.ndarray) -> np.ndarray:
    return np.array(["".join(_NUCS_A[row]) for row in nuc_ids], dtype=object)


def predict_chunked(model, x_str, chunk: int = 500) -> np.ndarray:
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(y_true, y_hat):
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(y_true[mask], y_hat[mask])[0])


def train_surrogate(X: np.ndarray, y: np.ndarray):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG["hidden_nodes"],
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    return model, test_df


def run_instance(k: int, pct: int, n_lib: int, gt_key: str) -> tuple:
    """Returns (rho_rand, rho_eval) for one instance and GT key."""
    mut_dir   = os.path.join(OUT_BASE, f"instance_{k:02d}", f"mut{pct:02d}")
    rand_path = os.path.join(mut_dir, f"lib_{n_lib}.npz")
    eval_path = os.path.join(mut_dir, f"eval_uniform_{gt_key}_lib{n_lib}.npz")

    d       = np.load(rand_path)
    nuc_ids = d["nuc_ids"]
    y_all   = d[f"scores_{gt_key}"].astype(np.float32)
    X       = np.eye(4, dtype=np.float32)[nuc_ids]

    model, test_df = train_surrogate(X, y_all.reshape(-1, 1))

    x_col     = "x" if "x" in test_df.columns else "X"
    y_col     = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
    x_test    = np.asarray(test_df[x_col])
    y_test    = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_test = predict_chunked(model, x_test)
    rho_rand  = _rho(y_test, yhat_test)

    d_ev       = np.load(eval_path)
    y_eval     = d_ev[f"scores_{gt_key}"].astype(float)
    x_eval_str = nuc_ids_to_str(d_ev["nuc_ids"])
    yhat_eval  = predict_chunked(model, x_eval_str)
    rho_eval   = _rho(y_eval, yhat_eval)

    return rho_rand, rho_eval


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    return float(np.nanmean(a)), (float(np.nanstd(a, ddof=1) / np.sqrt(len(a))) if len(a) >= 2 else 0.0)


def plot_bar(results: dict, gt_key: str, out_path: str, save_svg: bool = False):
    """
    results[pct][n_lib] = {"rand": [...], "eval": [...]}
    x-axis: library size groups (3)
    within each: mutation rate sub-groups (3), each with 2 bars (rand/eval)
    """
    BAR_W    = 0.12
    PAIR_GAP = 0.03
    MUT_GAP  = 0.10
    LIB_GAP  = 0.35

    mut_colors = {10: "#4C72B0", 25: "#55A868", 50: "#C44E52"}

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    xtick_pos, xtick_labs = [], []
    minor_pos, minor_labs = [], []

    x = 0.0
    for n_lib in LIB_SIZES:
        lib_start = x
        for pct in MUT_RATES_PCT:
            color     = mut_colors[pct]
            rand_vals = results[pct][n_lib]["rand"]
            eval_vals = results[pct][n_lib]["eval"]
            r_mean, r_sem = _stats(rand_vals)
            e_mean, e_sem = _stats(eval_vals)
            ekw = {"linewidth": 1.2, "capthick": 1.2, "zorder": 5}

            rx = x
            if np.isfinite(r_mean):
                ax.bar(rx, r_mean, width=BAR_W, color=color, alpha=0.75,
                       yerr=r_sem, capsize=3, error_kw=ekw, zorder=3)

            ex = x + BAR_W + PAIR_GAP
            if np.isfinite(e_mean):
                ax.bar(ex, e_mean, width=BAR_W, color=color, alpha=0.75,
                       hatch="///", edgecolor="white",
                       yerr=e_sem, capsize=3, error_kw=ekw, zorder=3)

            minor_pos.append(x + BAR_W + PAIR_GAP / 2)
            minor_labs.append(MUT_LABELS[pct])
            x += 2 * BAR_W + PAIR_GAP + MUT_GAP

        lib_end = x - MUT_GAP
        xtick_pos.append((lib_start + lib_end) / 2)
        xtick_labs.append(f"lib size = {LIB_LABELS[n_lib]}")

        if n_lib != LIB_SIZES[-1]:
            ax.axvline(lib_end + LIB_GAP / 2, color="gray", linewidth=0.8,
                       linestyle="--", alpha=0.4)
        x += LIB_GAP

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labs, fontsize=12, fontweight="bold")
    ax.set_xticks(minor_pos, minor=True)
    ax.set_xticklabels(minor_labs, minor=True, fontsize=8, color="dimgray")
    ax.tick_params(axis="x", which="minor", length=0, pad=2)

    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlim(-0.2, x - LIB_GAP + 0.2)
    ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    mut_patches = [Patch(facecolor=mut_colors[p], alpha=0.75, label=f"mut {MUT_LABELS[p]}")
                   for p in MUT_RATES_PCT]
    rand_patch  = Patch(facecolor="gray", alpha=0.75, label="Random (MAVE test split)")
    eval_patch  = Patch(facecolor="gray", alpha=0.75, hatch="///",
                        edgecolor="white", label="Eval (curated library)")

    ax.legend(handles=mut_patches + [rand_patch, eval_patch],
              loc="lower right", fontsize=9, ncol=3, framealpha=0.9)

    n_inst = len(next(iter(next(iter(results.values())).values()))["rand"])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    msg = out_path
    if save_svg:
        fig.savefig(os.path.splitext(out_path)[0] + ".svg", bbox_inches="tight")
        msg = f"{out_path} / .svg"
    plt.close(fig)
    print(f"\n[saved] bar chart → {msg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json",
                        default=os.path.join(_HERE, "outputs", "rho_results.json"))
    parser.add_argument("--out_dir",
                        default=os.path.join(_HERE, "outputs", "plots"))
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    # Load or init results cache — keyed by gt_key
    if os.path.isfile(args.out_json):
        with open(args.out_json) as f:
            all_results = json.load(f)
        print(f"[resume] loaded cached results from {args.out_json}")
    else:
        all_results = {}

    for gt_key in GT_KEYS:
        if gt_key not in all_results:
            all_results[gt_key] = {}

        # convert str keys back to int
        results = {int(p): {int(n): v for n, v in d.items()}
                   for p, d in all_results[gt_key].items()}
        # fill any missing combos
        for pct in MUT_RATES_PCT:
            if pct not in results:
                results[pct] = {}
            for n_lib in LIB_SIZES:
                if n_lib not in results[pct]:
                    results[pct][n_lib] = {"rand": [], "eval": []}

        print(f"\n{'#'*60}")
        print(f"  GT key: {gt_key}")
        print(f"{'#'*60}")

        for pct in MUT_RATES_PCT:
            for n_lib in LIB_SIZES:
                n_done = len(results[pct][n_lib]["rand"])
                if n_done >= N_INSTANCES:
                    print(f"[skip] {gt_key} mut{pct} lib{n_lib}: already have {n_done} instances")
                    continue

                print(f"\n{'='*55}")
                print(f"  {gt_key}  mut {pct}%  lib_size = {n_lib:,}")
                print(f"{'='*55}")

                for k in range(n_done, N_INSTANCES):
                    print(f"  instance {k:02d} …", flush=True)
                    try:
                        rho_r, rho_e = run_instance(k, pct, n_lib, gt_key)
                        results[pct][n_lib]["rand"].append(rho_r)
                        results[pct][n_lib]["eval"].append(rho_e)
                        print(f"    ρ_rand={rho_r:.4f}  ρ_eval={rho_e:.4f}", flush=True)
                    except Exception as exc:
                        print(f"    FAILED: {exc}")
                        results[pct][n_lib]["rand"].append(float("nan"))
                        results[pct][n_lib]["eval"].append(float("nan"))

                    # Save after every instance
                    all_results[gt_key] = results
                    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
                    with open(args.out_json, "w") as f:
                        json.dump(all_results, f, indent=2)

        out_path = os.path.join(args.out_dir, f"spearman_bar_{gt_key}.png")
        plot_bar(results, gt_key, out_path, save_svg=args.svg)


if __name__ == "__main__":
    main()
