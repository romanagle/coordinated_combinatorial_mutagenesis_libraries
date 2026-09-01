"""
mRNA_RBP/plot_cross_mutrate.py

Reads cross_mutrate_results.json (from cross_mutrate_eval.py or
cross_mutrate_residualbind.py) and plots the cross-mutation-rate
generalization gap: surrogates trained at one mutation rate evaluated on a
matched-size test set from the SAME rate vs. from each OTHER rate.

Two figures:
  (a) cross_mutrate_libsize_<cfg>.png -- one per surrogate config, 3 panels
      (one per train mutation rate), x = lib size, lines = {same-rate, cross
      -> other rates}, mean +/- std across instances.
  (b) cross_mutrate_heatmap.png -- 3x3 train x test heatmap of mean Spearman
      |rho| at the largest lib size for nonlinear additive + pairwise.

Works unmodified for both the synthetic-GT results (10 instances) and the
ResidualBind results (1 instance -- _stats() degrades std to 0.0 for
single-element lists, so the same code produces a flat line / single-value
heatmap with no extra logic).

    Usage:
    python mRNA_RBP/plot_cross_mutrate.py
    python mRNA_RBP/plot_cross_mutrate.py \
        --results_json "mRNA_RBP/outputs/ground_truth_collections/ResidualBind oracle MSI1/libraries_used_for_figures/cross_mutrate_results.json" \
        --gt_key residualbind_ensemble --out_prefix residualbind_
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.src.evaluate import make_ssm_deltas_from_scores
from mRNA_RBP.scripts.pipeline.generate_libraries import MUT_RATES_PCT, LIB_SIZES
from mRNA_RBP.scripts.figures.core.provenance import stamp_figure

DEFAULT_JSON = os.path.join(_HERE, "outputs", "cross_mutrate_results.json")
OUT_DIR      = os.path.join(_HERE, "outputs", "notebook_plots")
DEFAULT_GT_KEY = "nonlin_additive_pairwise"
CFG_NAMES = [
    "nonlinear additive + pairwise",
]
MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}
MIXED_LABEL = "mixed"


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, np.nan
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) >= 2 else 0.0)


def _rate_key(rate):
    return str(rate)


def _rate_dir(rate):
    return "mut_mixed" if str(rate) == MIXED_LABEL else f"mut{int(rate):02d}"


def _coef_rate_tag(rate):
    return "mut_mixed" if str(rate) == MIXED_LABEL else f"mut{int(rate):02d}"


def _rate_tick(rate):
    return "mixed" if str(rate) == MIXED_LABEL else f"{int(rate)}%"


def _safe_cfg(cfg_name):
    return cfg_name.replace(" ", "_").replace("+", "p")


def _heatmap_rates(cache, gt_key, cfg_names):
    rates = list(MUT_RATES_PCT)
    for cfg_name in cfg_names:
        sur = cache.get("cross", {}).get(gt_key, {}).get(cfg_name, {})
        if MIXED_LABEL in sur:
            rates.append(MIXED_LABEL)
            break
        for entry_by_size in sur.values():
            for entry in entry_by_size.values():
                if MIXED_LABEL in entry.get("cross", {}):
                    rates.append(MIXED_LABEL)
                    return rates
    return rates


def _rho_abs(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    return float(abs(spearmanr(y_true[mask], y_pred[mask])[0]))


def _instance_dirs(out_base):
    if not out_base or not os.path.isdir(out_base):
        return []
    return [
        os.path.join(out_base, name)
        for name in sorted(os.listdir(out_base))
        if name.startswith("instance_") and os.path.isdir(os.path.join(out_base, name))
    ]


def _ssm_delta_from_file(ssm_path, gt_key):
    ssm = np.load(ssm_path)
    nuc_ids = ssm["nuc_ids"]
    scores = ssm[f"scores_{gt_key}"].astype(float)
    L = nuc_ids.shape[1]

    wt_idx = np.array([np.bincount(nuc_ids[:, pos], minlength=4).argmax()
                       for pos in range(L)], dtype=np.uint8)
    wt_key = f"wt_score_{gt_key}"
    wt_activity = float(ssm[wt_key][0]) if wt_key in ssm.files else 0.0
    delta = make_ssm_deltas_from_scores(
        nuc_ids, scores, wt_idx=wt_idx, wt_activity=wt_activity
    )
    return delta, wt_activity


def _predict_ssm_from_delta(nuc_ids, delta, wt_activity=0.0):
    cols = np.arange(nuc_ids.shape[1])
    return delta[nuc_ids, cols].sum(axis=1) + float(wt_activity)


def _predict_alpha_only(nuc_ids, alpha):
    cols = np.arange(nuc_ids.shape[1])[None, :]
    return alpha[cols, nuc_ids].sum(axis=1)


def _ssm_by_test_mutrate(out_base, gt_key, lib_size, rates=None):
    """Mean SSM Spearman |rho| per mutation-rate library across instances."""
    rates = rates or MUT_RATES_PCT
    by_rate = {str(rate): [] for rate in rates}
    for inst_dir in _instance_dirs(out_base):
        ssm_path = os.path.join(inst_dir, "ssm.npz")
        if not os.path.exists(ssm_path):
            continue
        try:
            delta, wt_activity = _ssm_delta_from_file(ssm_path, gt_key)
        except KeyError:
            continue

        for rate in rates:
            lib_path = os.path.join(inst_dir, _rate_dir(rate), f"lib_{lib_size}.npz")
            if not os.path.exists(lib_path):
                continue
            lib = np.load(lib_path)
            score_key = f"scores_{gt_key}"
            if score_key not in lib:
                continue
            y_pred = _predict_ssm_from_delta(lib["nuc_ids"], delta, wt_activity)
            by_rate[str(rate)].append(_rho_abs(lib[score_key].astype(float), y_pred))

    means = []
    for rate in rates:
        m, _ = _stats(by_rate[str(rate)])
        means.append(m)
    return np.asarray(means, dtype=float)


def _coef_path(coef_dir, k, train_rate, lib_size, cfg_name, gt_key):
    return os.path.join(
        coef_dir,
        f"coefs_k{k:02d}_{_coef_rate_tag(train_rate)}_lib{lib_size}_"
        f"{_safe_cfg(cfg_name)}_{gt_key}.npz",
    )


def _surrogate_ssm_by_test_mutrate(out_base, gt_key, cfg_name, lib_size, rates):
    """SSM baseline from the surrogate's learned additive alpha weights.

    For each test-rate column, use the alpha weights learned by the same
    surrogate configuration on the matching training library, and evaluate
    alpha-only predictions on that library. This keeps the ResidualBind SSM
    row tied to surrogate-learned additive effects rather than a ResidualBind
    single-mutant oracle.
    """
    if not out_base:
        return None
    coef_dir = os.path.join(out_base, "surrogate_coefs")
    if not os.path.isdir(coef_dir):
        return None

    by_rate = {str(rate): [] for rate in rates}
    for inst_dir in _instance_dirs(out_base):
        try:
            k = int(os.path.basename(inst_dir).split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for rate in rates:
            coef_path = _coef_path(coef_dir, k, rate, lib_size, cfg_name, gt_key)
            lib_path = os.path.join(inst_dir, _rate_dir(rate), f"lib_{lib_size}.npz")
            if not os.path.exists(coef_path) or not os.path.exists(lib_path):
                continue
            coefs = np.load(coef_path)
            if "alpha" not in coefs:
                continue
            lib = np.load(lib_path)
            score_key = f"scores_{gt_key}"
            if score_key not in lib:
                continue
            y_pred = _predict_alpha_only(lib["nuc_ids"], coefs["alpha"])
            by_rate[str(rate)].append(_rho_abs(lib[score_key].astype(float), y_pred))

    means = []
    for rate in rates:
        m, _ = _stats(by_rate[str(rate)])
        means.append(m)
    out = np.asarray(means, dtype=float)
    return out if np.isfinite(out).any() else None


def plot_lines(cache, gt_key, cfg_name, out_stem, save_svg=False,
               source_paths=None, library_status=None):
    sur = cache.get("cross", {}).get(gt_key, {}).get(cfg_name, {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    x = np.array(LIB_SIZES, dtype=float)

    for ax, train_pct in zip(axes, MUT_RATES_PCT):
        spct = str(train_pct)
        other_pcts = [p for p in MUT_RATES_PCT if p != train_pct]

        series = [("same_rate", "-", "o", "#4C72B0", f"same-rate ({train_pct}%)")]
        for op in other_pcts:
            series.append((("cross", str(op)), "--", "s", MUT_COLORS[op], f"cross -> {op}%"))

        for key, ls, marker, color, label in series:
            means, stds = [], []
            for n_lib in LIB_SIZES:
                entry = sur.get(spct, {}).get(str(n_lib), {})
                vals = (entry.get("same_rate", []) if key == "same_rate"
                        else entry.get("cross", {}).get(key[1], []))
                m, s = _stats(vals)
                means.append(m)
                stds.append(s)
            means, stds = np.array(means), np.array(stds)
            valid = np.isfinite(means)
            if not valid.any():
                continue
            ax.plot(x[valid], means[valid], color=color, ls=ls, marker=marker,
                    markersize=5, linewidth=1.8, label=label)
            ax.fill_between(x[valid], (means - stds)[valid], (means + stds)[valid],
                            color=color, alpha=0.12)

        ax.set_xscale("log")
        ax.set_xticks(LIB_SIZES)
        ax.set_xticklabels(["200", "2K", "20K"], fontsize=9)
        ax.set_xlabel("Training library size", fontsize=10)
        ax.set_title(f"trained at mut rate = {train_pct}%", fontsize=10)
        ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, axis="y", ls="--", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Spearman |ρ|  (mean ± std)", fontsize=10)
    fig.suptitle(f"Cross-mutation-rate generalization — {cfg_name}", fontsize=11, y=1.02)
    stamp_figure(fig, library_status=library_status, source_paths=source_paths)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{out_stem}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if save_svg:
        svg = os.path.join(OUT_DIR, f"{out_stem}.svg")
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")


def plot_heatmap_grid(cache, gt_key, lib_size=20_000, cfg_names=CFG_NAMES,
                      out_base=None,
                      out_stem="cross_mutrate_heatmap", save_svg=False,
                      source_paths=None, library_status=None):
    rates = _heatmap_rates(cache, gt_key, cfg_names)
    train_rates = rates
    test_rates = rates
    fallback_ssm_row = (_ssm_by_test_mutrate(out_base, gt_key, lib_size, test_rates)
                        if out_base else None)
    has_fallback_ssm = fallback_ssm_row is not None and np.isfinite(fallback_ssm_row).any()
    cfg_ssm_rows = [
        _surrogate_ssm_by_test_mutrate(out_base, gt_key, cfg_name, lib_size, test_rates)
        for cfg_name in cfg_names
    ]
    has_cfg_ssm = any(row is not None and np.isfinite(row).any() for row in cfg_ssm_rows)
    has_ssm = has_cfg_ssm or has_fallback_ssm
    n_rows = len(train_rates) + (1 if has_ssm else 0)

    fig, axes = plt.subplots(1, len(cfg_names), figsize=(4.2 * len(cfg_names), 4.8),
                             sharey=True)
    sn = str(lib_size)
    im = None

    for ax, cfg_name, cfg_ssm_row in zip(axes, cfg_names, cfg_ssm_rows):
        sur = cache.get("cross", {}).get(gt_key, {}).get(cfg_name, {})
        M = np.full((n_rows, len(test_rates)), np.nan)
        for i, train_rate in enumerate(train_rates):
            entry = sur.get(_rate_key(train_rate), {}).get(sn, {})
            for j, test_rate in enumerate(test_rates):
                vals = (entry.get("same_rate", []) if str(test_rate) == str(train_rate)
                        else entry.get("cross", {}).get(_rate_key(test_rate), []))
                m, _ = _stats(vals)
                M[i, j] = m
        if has_ssm:
            # Prefer the true oracle's own single-mutant (SSM) scan when available
            # -- only fall back to a surrogate's learned alpha-only reconstruction
            # when no real oracle SSM data exists for this gt_key.
            M[-1, :] = fallback_ssm_row if has_fallback_ssm else cfg_ssm_row

        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(test_rates)))
        ax.set_xticklabels([_rate_tick(rate) for rate in test_rates])
        ax.set_yticks(range(n_rows))
        ylabels = [_rate_tick(rate) for rate in train_rates] + (["SSM"] if has_ssm else [])
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("test mut rate")
        if ax is axes[0]:
            ax.set_ylabel("train mut rate")
        ax.set_title(cfg_name, fontsize=9)
        for i in range(n_rows):
            for j in range(len(test_rates)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            color="white" if M[i, j] < 0.6 else "black", fontsize=8)

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="mean Spearman |ρ|")
    fig.suptitle(f"Train × test mutation-rate generalization (lib_size={lib_size:,})", y=1.05)
    stamp_figure(fig, library_status=library_status, source_paths=source_paths)

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{out_stem}.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    if save_svg:
        svg = os.path.join(OUT_DIR, f"{out_stem}.svg")
        fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", default=DEFAULT_JSON)
    parser.add_argument("--gt_key", default=DEFAULT_GT_KEY)
    parser.add_argument("--out_prefix", default="")
    parser.add_argument("--out_base", default=None,
                        help="Library root for computing the SSM row. "
                             "Defaults to the directory containing results_json.")
    parser.add_argument("--heatmap_lib_size", type=int, default=20_000)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    parser.add_argument("--out_dir", default=None,
                        help="Override output directory (default: outputs/notebook_plots)")
    args = parser.parse_args()

    if args.out_dir:
        global OUT_DIR
        OUT_DIR = args.out_dir

    with open(args.results_json) as f:
        cache = json.load(f)
    source_paths = [args.results_json]
    out_base = args.out_base or os.path.dirname(os.path.abspath(args.results_json))
    if out_base:
        source_paths.append(out_base)
    library_status = None

    for cfg_name in CFG_NAMES:
        safe = cfg_name.replace(" ", "_").replace("+", "p")
        plot_lines(cache, args.gt_key, cfg_name,
                   out_stem=f"{args.out_prefix}cross_mutrate_libsize_{safe}",
                   save_svg=args.svg,
                   source_paths=source_paths,
                   library_status=library_status)

    plot_heatmap_grid(cache, args.gt_key, lib_size=args.heatmap_lib_size,
                      out_base=out_base,
                      out_stem=f"{args.out_prefix}cross_mutrate_heatmap",
                      save_svg=args.svg,
                      source_paths=source_paths,
                      library_status=library_status)


if __name__ == "__main__":
    main()
