"""
mRNA_RBP/bar_surrogate_models_type3.py

One-instance, no-cache model comparison with the type3 eval library. This
intentionally does not read lib_size_spearman_results_type3.json because old
cached scores may have been computed with stale pairwise-coupling initialization.

The script reuses existing library nuc_ids, but recomputes all GT scores from
the current MrnaRbpGroundTruth before training/evaluation.

Models (x-axis groups):
  SSM | Additive | Additive+Pairwise | NL Additive only

Eval libraries (4 bars per group, centered):
  gray  (#8C8C8C) = random holdout / random 10% library
  blue  (#4C72B0) = uniform eval (type 2)
  orange(#DD8452) = targeted pairwise
  green (#55A868) = saturated additive + pw

Run with the squid conda environment:
  /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/bar_surrogate_models_type3.py

Output: mRNA_RBP/outputs/notebook_plots/model_comparison_bar_type3.png / .svg
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA,
    activity_balanced_path,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_ssm_deltas, predict_ssm
from mRNA_RBP.lib_size_spearman import (
    SURROGATE_CONFIGS, _rho, nuc_ids_to_str,
    predict_chunked, train_surrogate,
)

INSTANCE  = 0
GT_KEY    = "nonlin_additive_pairwise"
MUT_PCT   = 10
LIB_SIZE  = 20_000

OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots")

COLOR_RANDOM   = "#8C8C8C"
COLOR_TYPE2    = "#4C72B0"
COLOR_PAIRWISE = "#DD8452"
COLOR_TYPE3    = "#55A868"
BAR_WIDTH      = 0.19
EVAL_LABELS = {
    "random": "Random holdout",
    "type2": "Uniform eval (type 2)",
    "pairwise": "Targeted pairwise",
    "type3": "Saturated additive + pw",
}
EVAL_COLORS = {
    "random": COLOR_RANDOM,
    "type2": COLOR_TYPE2,
    "pairwise": COLOR_PAIRWISE,
    "type3": COLOR_TYPE3,
}
EVAL_KEYS = ["random", "type2", "pairwise", "type3"]

SURROGATE_MAP = {
    "Additive":            "additive",
    "Additive\n+Pairwise": "additive + pairwise",
    "NL Additive\nonly":   "nonlinear additive",
    "NL Additive\n+Pairwise": "nonlinear additive + pairwise",
}


def _stats(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan, 0.0
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) if len(a) >= 2 else 0.0)


def _annotate(ax, bars):
    for bar in bars:
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + 0.01 if h >= 0 else h - 0.03,
                    f"{h:.2f}",
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=7.5)


def _rmse(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_hat = np.asarray(y_hat, dtype=float).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_hat[mask]) ** 2)))


def _metric_value(y_true, y_hat, metric):
    if metric == "rmse":
        return _rmse(y_true, y_hat)
    return _rho(y_true, y_hat)


def _score_nuc_ids(gt, nuc_ids, chunk=100_000):
    vals = []
    for start in range(0, len(nuc_ids), chunk):
        ids = nuc_ids[start:start + chunk]
        x = np.eye(4, dtype=np.float32)[ids]
        vals.append(gt.score_all(x)[GT_KEY].astype(float))
    return np.concatenate(vals)


def _eval_model(model, nuc_ids, y_true, metric):
    y_hat = predict_chunked(model, nuc_ids_to_str(nuc_ids))
    return _metric_value(y_true, y_hat, metric)


def _load_nuc_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)["nuc_ids"]


def _train_mask(train_nuc_ids, eval_ids):
    eval_seqs = set()
    for ids in eval_ids.values():
        for row in ids:
            eval_seqs.add(row.tobytes())
    return np.array([row.tobytes() not in eval_seqs for row in train_nuc_ids], dtype=bool)


def _synthetic_data():
    inst_dir = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}")
    train_path = os.path.join(inst_dir, f"mut{MUT_PCT:02d}", f"lib_{LIB_SIZE}.npz")
    eval_paths = {
        "type2": activity_balanced_path(inst_dir),
        "pairwise": os.path.join(inst_dir, "pairwise_lib.npz"),
        "type3": os.path.join(inst_dir, "type3.npz"),
    }

    gt = MrnaRbpGroundTruth(
        SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=INSTANCE, stem_sigma=STEM_SIGMA
    )
    eval_ids = {name: _load_nuc_ids(path) for name, path in eval_paths.items()}
    train_ids = _load_nuc_ids(train_path)[:LIB_SIZE]
    eval_scores = {name: _score_nuc_ids(gt, ids) for name, ids in eval_ids.items()}
    train_scores = _score_nuc_ids(gt, train_ids)
    delta_W = make_ssm_deltas(gt.wt_one_hot(), gt)
    return train_ids, train_scores, eval_ids, eval_scores, delta_W


def _cache_data(score_cache):
    d = np.load(score_cache)
    eval_ids = {name: d[f"nuc_ids_{name}"] for name in ("type2", "pairwise", "type3")}
    eval_scores = {name: d[f"scores_{name}"].astype(float) for name in eval_ids}
    train_ids = d["nuc_ids_train"][:LIB_SIZE]
    train_scores = d["scores_train"][:LIB_SIZE].astype(float)
    delta_W = d["ssm_delta"].astype(float)
    return train_ids, train_scores, eval_ids, eval_scores, delta_W


def _fresh_values(score_cache=None, oracle_label="Synthetic GT", metric="spearman"):
    if score_cache:
        train_ids, train_scores, eval_ids, eval_scores, delta_W = _cache_data(score_cache)
    else:
        train_ids, train_scores, eval_ids, eval_scores, delta_W = _synthetic_data()

    library_sizes = {name: len(ids) for name, ids in eval_ids.items()}
    values = {}
    ssm = {}
    for name, ids in eval_ids.items():
        x = np.eye(4, dtype=np.float32)[ids]
        y_true = eval_scores[name]
        y_hat = predict_ssm(x, delta_W, wt_activity=0.0)
        rho = _metric_value(y_true, y_hat, metric)
        # SSM is constant on the pairwise stem library by construction; plot as 0
        # rather than dropping the bar because Spearman is undefined.
        if metric == "spearman" and not np.isfinite(rho):
            rho = 0.0
        ssm[name] = rho
    values["SSM"] = ssm

    mask = _train_mask(train_ids, eval_ids)
    n_removed = int((~mask).sum())
    train_ids = train_ids[mask]
    train_scores = train_scores[mask]
    if n_removed:
        print(f"[dedup] removed {n_removed} train seqs found in eval libs")
    x_train = np.eye(4, dtype=np.float32)[train_ids]
    y_train = train_scores.reshape(-1, 1)
    library_sizes["random"] = len(train_ids)
    yhat_ssm_train = predict_ssm(x_train, delta_W, wt_activity=0.0)
    values["SSM"]["random"] = _metric_value(y_train.ravel(), yhat_ssm_train, metric)

    for label, cfg_key in SURROGATE_MAP.items():
        cfg = SURROGATE_CONFIGS[cfg_key]
        print(f"[train] {label.replace(chr(10), ' ')}  n={len(train_ids):,}")
        wrapper, model, test_df = train_surrogate(x_train, y_train, cfg)
        x_col = "x" if "x" in test_df.columns else "X"
        y_col = "y" if "y" in test_df.columns else next(
            c for c in test_df.columns if c.startswith("y"))
        y_test = np.asarray(test_df[y_col], dtype=float).ravel()
        yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
        entry = {"random": _metric_value(y_test, yhat_test, metric)}
        entry.update({
            name: _eval_model(model, ids, eval_scores[name], metric)
            for name, ids in eval_ids.items()
        })
        values[label] = entry
        print(f"        rand={values[label]['random']:+.3f}  "
              f"activity-balanced={values[label]['type2']:+.3f}  "
              f"pw={values[label]['pairwise']:+.3f}  "
              f"t3={values[label]['type3']:+.3f}")
        del wrapper, model
        tf.keras.backend.clear_session()

    print(f"[oracle] {oracle_label}  metric={metric}")
    return values, library_sizes


def _metric_pair(y_true, y_hat):
    rho = _rho(y_true, y_hat)
    return {
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "rmse": _rmse(y_true, y_hat),
    }


def _fresh_metric_values(score_cache=None, oracle_label="Synthetic GT"):
    if score_cache:
        train_ids, train_scores, eval_ids, eval_scores, delta_W = _cache_data(score_cache)
    else:
        train_ids, train_scores, eval_ids, eval_scores, delta_W = _synthetic_data()

    library_sizes = {name: len(ids) for name, ids in eval_ids.items()}
    values = {"spearman": {}, "rmse": {}}

    def store(label, key, y_true, y_hat):
        vals = _metric_pair(y_true, y_hat)
        if key == "pairwise" and label == "SSM" and not np.isfinite(vals["spearman"]):
            vals["spearman"] = 0.0
        for metric, value in vals.items():
            values[metric].setdefault(label, {})[key] = value

    for name, ids in eval_ids.items():
        x = np.eye(4, dtype=np.float32)[ids]
        store("SSM", name, eval_scores[name], predict_ssm(x, delta_W, wt_activity=0.0))

    mask = _train_mask(train_ids, eval_ids)
    n_removed = int((~mask).sum())
    train_ids = train_ids[mask]
    train_scores = train_scores[mask]
    if n_removed:
        print(f"[dedup] removed {n_removed} train seqs found in eval libs")
    x_train = np.eye(4, dtype=np.float32)[train_ids]
    y_train = train_scores.reshape(-1, 1)
    library_sizes["random"] = len(train_ids)
    store("SSM", "random", y_train.ravel(),
          predict_ssm(x_train, delta_W, wt_activity=0.0))

    for label, cfg_key in SURROGATE_MAP.items():
        cfg = SURROGATE_CONFIGS[cfg_key]
        print(f"[train] {label.replace(chr(10), ' ')}  n={len(train_ids):,}")
        wrapper, model, test_df = train_surrogate(x_train, y_train, cfg)
        x_col = "x" if "x" in test_df.columns else "X"
        y_col = "y" if "y" in test_df.columns else next(
            c for c in test_df.columns if c.startswith("y"))
        y_test = np.asarray(test_df[y_col], dtype=float).ravel()
        yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
        store(label, "random", y_test, yhat_test)
        for name, ids in eval_ids.items():
            y_hat = predict_chunked(model, nuc_ids_to_str(ids))
            store(label, name, eval_scores[name], y_hat)
        print(f"        spearman rand={values['spearman'][label]['random']:+.3f}  "
              f"type2={values['spearman'][label]['type2']:+.3f}  "
              f"pw={values['spearman'][label]['pairwise']:+.3f}  "
              f"t3={values['spearman'][label]['type3']:+.3f}")
        print(f"        rmse     rand={values['rmse'][label]['random']:+.3f}  "
              f"type2={values['rmse'][label]['type2']:+.3f}  "
              f"pw={values['rmse'][label]['pairwise']:+.3f}  "
              f"t3={values['rmse'][label]['type3']:+.3f}")
        del wrapper, model
        tf.keras.backend.clear_session()

    print(f"[oracle] {oracle_label}  metrics=spearman,rmse")
    return values, library_sizes


def _annotate_bars(ax, bars, metric):
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        if metric == "rmse":
            offset = max(ax.get_ylim()[1] * 0.012, 0.01)
            y = h + offset
            va = "bottom"
        else:
            y = h + 0.018 if h >= 0 else h - 0.035
            va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{h:.2f}",
                ha="center", va=va, fontsize=7.2)


def _save_combined_plot(values, library_sizes, out_stem, title_prefix="Synthetic GT",
                        save_svg=False):
    labels = ["SSM"] + list(SURROGATE_MAP.keys())
    x = np.arange(len(labels))
    offsets = {
        "random": -1.5 * BAR_WIDTH,
        "type2": -0.5 * BAR_WIDTH,
        "pairwise": 0.5 * BAR_WIDTH,
        "type3": 1.5 * BAR_WIDTH,
    }

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 8.0), sharex=True)
    fig.subplots_adjust(top=0.86, bottom=0.12, hspace=0.18)

    legend_handles = None
    for ax, metric in zip(axes, ["spearman", "rmse"]):
        metric_values = values[metric]
        all_heights = []
        for key in EVAL_KEYS:
            heights = np.array([metric_values[label][key] for label in labels], dtype=float)
            all_heights.extend(heights[np.isfinite(heights)].tolist())
            bars = ax.bar(
                x + offsets[key],
                heights,
                BAR_WIDTH,
                color=EVAL_COLORS[key],
                label=f"{EVAL_LABELS[key]} (n={library_sizes[key]:,})",
            )
            _annotate_bars(ax, bars, metric)
            if legend_handles is None:
                legend_handles = bars

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.grid(True, axis="y", linestyle="--", alpha=0.22)
        ax.tick_params(axis="y", labelsize=9)
        if metric == "spearman":
            ax.set_ylabel("Spearman rho", fontsize=11)
            ymin = min(-0.15, min(all_heights) - 0.10)
            ax.set_ylim(ymin, 1.12)
            ax.set_title("Spearman", fontsize=11, pad=6)
        else:
            ymax = max(max(all_heights), 1e-6)
            ax.set_ylim(0, ymax * 1.22)
            ax.set_ylabel("RMSE", fontsize=11)
            ax.set_title("RMSE", fontsize=11, pad=6)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=11)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4,
               fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title_prefix, fontsize=13, y=0.985)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    msg = out
    if save_svg:
        fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
        msg = f"{out} / {out_stem}.svg"
    plt.close(fig)
    print(f"Saved → {msg}")


def _save_full_plot(values, library_sizes, out_stem, title_prefix="", metric="spearman", save_svg=False):
    labels = ["SSM"] + list(SURROGATE_MAP.keys())
    means_rand = np.array([values[label]["random"] for label in labels], dtype=float)
    means_t2 = np.array([values[label]["type2"] for label in labels], dtype=float)
    means_pw = np.array([values[label]["pairwise"] for label in labels], dtype=float)
    means_t3 = np.array([values[label]["type3"] for label in labels], dtype=float)
    stds_rand = np.zeros_like(means_rand)
    stds_t2 = np.zeros_like(means_t2)
    stds_pw = np.zeros_like(means_pw)
    stds_t3 = np.zeros_like(means_t3)

    print(f"\nFresh one-instance values (metric={metric}):")
    for label in labels:
        print(f"{label.replace(chr(10), ' '):30s}  "
              f"rand={values[label]['random']:+.3f}  "
              f"activity-balanced={values[label]['type2']:+.3f}  "
              f"pw={values[label]['pairwise']:+.3f}  "
              f"t3={values[label]['type3']:+.3f}")

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    fig.subplots_adjust(bottom=0.22)

    bars_rand = ax.bar(x - 1.5 * BAR_WIDTH, means_rand, BAR_WIDTH,
                     color=COLOR_RANDOM,
                     label=f"Random holdout (n={library_sizes['random']:,})",
                     yerr=stds_rand, capsize=4,
                     error_kw=dict(elinewidth=1.2, ecolor="black"))
    bars_t2 = ax.bar(x - 0.5 * BAR_WIDTH, means_t2, BAR_WIDTH,
                     color=COLOR_TYPE2,
                     label=f"Uniform eval (type 2) (n={library_sizes['type2']:,})",
                     yerr=stds_t2,  capsize=4,
                     error_kw=dict(elinewidth=1.2, ecolor="black"))
    bars_pw = ax.bar(x + 0.5 * BAR_WIDTH, means_pw, BAR_WIDTH,
                     color=COLOR_PAIRWISE,
                     label=f"Targeted pairwise (n={library_sizes['pairwise']:,})",
                     yerr=stds_pw, capsize=4,
                     error_kw=dict(elinewidth=1.2, ecolor="black"))
    bars_t3 = ax.bar(x + 1.5 * BAR_WIDTH, means_t3, BAR_WIDTH,
                     color=COLOR_TYPE3,
                     label=f"Saturated additive + pw (n={library_sizes['type3']:,})",
                     yerr=stds_t3, capsize=4,
                     error_kw=dict(elinewidth=1.2, ecolor="black"))

    for bars in [bars_rand, bars_t2, bars_pw, bars_t3]:
        _annotate(ax, bars)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("RMSE" if metric == "rmse" else "Spearman ρ", fontsize=11)
    if metric == "rmse":
        ymax = max(float(np.nanmax([means_rand, means_t2, means_pw, means_t3])), 1e-6)
        ax.set_ylim(0, ymax * 1.18)
    else:
        ax.set_ylim(-0.25, 1.12)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85)
    if title_prefix:
        ax.set_title(title_prefix, fontsize=11, pad=8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    msg = out
    if save_svg:
        fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
        msg = f"{out} / {out_stem}.svg"
    plt.close(fig)
    print(f"\nSaved → {msg}")


def _save_focused_plot(values, library_sizes, out_stem, title_prefix="", metric="spearman", save_svg=False):
    label = "NL Additive\n+Pairwise"
    keys = ["random", "type2", "pairwise", "type3"]
    means = np.array([values[label][key] for key in keys], dtype=float)
    colors = [COLOR_RANDOM, COLOR_TYPE2, COLOR_PAIRWISE, COLOR_TYPE3]
    x = np.arange(len(keys))

    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    fig.subplots_adjust(bottom=0.22)
    bars = ax.bar(x, means, width=0.58, color=colors)
    _annotate(ax, bars)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"Random holdout\n(n={library_sizes['random']:,})",
        f"Uniform eval\n(type 2)\n(n={library_sizes['type2']:,})",
        f"Targeted\npairwise\n(n={library_sizes['pairwise']:,})",
        f"Saturated additive\n+ pw\n(n={library_sizes['type3']:,})",
    ], fontsize=9)
    ax.set_ylabel("RMSE" if metric == "rmse" else "Spearman ρ", fontsize=11)
    if metric == "rmse":
        ymax = max(float(np.nanmax(means)), 1e-6)
        ax.set_ylim(0, ymax * 1.18)
    else:
        ax.set_ylim(-0.25, 1.12)
    ax.tick_params(axis="y", labelsize=9)
    metric_label = "RMSE" if metric == "rmse" else "Type3 evaluation breakdown"
    title = f"Nonlinear additive+pairwise surrogate\n{metric_label}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    ax.set_title(title,
                 fontsize=11, pad=8)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, out_stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    msg = out
    if save_svg:
        fig.savefig(out.replace(".png", ".svg"), bbox_inches="tight")
        msg = f"{out} / {out_stem}.svg"
    plt.close(fig)
    print(f"Saved → {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_stem", default="model_comparison_bar_type3",
                        help="Output filename stem for the combined chart")
    parser.add_argument("--focused_out_stem",
                        default="model_comparison_bar_type3_nonlinear_additive_pairwise",
                        help=argparse.SUPPRESS)
    parser.add_argument("--score_cache", default=None,
                        help="Optional precomputed oracle score cache from score_residualbind_type3.py")
    parser.add_argument("--oracle_label", default="Synthetic GT",
                        help="Label used in plot titles and logs")
    parser.add_argument("--metric", default="spearman",
                        choices=["spearman", "rmse"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()

    values, library_sizes = _fresh_metric_values(args.score_cache, args.oracle_label)
    _save_combined_plot(values, library_sizes, args.out_stem, args.oracle_label,
                        save_svg=args.svg)


if __name__ == "__main__":
    main()
