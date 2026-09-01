"""
Generate ResidualBind VTS1 result figures from a freshly completed
libraries_used_for_figures/lib_size_spearman_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy.stats import spearmanr

from mRNA_RBP.src.evaluate import make_ssm_deltas_from_scores


REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from mRNA_RBP.scripts.figures.core.provenance import stamp_figure  # noqa: E402
COLLECTION_DIR = (
    REPO
    / "mRNA_RBP"
    / "outputs"
    / "ground_truth_collections"
    / "ResidualBind oracle VTS1"
)
FIG_DIR = COLLECTION_DIR / "figures"
LIB_DIR = COLLECTION_DIR / "libraries_used_for_figures"
JSON_PATH = LIB_DIR / "lib_size_spearman_results.json"
ORACLE_KEY = "vts1_residualbind"
SCORE_KEY = "scores_vts1_residualbind"
LIB_SIZES = (200, 2000, 20000)
MUT_RATE_PLOT = 10

SPLITS = {
    "rand": ("-", "o", "#4C72B0", "random holdout"),
    "activity_balanced": ("-", "P", "#DD8452", "activity-balanced"),
    "pairwise": (":", "D", "#8172B2", "targeted pairwise"),
    "type3": ("-.", "s", "#55A868", "type3 exhaustive"),
}
MODEL_LABELS = {
    "additive": "Additive",
    "additive + pairwise": "Additive\n+Pairwise",
    "nonlinear additive": "NL Additive",
    "nonlinear additive + pairwise": "NL Additive\n+Pairwise",
}
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3}


def _as_float(value):
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.nanmean(arr)) if len(arr) else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _surrogate(cache):
    return cache.get("surrogate", {}).get(ORACLE_KEY, {})


def _rho(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(y_true[mask], y_hat[mask])[0])


def _wt_ids(seq: str) -> np.ndarray:
    return np.asarray([NUC_TO_IDX[c] for c in seq.strip().upper()], dtype=np.uint8)


def _ssm_deltas():
    seq = (LIB_DIR / "wt_seq.txt").read_text().strip()
    wt = _wt_ids(seq)
    d = np.load(LIB_DIR / "ssm.npz")
    scores = d[SCORE_KEY].astype(float)
    wt_key = SCORE_KEY.replace("scores_", "wt_score_", 1)
    wt_activity = float(d[wt_key][0]) if wt_key in d.files else 0.0
    return make_ssm_deltas_from_scores(
        d["nuc_ids"], scores, wt_idx=wt, wt_activity=wt_activity
    ), wt_activity


def _predict_ssm(nuc_ids: np.ndarray, delta: np.ndarray, wt_activity: float) -> np.ndarray:
    x = np.eye(4, dtype=np.float32)[nuc_ids.astype(np.uint8)]
    return np.einsum("nla,al->n", x.astype(float), delta.astype(float)) + wt_activity


def _ssm_model_values():
    delta, wt_activity = _ssm_deltas()
    paths = {
        "rand": LIB_DIR / "random_libraries" / "mut10_lib_20000.npz",
        "activity_balanced": LIB_DIR / "activity_balanced.npz",
        "pairwise": LIB_DIR / "pairwise_lib.npz",
        "type3": LIB_DIR / "type3.npz",
    }
    values = {}
    library_sizes = {}
    for key, path in paths.items():
        d = np.load(path)
        y_true = d[SCORE_KEY].astype(float)
        y_hat = _predict_ssm(d["nuc_ids"], delta, wt_activity)
        rho = _rho(y_true, y_hat)
        if key == "pairwise" and not np.isfinite(rho):
            rho = 0.0
        values[key] = rho
        library_sizes[key] = len(y_true)
    return values, library_sizes


def save_rho_vs_libsize(cache):
    sur = _surrogate(cache)
    cfg = "nonlinear additive + pairwise"
    if cfg not in sur:
        cfg = "nonlinear additive"

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.asarray(LIB_SIZES, dtype=float)
    mut = str(MUT_RATE_PLOT)
    for split, (ls, marker, color, label) in SPLITS.items():
        means = []
        for size in LIB_SIZES:
            vals = sur.get(cfg, {}).get(mut, {}).get(str(size), {}).get(split, np.nan)
            means.append(_as_float(vals))
        means = np.asarray(means, dtype=float)
        valid = np.isfinite(means)
        if valid.any():
            ax.plot(
                x[valid],
                means[valid],
                color=color,
                ls=ls,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                label=label,
            )

    ax.set_xscale("log")
    ax.set_xticks(LIB_SIZES)
    ax.set_xticklabels(["200", "2K", "20K"], fontsize=9)
    ax.set_xlabel("Training library size", fontsize=10)
    ax.set_ylabel("Spearman rho", fontsize=10)
    ax.set_ylim(-0.2, 1.08)
    ax.axhline(0, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.grid(True, axis="y", ls="--", alpha=0.25)
    ax.set_title(f"{MODEL_LABELS.get(cfg, cfg).replace(chr(10), ' + ')} | mut rate = {MUT_RATE_PLOT}%", fontsize=10)
    handles = [
        mlines.Line2D([], [], color=color, ls=ls, marker=marker, markersize=5, linewidth=1.8, label=label)
        for _, (ls, marker, color, label) in SPLITS.items()
    ]
    ax.legend(handles=handles, fontsize=9, framealpha=0.9)
    stamp_figure(
        fig,
        library_status="fresh",
        source_paths=[
            JSON_PATH,
            LIB_DIR / "random_libraries" / "mut10_lib_20000.npz",
            LIB_DIR / "activity_balanced.npz",
            LIB_DIR / "pairwise_lib.npz",
            LIB_DIR / "type3.npz",
        ],
    )
    libsize_dir = FIG_DIR / "library_size_sweep"
    libsize_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(libsize_dir / "rho_vs_libsize_type3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison(cache):
    sur = _surrogate(cache)
    mut = str(MUT_RATE_PLOT)
    size = "20000"
    models = ["SSM"] + [m for m in MODEL_LABELS if m in sur]
    eval_keys = ["rand", "activity_balanced", "pairwise", "type3"]
    eval_labels = {
        "rand": "Random holdout",
        "activity_balanced": "Activity-balanced",
        "pairwise": "Targeted pairwise",
        "type3": "Type3 exhaustive",
    }
    eval_colors = {
        "rand": "#8C8C8C",
        "activity_balanced": "#4C72B0",
        "pairwise": "#DD8452",
        "type3": "#55A868",
    }
    ssm_values, library_sizes = _ssm_model_values()

    x = np.arange(len(models), dtype=float)
    width = 0.18
    offsets = (np.arange(len(eval_keys)) - (len(eval_keys) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(11.5, 4.7))
    fig.subplots_adjust(bottom=0.22)
    for offset, key in zip(offsets, eval_keys):
        vals = []
        for model in models:
            if model == "SSM":
                val = ssm_values.get(key, np.nan)
            else:
                val = sur.get(model, {}).get(mut, {}).get(size, {}).get(key, np.nan)
            vals.append(_as_float(val))
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=eval_colors[key],
            label=f"{eval_labels[key]} (n={library_sizes[key]:,})",
            alpha=0.9,
        )
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                y = val + 0.018 if val >= 0 else val - 0.035
                va = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{val:.2f}", ha="center", va=va, fontsize=7.2)

    ax.set_xticks(x)
    ax.set_xticklabels(["SSM"] + [MODEL_LABELS[m] for m in models if m != "SSM"], fontsize=12)
    ax.set_ylabel("Spearman rho", fontsize=11)
    ax.set_ylim(-0.25, 1.12)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(True, axis="y", linestyle="--", alpha=0.22)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_title(f"ResidualBind VTS1 | mut {MUT_RATE_PLOT}% | n=20K", fontsize=11, pad=8)
    stamp_figure(
        fig,
        library_status="fresh",
        source_paths=[
            JSON_PATH,
            LIB_DIR / "ssm.npz",
            LIB_DIR / "random_libraries" / "mut10_lib_20000.npz",
            LIB_DIR / "activity_balanced.npz",
            LIB_DIR / "pairwise_lib.npz",
            LIB_DIR / "type3.npz",
        ],
    )
    model_comparison_dir = FIG_DIR / "model_comparison"
    model_comparison_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(model_comparison_dir / "model_comparison_bar_type3.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with JSON_PATH.open() as fh:
        cache = json.load(fh)
    save_rho_vs_libsize(cache)
    save_model_comparison(cache)
    print(f"Saved ResidualBind VTS1 result figures -> {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
