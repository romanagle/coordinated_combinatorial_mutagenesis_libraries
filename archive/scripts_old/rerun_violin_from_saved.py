"""rerun_violin_from_saved.py

Regenerate the Spearman ρ violin plot from previously saved sequence data,
applying a fixed uniformize_by_histogram (equal-width score bins, per_bin set
from target_n rather than the sparsest bin's count).

Skips library generation and GT scoring entirely — loads X_rand, GT scores,
X_pool, and pool GT scores from each seq_XXX directory, re-uniformizes the
pool GT scores to build new eval libraries, re-trains all surrogate models,
and recomputes ρ for both the random test split and the new eval library.

Usage:
    python scripts/rerun_violin_from_saved.py \\
        --save_dir   outputs/postabrcms/vts1high/spearman_violin_data \\
        --experiment RNCMPT00111 \\
        --out_path   outputs/postabrcms/vts1high/spearman_violin_run0.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.insert(0, '/home/nagle/final_version/squid-manuscript/squid')
sys.path.insert(0, '/home/nagle/final_version/squid-nn/squid')
sys.path.insert(0, '/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from ground_truth import uniformize_by_histogram

# ---------------------------------------------------------------------------
# Config  (must match spearman_violin.py)
# ---------------------------------------------------------------------------

GT_KEYS = [
    "additive",
    "additive_pairwise",
    "nonlin_additive",
    "nonlin_additive_pairwise",
]

SURROGATE_CONFIGS = {
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 12,
    },
    "pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "additive_GE": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 12, "hidden_nodes": 50,
    },
    "pairwise_GE": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.1, "hidden_nodes": 50,
    },
}

GT_LABELS = {
    "additive":                 "Additive GT",
    "additive_pairwise":        "Additive+Pair GT",
    "nonlin_additive":          "Nonlin Add GT",
    "nonlin_additive_pairwise": "Nonlin Add+Pair GT",
}

CFG_LABELS = {
    "additive":    "Additive",
    "pairwise":    "Pairwise",
    "additive_GE": "Add GE",
    "pairwise_GE": "Pair GE",
}

NUCS      = ['A', 'C', 'G', 'U']
_NUCS_ARR = np.array(list("ACGU"))

BLUE   = "#4C72B0"
ORANGE = "#DD8452"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_ARR[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _load_seq_dir(seq_dir):
    """Return (wt_oh, X_rand, gt_rand, X_pool, gt_pool) from a saved seq directory."""
    wt_oh = np.load(os.path.join(seq_dir, "sequence_onehot.npy")).astype(np.float32)

    nuc_ids_rand = np.load(os.path.join(seq_dir, "X_rand_nucids.npy"))
    X_rand = np.eye(4, dtype=np.float32)[nuc_ids_rand]

    nuc_ids_pool = np.load(os.path.join(seq_dir, "X_pool_nucids.npy"))
    X_pool = np.eye(4, dtype=np.float32)[nuc_ids_pool]

    gt_rand = dict(np.load(os.path.join(seq_dir, "gt_scores_rand.npz")))
    gt_pool = dict(np.load(os.path.join(seq_dir, "gt_scores_pool.npz")))

    return wt_oh, X_rand, gt_rand, X_pool, gt_pool


def _train_surrogate(X_rand, y_rand, cfg_name, cfg):
    N   = X_rand.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nl_pair = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs, patience = (1000, 50) if is_nl_pair else (500, 25)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X_rand.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X_rand, y_rand.reshape(-1, 1),
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    return model, test_df


def _predict_chunked(model, x_str, chunk=500):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(yhat, ytrue):
    m = np.isfinite(yhat) & np.isfinite(ytrue)
    return float(spearmanr(yhat[m], ytrue[m])[0]) if m.sum() >= 3 else np.nan


# ---------------------------------------------------------------------------
# Per-sequence re-run
# ---------------------------------------------------------------------------

def run_one_seq(seq_dir, s_idx, target_n, seed_offset):
    print(f"\n  [seq {s_idx}] loading saved data …", flush=True)
    wt_oh, X_rand, gt_rand, X_pool, gt_pool = _load_seq_dir(seq_dir)
    seq_str = "".join(_NUCS_ARR[np.argmax(wt_oh, axis=1)])
    print(f"  [seq {s_idx}] {seq_str}", flush=True)

    results = {}
    n_gt  = len(GT_KEYS)
    n_cfg = len(SURROGATE_CONFIGS)
    model_count = 0

    for gt_key in GT_KEYS:
        y_rand = np.asarray(gt_rand[gt_key], dtype=float).ravel()
        y_pool = np.asarray(gt_pool[gt_key], dtype=float).ravel()

        m = np.isfinite(y_rand)
        X_use, y_use = X_rand[m], y_rand[m]

        if y_use.size < 50 or np.nanstd(y_use) == 0:
            print(f"  [seq {s_idx}] SKIP {gt_key}: insufficient variance", flush=True)
            results.setdefault(gt_key, {})
            for cfg_name in SURROGATE_CONFIGS:
                results[gt_key][cfg_name] = (np.nan, np.nan)
            continue

        # Re-uniformize pool scores with fixed method
        seed_uni = seed_offset + abs(hash(gt_key)) % 10_000
        y_uni, keep_idx = uniformize_by_histogram(
            y_pool, X=None, n_bins=200, clip_hi=98,
            target_n=target_n, seed=seed_uni,
        )
        X_eval = X_pool[keep_idx]
        y_eval = y_uni.astype(float)
        X_eval_str = _to_str(X_eval)
        print(f"  [seq {s_idx}] {gt_key}: eval lib {len(y_eval):,} seqs "
              f"[{y_eval.min():.3f}, {y_eval.max():.3f}]", flush=True)

        results.setdefault(gt_key, {})
        for cfg_name, cfg in SURROGATE_CONFIGS.items():
            model_count += 1
            print(f"  [seq {s_idx}] model {model_count}/{n_gt*n_cfg}  "
                  f"GT={gt_key}  surrogate={cfg_name}", flush=True)
            model, test_df = _train_surrogate(X_use, y_use, cfg_name, cfg)

            x_test = np.asarray(test_df["x"])
            y_test = np.asarray(test_df["y"], dtype=float).ravel()

            try:
                rho_r = _rho(_predict_chunked(model, x_test), y_test)
            except Exception:
                rho_r = np.nan
            try:
                rho_e = _rho(_predict_chunked(model, X_eval_str), y_eval)
            except Exception:
                rho_e = np.nan

            results[gt_key][cfg_name] = (rho_r, rho_e)
            print(f"        ρ_rand={rho_r:.4f}  ρ_eval={rho_e:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_violin(accum, out_path, experiment="", n_seqs=10):
    subplot_order = [GT_KEYS[0], GT_KEYS[2], GT_KEYS[1], GT_KEYS[3]]
    cfg_names = list(SURROGATE_CONFIGS.keys())
    VIOLIN_W  = 0.42

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for sp_idx, gt_key in enumerate(subplot_order):
        ax = axes.flatten()[sp_idx]
        positions, data_lists, colors = [], [], []
        xtick_pos, xtick_labs = [], []

        x_pos = 0.0
        for cfg_name in cfg_names:
            rho_rnd = np.asarray(accum[gt_key][cfg_name]["rand"], dtype=float)
            rho_cur = np.asarray(accum[gt_key][cfg_name]["eval"], dtype=float)
            rho_rnd = rho_rnd[np.isfinite(rho_rnd)]
            rho_cur = rho_cur[np.isfinite(rho_cur)]
            positions.extend([x_pos, x_pos + 0.55])
            data_lists.extend([rho_rnd, rho_cur])
            colors.extend([BLUE, ORANGE])
            xtick_pos.append(x_pos + 0.275)
            xtick_labs.append(CFG_LABELS.get(cfg_name, cfg_name))
            x_pos += 1.4

        for pos, data, color in zip(positions, data_lists, colors):
            if data.size == 0:
                continue
            if data.size >= 3:
                parts = ax.violinplot(data, positions=[pos], widths=VIOLIN_W,
                                      showmedians=False, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.65)

            rng_j  = np.random.default_rng(42)
            jitter = rng_j.uniform(-0.08, 0.08, size=data.size)
            ax.scatter(np.full(data.size, pos) + jitter, data,
                       color=color, s=25, zorder=7, alpha=0.8, linewidths=0)

            if data.size >= 2:
                mean = np.nanmean(data)
                sem  = np.nanstd(data, ddof=1) / np.sqrt(data.size)
                ax.errorbar([pos], [mean], yerr=[[sem], [sem]], fmt="o",
                            color="black", markersize=5, linewidth=2.0,
                            capsize=5, capthick=2.0, zorder=10)
            elif data.size == 1:
                ax.scatter([pos], data, color="black", s=40, zorder=10)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labs, fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_title(GT_LABELS.get(gt_key, gt_key), fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if xtick_pos:
            ax.set_xlim(-0.5, x_pos - 0.15)

    fig.legend(
        handles=[
            Patch(facecolor=BLUE,   alpha=0.65, label="Random (MAVE test split)"),
            Patch(facecolor=ORANGE, alpha=0.65, label="Eval (curated library)"),
            plt.Line2D([0], [0], color="black", marker="o", linewidth=2,
                       markersize=5, label="Mean ± SEM"),
        ],
        loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0),
    )
    fig.suptitle(
        f"{experiment} — Spearman ρ  (N={n_seqs} random sequences, L=41)\n"
        "Left column: additive GT types   |   Right column: nonlinear GT types",
        fontsize=12,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] violin → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",
                        default="outputs/postabrcms/vts1high/spearman_violin_data")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--target_n",  type=int, default=20_000)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--out_path",
                        default="outputs/postabrcms/vts1high/spearman_violin_run0.png")
    args = parser.parse_args()

    seq_dirs = sorted(
        d for d in [
            os.path.join(args.save_dir, f"seq_{i:03d}") for i in range(100)
        ]
        if os.path.isdir(d)
    )
    print(f"Found {len(seq_dirs)} saved sequences in {args.save_dir}")

    accum = {
        gt_key: {cfg_name: {"rand": [], "eval": []} for cfg_name in SURROGATE_CONFIGS}
        for gt_key in GT_KEYS
    }

    for s_idx, seq_dir in enumerate(seq_dirs):
        seed_offset = args.seed + s_idx * 100_000 + 20_000
        res = run_one_seq(seq_dir, s_idx, args.target_n, seed_offset)
        for gt_key in GT_KEYS:
            for cfg_name in SURROGATE_CONFIGS:
                rho_r, rho_e = res.get(gt_key, {}).get(cfg_name, (np.nan, np.nan))
                accum[gt_key][cfg_name]["rand"].append(rho_r)
                accum[gt_key][cfg_name]["eval"].append(rho_e)

        print(f"\n  --- seq {s_idx} summary ---")
        for gt_key in GT_KEYS:
            row = "  ".join(
                f"{c}: r={accum[gt_key][c]['rand'][-1]:.3f}/e={accum[gt_key][c]['eval'][-1]:.3f}"
                for c in SURROGATE_CONFIGS
            )
            print(f"    {gt_key}: {row}")

    plot_violin(accum, args.out_path, experiment=args.experiment, n_seqs=len(seq_dirs))


if __name__ == "__main__":
    main()
