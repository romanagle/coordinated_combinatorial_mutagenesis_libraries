"""run_gt_configs.py

Run the Spearman ρ violin experiment across three GT initialization configs,
re-using the saved X_rand / X_pool sequences from spearman_violin.py so that
library generation is skipped entirely.

For each config, GT parameters are re-initialized from the saved wt_onehot,
X_rand and X_pool are re-scored, the pool is re-uniformized, and all 4
surrogate models are re-trained and evaluated.

GT configs
----------
signed_gaussian  : additive ~ N(0,0.5) signed; pairwise ~ dense all-pairs
                   N(0,0.3) signed (mutations can be beneficial)
absneg_adjacent  : additive ~ -|N(0,0.5)|; pairwise ~ adjacent-only (40 pairs)
                   N(0,0.2) signed
absneg_dense     : additive ~ -|N(0,0.5)|; pairwise ~ dense all-pairs
                   -|N(0,0.3)| (all-negative)

Usage:
    nohup python scripts/run_gt_configs.py \\
        --save_dir outputs/postabrcms/vts1high/spearman_violin_data \\
        --out_dir  outputs/postabrcms/vts1high/gt_configs \\
        --experiment RNCMPT00111 \\
        > /tmp/run_gt_configs.log 2>&1 &
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.insert(0, '/home/nagle/final_version/squid-manuscript/squid')
sys.path.insert(0, '/home/nagle/final_version/squid-nn/squid')
sys.path.insert(0, '/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from ground_truth import (
    init_additive_noWT, additive_affinity_noWT,
    init_pairwise_gaussian, init_pairwise_adjacent_noWT,
    init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
)

# ---------------------------------------------------------------------------
# GT config definitions
# ---------------------------------------------------------------------------

GT_CONFIGS = {
    "signed_gaussian": dict(
        additive=dict(sigma=0.5, l1_w=0.0, signed=True),
        pairwise=dict(kind="gaussian", sigma=0.3, signed=True),
        label="Signed Gaussian",
    ),
    "absneg_adjacent": dict(
        additive=dict(sigma=0.5, l1_w=0.1, signed=False),
        pairwise=dict(kind="adjacent", sigma_P=0.2),
        label="AbsNeg Add + Adjacent PW",
    ),
    "absneg_dense": dict(
        additive=dict(sigma=0.5, l1_w=0.1, signed=False),
        pairwise=dict(kind="gaussian", sigma=0.3, signed=False),
        label="AbsNeg Add + Dense AbsNeg PW",
    ),
}

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

CFG_LABELS = {
    "additive": "Additive", "pairwise": "Pairwise",
    "additive_GE": "Add GE",  "pairwise_GE": "Pair GE",
}

NUCS      = ['A', 'C', 'G', 'U']
_NUCS_ARR = np.array(list("ACGU"))
BLUE      = "#4C72B0"
ORANGE    = "#DD8452"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_str(X):
    return np.array(["".join(_NUCS_ARR[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=500):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(yhat, ytrue):
    m = np.isfinite(yhat) & np.isfinite(ytrue)
    return float(spearmanr(yhat[m], ytrue[m])[0]) if m.sum() >= 3 else np.nan


def _init_gt(rng, wt_oh, X_pool, gt_cfg, seed):
    """Initialise GT parameters and calibrate sigmoid nonlinearity."""
    add_kw = {k: v for k, v in gt_cfg["additive"].items()}
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_oh, **add_kw)

    pw = gt_cfg["pairwise"]
    pw_kind = pw["kind"]

    if pw_kind == "gaussian":
        edges, J = init_pairwise_gaussian(
            rng, wt_oh, sigma=pw["sigma"], signed=pw.get("signed", False)
        )
        pairwise_kwargs = dict(edges=edges, J=J)
    elif pw_kind == "adjacent":
        P_mut = init_pairwise_adjacent_noWT(rng, wt_oh, mut_map, sigma_P=pw["sigma_P"])
        pairwise_kwargs = dict(P_mut=P_mut)
    else:
        raise ValueError(f"Unknown pairwise kind: {pw_kind}")

    # Calibrate sigmoid nonlinearity on X_pool (50k seqs)
    s_add_ref = additive_affinity_noWT(X_pool, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kw = init_sigmoid_nonlin(s_add_ref)

    # Also compute _norm_std_addpair so the pairwise GT is correctly normalised
    from ground_truth import pairwise_potts_energy, pairwise_adjacent_noWT
    if "edges" in pairwise_kwargs:
        s_pair_ref = pairwise_potts_energy(X_pool, pairwise_kwargs["edges"],
                                           pairwise_kwargs["J"], b=0.0).reshape(-1)
    else:
        s_pair_ref = pairwise_adjacent_noWT(X_pool, pairwise_kwargs["P_mut"],
                                             mut_map, b=0.0).reshape(-1)
    s_addpair_ref = s_add_ref + s_pair_ref
    nonlin_kw["_norm_std_addpair"] = float(np.std(s_addpair_ref)) + 1e-8

    return W_mut, mut_map, b0, pairwise_kwargs, nonlin_kw


def _score_library(X, W_mut, mut_map, b0, pairwise_kwargs, nonlin_kw):
    return compute_gt_scores_for_library_potts(
        X, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kw,
        **pairwise_kwargs,
    )


def _train_surrogate(X, y, cfg_name, cfg):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nlp = (cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear")
    epochs, patience = (1000, 50) if is_nlp else (500, 25)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"], regression_type=cfg["regression_type"],
        linearity=cfg["linearity"], noise=cfg["noise"],
        noise_order=cfg["noise_order"], reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y.reshape(-1, 1),
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    return model, test_df

# ---------------------------------------------------------------------------
# Per-sequence per-config run
# ---------------------------------------------------------------------------

def run_one_seq(seq_dir, s_idx, gt_cfg_name, gt_cfg, target_n, seed_offset):
    wt_oh = np.load(os.path.join(seq_dir, "sequence_onehot.npy")).astype(np.float32)
    nuc_ids_rand = np.load(os.path.join(seq_dir, "X_rand_nucids.npy"))
    X_rand = np.eye(4, dtype=np.float32)[nuc_ids_rand]
    nuc_ids_pool = np.load(os.path.join(seq_dir, "X_pool_nucids.npy"))
    X_pool = np.eye(4, dtype=np.float32)[nuc_ids_pool]

    seq_str = "".join(_NUCS_ARR[np.argmax(wt_oh, axis=1)])
    print(f"  [seq {s_idx} | {gt_cfg_name}] {seq_str}", flush=True)

    rng = np.random.default_rng(seed_offset)
    W_mut, mut_map, b0, pw_kw, nonlin_kw = _init_gt(rng, wt_oh, X_pool, gt_cfg, seed_offset)

    gt_rand = _score_library(X_rand, W_mut, mut_map, b0, pw_kw, nonlin_kw)
    gt_pool = _score_library(X_pool, W_mut, mut_map, b0, pw_kw, nonlin_kw)

    results = {}
    n_total = len(GT_KEYS) * len(SURROGATE_CONFIGS)
    model_count = 0

    for gt_key in GT_KEYS:
        y_rand = np.asarray(gt_rand[gt_key], dtype=float).ravel()
        y_pool = np.asarray(gt_pool[gt_key], dtype=float).ravel()

        m = np.isfinite(y_rand)
        X_use, y_use = X_rand[m], y_rand[m]

        if y_use.size < 50 or np.nanstd(y_use) == 0:
            print(f"  SKIP {gt_key}: no variance", flush=True)
            results[gt_key] = {c: (np.nan, np.nan) for c in SURROGATE_CONFIGS}
            continue

        seed_uni = seed_offset + abs(hash(gt_key)) % 10_000
        y_uni, keep_idx = uniformize_by_histogram(
            y_pool, X=None, n_bins=200, clip_hi=98,
            target_n=target_n, seed=seed_uni,
        )
        X_eval     = X_pool[keep_idx]
        y_eval     = y_uni.astype(float)
        X_eval_str = _to_str(X_eval)
        print(f"  [seq {s_idx} | {gt_cfg_name}] {gt_key}: eval {len(y_eval):,} seqs "
              f"[{y_eval.min():.3f}, {y_eval.max():.3f}]", flush=True)

        results[gt_key] = {}
        for cfg_name, cfg in SURROGATE_CONFIGS.items():
            model_count += 1
            print(f"  [seq {s_idx} | {gt_cfg_name}] {model_count}/{n_total} "
                  f"GT={gt_key} surr={cfg_name}", flush=True)
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

def plot_violin(accum, out_path, title=""):
    subplot_order = [GT_KEYS[0], GT_KEYS[2], GT_KEYS[1], GT_KEYS[3]]
    cfg_names = list(SURROGATE_CONFIGS.keys())
    VIOLIN_W  = 0.42
    n_seqs    = len(next(iter(next(iter(accum.values())).values()))["rand"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for sp_idx, gt_key in enumerate(subplot_order):
        ax = axes.flatten()[sp_idx]
        x_pos = 0.0
        xtick_pos, xtick_labs = [], []

        for cfg_name in cfg_names:
            rho_rnd = np.asarray(accum[gt_key][cfg_name]["rand"], dtype=float)
            rho_cur = np.asarray(accum[gt_key][cfg_name]["eval"], dtype=float)
            rho_rnd = rho_rnd[np.isfinite(rho_rnd)]
            rho_cur = rho_cur[np.isfinite(rho_cur)]

            for pos, data, color in [(x_pos, rho_rnd, BLUE), (x_pos + 0.55, rho_cur, ORANGE)]:
                if data.size >= 3:
                    parts = ax.violinplot(data, positions=[pos], widths=VIOLIN_W,
                                          showmedians=False, showextrema=False)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(color); pc.set_alpha(0.65)
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=data.size)
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

            xtick_pos.append(x_pos + 0.275)
            xtick_labs.append(CFG_LABELS.get(cfg_name, cfg_name))
            x_pos += 1.4

        ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_labs, fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=10)
        ax.set_title(GT_LABELS.get(gt_key, gt_key), fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, linewidth=0.8, color="gray", linestyle="--", alpha=0.4)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.set_xlim(-0.5, x_pos - 0.15)

    fig.legend(handles=[
        Patch(facecolor=BLUE,   alpha=0.65, label="Random (MAVE test split)"),
        Patch(facecolor=ORANGE, alpha=0.65, label="Eval (curated library)"),
        plt.Line2D([0], [0], color="black", marker="o", linewidth=2,
                   markersize=5, label="Mean ± SEM"),
    ], loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",   default="outputs/postabrcms/vts1high/spearman_violin_data")
    parser.add_argument("--out_dir",    default="outputs/postabrcms/vts1high/gt_configs")
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--target_n",  type=int, default=20_000)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--configs",   nargs="+",
                        default=list(GT_CONFIGS.keys()),
                        help="Subset of configs to run")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seq_dirs = sorted(
        d for d in [os.path.join(args.save_dir, f"seq_{i:03d}") for i in range(100)]
        if os.path.isdir(d)
    )
    print(f"Found {len(seq_dirs)} sequences", flush=True)

    for gt_cfg_name in args.configs:
        if gt_cfg_name not in GT_CONFIGS:
            print(f"Unknown config {gt_cfg_name}, skipping", flush=True)
            continue

        gt_cfg = GT_CONFIGS[gt_cfg_name]
        print(f"\n{'='*65}", flush=True)
        print(f"  GT config: {gt_cfg_name}  ({gt_cfg['label']})", flush=True)
        print(f"{'='*65}", flush=True)

        accum = {
            gt_key: {c: {"rand": [], "eval": []} for c in SURROGATE_CONFIGS}
            for gt_key in GT_KEYS
        }

        for s_idx, seq_dir in enumerate(seq_dirs):
            seed_offset = args.seed + s_idx * 100_000 + abs(hash(gt_cfg_name)) % 1_000_000
            res = run_one_seq(seq_dir, s_idx, gt_cfg_name, gt_cfg, args.target_n, seed_offset)

            for gt_key in GT_KEYS:
                for cfg_name in SURROGATE_CONFIGS:
                    rho_r, rho_e = res.get(gt_key, {}).get(cfg_name, (np.nan, np.nan))
                    accum[gt_key][cfg_name]["rand"].append(rho_r)
                    accum[gt_key][cfg_name]["eval"].append(rho_e)

            # Checkpoint after each sequence
            ckpt = os.path.join(args.out_dir, f"{gt_cfg_name}_rho.json")
            with open(ckpt, "w") as f:
                def _ser(o):
                    if isinstance(o, dict):   return {k: _ser(v) for k, v in o.items()}
                    if isinstance(o, list):   return [_ser(x) for x in o]
                    if isinstance(o, float):  return None if (np.isnan(o) or np.isinf(o)) else o
                    if isinstance(o, (np.floating,)): v=float(o); return None if (np.isnan(v) or np.isinf(v)) else v
                    return o
                json.dump(_ser(accum), f, indent=2)

        n_seqs = len(seq_dirs)
        out_path = os.path.join(args.out_dir, f"{gt_cfg_name}_violin.png")
        plot_violin(
            accum, out_path,
            title=f"{args.experiment} — {gt_cfg['label']}\n"
                  f"Spearman ρ  (N={n_seqs} sequences, L=41)",
        )

    print("\nAll configs complete.", flush=True)


if __name__ == "__main__":
    main()
