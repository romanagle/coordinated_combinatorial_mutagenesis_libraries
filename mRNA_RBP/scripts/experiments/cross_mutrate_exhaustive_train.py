"""Train on exhaustive singles+doubles and test across mutation rates.

This is the saturated-training counterpart to ``cross_mutrate_eval.py``.
The nonlinear additive + pairwise surrogate is trained on ``type3.npz``
(all 1- and 2-mutant sequences), then evaluated on deterministic 4,000-row
samples from the 5%, 10%, and 25% mutation-rate pools.  Results are resumable
and are plotted as a fourth row under the existing random-trained 3x3 matrix.

Run in the squid environment, for example::

    python mRNA_RBP/scripts/experiments/cross_mutrate_exhaustive_train.py \
      --out_base mRNA_RBP/outputs --gt_key nonlin_additive_pairwise \
      --random_cross_json mRNA_RBP/outputs/cross_mutrate_results.json \
      --n_instances 10
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(_HERE))

from mRNA_RBP.scripts.pipeline.train_surrogate import (  # noqa: E402
    CFG_KEY, extract_coefs, nuc_ids_to_str, predict_chunked, train_surrogate,
)

RATES = (5, 10, 25)
RANDOM_LIBRARY_SIZE = "20000"


def rho_abs(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return float("nan")
    return float(abs(spearmanr(y_true[mask], y_pred[mask])[0]))


def deterministic_test(pool_path, gt_key, n, seed):
    pool = np.load(pool_path)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool["nuc_ids"]), size=min(n, len(pool["nuc_ids"])), replace=False)
    return pool["nuc_ids"][idx], pool[f"scores_{gt_key}"][idx].astype(float)


def random_matrix(path, gt_key):
    block = json.load(open(path))["cross"][gt_key][CFG_KEY]
    matrix = np.full((3, 3), np.nan)
    for row, train_rate in enumerate(RATES):
        entry = block[str(train_rate)][RANDOM_LIBRARY_SIZE]
        matrix[row, row] = np.nanmean(entry["same_rate"])
        for col, test_rate in enumerate(RATES):
            if row != col:
                matrix[row, col] = np.nanmean(entry["cross"][str(test_rate)])
    return matrix


def plot_comparison(random_cross_json, results, gt_key, out_png):
    random = random_matrix(random_cross_json, gt_key)
    exhaustive = np.array([
        np.nanmean(results["test_rho"][str(rate)]) for rate in RATES
    ])[None, :]
    matrix = np.vstack([random, exhaustive])

    fig, ax = plt.subplots(figsize=(5.7, 5.6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for row in range(4):
        for col in range(3):
            value = matrix[row, col]
            color = "white" if value < 0.6 else "#111827"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color)
    ax.axhline(2.5, color="white", linewidth=4)
    ax.set_xticks(range(3), [f"{r}%" for r in RATES])
    exhaustive_label = ("all doubles" if results.get("mutation_orders") == [2]
                        else "all singles\n+ all doubles")
    ax.set_yticks(range(4), ["5%", "10%", "25%", exhaustive_label])
    ax.set_xlabel("Test mutation rate")
    ax.set_ylabel("Training library")
    ax.set_title("Cross-mutation-rate generalization\nNonlinear additive + pairwise")
    fig.colorbar(im, ax=ax, label="Spearman |ρ|")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_base", required=True)
    parser.add_argument("--gt_key", required=True)
    parser.add_argument("--random_cross_json", required=True)
    parser.add_argument("--n_instances", type=int, default=1)
    parser.add_argument("--test_n", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--mutation_orders", nargs="+", type=int, default=[1, 2],
                        choices=[1, 2], help="Mutation orders retained from type3.npz")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_png", default=None)
    args = parser.parse_args()

    orders = sorted(set(args.mutation_orders))
    order_tag = "double_only" if orders == [2] else "singles_doubles"

    out_json = args.out_json or os.path.join(
        args.out_base, ("cross_mutrate_exhaustive_train_results.json"
                        if orders == [1, 2]
                        else f"cross_mutrate_exhaustive_train_{order_tag}_results.json"))
    out_png = args.out_png or os.path.join(
        args.out_base, ("cross_mutrate_exhaustive_train_heatmap.png"
                        if orders == [1, 2]
                        else f"cross_mutrate_exhaustive_train_{order_tag}_heatmap.png"))
    results = json.load(open(out_json)) if os.path.isfile(out_json) else {
        "model": CFG_KEY,
        "training_library": ("exhaustive doubles only" if orders == [2]
                             else "exhaustive singles and doubles"),
        "mutation_orders": orders,
        "gt_key": args.gt_key,
        "test_n": args.test_n,
        "test_rho": {str(rate): [] for rate in RATES},
        "instances": [],
    }

    for k in range(len(results["instances"]), args.n_instances):
        inst_dir = os.path.join(args.out_base, f"instance_{k:02d}")
        type3_path = os.path.join(inst_dir, "type3.npz")
        if not os.path.isfile(type3_path):
            print(f"[stop] missing {type3_path}")
            break
        train = np.load(type3_path)
        keep = np.isin(train["rate_labels"], orders)
        nuc_ids = train["nuc_ids"][keep]
        X = np.eye(4, dtype=np.float32)[nuc_ids]
        y_raw = train[f"scores_{args.gt_key}"][keep].astype(np.float32)
        y_std = float(y_raw.std()) or 1.0
        y = ((y_raw - float(y_raw.mean())) / y_std).reshape(-1, 1)
        print(f"[train] instance {k:02d}, n={len(nuc_ids):,}", flush=True)
        wrapper, model, _ = train_surrogate(
            X, y, epochs=args.epochs, patience=args.patience)

        instance_rho = {}
        for rate in RATES:
            pool_path = os.path.join(inst_dir, f"mut{rate:02d}", "pool_2M.npz")
            test_ids, y_true = deterministic_test(
                pool_path, args.gt_key, args.test_n, seed=731_000 + k * 100 + rate)
            pred = predict_chunked(model, nuc_ids_to_str(test_ids))
            value = rho_abs(y_true, pred)
            results["test_rho"][str(rate)].append(value)
            instance_rho[str(rate)] = value
            print(f"  test {rate:02d}%: |rho|={value:.4f}", flush=True)

        coef_dir = os.path.join(args.out_base, f"surrogate_coefs_exhaustive_{order_tag}")
        os.makedirs(coef_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(coef_dir, f"coefs_k{k:02d}_{order_tag}_{args.gt_key}.npz"),
            **extract_coefs(wrapper),
        )
        results["instances"].append({"k": k, "n_train": len(nuc_ids), "rho": instance_rho})
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        tf.keras.backend.clear_session()

    plot_comparison(args.random_cross_json, results, args.gt_key, out_png)
    print(f"[done] {out_json}\n[figure] {out_png}")


if __name__ == "__main__":
    main()
