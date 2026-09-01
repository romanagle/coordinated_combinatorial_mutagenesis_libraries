"""Train one surrogate on activity-balanced or equal-rate mixed libraries."""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(_HERE))

from mRNA_RBP.scripts.pipeline.train_surrogate import (  # noqa: E402
    CFG_KEY, extract_coefs, nuc_ids_to_str, predict_chunked, train_surrogate,
)

RATES = (5, 10, 25)


def rho_abs(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(abs(spearmanr(y_true[mask], y_pred[mask])[0])) if mask.sum() >= 3 else float("nan")


def row_keys(rows):
    return {row.tobytes() for row in rows}


def materialize_mixed(inst_dir, gt_key, n_total=20_000):
    path = os.path.join(inst_dir, "mut_mixed", f"lib_{n_total}.npz")
    if os.path.isfile(path):
        return path
    counts = [n_total // 3 + (i < n_total % 3) for i in range(3)]
    pieces = []
    scores = []
    labels = []
    for rate, count in zip(RATES, counts):
        pool = np.load(os.path.join(inst_dir, f"mut{rate:02d}", "pool_2M.npz"))
        rng = np.random.default_rng(880_000 + rate + int(os.path.basename(inst_dir).split("_")[-1]) * 100)
        idx = rng.choice(len(pool["nuc_ids"]), count, replace=False)
        pieces.append(pool["nuc_ids"][idx])
        scores.append(pool[f"scores_{gt_key}"][idx])
        labels.append(np.full(count, rate, dtype=np.uint8))
    nuc_ids = np.concatenate(pieces)
    y = np.concatenate(scores)
    rate_labels = np.concatenate(labels)
    rng = np.random.default_rng(991_337)
    order = rng.permutation(len(nuc_ids))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, nuc_ids=nuc_ids[order], rate_labels=rate_labels[order],
                        **{f"scores_{gt_key}": y[order]})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_base", required=True)
    parser.add_argument("--gt_key", required=True)
    parser.add_argument("--design", required=True, choices=["activity_balanced", "mixed_equal_rates"])
    parser.add_argument("--n_instances", type=int, default=1)
    parser.add_argument("--train_n", type=int, default=20_000)
    parser.add_argument("--test_n", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    out_json = os.path.join(args.out_base, f"cross_mutrate_{args.design}_train_results.json")
    results = json.load(open(out_json)) if os.path.isfile(out_json) else {
        "model": CFG_KEY, "design": args.design, "gt_key": args.gt_key,
        "target_train_n": args.train_n, "test_rho": {str(r): [] for r in RATES},
        "test_n": {str(r): [] for r in RATES}, "instances": [],
    }

    for k in range(len(results["instances"]), args.n_instances):
        inst_dir = os.path.join(args.out_base, f"instance_{k:02d}")
        if args.design == "activity_balanced":
            train_path = os.path.join(inst_dir, "activity_balanced.npz")
        else:
            train_path = materialize_mixed(inst_dir, args.gt_key, args.train_n)
        d = np.load(train_path)
        train_ids = d["nuc_ids"]
        y_raw = d[f"scores_{args.gt_key}"].astype(np.float32)
        X = np.eye(4, dtype=np.float32)[train_ids]
        y_std = float(y_raw.std()) or 1.0
        y = ((y_raw - float(y_raw.mean())) / y_std).reshape(-1, 1)
        print(f"[train] {args.design} instance {k:02d}, n={len(train_ids):,}", flush=True)
        wrapper, model, test_df = train_surrogate(X, y, epochs=args.epochs, patience=args.patience)
        x_col = "x" if "x" in test_df.columns else "X"
        y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
        heldout_rho = rho_abs(np.asarray(test_df[y_col], float).ravel(),
                              predict_chunked(model, np.asarray(test_df[x_col])))

        used = row_keys(train_ids)
        instance_rho = {}
        instance_n = {}
        for rate in RATES:
            pool = np.load(os.path.join(inst_dir, f"mut{rate:02d}", "pool_2M.npz"))
            eligible = np.array([i for i, row in enumerate(pool["nuc_ids"]) if row.tobytes() not in used])
            rng = np.random.default_rng(731_000 + k * 100 + rate)
            idx = rng.choice(eligible, min(args.test_n, len(eligible)), replace=False)
            pred = predict_chunked(model, nuc_ids_to_str(pool["nuc_ids"][idx]))
            value = rho_abs(pool[f"scores_{args.gt_key}"][idx].astype(float), pred)
            results["test_rho"][str(rate)].append(value)
            results["test_n"][str(rate)].append(len(idx))
            instance_rho[str(rate)] = value
            instance_n[str(rate)] = len(idx)
            print(f"  test {rate:02d}%: |rho|={value:.4f}, unseen n={len(idx):,}", flush=True)

        coef_dir = os.path.join(args.out_base, f"surrogate_coefs_{args.design}")
        os.makedirs(coef_dir, exist_ok=True)
        np.savez_compressed(os.path.join(coef_dir, f"coefs_k{k:02d}_{args.design}_{args.gt_key}.npz"),
                            **extract_coefs(wrapper))
        results["instances"].append({"k": k, "n_train": len(train_ids),
                                     "heldout_rho": heldout_rho,
                                     "rho": instance_rho, "test_n": instance_n})
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        tf.keras.backend.clear_session()
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
