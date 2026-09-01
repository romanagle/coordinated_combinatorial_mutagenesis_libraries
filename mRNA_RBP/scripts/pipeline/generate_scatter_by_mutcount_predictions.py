"""
mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py

Oracle-agnostic "scatter-by-mutation-count" prediction artifact generator.
Trains the nonlinear additive+pairwise surrogate once per mutation rate
(5/10/25%) directly from the standardized pipeline's own output tree
(``out_base/instance_00/...``, produced by generate_libraries.py) and saves
a reusable prediction artifact consumed by a scatter-by-mutcount plot script.

Generalizes generate_residualbind_vts1_scatter_predictions.py, which was
hardcoded to VTS1 and, worse, read from an already-*curated*
``ground_truth_collections/ResidualBind oracle VTS1/`` copy rather than the
live pipeline output -- i.e. it implicitly depended on a manual curation
step happening first. This version has no such dependency: it runs directly
against whatever `mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh` already produced.

Usage:
    python mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py --oracle hur --wt_activity high
    python mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py --oracle vts1 --wt_activity high
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-manuscript", "squid"))

from mRNA_RBP.scripts.pipeline.lib_size_spearman import (
    SURROGATE_CONFIGS,
    _exclude_eval_seqs,
    _rho,
    nuc_ids_to_str,
    predict_chunked,
    train_surrogate,
)
from mRNA_RBP.src.oracles import (
    MRNA_ORACLE,
    default_output_base,
    normalize_oracle_name,
    primary_gt_key,
)

MUT_RATES = (5, 10, 25)
MODEL_NAME = "nonlinear additive + pairwise"


def _r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() < 2:
        return np.nan
    return float(pearsonr(y[mask], yhat[mask])[0] ** 2)


def _rmse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2)))


def run_one(pct: int, inst_dir: str, score_key: str):
    rand_path = os.path.join(inst_dir, f"mut{pct:02d}", "lib_20000.npz")
    activity_path = os.path.join(inst_dir, "activity_balanced.npz")

    rand = np.load(rand_path)
    activity = np.load(activity_path)

    nuc_ids = rand["nuc_ids"].astype(np.uint8)
    y_all = rand[f"scores_{score_key}"].astype(float).reshape(-1, 1)
    activity_ids = activity["nuc_ids"].astype(np.uint8)

    keep_ids, n_removed = _exclude_eval_seqs(nuc_ids, activity_ids)
    if n_removed:
        eval_seqs = {row.tobytes() for row in activity_ids}
        keep_mask = np.array([row.tobytes() not in eval_seqs for row in nuc_ids], dtype=bool)
        y_all = y_all[keep_mask]
        nuc_ids = keep_ids

    x_train = np.eye(4, dtype=np.float32)[nuc_ids]
    cfg = SURROGATE_CONFIGS[MODEL_NAME]
    print(f"training scatter-by-mutcount surrogate mut{pct:02d}% (N={len(nuc_ids):,})")
    _, model, test_df = train_surrogate(x_train, y_all, cfg)

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
    y_rand = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_rand = predict_chunked(model, np.asarray(test_df[x_col]))

    y_activity = activity[f"scores_{score_key}"].astype(float)
    yhat_activity = predict_chunked(model, nuc_ids_to_str(activity_ids))
    rate_labels = activity["rate_labels"].astype(int)

    rho_rand = _rho(y_rand, yhat_rand)
    if np.isfinite(rho_rand) and rho_rand < 0:
        yhat_rand = -yhat_rand
        yhat_activity = -yhat_activity
        rho_rand = _rho(y_rand, yhat_rand)

    tf.keras.backend.clear_session()
    return {
        f"y_rand_{pct}": y_rand,
        f"yhat_rand_{pct}": yhat_rand,
        f"y_activity_{pct}": y_activity,
        f"yhat_activity_{pct}": yhat_activity,
        f"rate_labels_{pct}": rate_labels,
        f"rho_rand_{pct}": np.asarray(rho_rand, dtype=float),
        f"rho_activity_{pct}": np.asarray(_rho(y_activity, yhat_activity), dtype=float),
        f"r2_rand_{pct}": np.asarray(_r2(y_rand, yhat_rand), dtype=float),
        f"r2_activity_{pct}": np.asarray(_r2(y_activity, yhat_activity), dtype=float),
        f"rmse_rand_{pct}": np.asarray(_rmse(y_rand, yhat_rand), dtype=float),
        f"rmse_activity_{pct}": np.asarray(_rmse(y_activity, yhat_activity), dtype=float),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=MRNA_ORACLE,
                        choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1",
                                 "hur", "residualbind_hur", "qki", "residualbind_qki",
                                 "twister", "twister_ribozyme", "deepsquid_hur", "deepsquid_vts1",
                                 "mrna_negative_control", "negative_control"])
    parser.add_argument("--wt_activity", choices=["high", "low"], default="high",
                        help="Natural-probe WT sequence context (VTS1/HuR/QKI)")
    parser.add_argument("--out_base", default=None,
                        help="Library root. Defaults to outputs for synthetic GT or runs/<family>/<target>/<wt> for biological oracles")
    parser.add_argument("--out", default=None,
                        help="Output npz path. Defaults to <out_base>/scatter_by_mutcount_predictions.npz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    out_base = args.out_base or default_output_base(_HERE, oracle_name, args.wt_activity)
    inst_dir = os.path.join(out_base, "instance_00")
    score_key = primary_gt_key(oracle_name)
    out_path = args.out or os.path.join(out_base, "scatter_by_mutcount_predictions.npz")

    if os.path.exists(out_path) and not args.force:
        raise FileExistsError(f"{out_path} already exists; pass --force to overwrite")

    payload = {
        "mut_rates": np.asarray(MUT_RATES, dtype=np.int16),
        "model_name": np.asarray(MODEL_NAME),
        "score_key": np.asarray(f"scores_{score_key}"),
    }
    for pct in MUT_RATES:
        payload.update(run_one(pct, inst_dir, score_key))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved scatter prediction artifact -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
