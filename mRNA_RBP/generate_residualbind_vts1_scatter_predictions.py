"""
Train once for the ResidualBind VTS1 scatter-by-mutation-count figure.

This script writes a reusable prediction artifact consumed by
`mRNA_RBP/plots/plot_residualbind_vts1_scatter_by_mutcount.py`. Plotting must
not retrain models; rerun this script only when the source libraries or model
configuration intentionally change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "squid-nn"))
sys.path.insert(0, str(REPO / "squid-manuscript" / "squid"))

from mRNA_RBP.lib_size_spearman import (  # noqa: E402
    SURROGATE_CONFIGS,
    _exclude_eval_seqs,
    _rho,
    nuc_ids_to_str,
    predict_chunked,
    train_surrogate,
)


COLLECTION_DIR = (
    REPO
    / "mRNA_RBP"
    / "outputs"
    / "ground_truth_collections"
    / "ResidualBind oracle VTS1"
)
LIB_DIR = COLLECTION_DIR / "libraries_used_for_figures"
DEFAULT_OUT = LIB_DIR / "scatter_by_mutcount_predictions.npz"
SCORE_KEY = "scores_vts1_residualbind"
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


def run_one(pct: int):
    rand_path = LIB_DIR / "random_libraries" / f"mut{pct:02d}_lib_20000.npz"
    activity_path = LIB_DIR / "activity_balanced.npz"

    rand = np.load(rand_path)
    activity = np.load(activity_path)

    nuc_ids = rand["nuc_ids"].astype(np.uint8)
    y_all = rand[SCORE_KEY].astype(float).reshape(-1, 1)
    activity_ids = activity["nuc_ids"].astype(np.uint8)

    keep_ids, n_removed = _exclude_eval_seqs(nuc_ids, activity_ids)
    if n_removed:
        eval_seqs = {row.tobytes() for row in activity_ids}
        keep_mask = np.array([row.tobytes() not in eval_seqs for row in nuc_ids], dtype=bool)
        y_all = y_all[keep_mask]
        nuc_ids = keep_ids

    x_train = np.eye(4, dtype=np.float32)[nuc_ids]
    cfg = SURROGATE_CONFIGS[MODEL_NAME]
    print(f"training ResidualBind VTS1 scatter model mut{pct:02d}% (N={len(nuc_ids):,})")
    _, model, test_df = train_surrogate(x_train, y_all, cfg)

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
    y_rand = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_rand = predict_chunked(model, np.asarray(test_df[x_col]))

    y_activity = activity[SCORE_KEY].astype(float)
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} already exists; pass --force to overwrite")

    payload = {
        "mut_rates": np.asarray(MUT_RATES, dtype=np.int16),
        "model_name": np.asarray(MODEL_NAME),
        "score_key": np.asarray(SCORE_KEY),
    }
    for pct in MUT_RATES:
        payload.update(run_one(pct))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Saved scatter prediction artifact -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
