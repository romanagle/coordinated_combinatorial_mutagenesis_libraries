"""
mRNA_RBP/train_surrogate_vts1_mut10.py

Trains the nonlinear additive+pairwise MAVE-NN surrogate on the
ResidualBind-scored fixed mut=10%/lib=20,000 VTS1 library (see
generate_vts1_mut10_library.py and score_residualbind_vts1_mut10.py), then
saves the trained model to disk so it can be reloaded as a standalone oracle
(SurrogateMAVENNOracle in oracles.py) without retraining. Mirrors
train_surrogate_varied_mutrate.py for the "other surrogate" comparison
condition in plot_coefficients_compare.py.

Run with the squid conda environment:
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/train_surrogate_vts1_mut10.py
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/train_surrogate_vts1_mut10.py --oracle vts1_residualbind

Reads:  mRNA_RBP/outputs_<oracle>/instance_00/vts1_mut10/lib_20000.npz
Writes: mRNA_RBP/outputs_<oracle>/surrogate_models/vts1_mut10_nonlin_pairwise/
            mavenn_model_pairwise.pickle / .h5   -- reloadable MAVE-NN model
            mave_df.csv                          -- train/val/test split dataframe
            meta.npz                             -- y_mean, y_std, score_key, seq, train_n, rho_test
        mRNA_RBP/outputs_<oracle>/surrogate_coefs/
            coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_<oracle>.npz
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.generate_varied_mutrate_library import VTS1_SEQ, VTS1_STEM_PAIRS
from mRNA_RBP.train_surrogate import (
    train_surrogate, extract_coefs, nuc_ids_to_str, predict_chunked, _rho,
)
from mRNA_RBP.oracles import (
    SurrogateMAVENNOracle, SURROGATE_SCORE_KEY, RESIDUALBIND_ORACLE,
    default_output_base, normalize_oracle_name, primary_gt_key,
)

TRAIN_N   = 20_000
SEED      = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=RESIDUALBIND_ORACLE,
                        choices=[RESIDUALBIND_ORACLE, "vts1_residualbind"])
    parser.add_argument("--train_n", type=int, default=TRAIN_N)
    parser.add_argument("--src_path", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--coef_path", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    src_score_key = primary_gt_key(oracle_name)
    out_base = default_output_base(_HERE, oracle_name)
    args.src_path = args.src_path or os.path.join(
        out_base, "instance_00", "vts1_mut10", "lib_20000.npz")
    args.model_dir = args.model_dir or os.path.join(
        out_base, "surrogate_models", "vts1_mut10_nonlin_pairwise")
    args.coef_path = args.coef_path or os.path.join(
        out_base, "surrogate_coefs",
        f"coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_{oracle_name}.npz")

    d = np.load(args.src_path)
    nuc_ids_all = d["nuc_ids"]
    y_all = d[f"scores_{src_score_key}"]
    print(f"Loaded pool: n={len(nuc_ids_all):,}")

    rng = np.random.default_rng(SEED)
    n_draw = min(args.train_n, len(nuc_ids_all))
    idx = rng.choice(len(nuc_ids_all), size=n_draw, replace=False)
    idx.sort()
    nuc_ids = nuc_ids_all[idx]
    y_raw = y_all[idx].astype(np.float32)
    print(f"Training subsample: n={len(nuc_ids):,}")

    X = np.eye(4, dtype=np.float32)[nuc_ids]
    y_mean = float(y_raw.mean())
    y_std  = float(y_raw.std())
    if y_std < 1e-8:
        y_std = 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    os.makedirs(args.model_dir, exist_ok=True)
    print("\nTraining nonlinear additive+pairwise surrogate...")
    wrapper, model, test_df = train_surrogate(
        X, y, save_dir=args.model_dir, epochs=args.epochs, patience=args.patience
    )

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
    rho_test = _rho(
        np.asarray(test_df[y_col], dtype=float).ravel(),
        predict_chunked(model, np.asarray(test_df[x_col])),
    )
    print(f"\nHeld-out test rho: {rho_test:+.3f}  (n_test={len(test_df):,})")

    coefs = extract_coefs(wrapper)
    coefs["y_mean"]   = np.float32(y_mean)
    coefs["y_std"]    = np.float32(y_std)
    coefs["rho_test"] = np.float32(rho_test)
    coefs["train_n"]  = np.int64(len(nuc_ids))
    os.makedirs(os.path.dirname(args.coef_path), exist_ok=True)
    np.savez_compressed(args.coef_path, **coefs)
    print(f"Saved coefs -> {args.coef_path}")

    np.savez_compressed(
        os.path.join(args.model_dir, "meta.npz"),
        y_mean=np.float32(y_mean), y_std=np.float32(y_std),
        score_key=np.array([SURROGATE_SCORE_KEY]), seq=np.array([VTS1_SEQ]),
        train_n=np.int64(len(nuc_ids)), rho_test=np.float32(rho_test),
    )
    print(f"Saved model -> {args.model_dir}")
    tf.keras.backend.clear_session()

    # ── Smoke test the reloaded oracle ───────────────────────────────────────
    print("\nReloading as SurrogateMAVENNOracle and verifying...")
    oracle = SurrogateMAVENNOracle(seq=VTS1_SEQ, stem_pairs=VTS1_STEM_PAIRS, model_dir=args.model_dir)
    wt_oh = oracle.wt_one_hot()
    wt_score = oracle(wt_oh[None, :, :])[0]
    print(f"  WT score (should be ~0): {wt_score:+.6f}")

    check_n = min(2000, len(nuc_ids))
    check_idx = rng.choice(len(nuc_ids), size=check_n, replace=False)
    x_check = X[check_idx]
    y_check_true = y_raw[check_idx]
    y_check_pred = oracle(x_check)
    rho_check = _rho(y_check_true, y_check_pred)
    print(f"  rho(surrogate vs ResidualBind) on {check_n} training-pool seqs: {rho_check:+.3f}")
    print(f"  score key: {oracle.score_key}")


if __name__ == "__main__":
    main()
