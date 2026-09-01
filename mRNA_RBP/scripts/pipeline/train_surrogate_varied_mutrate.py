"""
mRNA_RBP/scripts/pipeline/train_surrogate_varied_mutrate.py

Layer-2 "deep squid" trainer. Trains a nonlinear additive+pairwise MAVE-NN
surrogate on the varied-mutation-rate pool (mutation counts 3-10; see
generate_varied_mutrate_library.py), labeled by the real ResidualBind oracle
scores already baked into that pool, then saves the trained model to disk so
it can be reloaded as a standalone stand-in oracle (SurrogateMAVENNOracle in
oracles.py) without retraining -- this saved model *is* "deep squid": a
simpler, fully-interpretable surrogate distilled from the real (black-box)
ResidualBind ensemble.

Each real oracle (MSI1, VTS1, HuR, QKI, ...) gets its own distinct,
independently-trained deep-squid surrogate (different oracle, different
sequence/structure) -- saved score_key is "deepsquid_<short_name>"
(see oracles.deepsquid_score_key) so none are ever conflated.

Run with the squid conda environment (needs mavenn/tensorflow, and a
torch install able to load the real oracle -- see the LD_LIBRARY_PATH note
in oracles.py's ResidualBindEnsembleOracle):
    python mRNA_RBP/scripts/pipeline/train_surrogate_varied_mutrate.py --oracle residualbind_ensemble
    python mRNA_RBP/scripts/pipeline/train_surrogate_varied_mutrate.py --oracle vts1_residualbind

Reads:  <out_base>/instance_00/varied_mutrate/pool.npz
Writes: <out_base>/surrogate_models/varied_mutrate_nonlin_pairwise/
            mavenn_model_pairwise.pickle / .h5   -- reloadable MAVE-NN model
            mave_df.csv                          -- train/val/test split dataframe
            meta.npz                             -- y_mean, y_std, score_key, seq, train_n, rho_test
        <out_base>/surrogate_coefs/
            coefs_varied_mutrate_nonlinear_additive_p_pairwise_deepsquid_<rbp>.npz
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "squid-manuscript", "squid"))

from mRNA_RBP.scripts.pipeline.train_surrogate import (
    train_surrogate, extract_coefs, nuc_ids_to_str, predict_chunked, _rho,
)
from mRNA_RBP.src.oracles import (
    SurrogateMAVENNOracle,
    default_output_base, normalize_oracle_name, primary_gt_key,
    deepsquid_score_key, sequence_config_for_oracle,
)

TRAIN_N = 200_000
SEED    = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True,
                        help="residualbind_ensemble/msi1 (source real oracle) "
                             "or vts1_residualbind/vts1")
    parser.add_argument("--train_n", type=int, default=TRAIN_N)
    parser.add_argument("--src_path", default=None)
    parser.add_argument("--out_base", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--coef_path", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--wt_activity", choices=["high", "low"], default="high",
                        help="Natural-probe WT sequence context (VTS1/HuR/QKI)")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    seq, stem_pairs, _motif_positions = sequence_config_for_oracle(oracle_name, args.wt_activity)
    src_score_key = primary_gt_key(oracle_name)
    deepsquid_key = deepsquid_score_key(oracle_name)

    out_base = args.out_base or default_output_base(_HERE, oracle_name, args.wt_activity)
    src_path = args.src_path or os.path.join(
        out_base, "instance_00", "varied_mutrate", "pool.npz")
    model_dir = args.model_dir or os.path.join(
        out_base, "surrogate_models", "varied_mutrate_nonlin_pairwise")
    coef_path = args.coef_path or os.path.join(
        out_base, "surrogate_coefs",
        f"coefs_varied_mutrate_nonlinear_additive_p_pairwise_{deepsquid_key}.npz")

    d = np.load(src_path)
    nuc_ids_all = d["nuc_ids"]
    mut_counts_all = d["mut_counts"] if "mut_counts" in d.files else None
    y_all = d[f"scores_{src_score_key}"]
    print(f"Loaded pool: n={len(nuc_ids_all):,}  (labels: {src_score_key})")

    rng = np.random.default_rng(SEED)
    n_draw = min(args.train_n, len(nuc_ids_all))
    idx = rng.choice(len(nuc_ids_all), size=n_draw, replace=False)
    idx.sort()
    nuc_ids = nuc_ids_all[idx]
    mut_counts = mut_counts_all[idx] if mut_counts_all is not None else None
    y_raw = y_all[idx].astype(np.float32)
    print(f"Training subsample: n={len(nuc_ids):,}")
    if mut_counts is not None:
        vals, cnts = np.unique(mut_counts, return_counts=True)
        print("  mut_count distribution: " + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnts)))

    X = np.eye(4, dtype=np.float32)[nuc_ids]
    y_mean = float(y_raw.mean())
    y_std  = float(y_raw.std())
    if y_std < 1e-8:
        y_std = 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    os.makedirs(model_dir, exist_ok=True)
    print(f"\nTraining nonlinear additive+pairwise surrogate ({deepsquid_key})...", flush=True)
    wrapper, model, test_df = train_surrogate(
        X, y, save_dir=model_dir, epochs=args.epochs, patience=args.patience
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
    os.makedirs(os.path.dirname(coef_path), exist_ok=True)
    np.savez_compressed(coef_path, **coefs)
    print(f"Saved coefs -> {coef_path}")

    np.savez_compressed(
        os.path.join(model_dir, "meta.npz"),
        y_mean=np.float32(y_mean), y_std=np.float32(y_std),
        score_key=np.array([deepsquid_key]), seq=np.array([seq]),
        train_n=np.int64(len(nuc_ids)), rho_test=np.float32(rho_test),
    )
    print(f"Saved model -> {model_dir}")
    tf.keras.backend.clear_session()

    # Smoke test the reloaded oracle
    print("\nReloading as SurrogateMAVENNOracle and verifying...")
    oracle = SurrogateMAVENNOracle(seq=seq, stem_pairs=stem_pairs, model_dir=model_dir)
    wt_oh = oracle.wt_one_hot()
    wt_score = oracle(wt_oh[None, :, :])[0]
    print(f"  WT score (should be ~0): {wt_score:+.6f}")

    check_n = min(2000, len(nuc_ids))
    check_idx = rng.choice(len(nuc_ids), size=check_n, replace=False)
    x_check = X[check_idx]
    y_check_true = y_raw[check_idx]
    y_check_pred = oracle(x_check)
    rho_check = _rho(y_check_true, y_check_pred)
    print(f"  rho(deep-squid vs real oracle) on {check_n} training-pool seqs: {rho_check:+.3f}")
    print(f"  score key: {oracle.score_key}")


if __name__ == "__main__":
    main()
