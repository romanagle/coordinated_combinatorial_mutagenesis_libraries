"""
mRNA_RBP/train_twister_deepsquid.py

"Deep squid" trainer for the Twister ribozyme, adapted from
train_surrogate_varied_mutrate.py. For MSI1/VTS1/HuR/QKI, deep squid is a
*distillation* of a live black-box ResidualBind ensemble (which can already
score arbitrary sequences on its own). Twister has no such live oracle -- the
real data (Kobori & Yokobayashi 2016) is a finite, fixed table of WT + all
144 single + 10,152 double mutants (see parse_twister_data.py), with no way
to score anything outside that set. So here, deep squid is trained directly
on the real measurements and *is* the only scorable oracle: a nonlinear
additive+pairwise MAVE-NN surrogate that smooths/interpolates the real
landscape into something `generate_libraries.py`/`lib_size_spearman.py` can
draw arbitrary libraries from, exactly like any other oracle in this
pipeline (via oracles.SurrogateMAVENNOracle).

Unlike the 200K-subsample MSI1/VTS1 case, there's no larger pool to draw
from -- this trains on the full 10,297-sequence real dataset. MAVE-NN's
internal train/val/test split inside train_surrogate() still gives an
honest held-out rho.

Run with the squid conda environment (mavenn/tensorflow):
    python mRNA_RBP/train_twister_deepsquid.py

Reads:  mRNA_RBP/outputs_twister_ribozyme/instance_00/varied_mutrate/pool.npz
Writes: mRNA_RBP/outputs_twister_ribozyme/surrogate_models/deepsquid_twister/
            mavenn_model_pairwise.pickle / .h5
            mave_df.csv
            meta.npz          -- y_mean, y_std, score_key, seq, train_n, rho_test
        mRNA_RBP/outputs_twister_ribozyme/surrogate_coefs/
            coefs_varied_mutrate_nonlinear_additive_p_pairwise_deepsquid_twister.npz
"""

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.train_surrogate import (
    train_surrogate, extract_coefs, predict_chunked, _rho,
)
from mRNA_RBP.oracles import (
    SurrogateMAVENNOracle,
    TWISTER_ORACLE, TWISTER_SCORE_KEY,
    default_output_base, sequence_config_for_oracle,
)

SEED = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=None,
                        help="Subsample cap (default: use all 10,297 real points)")
    parser.add_argument("--src_path", default=None)
    parser.add_argument("--out_base", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--coef_path", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    seq, stem_pairs, _motif_positions = sequence_config_for_oracle(TWISTER_ORACLE)

    out_base = args.out_base or default_output_base(_HERE, TWISTER_ORACLE)
    src_path = args.src_path or os.path.join(
        out_base, "instance_00", "varied_mutrate", "pool.npz")
    model_dir = args.model_dir or os.path.join(
        out_base, "surrogate_models", "deepsquid_twister")
    coef_path = args.coef_path or os.path.join(
        out_base, "surrogate_coefs",
        f"coefs_varied_mutrate_nonlinear_additive_p_pairwise_{TWISTER_SCORE_KEY}.npz")

    d = np.load(src_path, allow_pickle=True)
    nuc_ids_all = d["nuc_ids"]
    mut_counts_all = d["mut_counts"] if "mut_counts" in d.files else None
    y_all = d["scores_ra"]
    assert str(d["wt_seq"]) == seq, (
        f"pool.npz wt_seq {str(d['wt_seq'])!r} != registered TWISTER_SEQ {seq!r}")
    print(f"Loaded real Twister dataset: n={len(nuc_ids_all):,}  (labels: scores_ra)")

    rng = np.random.default_rng(SEED)
    n_draw = min(args.train_n, len(nuc_ids_all)) if args.train_n else len(nuc_ids_all)
    idx = rng.choice(len(nuc_ids_all), size=n_draw, replace=False)
    idx.sort()
    nuc_ids = nuc_ids_all[idx]
    mut_counts = mut_counts_all[idx] if mut_counts_all is not None else None
    y_raw = y_all[idx].astype(np.float32)
    print(f"Training set: n={len(nuc_ids):,}")
    if mut_counts is not None:
        vals, cnts = np.unique(mut_counts, return_counts=True)
        print("  mut_count distribution: " + "  ".join(f"{v}:{c}" for v, c in zip(vals, cnts)))

    X = np.eye(4, dtype=np.float32)[nuc_ids]
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std())
    if y_std < 1e-8:
        y_std = 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    os.makedirs(model_dir, exist_ok=True)
    print(f"\nTraining nonlinear additive+pairwise surrogate ({TWISTER_SCORE_KEY})...", flush=True)
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
    coefs["y_mean"] = np.float32(y_mean)
    coefs["y_std"] = np.float32(y_std)
    coefs["rho_test"] = np.float32(rho_test)
    coefs["train_n"] = np.int64(len(nuc_ids))
    os.makedirs(os.path.dirname(coef_path), exist_ok=True)
    np.savez_compressed(coef_path, **coefs)
    print(f"Saved coefs -> {coef_path}")

    np.savez_compressed(
        os.path.join(model_dir, "meta.npz"),
        y_mean=np.float32(y_mean), y_std=np.float32(y_std),
        score_key=np.array([TWISTER_SCORE_KEY]), seq=np.array([seq]),
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
    print(f"  rho(deep-squid vs real RA) on {check_n} points: {rho_check:+.3f}")
    print(f"  score key: {oracle.score_key}")


if __name__ == "__main__":
    main()
