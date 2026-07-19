"""
mRNA_RBP/plots/train_deepsquid_surrogate_mut10.py

Trains a nonlinear additive+pairwise MAVE-NN surrogate on a 20K-sequence,
10%-mutation-rate library labeled by "deep squid" (the 200K-varied-mutrate
surrogate, treated as an interpretable stand-in oracle via
oracles.SurrogateMAVENNOracle) -- NOT the raw ResidualBind ensemble.

Reuses the exact same nuc_ids already drawn for the ResidualBind-ensemble
mut=10%/lib=20,000 libraries (so the sequences are identical across the
"oracle_vs_surrogate" and "deep-squid oracle_vs_surrogate" comparisons),
just re-labels them with deep squid instead of ResidualBind.

Reads (nuc_ids only, existing labels ignored):
  MSI1  outputs_vts1_residualbind/outputs_residualbind_ensemble/instance_00/mut10/lib_20000.npz
  VTS1  outputs_vts1_residualbind/instance_00/vts1_mut10/lib_20000.npz

Writes (SURROGATE_ORACLE = "surrogate_varied_mutrate" naming convention):
  outputs_surrogate_varied_mutrate/instance_00/mut10/lib_20000.npz            (MSI1, scored)
  outputs_surrogate_varied_mutrate/instance_00/vts1_mut10/lib_20000.npz       (VTS1, scored)
  outputs_surrogate_varied_mutrate/surrogate_models/mut10_lib20000_<rbp>_nonlin_pairwise/
  outputs_surrogate_varied_mutrate/surrogate_coefs/
      coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_msi1.npz
      coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_vts1.npz

Usage (squid env -- needs mavenn/tensorflow):
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/plots/train_deepsquid_surrogate_mut10.py
"""

import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.oracles import SurrogateMAVENNOracle
from mRNA_RBP.train_surrogate import train_surrogate, extract_coefs, predict_chunked, _rho

SEED    = 0
TRAIN_N = 20_000

MSI1_SEQ        = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS = [(8, 30), (9, 29), (10, 28), (11, 27), (12, 26), (13, 25), (14, 24), (15, 23)]
MSI1_DEEPSQUID_MODEL_DIR = os.path.join(
    _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
    "surrogate_models", "varied_mutrate_nonlin_pairwise")
MSI1_SRC = os.path.join(
    _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
    "instance_00", "mut10", "lib_20000.npz")

VTS1_SEQ        = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
VTS1_STEM_PAIRS = [(16, 21), (15, 22), (14, 23), (25, 31), (24, 32)]
VTS1_DEEPSQUID_MODEL_DIR = os.path.join(
    _HERE, "outputs_vts1_residualbind", "surrogate_models", "varied_mutrate_nonlin_pairwise")
VTS1_SRC = os.path.join(
    _HERE, "outputs_vts1_residualbind", "instance_00", "vts1_mut10", "lib_20000.npz")

OUT_BASE   = os.path.join(_HERE, "outputs_surrogate_varied_mutrate")
COEF_DIR   = os.path.join(OUT_BASE, "surrogate_coefs")


def run_one(label, seq, stem_pairs, deepsquid_model_dir, src_path, lib_dst, coef_path, model_dir):
    if os.path.exists(coef_path):
        print(f"[skip] {label} already trained -> {coef_path}")
        return

    print(f"\n=== {label}: labeling {TRAIN_N} seqs with deep squid ===")
    d = np.load(src_path)
    nuc_ids = d["nuc_ids"]
    assert len(nuc_ids) == TRAIN_N, f"expected {TRAIN_N} seqs, got {len(nuc_ids)}"

    oracle = SurrogateMAVENNOracle(seq=seq, stem_pairs=stem_pairs, model_dir=deepsquid_model_dir)
    X = np.eye(4, dtype=np.float32)[nuc_ids]
    y_raw = np.concatenate([
        oracle(X[s:s + 4096]) for s in range(0, len(X), 4096)
    ]).astype(np.float32)

    os.makedirs(os.path.dirname(lib_dst), exist_ok=True)
    np.savez_compressed(lib_dst, nuc_ids=nuc_ids, scores_deepsquid=y_raw)
    print(f"  Scored library -> {lib_dst}  "
          f"(mean={y_raw.mean():+.4f} std={y_raw.std():.4f})")

    y_mean = float(y_raw.mean())
    y_std  = float(y_raw.std()) if y_raw.std() > 1e-8 else 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    os.makedirs(model_dir, exist_ok=True)
    print(f"  Training nonlinear additive+pairwise surrogate on deep-squid labels...")
    wrapper, model, test_df = train_surrogate(
        X, y, save_dir=model_dir, epochs=1000, patience=50,
    )

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(c for c in test_df.columns if c.startswith("y"))
    rho_test = _rho(
        np.asarray(test_df[y_col], dtype=float).ravel(),
        predict_chunked(model, np.asarray(test_df[x_col])),
    )
    print(f"  Held-out test rho: {rho_test:+.3f}  (n_test={len(test_df):,})")

    coefs = extract_coefs(wrapper)
    coefs["y_mean"]   = np.float32(y_mean)
    coefs["y_std"]    = np.float32(y_std)
    coefs["rho_test"] = np.float32(rho_test)
    coefs["train_n"]  = np.int64(len(nuc_ids))
    os.makedirs(os.path.dirname(coef_path), exist_ok=True)
    np.savez_compressed(coef_path, **coefs)
    print(f"  Saved coefs -> {coef_path}")
    tf.keras.backend.clear_session()


def main():
    run_one(
        "MSI1", MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_DEEPSQUID_MODEL_DIR, MSI1_SRC,
        lib_dst=os.path.join(OUT_BASE, "instance_00", "mut10", "lib_20000.npz"),
        coef_path=os.path.join(
            COEF_DIR, "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_msi1.npz"),
        model_dir=os.path.join(OUT_BASE, "surrogate_models", "mut10_lib20000_msi1_nonlin_pairwise"),
    )
    run_one(
        "VTS1", VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_DEEPSQUID_MODEL_DIR, VTS1_SRC,
        lib_dst=os.path.join(OUT_BASE, "instance_00", "vts1_mut10", "lib_20000.npz"),
        coef_path=os.path.join(
            COEF_DIR, "coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_vts1.npz"),
        model_dir=os.path.join(OUT_BASE, "surrogate_models", "mut10_lib20000_vts1_nonlin_pairwise"),
    )


if __name__ == "__main__":
    main()
