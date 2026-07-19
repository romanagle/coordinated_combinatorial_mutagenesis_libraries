"""
mRNA_RBP/experiments/train_surrogate_from_score_cache.py

Trains the nonlinear additive+pairwise MAVE-NN surrogate on labels that were
already computed by score_residualbind_type3.py, instead of calling
build_oracle() directly.

Why this split exists: scoring with a ResidualBind ensemble oracle requires
torch (rbp_bench), which only loads cleanly in the toehold_gpu conda env on
this machine; MAVE-NN surrogate training requires tensorflow, which only
loads cleanly in the squid conda env (py3.7). The two environments' torch/TF
builds conflict, so scoring and training run as separate steps connected by
an intermediate score-cache npz (see score_residualbind_type3.py).

Usage (squid env):
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/experiments/train_surrogate_from_score_cache.py \
        --score_cache mRNA_RBP/outputs/residualbind_type3_scores_vts1_instance00.npz \
        --gt_key vts1_residualbind \
        --out mRNA_RBP/outputs/surrogate_coefs/coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401 -- must load before tensorflow, see below
import numpy as np
import tensorflow as tf

# mRNA_RBP.seq_utils (imported transitively below) pulls in matplotlib via
# squid.ink; if tensorflow loads first it pins the system libstdc++.so.6 in
# the process, and matplotlib's compiled extension (needs a newer GLIBCXX
# from the conda env's libstdc++) then fails to import. Importing matplotlib
# before tensorflow avoids the conflict.

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.train_surrogate import (
    train_surrogate, extract_coefs, nuc_ids_to_str, predict_chunked, _rho,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_cache", required=True)
    parser.add_argument("--gt_key", required=True,
                        help="Key suffix in the cache, e.g. scores_train uses this "
                             "only for logging -- the cache's 'train'/'type2'/"
                             "'pairwise' arrays are used directly")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    d = np.load(args.score_cache)
    nuc_ids = d["nuc_ids_train"]
    X       = np.eye(4, dtype=np.float32)[nuc_ids]
    y_raw   = d["scores_train"].astype(np.float32)
    y_mean  = float(y_raw.mean())
    y_std   = float(y_raw.std()) or 1.0
    y       = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    print(f"[train] N={len(nuc_ids):,}  gt_key={args.gt_key}  "
          f"y_mean={y_mean:.4f}  y_std={y_std:.4f}")
    wrapper, model, test_df = train_surrogate(X, y)

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        c for c in test_df.columns if c.startswith("y"))
    rho_rand = _rho(
        np.asarray(test_df[y_col], dtype=float).ravel(),
        predict_chunked(model, np.asarray(test_df[x_col])),
    )

    rho_t2 = float("nan")
    if "nuc_ids_type2" in d:
        str_t2 = nuc_ids_to_str(d["nuc_ids_type2"])
        p_t2   = predict_chunked(model, str_t2)
        rho_t2 = _rho(d["scores_type2"].astype(float), p_t2)

    rho_pw = float("nan")
    if "nuc_ids_pairwise" in d:
        str_pw = nuc_ids_to_str(d["nuc_ids_pairwise"])
        p_pw   = predict_chunked(model, str_pw)
        rho_pw = _rho(d["scores_pairwise"].astype(float), p_pw)

    print(f"[train] rho_rand={rho_rand:+.4f}  rho_activity_balanced={rho_t2:+.4f}  rho_pw={rho_pw:+.4f}")

    coefs = extract_coefs(wrapper)
    coefs["y_mean"] = np.float32(y_mean)
    coefs["y_std"]  = np.float32(y_std)
    tf.keras.backend.clear_session()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **coefs)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
