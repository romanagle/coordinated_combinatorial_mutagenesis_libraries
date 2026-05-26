"""
mRNA_RBP/train_surrogate.py

Train the nonlinear additive+pairwise SQUID/MAVE-NN surrogate and save coefs.

Saves coefs in the same npz format consumed by plot_scatter_grid.py,
plot_pairwise_heatmaps.py, and plot_coefficients.py:
  coefs_k{k:02d}_mut{pct:02d}_lib{N}_nonlinear_additive_p_pairwise_nonlin_additive_pairwise.npz
  keys: alpha (L,4), J (L,4,L,4)

Run with the squid conda environment:
  /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/train_surrogate.py

Usage:
    python mRNA_RBP/train_surrogate.py
    python mRNA_RBP/train_surrogate.py --n_instances 1 --lib_sizes 20000
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "scripts"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

import squid.surrogate_zoo
from mRNA_RBP.generate_libraries import (
    SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA,
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES, GT_KEYS,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import make_pairwise_dataset, make_ssm_deltas, predict_ssm

COEF_DIR  = os.path.join(_HERE, "outputs", "surrogate_coefs")
JSON_PATH = os.path.join(_HERE, "outputs", "lib_size_spearman_results.json")
GT_KEY    = "nonlin_additive_pairwise"
NUCS      = ["A", "C", "G", "U"]
_NUCS_A   = np.array(list("ACGU"))

CFG = {
    "gpmap": "pairwise", "linearity": "nonlinear",
    "regression_type": "GE", "noise": "Gaussian",
    "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
}
CFG_KEY = "nonlinear additive + pairwise"


def nuc_ids_to_str(nuc_ids: np.ndarray) -> np.ndarray:
    return np.array(["".join(_NUCS_A[row]) for row in nuc_ids], dtype=object)


def predict_chunked(model, x_str, chunk: int = 500) -> np.ndarray:
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(y_true, y_hat):
    mask = np.isfinite(y_true) & np.isfinite(y_hat)
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(y_true[mask], y_hat[mask])[0])


def train_surrogate(X: np.ndarray, y: np.ndarray):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=CFG["gpmap"],
        regression_type=CFG["regression_type"],
        linearity=CFG["linearity"],
        noise=CFG["noise"],
        noise_order=CFG["noise_order"],
        reg_strength=CFG["reg_strength"],
        hidden_nodes=CFG["hidden_nodes"],
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    return wrapper, model, test_df


def extract_coefs(wrapper) -> dict:
    try:
        params = wrapper.get_params(gauge="consensus")
    except Exception:
        return {}
    result = {}
    if len(params) > 1 and params[1] is not None:
        result["alpha"] = np.asarray(params[1], dtype=np.float32)
    if len(params) > 2 and params[2] is not None:
        J = np.asarray(params[2], dtype=np.float32)
        result["J"] = J
    return result


def train_one(k: int, pct: int, n_lib: int, gt: MrnaRbpGroundTruth):
    lib_path = os.path.join(OUT_BASE, f"instance_{k:02d}",
                            f"mut{pct:02d}", f"lib_{n_lib}.npz")
    if not os.path.isfile(lib_path):
        print(f"  [skip] missing {lib_path}")
        return None

    d       = np.load(lib_path)
    nuc_ids = d["nuc_ids"]
    X       = np.eye(4, dtype=np.float32)[nuc_ids]
    y_raw   = gt.score_all(X)[GT_KEY].astype(np.float32)
    y_mean  = float(y_raw.mean())
    y_std   = float(y_raw.std())
    if y_std < 1e-8:
        y_std = 1.0
    y = ((y_raw - y_mean) / y_std).reshape(-1, 1)

    try:
        wrapper, model, test_df = train_surrogate(X, y)
    except Exception as e:
        print(f"  [FAIL] {e}")
        tf.keras.backend.clear_session()
        return None

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        c for c in test_df.columns if c.startswith("y"))
    rho_rand = _rho(
        np.asarray(test_df[y_col], dtype=float).ravel(),
        predict_chunked(model, np.asarray(test_df[x_col])),
    )

    type2_path = os.path.join(OUT_BASE, f"instance_{k:02d}", "type2.npz")
    rho_t2   = float("nan")
    yhat_t2  = np.array([], dtype=np.float32)
    y_t2_arr = np.array([], dtype=np.float32)
    if os.path.isfile(type2_path):
        d2       = np.load(type2_path)
        str_t2   = nuc_ids_to_str(d2["nuc_ids"])
        y_t2     = d2[f"scores_{GT_KEY}"].astype(float)
        p_t2     = predict_chunked(model, str_t2)
        rho_t2   = _rho(y_t2, p_t2)
        yhat_t2  = (p_t2 * y_std + y_mean).astype(np.float32)
        y_t2_arr = y_t2.astype(np.float32)

    wt_oh = gt.wt_one_hot()
    x_pw, y_pw_gt, _, _ = make_pairwise_dataset(wt_oh, gt)
    str_pw = nuc_ids_to_str(np.argmax(x_pw, axis=2).astype(np.uint8))
    rho_pw = _rho(y_pw_gt, predict_chunked(model, str_pw))

    delta_W    = make_ssm_deltas(wt_oh, gt)
    rho_ssm_pw = _rho(y_pw_gt, predict_ssm(x_pw, delta_W))

    print(f"  k={k} mut={pct:02d}% lib={n_lib:6d}  "
          f"ρ_rand={rho_rand:+.3f}  ρ_t2={rho_t2:+.3f}  "
          f"ρ_pw={rho_pw:+.3f}  (ssm_pw={rho_ssm_pw:+.3f})")

    yhat_lib = (predict_chunked(model, nuc_ids_to_str(nuc_ids)) * y_std + y_mean).astype(np.float32)

    coefs = extract_coefs(wrapper)
    coefs["y_mean"]  = np.float32(y_mean)
    coefs["y_std"]   = np.float32(y_std)
    coefs["yhat_lib"] = yhat_lib
    coefs["y_lib"]    = y_raw.astype(np.float32)
    coefs["yhat_t2"]  = yhat_t2
    coefs["y_t2"]     = y_t2_arr
    tf.keras.backend.clear_session()

    return {
        **coefs,
        "rho_rand": rho_rand, "rho_t2": rho_t2, "rho_pw": rho_pw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_instances", type=int, default=1)
    parser.add_argument("--lib_sizes", nargs="+", type=int, default=LIB_SIZES)
    args = parser.parse_args()

    os.makedirs(COEF_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)

    cache = {}
    if os.path.isfile(JSON_PATH):
        with open(JSON_PATH) as f:
            cache = json.load(f)

    for k in range(args.n_instances):
        inst_dir = os.path.join(OUT_BASE, f"instance_{k:02d}")
        if not os.path.isdir(inst_dir):
            print(f"[skip] instance {k:02d}: no data")
            continue
        gt = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS,
                                 seed=k, stem_sigma=STEM_SIGMA)
        for pct in MUT_RATES_PCT:
            for n_lib in args.lib_sizes:
                print(f"[train SQUID] k={k} mut={pct}% lib={n_lib}")
                res = train_one(k, pct, n_lib, gt)
                if res is None:
                    continue

                stem = (f"coefs_k{k:02d}_mut{pct:02d}_lib{n_lib}"
                        "_nonlinear_additive_p_pairwise_nonlin_additive_pairwise")
                arrays = {key: val for key, val in res.items()
                          if key in ("alpha", "J", "y_mean", "y_std",
                                     "yhat_lib", "y_lib", "yhat_t2", "y_t2")}
                np.savez_compressed(os.path.join(COEF_DIR, stem + ".npz"), **arrays)

                cache_key = f"{CFG_KEY}|{GT_KEY}|{pct}|{n_lib}|{k}"
                cache[cache_key] = {
                    "cfg": CFG_KEY, "gt_key": GT_KEY,
                    "pct": pct, "n_lib": n_lib, "k": k,
                    "rho_rand": res["rho_rand"],
                    "rho_t2":   res["rho_t2"],
                    "rho_pw":   res["rho_pw"],
                }

        with open(JSON_PATH, "w") as f:
            json.dump(cache, f, indent=2)

    print(f"\nDone. Coefs in {COEF_DIR}/")


if __name__ == "__main__":
    main()
