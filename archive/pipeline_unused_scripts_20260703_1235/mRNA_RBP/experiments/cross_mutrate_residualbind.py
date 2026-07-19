"""mRNA_RBP/cross_mutrate_residualbind.py

Step B of the ResidualBind cross-mutation-rate experiment: trains real
surrogates against ResidualBind-derived labels (precomputed by
score_residualbind_cross_mutrate.py, for either the MSI1 or VTS1 ensemble --
selected via --oracle) and evaluates the same same-rate vs. cross-rate
Spearman rho comparison as cross_mutrate_eval.py -- but via actual training +
actual model predictions (predict_chunked), not a post-hoc coefficient
reconstruction, since no ResidualBind-labeled surrogate has been trained
before. There's therefore no sign ambiguity to work around here (unlike
cross_mutrate_eval.py's phi-reconstruction trick): plain signed Spearman rho
is used throughout.

Scope (confirmed with user): instance 0 only -- ResidualBind is a single
fixed model, so repeat "instances" would only capture sampling/training
noise, not independent ground-truth realizations. Still covers all 3
mutation rates x 3 lib sizes x 4 surrogate configs (36 conditions).

Run with the squid conda environment (TF/mavenn), NOT the torch env used for
scoring:

    LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH \
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/experiments/cross_mutrate_residualbind.py --oracle vts1

Reads:  mRNA_RBP/outputs_<oracle>/instance_00/mut{pct}/lib_{n}.npz
        (written by score_residualbind_cross_mutrate.py)
Writes: mRNA_RBP/outputs_<oracle>/cross_mutrate_results.json
        (same schema as cross_mutrate_eval.py's output, single-instance lists)
        mRNA_RBP/outputs_<oracle>/surrogate_coefs/*.npz
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-nn"))
sys.path.insert(0, os.path.join(_HERE, "..", "squid-manuscript", "squid"))

from mRNA_RBP.generate_libraries import MUT_RATES_PCT, LIB_SIZES
from mRNA_RBP.oracles import default_output_base, normalize_oracle_name, primary_gt_key
from mRNA_RBP.lib_size_spearman import (
    SURROGATE_CONFIGS, _rho, nuc_ids_to_str, predict_chunked, train_surrogate,
    extract_surrogate_coefs, save_coefs,
)

INSTANCE = 0


def _load_scored(path, gt_key):
    d = np.load(path)
    return d["nuc_ids"], d[f"scores_{gt_key}"].astype(np.float32)


def run_condition(train_pct, n_lib, cfg_name, rb_base, test_frac, gt_key):
    train_path = os.path.join(rb_base, f"instance_{INSTANCE:02d}",
                              f"mut{train_pct:02d}", f"lib_{n_lib}.npz")
    if not os.path.exists(train_path):
        print(f"    [skip] missing {train_path}")
        return float("nan"), {}, {}, 0

    nuc_ids, y = _load_scored(train_path, gt_key)
    X = np.eye(4, dtype=np.float32)[nuc_ids]
    y = y.reshape(-1, 1)

    cfg = SURROGATE_CONFIGS[cfg_name]
    try:
        wrapper, model, test_df = train_surrogate(X, y, cfg)
    except Exception as e:
        print(f"    [FAIL] training: {e}")
        tf.keras.backend.clear_session()
        return float("nan"), {}, {}, 0

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        c for c in test_df.columns if c.startswith("y"))
    y_test = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
    rho_same = _rho(y_test, yhat_test)
    actual_n_train = len(nuc_ids)

    rho_cross = {}
    test_n_used = 0
    for other_pct in MUT_RATES_PCT:
        if other_pct == train_pct:
            continue
        other_path = os.path.join(rb_base, f"instance_{INSTANCE:02d}",
                                  f"mut{other_pct:02d}", "lib_20000.npz")
        if not os.path.exists(other_path):
            rho_cross[other_pct] = float("nan")
            continue
        other_ids, other_y = _load_scored(other_path, gt_key)
        test_n = max(1, round(test_frac * actual_n_train))
        test_n = min(test_n, len(other_ids))
        seed = train_pct * 1_000 + n_lib + other_pct * 7 + 555_001
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(other_ids), size=test_n, replace=False)
        x_other_str = nuc_ids_to_str(other_ids[idx])
        yhat_other = predict_chunked(model, x_other_str)
        rho_cross[other_pct] = _rho(other_y[idx], yhat_other)
        test_n_used = test_n

    coefs = extract_surrogate_coefs(wrapper, cfg)
    tf.keras.backend.clear_session()
    return rho_same, rho_cross, coefs, test_n_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True,
                        help="residualbind_msi1 / msi1 or residualbind_vts1 / vts1")
    parser.add_argument("--rb_base", default=None,
                        help="Defaults to outputs_<oracle>/")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--coef_dir", default=None)
    parser.add_argument("--mut_rates", nargs="+", type=int, default=MUT_RATES_PCT)
    parser.add_argument("--lib_sizes", nargs="+", type=int, default=LIB_SIZES)
    parser.add_argument("--test_frac", type=float, default=0.2)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    gt_key = primary_gt_key(oracle_name)
    rb_base = args.rb_base or default_output_base(_HERE, oracle_name)
    out_json = args.out_json or os.path.join(rb_base, "cross_mutrate_results.json")
    coef_dir = args.coef_dir or os.path.join(rb_base, "surrogate_coefs")

    cache = json.load(open(out_json)) if os.path.isfile(out_json) else {}
    cache.setdefault("cross", {}).setdefault(gt_key, {})

    def save():
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(cache, f, indent=2)

    for cfg_name in SURROGATE_CONFIGS:
        cache["cross"][gt_key].setdefault(cfg_name, {})

        for train_pct in args.mut_rates:
            spct = str(train_pct)
            cache["cross"][gt_key][cfg_name].setdefault(spct, {})
            other_pcts = [p for p in MUT_RATES_PCT if p != train_pct]

            for n_lib in args.lib_sizes:
                sn = str(n_lib)
                entry = cache["cross"][gt_key][cfg_name][spct].setdefault(
                    sn, {"same_rate": [], "cross": {}, "test_n": []})
                for op in other_pcts:
                    entry["cross"].setdefault(str(op), [])

                if len(entry["same_rate"]) >= 1:
                    print(f"[skip] {cfg_name} | mut{train_pct}% | lib={n_lib:,} (done)")
                    continue

                print(f"\n{'='*65}\n  {cfg_name} | mut{train_pct}% | "
                      f"lib={n_lib:,}\n{'='*65}", flush=True)

                rho_same, rho_cross, coefs, test_n = run_condition(
                    train_pct, n_lib, cfg_name, rb_base, args.test_frac, gt_key)

                entry["same_rate"].append(rho_same)
                for op in other_pcts:
                    entry["cross"][str(op)].append(rho_cross.get(op, float("nan")))
                entry["test_n"].append(test_n)

                cross_str = "  ".join(f"->{op}%={rho_cross.get(op, float('nan')):+.4f}"
                                      for op in other_pcts)
                print(f"  same={rho_same:+.4f}  {cross_str}", flush=True)

                save_coefs(coef_dir, INSTANCE, train_pct, n_lib, cfg_name, gt_key, coefs)
                save()

    print(f"\n[done] results saved to {out_json}")


if __name__ == "__main__":
    main()
