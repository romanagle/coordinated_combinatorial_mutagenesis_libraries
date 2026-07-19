"""Add a mixed mutation-rate condition to cross_mutrate_results.json.

The mixed condition is an equal-third blend of the existing 5%, 10%, and 25%
libraries.  This script:

  1. writes instance_XX/mut_mixed/lib_<N>.npz libraries;
  2. evaluates existing 5/10/25-trained coefficient files on the mixed test lib;
  3. trains mixed-library surrogates and evaluates them on 5/10/25/mixed tests.

It targets the heatmap use case, so by default it only runs lib_size=20,000.
Run in the squid environment with TensorFlow/MAVE-NN available.
"""

import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.cross_mutrate_eval import CFG_NAMES, phi_from_coefs, _rho_abs
from mRNA_RBP.generate_libraries import MUT_RATES_PCT, LIB_SIZES, N_INSTANCES
from mRNA_RBP.lib_size_spearman import (
    SURROGATE_CONFIGS,
    extract_surrogate_coefs,
    nuc_ids_to_str,
    predict_chunked,
    save_coefs,
    train_surrogate,
)

MIXED = "mixed"


def _safe_cfg(cfg_name):
    return cfg_name.replace(" ", "_").replace("+", "p")


def _rate_dir(rate):
    return "mut_mixed" if str(rate) == MIXED else "mut%02d" % int(rate)


def _coef_path(coef_dir, k, rate, n_lib, cfg_name, gt_key):
    if str(rate) == MIXED:
        rate_part = "mut_mixed"
    else:
        rate_part = "mut%02d" % int(rate)
    return os.path.join(
        coef_dir,
        "coefs_k%02d_%s_lib%d_%s_%s.npz"
        % (k, rate_part, n_lib, _safe_cfg(cfg_name), gt_key),
    )


def _load_npz(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _mix_counts(n_total, n_parts):
    base = n_total // n_parts
    counts = [base] * n_parts
    for i in range(n_total - base * n_parts):
        counts[i] += 1
    return counts


def create_mixed_libraries(out_base, lib_sizes, n_instances, rates=MUT_RATES_PCT):
    for k in range(n_instances):
        inst_dir = os.path.join(out_base, "instance_%02d" % k)
        if not os.path.isdir(inst_dir):
            continue

        for n_lib in lib_sizes:
            out_dir = os.path.join(inst_dir, "mut_mixed")
            out_path = os.path.join(out_dir, "lib_%d.npz" % n_lib)
            if os.path.exists(out_path):
                continue

            parts = []
            arrays = []
            for rate in rates:
                path = os.path.join(inst_dir, "mut%02d" % rate, "lib_%d.npz" % n_lib)
                arrays.append(_load_npz(path))

            counts = _mix_counts(n_lib, len(rates))
            rng = np.random.default_rng(910_000 + k * 1000 + n_lib)
            for arr, count in zip(arrays, counts):
                idx = rng.choice(len(arr["nuc_ids"]), size=count, replace=False)
                parts.append((arr, idx))

            nuc_ids = np.concatenate([arr["nuc_ids"][idx] for arr, idx in parts], axis=0)
            order = rng.permutation(len(nuc_ids))
            out = {"nuc_ids": nuc_ids[order]}

            common_keys = set(arrays[0].files)
            for arr in arrays[1:]:
                common_keys &= set(arr.files)
            for key in sorted(common_keys):
                if key == "nuc_ids":
                    continue
                if arrays[0][key].shape[:1] == (len(arrays[0]["nuc_ids"]),):
                    values = np.concatenate([arr[key][idx] for arr, idx in parts], axis=0)
                    out[key] = values[order]
                else:
                    out[key] = arrays[0][key]

            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(out_path, **out)
            print("[mixed-lib] wrote %s" % out_path, flush=True)


def _eval_coefs_on_library(coef_path, lib_path, gt_key):
    if not os.path.exists(coef_path) or not os.path.exists(lib_path):
        return float("nan")
    coefs = np.load(coef_path)
    alpha = coefs["alpha"]
    J = coefs["J"] if "J" in coefs else None
    lib = np.load(lib_path)
    X = np.eye(4, dtype=np.float32)[lib["nuc_ids"]]
    phi = phi_from_coefs(X, alpha, J)
    return _rho_abs(lib["scores_%s" % gt_key].astype(float), phi)


def fill_existing_train_to_mixed(cache, out_base, coef_dir, gt_key, cfg_names,
                                 lib_size, n_instances):
    mixed_key = MIXED
    mixed_lib_name = os.path.join("mut_mixed", "lib_%d.npz" % lib_size)

    for cfg_name in cfg_names:
        cfg_cache = cache.setdefault("cross", {}).setdefault(gt_key, {}).setdefault(cfg_name, {})
        for train_rate in MUT_RATES_PCT:
            entry = cfg_cache.setdefault(str(train_rate), {}).setdefault(
                str(lib_size), {"same_rate": [], "cross": {}, "test_n": []}
            )
            vals = entry.setdefault("cross", {}).setdefault(mixed_key, [])
            for k in range(len(vals), n_instances):
                coef_path = _coef_path(coef_dir, k, train_rate, lib_size, cfg_name, gt_key)
                lib_path = os.path.join(out_base, "instance_%02d" % k, mixed_lib_name)
                rho = _eval_coefs_on_library(coef_path, lib_path, gt_key)
                vals.append(rho)
                print("[numeric->mixed] %s train=%s k=%02d rho=%+.4f"
                      % (cfg_name, train_rate, k, rho), flush=True)


def train_existing_to_mixed(cache, out_base, gt_key, cfg_names, lib_size,
                            n_instances, test_frac):
    """Retrain numeric-rate models and evaluate them on the mixed test library.

    This is the exact path for ResidualBind, where the original cross-mutrate
    script stored scores from real model predictions rather than coefficient
    reconstruction.
    """
    for cfg_name in cfg_names:
        cfg_cache = cache.setdefault("cross", {}).setdefault(gt_key, {}).setdefault(cfg_name, {})
        for train_rate in MUT_RATES_PCT:
            entry = cfg_cache.setdefault(str(train_rate), {}).setdefault(
                str(lib_size), {"same_rate": [], "cross": {}, "test_n": []}
            )
            vals = entry.setdefault("cross", {}).setdefault(MIXED, [])
            for k in range(len(vals), n_instances):
                print("[train->mixed] %s train=%s k=%02d lib=%d"
                      % (cfg_name, train_rate, k, lib_size), flush=True)
                train_path = os.path.join(out_base, "instance_%02d" % k,
                                          _rate_dir(train_rate),
                                          "lib_%d.npz" % lib_size)
                mixed_path = os.path.join(out_base, "instance_%02d" % k,
                                          "mut_mixed", "lib_%d.npz" % lib_size)
                train = _load_npz(train_path)
                mixed = _load_npz(mixed_path)

                X = np.eye(4, dtype=np.float32)[train["nuc_ids"]]
                y = train["scores_%s" % gt_key].astype(np.float32).reshape(-1, 1)
                cfg = SURROGATE_CONFIGS[cfg_name]
                wrapper, model, _ = train_surrogate(X, y, cfg)

                test_n = max(1, round(test_frac * len(train["nuc_ids"])))
                test_n = min(test_n, len(mixed["nuc_ids"]))
                rng = np.random.default_rng(
                    880_000 + k * 1000 + int(train_rate) * 10 + lib_size)
                idx = rng.choice(len(mixed["nuc_ids"]), size=test_n, replace=False)
                yhat = predict_chunked(model, nuc_ids_to_str(mixed["nuc_ids"][idx]))
                rho = _rho_abs(mixed["scores_%s" % gt_key][idx].astype(float), yhat)
                vals.append(rho)
                entry.setdefault("test_n", []).append(test_n)

                coefs = extract_surrogate_coefs(wrapper, cfg)
                save_coefs(os.path.join(out_base, "surrogate_coefs"), k, train_rate,
                           lib_size, cfg_name, gt_key, coefs)
                tf.keras.backend.clear_session()
                print("  ->mixed=%+.4f" % rho, flush=True)


def _train_and_eval_mixed_one(out_base, gt_key, cfg_name, k, lib_size, test_frac):
    train_path = os.path.join(out_base, "instance_%02d" % k, "mut_mixed",
                              "lib_%d.npz" % lib_size)
    train = _load_npz(train_path)
    X = np.eye(4, dtype=np.float32)[train["nuc_ids"]]
    y = train["scores_%s" % gt_key].astype(np.float32).reshape(-1, 1)

    cfg = SURROGATE_CONFIGS[cfg_name]
    wrapper, model, test_df = train_surrogate(X, y, cfg)

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        c for c in test_df.columns if c.startswith("y"))
    y_test = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
    rho_same = abs(float(_rho_abs(y_test, yhat_test)))

    actual_n_train = len(train["nuc_ids"])
    test_n = max(1, round(test_frac * actual_n_train))
    rho_cross = {}
    for test_rate in MUT_RATES_PCT:
        path = os.path.join(out_base, "instance_%02d" % k, _rate_dir(test_rate),
                            "lib_%d.npz" % lib_size)
        lib = _load_npz(path)
        rng = np.random.default_rng(770_000 + k * 1000 + int(test_rate) * 10 + lib_size)
        idx = rng.choice(len(lib["nuc_ids"]), size=min(test_n, len(lib["nuc_ids"])),
                         replace=False)
        yhat = predict_chunked(model, nuc_ids_to_str(lib["nuc_ids"][idx]))
        rho_cross[str(test_rate)] = _rho_abs(lib["scores_%s" % gt_key][idx].astype(float),
                                             yhat)

    coefs = extract_surrogate_coefs(wrapper, cfg)
    tf.keras.backend.clear_session()
    return rho_same, rho_cross, coefs, test_n


def train_mixed_rows(cache, out_base, coef_dir, gt_key, cfg_names, lib_size,
                     n_instances, test_frac):
    for cfg_name in cfg_names:
        cfg_cache = cache.setdefault("cross", {}).setdefault(gt_key, {}).setdefault(cfg_name, {})
        entry = cfg_cache.setdefault(MIXED, {}).setdefault(
            str(lib_size), {"same_rate": [], "cross": {}, "test_n": []}
        )
        for test_rate in MUT_RATES_PCT:
            entry.setdefault("cross", {}).setdefault(str(test_rate), [])
        entry.setdefault("test_n", [])

        n_done = len(entry["same_rate"])
        for k in range(n_done, n_instances):
            print("[mixed-train] %s k=%02d lib=%d" % (cfg_name, k, lib_size), flush=True)
            same, cross, coefs, test_n = _train_and_eval_mixed_one(
                out_base, gt_key, cfg_name, k, lib_size, test_frac)
            entry["same_rate"].append(same)
            for test_rate in MUT_RATES_PCT:
                entry["cross"][str(test_rate)].append(cross.get(str(test_rate), float("nan")))
            entry["test_n"].append(test_n)
            save_coefs(coef_dir, k, 0, lib_size, cfg_name, gt_key, coefs)
            mixed_path = _coef_path(coef_dir, k, MIXED, lib_size, cfg_name, gt_key)
            legacy_path = os.path.join(
                coef_dir,
                "coefs_k%02d_mut00_lib%d_%s_%s.npz"
                % (k, lib_size, _safe_cfg(cfg_name), gt_key),
            )
            if os.path.exists(legacy_path) and not os.path.exists(mixed_path):
                os.replace(legacy_path, mixed_path)
            print("  same=%+.4f  ->5=%+.4f  ->10=%+.4f  ->25=%+.4f"
                  % (same, cross.get("5", float("nan")),
                     cross.get("10", float("nan")),
                     cross.get("25", float("nan"))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_base", default=os.path.join(_HERE, "outputs"))
    parser.add_argument("--results_json", default=None)
    parser.add_argument("--coef_dir", default=None)
    parser.add_argument("--gt_key", default="nonlin_additive_pairwise")
    parser.add_argument("--lib_size", type=int, default=20_000)
    parser.add_argument("--n_instances", type=int, default=N_INSTANCES)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--retrain_existing_to_mixed", action="store_true",
                        help="Train 5/10/25 models and evaluate mixed column "
                             "with real model predictions instead of coefficient reconstruction.")
    args = parser.parse_args()

    results_json = args.results_json or os.path.join(args.out_base, "cross_mutrate_results.json")
    coef_dir = args.coef_dir or os.path.join(args.out_base, "surrogate_coefs")

    create_mixed_libraries(args.out_base, [args.lib_size], args.n_instances)

    cache = json.load(open(results_json)) if os.path.isfile(results_json) else {}
    if args.retrain_existing_to_mixed:
        train_existing_to_mixed(cache, args.out_base, args.gt_key, CFG_NAMES,
                                args.lib_size, args.n_instances, args.test_frac)
        with open(results_json, "w") as f:
            json.dump(cache, f, indent=2)
    else:
        fill_existing_train_to_mixed(cache, args.out_base, coef_dir, args.gt_key,
                                     CFG_NAMES, args.lib_size, args.n_instances)
        with open(results_json, "w") as f:
            json.dump(cache, f, indent=2)
    if not args.skip_training:
        train_mixed_rows(cache, args.out_base, coef_dir, args.gt_key, CFG_NAMES,
                         args.lib_size, args.n_instances, args.test_frac)
        with open(results_json, "w") as f:
            json.dump(cache, f, indent=2)

    with open(results_json, "w") as f:
        json.dump(cache, f, indent=2)
    print("[done] updated %s" % results_json, flush=True)


if __name__ == "__main__":
    main()
