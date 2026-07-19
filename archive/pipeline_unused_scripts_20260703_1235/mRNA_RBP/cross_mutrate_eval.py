"""mRNA_RBP/cross_mutrate_eval.py

Post-hoc, retraining-free evaluation of how poorly a surrogate trained at one
mutation rate generalizes to sequences from a different mutation rate.

For every (instance k, train_pct, n_lib, surrogate cfg) condition already
trained by lib_size_spearman.py, reconstructs the MAVE-NN latent phenotype
phi(x) = alpha-sum + pairwise-sum directly from the saved coefficient npz (no
model reload, no retraining), and computes Spearman rho against the cached GT
scores of a freshly-drawn, matched-size test set pulled from each *other*
mutation rate's pool_2M.npz. The same-rate baseline is NOT recomputed -- it is
pulled through directly from lib_size_spearman_results.json's cached "rand"
value (MAVE-NN's own internal ~20% holdout split), keeping both numbers on
comparably-sized test sets without needing to reverse-engineer that split.

MAVE-NN's GE nonlinearity g(phi)->yhat is monotonic, so Spearman(yhat, y)
depends on phi only up to SIGN (verified empirically: linear-GE configs can
reconstruct phi with the opposite sign of the trained model's own output,
since gauge="consensus" does not fix the sign of g). All reported rho values
are therefore reported as abs(rho).

Resumable: results saved to JSON after every instance within a condition.

Usage:
    python mRNA_RBP/cross_mutrate_eval.py \
        --out_json mRNA_RBP/outputs/cross_mutrate_results.json
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES,
)
from mRNA_RBP.oracles import (
    MRNA_ORACLE,
    default_output_base,
    normalize_oracle_name,
    primary_gt_key,
)

# Mirrors the keys of SURROGATE_CONFIGS in lib_size_spearman.py:59-80. Kept as
# a local list (not imported) so this script never pulls in tensorflow/mavenn
# -- the whole point of the post-hoc design is to stay lightweight.
CFG_NAMES = [
    "additive",
    "additive + pairwise",
    "nonlinear additive",
    "nonlinear additive + pairwise",
]


def _safe_cfg(cfg_name: str) -> str:
    return cfg_name.replace(" ", "_").replace("+", "p")


def phi_from_coefs(X: np.ndarray, alpha: np.ndarray, J: np.ndarray = None) -> np.ndarray:
    """Reconstruct the MAVE-NN latent phenotype phi(x) from saved coefficients.

    phi(x) = sum_l alpha[l, x_l]  +  sum_{i<j} x_i . J[i,:,j,:] . x_j

    The pairwise sum runs over ALL L*(L-1)/2 upper-triangle position pairs,
    not just the GT's stem pairs, since cross-rate test sequences carry
    mutations scattered across all positions. This generalizes the existing
    single-stem-pair pattern in lib_size_spearman.py (xi[ei] @ J[ei,:,ej,:] @
    xi[ej], summed once per i<j pair, no 0.5 factor).
    """
    X = X.astype(np.float64)
    phi = np.einsum("nla,la->n", X, alpha.astype(np.float64))
    if J is not None:
        L = alpha.shape[0]
        iu, ju = np.triu_indices(L, k=1)
        Jp = J.astype(np.float64)[iu, :, ju, :]          # (P, 4, 4)
        phi = phi + np.einsum("npa,pab,npb->np",
                              X[:, iu, :], Jp, X[:, ju, :]).sum(axis=1)
    return phi


def _rho_abs(y_true: np.ndarray, phi: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(phi)
    if mask.sum() < 3:
        return float("nan")
    return float(abs(spearmanr(y_true[mask], phi[mask])[0]))


def _coef_path(coef_dir, k, pct, n_lib, cfg_name, gt_key):
    return os.path.join(
        coef_dir,
        f"coefs_k{k:02d}_mut{pct:02d}_lib{n_lib}_{_safe_cfg(cfg_name)}_{gt_key}.npz",
    )


def eval_condition(k: int, train_pct: int, n_lib: int, cfg_name: str, gt_key: str,
                   other_pct: int, coef_dir: str, out_base: str,
                   test_frac: float) -> tuple:
    """Cross-rate Spearman |rho| for one (k, train_pct, n_lib, cfg, other_pct).

    Returns (rho, test_n).
    """
    coef_path = _coef_path(coef_dir, k, train_pct, n_lib, cfg_name, gt_key)
    if not os.path.exists(coef_path):
        return float("nan"), 0
    coefs = np.load(coef_path)
    alpha = coefs["alpha"]
    J = coefs["J"] if "J" in coefs else None

    train_lib_path = os.path.join(out_base, f"instance_{k:02d}",
                                  f"mut{train_pct:02d}", f"lib_{n_lib}.npz")
    if not os.path.exists(train_lib_path):
        return float("nan"), 0
    actual_n_train = len(np.load(train_lib_path)["nuc_ids"])
    test_n = max(1, round(test_frac * actual_n_train))

    other_pool_path = os.path.join(out_base, f"instance_{k:02d}",
                                   f"mut{other_pct:02d}", "pool_2M.npz")
    if not os.path.exists(other_pool_path):
        return float("nan"), 0
    pool = np.load(other_pool_path)
    pool_n = len(pool["nuc_ids"])
    test_n = min(test_n, pool_n)

    seed = k * 100_000 + train_pct * 1_000 + n_lib + other_pct * 7 + 999_983
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool_n, size=test_n, replace=False)

    X = np.eye(4, dtype=np.float32)[pool["nuc_ids"][idx]]
    y_true = pool[f"scores_{gt_key}"][idx].astype(float)

    phi = phi_from_coefs(X, alpha, J)
    return _rho_abs(y_true, phi), test_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--coef_dir", default=None)
    parser.add_argument("--src_json", default=None,
                        help="lib_size_spearman_results.json to pull the "
                             "same-rate 'rand' baseline from")
    parser.add_argument("--gt_keys", nargs="+", default=None)
    parser.add_argument("--oracle", default=MRNA_ORACLE,
                        choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1"])
    parser.add_argument("--out_base", default=None,
                        help="Library root. Defaults to outputs or outputs_<oracle>")
    parser.add_argument("--n_instances", type=int, default=N_INSTANCES)
    parser.add_argument("--mut_rates", nargs="+", type=int, default=MUT_RATES_PCT)
    parser.add_argument("--lib_sizes", nargs="+", type=int, default=LIB_SIZES)
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Fraction of actual training lib size to draw as "
                             "the cross-rate test set (matches MAVE-NN's "
                             "internal test split fraction)")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    out_base = args.out_base or default_output_base(_HERE, oracle_name)
    gt_keys = args.gt_keys or [primary_gt_key(oracle_name)]
    out_json = args.out_json or os.path.join(out_base, "cross_mutrate_results.json")
    coef_dir = args.coef_dir or os.path.join(out_base, "surrogate_coefs")
    src_json = args.src_json or os.path.join(out_base, "lib_size_spearman_results.json")

    with open(src_json) as f:
        src_cache = json.load(f)

    cache = json.load(open(out_json)) if os.path.isfile(out_json) else {}
    cache.setdefault("cross", {})

    def save():
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(cache, f, indent=2)

    for gt_key in gt_keys:
        cache["cross"].setdefault(gt_key, {})
        src_sur = src_cache.get("surrogate", {}).get(gt_key, {})

        for cfg_name in CFG_NAMES:
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

                    n_done = min(
                        [len(entry["same_rate"])] +
                        [len(entry["cross"][str(op)]) for op in other_pcts]
                    )
                    if n_done >= args.n_instances:
                        print(f"[skip] {gt_key} | {cfg_name} | mut{train_pct}% | "
                              f"lib={n_lib:,} ({n_done} done)")
                        continue

                    print(f"\n{'='*65}\n  {gt_key} | {cfg_name} | "
                          f"mut{train_pct}% | lib={n_lib:,}\n{'='*65}")

                    for k in range(n_done, args.n_instances):
                        rand_list = (src_sur.get(cfg_name, {})
                                            .get(spct, {}).get(sn, {}).get("rand", []))
                        same_rho = rand_list[k] if k < len(rand_list) else float("nan")
                        entry["same_rate"].append(same_rho)

                        test_n_used = 0
                        cross_strs = []
                        for op in other_pcts:
                            rho, test_n_used = eval_condition(
                                k, train_pct, n_lib, cfg_name, gt_key, op,
                                coef_dir, out_base, args.test_frac,
                            )
                            entry["cross"][str(op)].append(rho)
                            cross_strs.append(f"->{op}%={rho:+.4f}")
                        entry["test_n"].append(test_n_used)

                        print(f"  instance {k:02d}  same={same_rho:+.4f}  "
                              + "  ".join(cross_strs), flush=True)
                        save()

    print(f"\n[done] results saved to {out_json}")


if __name__ == "__main__":
    main()
