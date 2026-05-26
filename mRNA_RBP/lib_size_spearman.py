"""mRNA_RBP/lib_size_spearman.py

For each of the 10 instances × 3 mutation rates × 4 library sizes × 4 surrogate
configs × 4 GT keys, trains a MAVE-NN surrogate and evaluates:
  - Spearman ρ on the MAVE-NN internal random holdout split
  - Spearman ρ on the uniformized eval library for that GT key

Also computes the saturated mutagenesis ρ per instance (additive SSM baseline).

After each surrogate is trained, extracts additive (and pairwise) coefficients
and saves them as npz alongside the JSON cache, enabling downstream coefficient
comparison plots.

Resumable: results are saved to JSON after every completed instance × condition.

Usage:
    python mRNA_RBP/lib_size_spearman.py \
        --out_json mRNA_RBP/outputs/lib_size_spearman_results.json \
        --out_coef mRNA_RBP/outputs/surrogate_coefs/
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
    OUT_BASE, N_INSTANCES, MUT_RATES_PCT, LIB_SIZES,
    SEQ, STEM_PAIRS, MOTIF_POSITIONS, GT_KEYS, TYPE2_TARGET_N, STEM_SIGMA,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.evaluate import (
    make_additive_dataset, make_pairwise_dataset,
    make_ssm_deltas, predict_ssm,
)
import mRNA_RBP.viz as viz

NUCS   = ["A", "C", "G", "U"]
_NUCS_A = np.array(list("ACGU"))

SURROGATE_CONFIGS = {
    "additive": {
        "gpmap": "additive", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.01,
    },
    "additive + pairwise": {
        "gpmap": "pairwise", "linearity": "linear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 0, "reg_strength": 0.1,
    },
    "nonlinear additive": {
        "gpmap": "additive", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "SkewedT",
        "noise_order": 2, "reg_strength": 0.01, "hidden_nodes": 50,
    },
    "nonlinear additive + pairwise": {
        "gpmap": "pairwise", "linearity": "nonlinear",
        "regression_type": "GE", "noise": "Gaussian",
        "noise_order": 2, "reg_strength": 0.005, "hidden_nodes": 10,
    },
}

MUT_LABELS = {5: "5%", 10: "10%", 25: "25%"}
LIB_LABELS = {200: "200", 2_000: "2K", 20_000: "20K"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_pool(wt_onehot: np.ndarray, n_seqs: int,
                   mut_count: int, rng) -> np.ndarray:
    """Generate (n_seqs, L) uint8 nuc_ids; each row is WT with exactly
    mut_count positions mutated to a uniformly random non-WT nucleotide.
    Returns deduplicated rows."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(np.uint8)
    buf    = np.tile(wt_idx[None, :], (n_seqs, 1))
    noise  = rng.random((n_seqs, L))
    pos    = np.argpartition(noise, mut_count, axis=1)[:, :mut_count]
    wt_at     = wt_idx[pos.ravel()]
    rand_nuc  = rng.integers(0, 3, size=n_seqs * mut_count, dtype=np.uint8)
    new_nucs  = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
    n_idx     = np.repeat(np.arange(n_seqs), mut_count)
    buf[n_idx, pos.ravel()] = new_nucs
    return np.unique(buf, axis=0)


def _exclude_eval_seqs(train_nuc_ids: np.ndarray,
                       *eval_arrays: np.ndarray) -> tuple:
    """Remove from train_nuc_ids any sequence that appears in any eval array.

    Returns (filtered_nuc_ids, n_removed).
    """
    eval_seqs = set()
    for arr in eval_arrays:
        if arr is not None and len(arr):
            for row in arr:
                eval_seqs.add(row.tobytes())
    if not eval_seqs:
        return train_nuc_ids, 0
    mask = np.array([row.tobytes() not in eval_seqs
                     for row in train_nuc_ids], dtype=bool)
    return train_nuc_ids[mask], int((~mask).sum())


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


def train_surrogate(X: np.ndarray, y: np.ndarray, cfg: dict):
    N   = X.shape[0]
    bsz = max(32, min(N // 150, 2048))
    lr  = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    is_nl_pair = cfg["gpmap"] == "pairwise" and cfg["linearity"] == "nonlinear"
    epochs  = 1000 if is_nl_pair else 500
    patience = 50  if is_nl_pair else 25

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg.get("hidden_nodes", 50),
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=epochs, batch_size=bsz,
        early_stopping=True, patience=patience,
        restore_best_weights=True, save_dir=None, verbose=0,
    )
    return wrapper, model, test_df


def extract_surrogate_coefs(wrapper, cfg: dict) -> dict:
    """Extract additive (and pairwise) weight matrices from a trained surrogate.

    Returns dict with:
      alpha : (L, 4) float32  additive weights (MAVE-NN params[1])
      beta  : {(i,j): (4,4)}  pairwise weights for stem pairs (pairwise models only)
      J     : (L, L, 4, 4)   full pairwise tensor (pairwise models only)
    """
    try:
        params = wrapper.get_params(gauge="consensus")
    except Exception:
        return {}

    result = {}
    if len(params) > 1 and params[1] is not None:
        result["alpha"] = np.asarray(params[1], dtype=np.float32)

    if cfg["gpmap"] == "pairwise" and len(params) > 2 and params[2] is not None:
        J = np.asarray(params[2], dtype=np.float32)  # (L, 4, L, 4) from MAVE-NN
        result["J"] = J
        beta = {}
        for (i, j) in STEM_PAIRS:
            if i < J.shape[0] and j < J.shape[2]:
                beta[(i, j)] = J[i, :, j, :]   # (4, 4)
        result["beta"] = beta

    return result


# ---------------------------------------------------------------------------
# Saturated mutagenesis ρ
# ---------------------------------------------------------------------------

def compute_saturated_rho(k: int) -> dict:
    """Spearman ρ for the SSM additive predictor on both type2 and pairwise eval libs.

    Returns {gt_key: {"type2": float, "pairwise": float}}.
    """
    gt      = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k, stem_sigma=STEM_SIGMA)
    wt_oh   = gt.wt_one_hot()
    delta_W = make_ssm_deltas(wt_oh, gt)

    inst_dir      = os.path.join(OUT_BASE, f"instance_{k:02d}")
    type2_path    = os.path.join(inst_dir, "type2.npz")
    pairwise_path = os.path.join(inst_dir, "pairwise_lib.npz")

    nan_entry = {"type2": float("nan"), "pairwise": float("nan")}
    if not os.path.exists(type2_path):
        return {gt_key: dict(nan_entry) for gt_key in GT_KEYS}

    ev2    = np.load(type2_path)
    X2     = np.eye(4, dtype=np.float32)[ev2["nuc_ids"]]
    yhat2  = predict_ssm(X2, delta_W, wt_activity=0.0)

    has_pw = os.path.exists(pairwise_path)
    if has_pw:
        evp   = np.load(pairwise_path)
        Xp    = np.eye(4, dtype=np.float32)[evp["nuc_ids"]]
        yhatp = predict_ssm(Xp, delta_W, wt_activity=0.0)

    result = {}
    for gt_key in GT_KEYS:
        rho_t2 = _rho(ev2[f"scores_{gt_key}"].astype(float), yhat2)
        rho_pw = _rho(evp[f"scores_{gt_key}"].astype(float), yhatp) if has_pw else float("nan")
        result[gt_key] = {"type2": rho_t2, "pairwise": rho_pw}
    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

TYPE1_EVAL_SIZES = [200, 2_000, 20_000]   # fixed type1 eval libs, always evaluated


def run_instance_condition(k: int, pct: int, n_lib: int,
                           cfg_name: str, gt_key: str) -> tuple:
    """Train one surrogate; return (rho_rand,
                                    rho_type1_200, rho_type1_2000, rho_type1_20000,
                                    rho_type2, rho_pairwise, coefs).

    All eval libraries are evaluated for every training condition regardless
    of the training lib size.
    """
    inst_dir      = os.path.join(OUT_BASE, f"instance_{k:02d}")
    rand_path     = os.path.join(inst_dir, f"mut{pct:02d}", f"lib_{n_lib}.npz")
    type2_path    = os.path.join(inst_dir, "type2.npz")
    pairwise_path = os.path.join(inst_dir, "pairwise_lib.npz")

    _nan7 = (float("nan"),) * 6 + ({},)

    if not os.path.exists(type2_path):
        print(f"      [SKIP] missing type2 for instance {k:02d}")
        return _nan7

    # Load eval nuc_ids for leakage removal
    type2_ids    = np.load(type2_path)["nuc_ids"]
    pairwise_ids = np.load(pairwise_path)["nuc_ids"] if os.path.exists(pairwise_path) else None

    gt = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS,
                             seed=k, stem_sigma=STEM_SIGMA)

    if os.path.exists(rand_path):
        d       = np.load(rand_path)
        nuc_ids = d["nuc_ids"][:n_lib]
    else:
        # Generate training library on the fly
        print(f"      [gen] {rand_path} missing — generating {n_lib} seqs from GT")
        wt_oh = gt.wt_one_hot()
        mc    = max(1, round(pct * wt_oh.shape[0] / 100))
        rng   = np.random.default_rng(k * 10_000 + pct * 100 + n_lib)
        nuc_ids = _generate_pool(wt_oh, n_lib, mc, rng)

    # Remove any training sequences that appear in eval libraries
    nuc_ids, n_removed = _exclude_eval_seqs(nuc_ids, type2_ids, pairwise_ids)
    if n_removed:
        print(f"      [dedup] removed {n_removed} train seqs found in eval libs")

    # Score after filtering (ensures y aligns with deduplicated nuc_ids)
    X     = np.eye(4, dtype=np.float32)[nuc_ids]
    y_all = gt.score_all(X)[gt_key].reshape(-1, 1)

    cfg = SURROGATE_CONFIGS[cfg_name]
    try:
        wrapper, model, test_df = train_surrogate(X, y_all, cfg)
    except Exception as e:
        print(f"      [FAIL] training: {e}")
        tf.keras.backend.clear_session()
        return _nan7

    x_col = "x" if "x" in test_df.columns else "X"
    y_col = "y" if "y" in test_df.columns else next(
        c for c in test_df.columns if c.startswith("y"))
    y_test    = np.asarray(test_df[y_col], dtype=float).ravel()
    yhat_test = predict_chunked(model, np.asarray(test_df[x_col]))
    rho_rand  = _rho(y_test, yhat_test)

    def _eval_on(path, key):
        ev      = np.load(path)
        y_ev    = ev[f"scores_{key}"].astype(float)
        x_str   = nuc_ids_to_str(ev["nuc_ids"])
        yhat_ev = predict_chunked(model, x_str)
        return _rho(y_ev, yhat_ev)

    rho_type1 = {}
    for n in TYPE1_EVAL_SIZES:
        p = os.path.join(inst_dir, f"type1_lib{n}.npz")
        rho_type1[n] = _eval_on(p, gt_key) if os.path.exists(p) else float("nan")

    rho_type2    = _eval_on(type2_path, gt_key)
    rho_pairwise = _eval_on(pairwise_path, gt_key) if os.path.exists(pairwise_path) else float("nan")

    coefs = extract_surrogate_coefs(wrapper, cfg)
    tf.keras.backend.clear_session()
    return (rho_rand,
            rho_type1[200], rho_type1[2_000], rho_type1[20_000],
            rho_type2, rho_pairwise,
            coefs)


def save_coefs(coef_dir: str, k: int, pct: int, n_lib: int,
               cfg_name: str, gt_key: str, coefs: dict):
    """Save surrogate coefficient arrays to npz."""
    if not coefs:
        return
    safe_cfg = cfg_name.replace(" ", "_").replace("+", "p")
    fname = f"coefs_k{k:02d}_mut{pct:02d}_lib{n_lib}_{safe_cfg}_{gt_key}.npz"
    os.makedirs(coef_dir, exist_ok=True)
    arrays = {k_: v for k_, v in coefs.items() if not isinstance(v, dict)}
    np.savez_compressed(os.path.join(coef_dir, fname), **arrays)


def generate_notebook_plots(k: int, pct: int, n_lib: int,
                            cfg_name: str, gt_key: str,
                            coefs: dict, plot_dir: str):
    """Coefficient comparison + structured eval plots for one (instance, condition)."""
    if not coefs or "alpha" not in coefs:
        return

    gt    = MrnaRbpGroundTruth(SEQ, STEM_PAIRS, MOTIF_POSITIONS, seed=k, stem_sigma=STEM_SIGMA)
    wt_oh = gt.wt_one_hot()
    L     = len(SEQ)
    os.makedirs(plot_dir, exist_ok=True)
    safe_cfg = cfg_name.replace(" ", "_").replace("+", "p")
    stem  = f"k{k:02d}_mut{pct:02d}_lib{n_lib}_{safe_cfg}_{gt_key}"

    sur_alpha = coefs["alpha"]
    sur_beta  = coefs.get("beta", {})

    # Coefficient comparison
    fig_gt, fig_sur = viz.plot_gt_vs_surrogate(gt, sur_alpha, sur_beta, L)
    fig_gt.savefig(os.path.join(plot_dir, f"{stem}_coef_gt.png"),
                   dpi=110, bbox_inches='tight')
    fig_gt.savefig(os.path.join(plot_dir, f"{stem}_coef_gt.svg"), bbox_inches='tight')
    fig_sur.savefig(os.path.join(plot_dir, f"{stem}_coef_sur.png"),
                    dpi=110, bbox_inches='tight')
    fig_sur.savefig(os.path.join(plot_dir, f"{stem}_coef_sur.svg"), bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close('all')

    # Stem block comparison (pairwise models only)
    if sur_beta:
        fig_blocks = viz.plot_stem_blocks(gt.beta, sur_beta, STEM_PAIRS)
        fig_blocks.savefig(os.path.join(plot_dir, f"{stem}_stem_blocks.png"),
                           dpi=110, bbox_inches='tight')
        fig_blocks.savefig(os.path.join(plot_dir, f"{stem}_stem_blocks.svg"), bbox_inches='tight')
        plt.close('all')

    # Structured additive eval
    x_add, y_add_gt, k_labels = make_additive_dataset(
        wt_oh, gt, ks=[2, 3, 5], n_per_k=300, seed=k)
    delta_W  = make_ssm_deltas(wt_oh, gt)
    y_add_ssm = predict_ssm(x_add, delta_W)

    # Surrogate prediction from alpha (additive part only — model was cleared after training)
    sur_add_pred = np.einsum("nla,la->n",
                             x_add.astype(np.float64),
                             sur_alpha.astype(np.float64))
    fig_add = viz.plot_additive_eval(y_add_gt, sur_add_pred.astype(np.float32),
                                     k_labels, y_ssm=y_add_ssm)
    fig_add.savefig(os.path.join(plot_dir, f"{stem}_additive_eval.png"),
                    dpi=110, bbox_inches='tight')
    fig_add.savefig(os.path.join(plot_dir, f"{stem}_additive_eval.svg"), bbox_inches='tight')
    plt.close('all')

    # Structured pairwise eval
    x_pw, y_pw_gt, pair_labels, nuc_combos = make_pairwise_dataset(wt_oh, gt)
    y_pw_ssm = predict_ssm(x_pw, delta_W)
    sur_pw_pred = np.einsum("nla,la->n",
                            x_pw.astype(np.float64),
                            sur_alpha.astype(np.float64))
    if sur_beta:
        J = coefs.get("J")
        if J is not None:
            J_np = np.asarray(J, dtype=np.float64)  # (L, 4, L, 4)
            for ni in range(len(x_pw)):
                xi = x_pw[ni]
                for (ei, ej) in STEM_PAIRS:
                    if ei < J_np.shape[0] and ej < J_np.shape[2]:
                        sur_pw_pred[ni] += float(
                            xi[ei] @ J_np[ei, :, ej, :] @ xi[ej])

    fig_pw = viz.plot_pairwise_eval(y_pw_gt, sur_pw_pred.astype(np.float32),
                                    pair_labels, nuc_combos, STEM_PAIRS,
                                    y_ssm=y_pw_ssm)
    fig_pw.savefig(os.path.join(plot_dir, f"{stem}_pairwise_eval.png"),
                   dpi=110, bbox_inches='tight')
    fig_pw.savefig(os.path.join(plot_dir, f"{stem}_pairwise_eval.svg"), bbox_inches='tight')
    plt.close('all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_json",
                        default=os.path.join(_HERE, "outputs",
                                             "lib_size_spearman_results.json"))
    parser.add_argument("--out_coef",
                        default=os.path.join(_HERE, "outputs", "surrogate_coefs"))
    parser.add_argument("--plots", action="store_true",
                        help="Generate notebook-style coefficient+eval plots")
    parser.add_argument("--plot_dir",
                        default=os.path.join(_HERE, "outputs", "notebook_plots"))
    parser.add_argument("--gt_keys", nargs="+", default=["nonlin_additive_pairwise"],
                        help="GT keys to train on (default: nonlin_additive_pairwise)")
    parser.add_argument("--n_instances", type=int, default=N_INSTANCES,
                        help="Number of instances to run (default: all 10)")
    parser.add_argument("--mut_rates", nargs="+", type=int, default=MUT_RATES_PCT,
                        help="Mutation rates to run (default: all 3)")
    parser.add_argument("--lib_sizes", nargs="+", type=int, default=LIB_SIZES,
                        help="Library sizes to run (default: all 4)")
    args = parser.parse_args()

    if os.path.isfile(args.out_json):
        with open(args.out_json) as f:
            cache = json.load(f)
        print(f"[resume] loaded cache from {args.out_json}")
    else:
        cache = {}

    def save():
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(cache, f, indent=2)

    # ── Saturated ρ per instance (all GT keys)
    if "saturated" not in cache:
        cache["saturated"] = {}
    for k in range(args.n_instances):
        key = str(k)
        existing = cache["saturated"].get(key, {})
        # Detect old flat-float format or missing pairwise split → recompute
        needs_update = (
            key not in cache["saturated"]
            or any(not isinstance(v, dict) or "pairwise" not in v
                   for v in existing.values())
        )
        if needs_update:
            rho_sat = compute_saturated_rho(k)
            cache["saturated"][key] = rho_sat
            print(f"[saturated] instance {k:02d}  " +
                  "  ".join(f"{gk}=t2:{v['type2']:.4f}/pw:{v['pairwise']:.4f}"
                             for gk, v in rho_sat.items()))
            save()

    # ── Surrogate results
    # cache["surrogate"][gt_key][cfg_name][str(pct)][str(n_lib)]
    #   = {"rand": [...], "eval": [...]}
    if "surrogate" not in cache:
        cache["surrogate"] = {}

    for gt_key in args.gt_keys:
        if gt_key not in cache["surrogate"]:
            cache["surrogate"][gt_key] = {}

        for cfg_name in SURROGATE_CONFIGS:
            if cfg_name not in cache["surrogate"][gt_key]:
                cache["surrogate"][gt_key][cfg_name] = {}

            for pct in args.mut_rates:
                spct = str(pct)
                if spct not in cache["surrogate"][gt_key][cfg_name]:
                    cache["surrogate"][gt_key][cfg_name][spct] = {}

                for n_lib in args.lib_sizes:
                    sn = str(n_lib)
                    if sn not in cache["surrogate"][gt_key][cfg_name][spct]:
                        cache["surrogate"][gt_key][cfg_name][spct][sn] = {
                            "rand": [],
                            "type1_200": [], "type1_2000": [], "type1_20000": [],
                            "type2": [], "pairwise": []}
                    entry = cache["surrogate"][gt_key][cfg_name][spct][sn]
                    # migrate old formats
                    for old_key in ("eval", "type1", "ssm_lib", "type3"):
                        entry.pop(old_key, None)
                    entry.setdefault("type1_200",   [])
                    entry.setdefault("type1_2000",  [])
                    entry.setdefault("type1_20000", [])
                    entry.setdefault("type2",    [])
                    entry.setdefault("pairwise", [])

                    n_done = min(len(entry["rand"]), len(entry["pairwise"]))
                    if n_done >= args.n_instances:
                        print(f"[skip] {gt_key}  {cfg_name}  "
                              f"mut{pct}%  lib{n_lib:,}  ({n_done} done)")
                        continue

                    print(f"\n{'='*65}")
                    print(f"  {gt_key}  |  {cfg_name}  |  "
                          f"mut {pct}%  |  lib_size = {n_lib:,}")
                    print(f"{'='*65}")

                    for k in range(n_done, args.n_instances):
                        print(f"  instance {k:02d} …", end="", flush=True)
                        rr, rt1_200, rt1_2k, rt1_20k, rt2, rp, coefs = \
                            run_instance_condition(k, pct, n_lib, cfg_name, gt_key)
                        entry["rand"].append(rr)
                        entry["type1_200"].append(rt1_200)
                        entry["type1_2000"].append(rt1_2k)
                        entry["type1_20000"].append(rt1_20k)
                        entry["type2"].append(rt2)
                        entry["pairwise"].append(rp)
                        print(f"  ρ_rand={rr:+.4f}  ρ_t2={rt2:+.4f}  ρ_pw={rp:+.4f}",
                              flush=True)

                        save_coefs(args.out_coef, k, pct, n_lib,
                                   cfg_name, gt_key, coefs)

                        if args.plots and coefs:
                            generate_notebook_plots(
                                k, pct, n_lib, cfg_name, gt_key,
                                coefs, args.plot_dir)

                        save()

    print(f"\n[done] results saved to {args.out_json}")


if __name__ == "__main__":
    main()
