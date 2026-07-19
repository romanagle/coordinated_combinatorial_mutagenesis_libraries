"""fragility_comparison_table.py

Trains a nonlinear additive+pairwise (pairwise_GE) MAVENN surrogate on a
10%-mutation-rate 20k library and compares Spearman rho across four GT
function configurations:

  1. Nonlinear additive only       — init_additive_noWT + auto sigmoid
  2. Dense Gaussian pairwise       — additive + all-pairs Gaussian coupling
  3. MrnaRbpGroundTruth, b=3      — high fragility (steep sigmoid)
  4. MrnaRbpGroundTruth, b=0.5    — low fragility (gentle sigmoid)

Configs 3 and 4 use mRNA_RBP/gt_init.py::MrnaRbpGroundTruth with the
stem-loop sequence and structure from mRNA_RBP/generate_libraries.py.
"b" is the sigmoid sensitivity in y = a*sigma(b*latent + c) + d.
Higher b -> steeper threshold -> high fragility.

Usage:
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \\
        scripts/fragility_comparison_table.py \\
        --out_dir outputs/fragility_table
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT,
    init_pairwise_gaussian,
    init_sigmoid_nonlin,
    additive_affinity_noWT,
    pairwise_potts_energy,
    compute_gt_scores_for_library_potts,
    uniformize_by_histogram,
)
from mRNA_RBP.gt_init import MrnaRbpGroundTruth

# ---------------------------------------------------------------------------
# mRNA_RBP sequence and structural annotation (from generate_libraries.py)
# ---------------------------------------------------------------------------

SEQ             = 'AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA'
STEM_PAIRS      = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
MOTIF_POSITIONS = [17, 18, 19, 20, 21]

NUCS    = ['A', 'C', 'G', 'U']
_NUCS_A = np.array(list("ACGU"))

# Surrogate config used for all four GT conditions (nonlinear add+pairwise).
SURROGATE_CFG = {
    "gpmap":           "pairwise",
    "linearity":       "nonlinear",
    "regression_type": "GE",
    "noise":           "SkewedT",
    "noise_order":     2,
    "reg_strength":    0.1,
    "hidden_nodes":    50,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    parts  = []
    CHUNK  = 50_000
    for start in range(0, n_seqs, CHUNK):
        nc       = min(CHUNK, n_seqs - start)
        noise    = rng.random((nc, L))
        pos      = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X        = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


def _to_str(X):
    return np.array(["".join(_NUCS_A[np.argmax(x, axis=1)]) for x in X], dtype=object)


def _predict_chunked(model, x_str, chunk=500):
    parts = []
    for s in range(0, len(x_str), chunk):
        parts.append(np.asarray(model.x_to_yhat(x_str[s:s + chunk]), dtype=float).ravel())
    return np.concatenate(parts)


def _rho(yhat, ytrue):
    m = np.isfinite(yhat) & np.isfinite(ytrue)
    return float(spearmanr(yhat[m], ytrue[m])[0]) if m.sum() >= 3 else np.nan


def train_surrogate(X, y):
    N       = X.shape[0]
    bsz     = max(32, min(N // 150, 2048))
    lr      = 5e-4 * min(1.0, (20_000 / N) ** 0.5)
    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape, num_tasks=1,
        gpmap=SURROGATE_CFG["gpmap"],
        regression_type=SURROGATE_CFG["regression_type"],
        linearity=SURROGATE_CFG["linearity"],
        noise=SURROGATE_CFG["noise"],
        noise_order=SURROGATE_CFG["noise_order"],
        reg_strength=SURROGATE_CFG["reg_strength"],
        hidden_nodes=SURROGATE_CFG["hidden_nodes"],
        alphabet=NUCS, deduplicate=True, gpu=True,
    )
    model, _, test_df = wrapper.train(
        X, y,
        learning_rate=lr, epochs=1000, batch_size=bsz,
        early_stopping=True, patience=50, restore_best_weights=True,
        save_dir=None, verbose=0,
    )
    print(f"    trained  (lr={lr:.2e}  bsz={bsz})")
    return model, test_df


# ---------------------------------------------------------------------------
# Pairwise library: all 16 nt combos per stem pair, all other positions at WT
# ---------------------------------------------------------------------------

def build_pairwise_library(wt_onehot, stem_pairs):
    """Return X (N,L,4) and pair_labels list for all 16 combos × each stem pair."""
    L      = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1).astype(int)
    seqs, labels = [], []
    for (si, sj) in stem_pairs:
        for ni in range(4):
            for nj in range(4):
                row      = wt_idx.copy()
                row[si]  = ni
                row[sj]  = nj
                seqs.append(row)
                labels.append((si, sj, ni, nj))
    nuc_ids = np.stack(seqs).astype(np.int32)
    X = np.eye(4, dtype=np.float32)[nuc_ids]
    return X, labels


# ---------------------------------------------------------------------------
# GT scoring wrappers
# ---------------------------------------------------------------------------

def score_additive_only(X, W_mut, mut_map, b0, nonlin_kw):
    """Config 1: nonlin additive, no pairwise."""
    gt = compute_gt_scores_for_library_potts(
        X, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kw,
    )
    return gt["nonlin_additive"].astype(float)


def score_gaussian_pairwise(X, W_mut, mut_map, b0, nonlin_kw, edges_g, J_g):
    """Config 2: nonlin additive + dense Gaussian pairwise."""
    gt = compute_gt_scores_for_library_potts(
        X, W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kw,
        edges=edges_g, J=J_g,
    )
    return gt["nonlin_additive_pairwise"].astype(float)


def score_mrna_rbp(X, gt_obj):
    """Configs 3 and 4: MrnaRbpGroundTruth (additive + WC-aware stem pairwise)."""
    return gt_obj.score_all(X)["nonlin_additive_pairwise"].astype(float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--out_dir", default="outputs/fragility_table")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    seq = SEQ.upper().replace("T", "U")
    oh_seq       = rna_to_one_hot(seq)
    wt_onehot, _ = remove_padding(oh_seq)
    L            = wt_onehot.shape[0]
    exact_mut_count = max(1, round(0.10 * L))

    print(f"Sequence : {seq}")
    print(f"L={L}  exact_mut_count={exact_mut_count} (10% mut rate)")
    print(f"Stem pairs      : {STEM_PAIRS}")
    print(f"Motif positions : {MOTIF_POSITIONS}")

    # ── Shared additive GT params (configs 1 and 2) ───────────────────────
    rng = np.random.default_rng(args.seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)

    # Gaussian pairwise (config 2)
    edges_gauss, J_gauss = init_pairwise_gaussian(
        rng, wt_onehot, sigma=0.3, wt_rowcol_zero=True, signed=False,
    )

    # ── Reference library to calibrate sigmoid (configs 1 and 2) ─────────
    print("\n[calibration] generating 50k reference library …")
    rng_ref = np.random.default_rng(args.seed + 1)
    X_ref   = generate_library(wt_onehot, 50_000, exact_mut_count, rng_ref)

    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kw = init_sigmoid_nonlin(s_add_ref)

    s_pair_gauss_ref = pairwise_potts_energy(X_ref, edges_gauss, J_gauss, b=0.0).reshape(-1)
    nonlin_kw_gauss  = dict(nonlin_kw)
    nonlin_kw_gauss["_norm_std_addpair"] = float(np.std(s_add_ref + s_pair_gauss_ref)) + 1e-8
    del X_ref, s_add_ref, s_pair_gauss_ref

    # ── MrnaRbpGroundTruth instances (configs 3 and 4) ────────────────────
    # Same sequence/structure, same additive weights (same seed), only b differs.
    gt_high = MrnaRbpGroundTruth(
        seq, STEM_PAIRS, MOTIF_POSITIONS,
        motif_sigma=3.0, bg_sigma=0.5, stem_additive_sigma=1.5,
        a=1.0, b=3.0, c=1.5, d=0.0, seed=args.seed,
    )
    gt_low = MrnaRbpGroundTruth(
        seq, STEM_PAIRS, MOTIF_POSITIONS,
        motif_sigma=3.0, bg_sigma=0.5, stem_additive_sigma=1.5,
        a=1.0, b=0.5, c=1.5, d=0.0, seed=args.seed,
    )
    print(f"\nMrnaRbpGroundTruth (b=3):   {gt_high}")
    print(f"MrnaRbpGroundTruth (b=0.5): {gt_low}")

    # ── Training library (20k) and eval pool (100k) ───────────────────────
    print("\n[library] generating 20k training library …")
    rng_lib  = np.random.default_rng(args.seed + 2)
    X_lib    = generate_library(wt_onehot, 20_000, exact_mut_count, rng_lib)

    print("[library] generating 100k eval pool …")
    rng_pool = np.random.default_rng(args.seed + 3)
    X_pool   = generate_library(wt_onehot, 100_000, exact_mut_count, rng_pool)

    # Pairwise library: all 16 nt combos per stem pair (WT elsewhere)
    X_pair, _ = build_pairwise_library(wt_onehot, STEM_PAIRS)
    print(f"[library] pairwise library: {len(X_pair)} seqs "
          f"({len(STEM_PAIRS)} pairs × 16 combos)")

    # ── Config definitions ─────────────────────────────────────────────────
    CONFIGS = [
        {
            "id":    1,
            "label": "Nonlinear Additive Only",
            "score_fn": lambda X: score_additive_only(X, W_mut, mut_map, b0, nonlin_kw),
        },
        {
            "id":    2,
            "label": "Gaussian Pairwise (dense, not sparse)",
            "score_fn": lambda X: score_gaussian_pairwise(
                X, W_mut, mut_map, b0, nonlin_kw_gauss, edges_gauss, J_gauss),
        },
        {
            "id":    3,
            "label": "MrnaRbpGroundTruth, High Fragility (b=3)",
            "score_fn": lambda X: score_mrna_rbp(X, gt_high),
        },
        {
            "id":    4,
            "label": "MrnaRbpGroundTruth, Low Fragility (b=0.5)",
            "score_fn": lambda X: score_mrna_rbp(X, gt_low),
        },
    ]

    # ── Per-config training and evaluation ────────────────────────────────
    rows = []
    for cfg in CONFIGS:
        print(f"\n{'='*62}")
        print(f"Config {cfg['id']}: {cfg['label']}")

        score_fn = cfg["score_fn"]

        # Score training library
        y_train = score_fn(X_lib).reshape(-1, 1)
        print(f"  y_train: [{y_train.min():.4f}, {y_train.max():.4f}]  "
              f"std={y_train.std():.4f}")

        if y_train.std() < 1e-8:
            print("  SKIP: zero variance in GT scores")
            rows.append({"id": cfg["id"], "label": cfg["label"],
                         "rho_random": float("nan"), "rho_eval": float("nan"),
                         "rho_pairwise": float("nan")})
            continue

        # Score eval pool and uniformize
        y_pool = score_fn(X_pool)
        y_eval_scores, keep_idx = uniformize_by_histogram(
            y_pool, X=None, n_bins=200, clip_hi=98, target_n=5_000, seed=args.seed,
        )
        X_eval = X_pool[keep_idx]
        y_eval = y_eval_scores.astype(float)
        print(f"  eval library: {len(y_eval):,} seqs  "
              f"[{y_eval.min():.4f}, {y_eval.max():.4f}]")

        # Score pairwise library (all 16 combos per stem pair)
        y_pair = score_fn(X_pair).astype(float)
        print(f"  pairwise library GT: [{y_pair.min():.4f}, {y_pair.max():.4f}]")

        # Train pairwise_GE surrogate
        print("  Training pairwise_GE surrogate …")
        model, test_df = train_surrogate(X_lib, y_train)

        # Spearman on MAVENN's random held-out test split
        cols  = list(test_df.columns)
        x_col = "x" if "x" in cols else "X"
        y_col = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        x_test    = np.asarray(test_df[x_col])
        y_test    = np.asarray(test_df[y_col], dtype=float).ravel()
        yhat_test = _predict_chunked(model, x_test)
        rho_rand  = _rho(yhat_test, y_test)

        # Spearman on uniformized eval library (type-2-like)
        yhat_eval = _predict_chunked(model, _to_str(X_eval))
        rho_eval  = _rho(yhat_eval, y_eval)

        # Spearman on pairwise library (stem epistasis)
        yhat_pair = _predict_chunked(model, _to_str(X_pair))
        rho_pair  = _rho(yhat_pair, y_pair)

        print(f"  rho_random={rho_rand:+.4f}   rho_eval={rho_eval:+.4f}   rho_pairwise={rho_pair:+.4f}")
        rows.append({
            "id":           cfg["id"],
            "label":        cfg["label"],
            "rho_random":   rho_rand,
            "rho_eval":     rho_eval,
            "rho_pairwise": rho_pair,
        })

    # ── Print table ────────────────────────────────────────────────────────
    W = 92
    print(f"\n{'='*W}")
    print("SPEARMAN rho — pairwise_GE surrogate, 10% mut rate, 20k training library")
    print(f"{'='*W}")
    print(f"{'#':<3}  {'GT landscape':<46}  {'rho_random':>10}  {'rho_eval':>10}  {'rho_pairwise':>12}")
    print("-" * W)
    for r in rows:
        rr = f"{r['rho_random']:>10.4f}"   if np.isfinite(r['rho_random'])   else f"{'NaN':>10}"
        re = f"{r['rho_eval']:>10.4f}"     if np.isfinite(r['rho_eval'])     else f"{'NaN':>10}"
        rp = f"{r['rho_pairwise']:>12.4f}" if np.isfinite(r['rho_pairwise']) else f"{'NaN':>12}"
        print(f"{r['id']:<3}  {r['label']:<46}  {rr}  {re}  {rp}")
    print(f"{'='*W}")
    print("\nrho_random   : Spearman rho on MAVENN held-out random test split")
    print("rho_eval     : Spearman rho on uniformized eval library (~5k seqs)")
    print("rho_pairwise : Spearman rho on all 16 nt combos per stem pair (WT elsewhere)")

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "fragility_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "rho_random", "rho_eval", "rho_pairwise"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[saved] {csv_path}")

    _plot_table(rows, os.path.join(args.out_dir, "fragility_comparison.png"))


def _plot_table(rows, path):
    labels   = [r["label"]        for r in rows]
    rho_eval = [r["rho_eval"]     for r in rows]
    rho_pair = [r["rho_pairwise"] for r in rows]

    short_labels = [
        "Nonlinear\nAdditive\nOnly",
        "Gaussian\nPairwise",
        "MrnaRbpGT\nHigh Frag\n(b=3)",
        "MrnaRbpGT\nLow Frag\n(b=0.5)",
    ]

    BLUE  = "#4878CF"
    GREEN = "#6ACC65"
    GRAY  = "#999999"

    x = np.arange(len(rows))
    w = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    panel_data = [
        (axes[0], rho_eval, BLUE,  "Uniformized eval library\n(random mut-rate pool)"),
        (axes[1], rho_pair, GREEN, "Pairwise library\n(all 16 nt combos per stem pair)"),
    ]

    for ax, vals, color, title in panel_data:
        bars = ax.bar(x, vals, w, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.012,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                )
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel("Spearman ρ", fontsize=11)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    fig.suptitle(
        "pairwise_GE surrogate  ·  10% mut-rate  ·  20k training library",
        fontsize=11, y=1.02,
    )

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


if __name__ == "__main__":
    main()
