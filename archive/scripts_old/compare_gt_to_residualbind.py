"""compare_gt_to_residualbind.py

Generate a 20k library (4 mutations, 41-nt WT sequence) and compare:
  - ResidualBind predictions (anchored so WT = 0)
  - All 4 synthetic GT score distributions (additive, add+pair, nonlin variants)

All scores are shifted so WT = 0 to enable apples-to-apples comparison.

Usage:
    python scripts/compare_gt_to_residualbind.py \\
        --experiment RNCMPT00111 \\
        --seq AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --seed 42 \\
        --out outputs/gt_vs_residualbind.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/residualbind')
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')

from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT, init_pairwise_potts_optionA, init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts, additive_affinity_noWT,
    pairwise_potts_energy,
)

import residualbind as rb
import helper


# ---------------------------------------------------------------------------
# Library generator (same as noisy_panel.py)
# ---------------------------------------------------------------------------

def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L, wt_idx = wt_onehot.shape[0], np.argmax(wt_onehot, axis=1)
    parts, CHUNK = [], 50_000
    for start in range(0, n_seqs, CHUNK):
        nc    = min(CHUNK, n_seqs - start)
        pos   = np.argpartition(rng.random((nc, L)), exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_idx[p_idx], rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# ResidualBind scorer
# ---------------------------------------------------------------------------

def score_with_residualbind(X_lib, wt_onehot, experiment, batch_size=512):
    """Return ResidualBind scores anchored at WT = 0."""
    data_path   = Path.home() / 'residualbind' / 'data' / 'RNAcompete_2013' / 'rnacompete2013.h5'
    save_path   = '/home/nagle/residualbind/weights/log_norm_seq'
    weights_path = os.path.join(save_path, experiment + '_weights.hdf5')

    rbp_index = helper.find_experiment_index(data_path, experiment)
    train, _, _ = helper.load_rnacompete_data(
        data_path, ss_type='seq', normalization='log_norm', rbp_index=rbp_index
    )
    input_shape = list(train['inputs'].shape)[1:]

    model = rb.ResidualBind(input_shape, num_class=1, weights_path=weights_path)
    model.load_weights()
    print(f"[residualbind] loaded {experiment}")

    # Score library (sequences are already 41-nt, no padding needed)
    scores_lib = model.predict(X_lib,  batch_size=batch_size).ravel()
    score_wt   = model.predict(wt_onehot[np.newaxis], batch_size=1).ravel()[0]
    scores_anchored = scores_lib - score_wt   # WT → 0
    print(f"[residualbind] WT score={score_wt:.4f}  "
          f"library range=[{scores_anchored.min():.4f}, {scores_anchored.max():.4f}]  "
          f"std={scores_anchored.std():.4f}")
    return scores_anchored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="RNCMPT00111")
    parser.add_argument("--seq",        default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--lib_size",   type=int, default=20_000)
    parser.add_argument("--mut_rate",   type=int, default=4)
    parser.add_argument("--out",        default="outputs/gt_vs_residualbind.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seq = args.seq.upper().replace("T", "U")
    oh_seq               = rna_to_one_hot(seq)
    wt_onehot, _         = remove_padding(oh_seq)
    L                    = wt_onehot.shape[0]
    exact_mut            = min(args.mut_rate, max(1, L // 2))
    print(f"[init] L={L}  exact_mut={exact_mut}  lib_size={args.lib_size:,}")

    # ── Generate library (shared for all GTs)
    rng     = np.random.default_rng(args.seed)
    X_lib   = generate_library(wt_onehot, args.lib_size, exact_mut, rng)
    print(f"[library] generated {len(X_lib):,} sequences")

    # ── Synthetic GT params (seed=42, matching noisy_panel.py)
    rng_gt  = np.random.default_rng(args.seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng_gt, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    # Calibrate sigmoid on a large reference library
    rng_ref   = np.random.default_rng(args.seed + 1)
    X_ref     = generate_library(wt_onehot, 200_000, exact_mut, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nk        = init_sigmoid_nonlin(s_add_ref)
    s_pair_ref = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
    nk["_norm_std_addpair"] = float(np.std(s_add_ref + s_pair_ref)) + 1e-8
    del X_ref, s_add_ref, s_pair_ref
    print(f"[sigmoid] ref_std={nk['_norm_std']:.4f}  ref_std_addpair={nk['_norm_std_addpair']:.4f}")

    # ── Score library with synthetic GTs
    gt_scores = compute_gt_scores_for_library_potts(
        X_lib,
        W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nk,
        edges=edges, J=J,
    )
    for k, v in gt_scores.items():
        print(f"  GT {k:<30s}  [{v.min():.4f}, {v.max():.4f}]  std={v.std():.4f}")

    # ── Score library with ResidualBind
    rb_scores = score_with_residualbind(X_lib, wt_onehot, args.experiment)

    # ── Plot overlaid distributions
    CURVES = [
        ("ResidualBind",              rb_scores,                        "#e41a1c", 2.5),
        ("Additive GT",               gt_scores["additive"],            "#377eb8", 1.5),
        ("Add+Pair GT",               gt_scores["additive_pairwise"],   "#4daf4a", 1.5),
        ("Nonlin Add GT",             gt_scores["nonlin_additive"],     "#984ea3", 1.5),
        ("Nonlin Add+Pair GT",        gt_scores["nonlin_additive_pairwise"], "#ff7f00", 2.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: overlaid KDE / histogram
    ax = axes[0]
    for label, scores, color, lw in CURVES:
        # Normalise to [0,1] for display so different scales are comparable
        lo, hi = np.percentile(scores, 1), np.percentile(scores, 99)
        scores_n = np.clip((scores - lo) / max(hi - lo, 1e-8), 0, 1)
        ax.hist(scores_n, bins=80, density=True, histtype="step",
                color=color, lw=lw, label=label, alpha=0.85)
    ax.set_xlabel("Score (normalised to [0, 1] per GT)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Score distributions — 20k library, 4 mutations", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Right: raw score distributions (actual units)
    ax = axes[1]
    for label, scores, color, lw in CURVES:
        ax.hist(scores, bins=80, density=True, histtype="step",
                color=color, lw=lw, label=label, alpha=0.85)
    ax.set_xlabel("Raw score (WT anchored at 0)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Raw score distributions", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"{args.experiment} — ResidualBind vs synthetic GT functions", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] {args.out}")

    # Also print statistics
    print("\n--- Distribution statistics ---")
    print(f"{'GT':<30s}  {'mean':>8s}  {'std':>8s}  {'p10':>8s}  {'p50':>8s}  {'p90':>8s}")
    for label, scores, _, _ in CURVES:
        p10, p50, p90 = np.percentile(scores, [10, 50, 90])
        print(f"{label:<30s}  {scores.mean():>8.4f}  {scores.std():>8.4f}  "
              f"{p10:>8.4f}  {p50:>8.4f}  {p90:>8.4f}")


if __name__ == "__main__":
    main()
