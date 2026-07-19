"""scripts/plot_eval_library_histograms.py

Generates Type 1 (3 per-lib-size eval libraries) and Type 2 (1 shared 20K
eval library) for each sequence, then plots score-distribution histograms
with bars coloured/stacked by mutation rate.

Type 1 target sizes match MAVE-NN's 20% test split:
    lib_size=200   → 40 seqs,  n_bins=40
    lib_size=2000  → 400 seqs, n_bins=200
    lib_size=20000 → 4000 seqs,n_bins=200

Type 2: 20 000 seqs, n_bins=200, sequences drawn from all 3 mutation rates.

Histograms are averaged across sequences and saved to:
    outputs/eval_library_histograms/type1_lib{N}_{gt_key}.png
    outputs/eval_library_histograms/type2_{gt_key}.png

Usage:
    python scripts/plot_eval_library_histograms.py \
        --seqs_file outputs/lib_size_saturated/sequences.txt \
        --out_dir   outputs/eval_library_histograms
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mRNA_RBP"))
sys.path.insert(0, "/home/nagle/final_version/squid-nn")
sys.path.append("/home/nagle/final_version/squid-manuscript/squid")
sys.path.append("/home/nagle/final_version/residualbind")

from seq_utils import rna_to_one_hot
from ground_truth import (
    init_additive_noWT, init_sigmoid_nonlin,
    additive_affinity_noWT, pairwise_potts_energy,
    apply_global_nonlin, uniformize_by_histogram,
)
from gt_init import init_wc_pairwise_sparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NONLIN_NAME     = "sigmoid"
MUT_RATES_PCT   = [5, 10, 25]          # percent
LIB_SIZES       = [200, 2_000, 20_000]
TYPE2_TARGET_N  = 20_000
N_POOL_PER_RATE = 300_000

GT_KEY     = "nonlin_additive_pairwise"
MUT_COLORS = {5: "#4C72B0", 10: "#DD8452", 25: "#55A868"}


# ---------------------------------------------------------------------------
# Pool generation
# ---------------------------------------------------------------------------

def _exact_mut_count(pct, L):
    return max(1, round(pct / 100.0 * L))


def _gen_nuc_ids(wt_nuc_ids, n, exact_mut_count, rng):
    L     = len(wt_nuc_ids)
    CHUNK = 50_000
    nids  = np.tile(wt_nuc_ids[None], (n, 1)).astype(np.uint8)
    for start in range(0, n, CHUNK):
        nc       = min(CHUNK, n - start)
        noise    = rng.random((nc, L))
        pos      = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        n_idx    = np.repeat(np.arange(nc), exact_mut_count)
        p_idx    = pos.ravel()
        wt_at    = wt_nuc_ids[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count, dtype=np.uint8)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc).astype(np.uint8)
        nids[start:start + nc][n_idx, p_idx] = new_nucs
    return nids


def _score_nuc_ids(nids, W_mut, mut_map, b0, edges, J,
                   nonlin_kwargs, ref_std, ref_std_ap, wt_nl,
                   gt_keys, A=4, CHUNK=50_000):
    N = len(nids)
    scores = {k: np.empty(N, dtype=np.float32) for k in gt_keys}
    for start in range(0, N, CHUNK):
        end     = min(start + CHUNK, N)
        X_chunk = np.eye(A, dtype=np.float32)[nids[start:end]]
        sa      = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp      = pairwise_potts_energy(X_chunk, edges, J, b=0.0).reshape(-1)
        sap     = sa + sp
        for k in gt_keys:
            if k == "additive":
                scores[k][start:end] = sa
            elif k == "additive_pairwise":
                scores[k][start:end] = sap
            elif k == "nonlin_additive":
                v = apply_global_nonlin(sa / ref_std, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                scores[k][start:end] = v
            elif k == "nonlin_additive_pairwise":
                v = apply_global_nonlin(sap / ref_std_ap, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
                scores[k][start:end] = v
    return scores


# ---------------------------------------------------------------------------
# Mixed eval library builder
# ---------------------------------------------------------------------------

def build_mixed_eval_library(wt_onehot, gt_params, nonlin_kwargs,
                              mut_rates_exact, target_n,
                              seed=42, n_pool_per_rate=N_POOL_PER_RATE):
    """
    Generate a uniform eval library scored under nonlin_additive_pairwise GT.

    Sequences are drawn from all mutation rates, pooled, scored with
    nonlin_additive_pairwise, then uniformized jointly.

    Returns:
        {"y_eval": (M,), "nuc_ids": (M, L), "rate_labels": (M,)}
    where rate_labels[i] is the mut_rate_pct of sequence i.
    n_bins = min(200, target_n).
    """
    W_mut   = gt_params["W_mut"]
    mut_map = gt_params["mut_map"]
    b0      = gt_params["b"]
    edges   = gt_params["edges"]
    J       = gt_params["J"]
    L, A    = wt_onehot.shape
    wt_idx  = np.argmax(wt_onehot, axis=1).astype(np.uint8)

    ref_std_ap = float(nonlin_kwargs["_norm_std_addpair"])
    wt_nl      = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))
    n_bins     = min(200, target_n)

    # Generate pool from each mutation rate
    all_nids, all_labels = [], []
    rng = np.random.default_rng(seed)
    for pct, emc in mut_rates_exact:
        nids = _gen_nuc_ids(wt_idx, n_pool_per_rate, emc, rng)
        all_nids.append(nids)
        all_labels.append(np.full(n_pool_per_rate, pct, dtype=np.uint8))

    combined_nids   = np.concatenate(all_nids,   axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    del all_nids, all_labels

    # Score combined pool under nonlin_additive_pairwise
    CHUNK = 50_000
    N     = len(combined_nids)
    scores = np.empty(N, dtype=np.float32)
    for start in range(0, N, CHUNK):
        end     = min(start + CHUNK, N)
        X_chunk = np.eye(A, dtype=np.float32)[combined_nids[start:end]]
        sa = additive_affinity_noWT(X_chunk, W_mut, mut_map, b=b0).reshape(-1)
        sp = pairwise_potts_energy(X_chunk, edges, J, b=0.0).reshape(-1)
        sap = sa + sp
        v = apply_global_nonlin(sap / ref_std_ap, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl
        scores[start:end] = v

    # Uniformize jointly
    y_uni, keep_idx = uniformize_by_histogram(
        scores.astype(float), X=None,
        n_bins=n_bins, clip_hi=98,
        target_n=target_n, seed=seed,
    )

    return {
        "y_eval":      y_uni,
        "nuc_ids":     combined_nids[keep_idx],
        "rate_labels": combined_labels[keep_idx],
    }


# ---------------------------------------------------------------------------
# GT initializer (mRNA_RBP params)
# ---------------------------------------------------------------------------

def init_gt(seq, seq_seed, eval_mut_count):
    wt_oh  = rna_to_one_hot(seq)
    L, A   = wt_oh.shape
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)

    rng_gt = np.random.default_rng(seq_seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_oh, sigma=0.5, l1_w=0.03, bias=0.0)
    edges, J = init_wc_pairwise_sparse(
        rng_gt, wt_oh, seq, p_edge=0.15, sigma_P=0.30, l1_P=0.40,
        edge_seed=int(seq_seed),
    )
    gt_params = {"W_mut": W_mut, "mut_map": mut_map, "b": b0, "edges": edges, "J": J}

    # Calibration pool at eval_mut_count
    rng_ref = np.random.default_rng(seq_seed + 1)
    calib   = _gen_nuc_ids(wt_idx, 200_000, eval_mut_count, rng_ref)
    X_cal   = np.eye(A, dtype=np.float32)[calib]
    sa_ref  = additive_affinity_noWT(X_cal, W_mut, mut_map, b=b0).reshape(-1)
    sp_ref  = pairwise_potts_energy(X_cal, edges, J, b=0.0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(sa_ref)
    nonlin_kwargs["_norm_std_addpair"] = float(np.std(sa_ref + sp_ref)) + 1e-8

    return wt_oh, gt_params, nonlin_kwargs


# ---------------------------------------------------------------------------
# Saturated mutagenesis library
# ---------------------------------------------------------------------------

def build_saturated_library(wt_onehot, gt_params, nonlin_kwargs):
    """
    Generate all 3×L single-mutant sequences, score with nonlin_additive_pairwise.

    Returns {"y_scores": (3L,), "positions": (3L,), "alt_nucs": (3L,)}
    """
    W_mut      = gt_params["W_mut"]
    mut_map    = gt_params["mut_map"]
    b0         = gt_params["b"]
    edges      = gt_params["edges"]
    J          = gt_params["J"]
    L, A       = wt_onehot.shape
    wt_idx     = np.argmax(wt_onehot, axis=1)
    ref_std_ap = float(nonlin_kwargs["_norm_std_addpair"])
    wt_nl      = float(apply_global_nonlin(np.array([[0.0]]), NONLIN_NAME, nonlin_kwargs))

    X_sat, positions, alt_nucs = [], [], []
    for pos in range(L):
        for nuc in range(A):
            if nuc == wt_idx[pos]:
                continue
            x = wt_onehot.copy()
            x[pos] = 0
            x[pos, nuc] = 1
            X_sat.append(x)
            positions.append(pos)
            alt_nucs.append(nuc)

    X_sat = np.stack(X_sat).astype(np.float32)   # (3L, L, A)
    sa = additive_affinity_noWT(X_sat, W_mut, mut_map, b=b0).reshape(-1)
    sp = pairwise_potts_energy(X_sat, edges, J, b=0.0).reshape(-1)
    sap = sa + sp
    y = apply_global_nonlin(sap / ref_std_ap, NONLIN_NAME, nonlin_kwargs).reshape(-1) - wt_nl

    return {
        "y_scores":  y.astype(float),
        "positions": np.array(positions, dtype=np.int32),
        "alt_nucs":  np.array(alt_nucs,  dtype=np.int32),
    }


def plot_saturated_histogram(sat_lib, title, out_path, n_bins_hist=40):
    """
    Histogram of saturated mutagenesis scores (3L single mutants).
    Bars are coloured by mutated nucleotide (A/C/G/U).
    """
    NUC_LABELS = ["A", "C", "G", "U"]
    NUC_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    y       = sat_lib["y_scores"]
    alt_nuc = sat_lib["alt_nucs"]
    lo, hi  = np.nanpercentile(y, 1), np.nanpercentile(y, 99)
    bins    = np.linspace(lo, hi, n_bins_hist + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bottoms = np.zeros(n_bins_hist)
    for nuc_idx in range(4):
        mask      = alt_nuc == nuc_idx
        counts, _ = np.histogram(y[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=bins[1] - bins[0],
               bottom=bottoms, color=NUC_COLORS[nuc_idx], alpha=0.85,
               align="edge", label=f"→{NUC_LABELS[nuc_idx]}  ({mask.sum():,})")
        bottoms += counts

    ax.set_xlabel("Nonlin Add+Pair GT score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9, loc="upper left", title="Mutated to")
    ax.text(0.98, 0.97, f"N={len(y):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Histogram plotting
# ---------------------------------------------------------------------------

def plot_histogram(lib, title, out_path, n_bins_hist=40):
    """
    lib: {"y_eval": (M,), "rate_labels": (M,)}
    Stacked histogram of nonlin_additive_pairwise GT scores, coloured by mutation rate.
    """
    y      = lib["y_eval"]
    labels = lib["rate_labels"]
    lo, hi = np.nanpercentile(y, 1), np.nanpercentile(y, 99)
    bins   = np.linspace(lo, hi, n_bins_hist + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bottoms = np.zeros(n_bins_hist)
    for pct in MUT_RATES_PCT:
        mask      = labels == pct
        counts, _ = np.histogram(y[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=bins[1] - bins[0],
               bottom=bottoms, color=MUT_COLORS[pct], alpha=0.85,
               align="edge", label=f"{pct}% mut  ({mask.sum():,} seqs)")
        bottoms += counts

    ax.set_xlabel("Nonlin Add+Pair GT score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.98, 0.97, f"N={len(y):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs_file",
                        default="outputs/lib_size_saturated/sequences.txt")
    parser.add_argument("--out_dir",
                        default="outputs/eval_library_histograms")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.seqs_file) as f:
        sequences = [l.strip().upper().replace("T", "U")
                     for l in f if l.strip() and not l.startswith("#")]
    print(f"Loaded {len(sequences)} sequences")

    # Calibration uses median mutation rate (10%) for sigmoid fit
    calib_mut_count = _exact_mut_count(10, len(rna_to_one_hot(sequences[0])))

    # mut_rates for mixed pool
    L_seq = len(rna_to_one_hot(sequences[0]))
    mut_rates_exact = [(pct, _exact_mut_count(pct, L_seq)) for pct in MUT_RATES_PCT]
    print(f"L={L_seq}  mut_rates_exact={mut_rates_exact}")

    for seq_idx, seq in enumerate(sequences):
        seq_seed = args.seed + seq_idx * 1_000_007
        print(f"\n[seq {seq_idx}]  {seq[:40]}…")

        wt_oh, gt_params, nonlin_kwargs = init_gt(seq, seq_seed, calib_mut_count)
        seq_dir = os.path.join(args.out_dir, f"seq_{seq_idx:02d}")

        # Saturated mutagenesis library
        sat_lib  = build_saturated_library(wt_oh, gt_params, nonlin_kwargs)
        L        = wt_oh.shape[0]
        plot_saturated_histogram(sat_lib,
            title    = f"Seq {seq_idx} — Saturated mutagenesis (N={3*L}, all single mutants)",
            out_path = os.path.join(seq_dir, "saturated.png"))

        # Type 1 — one library per lib_size
        for lib_size in LIB_SIZES:
            target_n = max(1, int(0.2 * lib_size))
            print(f"  Type1  lib_size={lib_size:,}  target_n={target_n}")
            lib = build_mixed_eval_library(
                wt_oh, gt_params, nonlin_kwargs,
                mut_rates_exact=mut_rates_exact,
                target_n=target_n,
                seed=seq_seed + lib_size,
            )
            title    = f"Seq {seq_idx} — Type 1  lib_size={lib_size:,}  (N={target_n})"
            out_path = os.path.join(seq_dir, f"type1_lib{lib_size}.png")
            plot_histogram(lib, title, out_path)

        # Type 2 — one shared library
        print(f"  Type2  target_n={TYPE2_TARGET_N:,}")
        lib2 = build_mixed_eval_library(
            wt_oh, gt_params, nonlin_kwargs,
            mut_rates_exact=mut_rates_exact,
            target_n=TYPE2_TARGET_N,
            seed=seq_seed + 999_999,
        )
        title    = f"Seq {seq_idx} — Type 2  (N={TYPE2_TARGET_N:,}, all mut rates)"
        out_path = os.path.join(seq_dir, "type2.png")
        plot_histogram(lib2, title, out_path)

    print("\n[done]")


if __name__ == "__main__":
    main()
