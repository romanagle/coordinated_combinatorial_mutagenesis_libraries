"""train_surrogate_gt.py

Pipeline:
  1. Generate a 20k library from the main sequence using the existing
     nonlinear additive+pairwise GT function.
  2. Train a MAVENN nonlinear additive+pairwise surrogate on those scores.
  3. Save the trained model and parameters.
  4. Produce a SurrogateGT object whose .score(X) method can be called on
     any library of one-hot sequences with the same sequence length.

Usage:
    python scripts/train_surrogate_gt.py \\
        --seq  AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA \\
        --seed 42 \\
        --out_dir outputs/surrogate_gt

Outputs (all in out_dir/):
    surrogate_gt.pkl       – pickled SurrogateGT (load and call .score)
    params_additive.npy    – (L,4) additive weight matrix
    params_pairwise.npy    – (L,4,L,4) pairwise weight tensor
    additive_weights.png   – 4xL heatmap of learned additive weights
    y_vs_yhat.png          – surrogate fit quality on training split
    library.npz            – the 20k library used for training (X, y)
"""

import argparse
import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')
sys.path.append('/home/nagle/final_version/squid-nn/squid')
sys.path.append('/home/nagle/final_version/residualbind')

import squid.surrogate_zoo
from scipy.stats import spearmanr

from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT,
    init_pairwise_potts_optionA,
    init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts,
    additive_affinity_noWT,
)

NUCS = ['A', 'C', 'G', 'U']

SURROGATE_CFG = {
    "gpmap":            "pairwise",
    "linearity":        "nonlinear",
    "regression_type":  "GE",
    "noise":            "Gaussian",
    "noise_order":      2,
    "reg_strength":     0.005,
    "hidden_nodes":     10,
}


# ---------------------------------------------------------------------------
# Library generation
# ---------------------------------------------------------------------------

def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    """Mutate exactly `exact_mut_count` positions per sequence."""
    L     = wt_onehot.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    CHUNK  = 50_000
    parts  = []
    for start in range(0, n_seqs, CHUNK):
        nc    = min(CHUNK, n_seqs - start)
        noise = rng.random((nc, L))
        pos   = np.argpartition(noise, exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        wt_at    = wt_idx[p_idx]
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_at, rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------

def onehot_to_str(x):
    return "".join(np.array(list("ACGU"))[np.argmax(x, axis=1)])


def train_surrogate(X, y, out_dir):
    """Train MAVENN nonlinear additive+pairwise surrogate. Returns mavenn_model."""
    cfg  = SURROGATE_CFG
    N    = X.shape[0]
    bsz  = max(32, min(N // 150, 2048))
    lr   = 5e-4 * min(1.0, (20_000 / N) ** 0.5)

    wrapper = squid.surrogate_zoo.SurrogateMAVENN(
        X.shape,
        num_tasks=1,
        gpmap=cfg["gpmap"],
        regression_type=cfg["regression_type"],
        linearity=cfg["linearity"],
        noise=cfg["noise"],
        noise_order=cfg["noise_order"],
        reg_strength=cfg["reg_strength"],
        hidden_nodes=cfg["hidden_nodes"],
        alphabet=NUCS,
        deduplicate=True,
        gpu=True,
    )

    model, mave_df, test_df = wrapper.train(
        X, y,
        learning_rate=lr,
        epochs=1000,
        batch_size=bsz,
        early_stopping=True,
        patience=50,
        restore_best_weights=True,
        save_dir=None,
        verbose=1,
    )

    # Spearman on held-out random split
    try:
        cols   = list(test_df.columns)
        X_col  = "x" if "x" in cols else "X"
        y_col  = "y" if "y" in cols else next(c for c in cols if c.startswith("y"))
        x_str  = np.asarray(test_df[X_col])
        y_test = np.asarray(test_df[y_col], dtype=float).ravel()
        y_hat  = np.asarray(model.x_to_yhat(x_str), dtype=float).ravel()
        mask   = np.isfinite(y_test) & np.isfinite(y_hat)
        rho, _ = spearmanr(y_test[mask], y_hat[mask])
        print(f"\n[train] Spearman ρ on test split: {rho:+.4f}")
    except Exception as e:
        print(f"[warn] rho calculation failed: {e}")
        rho = np.nan

    # y vs yhat plot
    _plot_y_vs_yhat(y_test, y_hat, rho, os.path.join(out_dir, "y_vs_yhat.png"))

    return model, wrapper, rho


def _plot_y_vs_yhat(y, yhat, rho, path):
    mask = np.isfinite(y) & np.isfinite(yhat)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y[mask], yhat[mask], s=3, alpha=0.3, color="#1f77b4")
    ax.set_xlabel("GT score (nonlin additive+pairwise)", fontsize=11)
    ax.set_ylabel("Surrogate prediction", fontsize=11)
    ax.set_title(f"Surrogate fit  (ρ = {rho:+.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------------------
# Parameter extraction and heatmap
# ---------------------------------------------------------------------------

def extract_params(wrapper):
    """Return (theta_lc, theta_lclc) from the trained surrogate."""
    params = wrapper.get_params(gauge="consensus")
    # params[0] = bias, params[1] = additive (L,4), params[2] = pairwise (L,4,L,4)
    theta_lc    = np.array(params[1])          # (L, 4)
    theta_lclc  = np.array(params[2]) if len(params) > 2 else None
    return theta_lc, theta_lclc


def plot_weight_heatmap(theta_lc, wt_onehot, path):
    """4×L heatmap of learned additive weights, yellow = WT."""
    L      = theta_lc.shape[0]
    wt_idx = np.argmax(wt_onehot, axis=1)
    W      = theta_lc.T           # (4, L)

    vmax = np.abs(W).max()
    vmax = vmax if vmax > 0 else 1.0

    fig_w = max(8, min(L * 0.35, 30))
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    for i, wt in enumerate(wt_idx):
        ax.add_patch(patches.Rectangle(
            (i - 0.5, wt - 0.5), 1, 1,
            linewidth=2.0, edgecolor="gold", facecolor="yellow",
            alpha=0.55, zorder=3,
        ))

    seq_str = "".join(NUCS[j] for j in wt_idx)
    step = max(1, L // 20)
    ax.set_xticks(range(0, L, step))
    ax.set_xticklabels([f"{i}\n{seq_str[i]}" for i in range(0, L, step)], fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels(NUCS, fontsize=11)
    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("Nucleotide", fontsize=11)
    ax.set_title("Surrogate learned additive weights  (yellow = wildtype)", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Weight", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {path}")


# ---------------------------------------------------------------------------
# SurrogateGT class — the new ground truth function
# ---------------------------------------------------------------------------

class SurrogateGT:
    """Wraps a trained MAVENN model as a ground truth function.

    Usage:
        gt = SurrogateGT.load("outputs/surrogate_gt/surrogate_gt.pkl")
        scores = gt.score(X_onehot)   # X: (N, L, 4) float32
    """

    def __init__(self, mavenn_model, seq_len):
        self.model   = mavenn_model
        self.seq_len = seq_len

    def _to_str(self, X):
        nucs = np.array(list("ACGU"))
        return np.array(
            ["".join(nucs[np.argmax(x, axis=1)]) for x in X],
            dtype=object,
        )

    def score(self, X):
        """Score a library. X: (N, L, 4) one-hot. Returns (N,) float array."""
        assert X.shape[1] == self.seq_len, (
            f"Sequence length mismatch: model trained on L={self.seq_len}, "
            f"got L={X.shape[1]}"
        )
        x_str = self._to_str(X)
        # Batch to avoid memory issues
        CHUNK  = 10_000
        parts  = []
        for start in range(0, len(x_str), CHUNK):
            batch = x_str[start:start + CHUNK]
            yhat  = np.asarray(self.model.x_to_yhat(batch), dtype=float).ravel()
            parts.append(yhat)
        return np.concatenate(parts)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[save] SurrogateGT → {path}")

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq",      default="AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--lib_size", type=int, default=20_000)
    parser.add_argument("--mut_rate", type=int, default=4,
                        help="Exact number of positions to mutate per sequence")
    parser.add_argument("--out_dir",  default="outputs/surrogate_gt")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seq = args.seq.upper().replace("T", "U")

    # ── 1. Sequence setup
    oh_seq               = rna_to_one_hot(seq)
    wt_onehot, _         = remove_padding(oh_seq)
    L                    = wt_onehot.shape[0]
    exact_mut_count      = min(args.mut_rate, max(1, L // 2))
    print(f"[init] seq length L={L}, exact_mut_count={exact_mut_count}")

    # ── 2. Initialise the same GT params used in the experiments (seed=42)
    rng = np.random.default_rng(args.seed)
    W_mut, mut_map, b0 = init_additive_noWT(rng, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
    edges, J = init_pairwise_potts_optionA(
        rng, wt_onehot,
        p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
    )

    # Sigmoid nonlinearity calibrated on a reference library
    rng_ref   = np.random.default_rng(args.seed + 1)
    X_ref     = generate_library(wt_onehot, 200_000, exact_mut_count, rng_ref)
    s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
    nonlin_kwargs = init_sigmoid_nonlin(s_add_ref)
    del X_ref, s_add_ref
    print(f"[init] sigmoid ref_std={nonlin_kwargs['_norm_std']:.4f}")

    # ── 3. Generate 20k library and score with nonlin_additive_pairwise GT
    print(f"\n[library] generating {args.lib_size:,} sequences …")
    rng_lib = np.random.default_rng(args.seed + 2)
    X_lib   = generate_library(wt_onehot, args.lib_size, exact_mut_count, rng_lib)

    gt_scores = compute_gt_scores_for_library_potts(
        X_lib,
        W_mut=W_mut, mut_map=mut_map, b0=b0,
        nonlin_name="sigmoid", nonlin_kwargs=nonlin_kwargs,
        edges=edges, J=J,
    )
    y = gt_scores["nonlin_additive_pairwise"].astype(float).reshape(-1, 1)
    print(f"[library] y range: [{y.min():.4f}, {y.max():.4f}]  std={y.std():.4f}")

    # Save library
    np.savez(os.path.join(args.out_dir, "library.npz"), X=X_lib, y=y)
    print(f"[save] library → {args.out_dir}/library.npz")

    # ── 4. Train surrogate
    print("\n[surrogate] training nonlinear additive+pairwise …")
    mavenn_model, wrapper, rho = train_surrogate(X_lib, y, args.out_dir)

    # ── 5. Extract and save parameters
    theta_lc, theta_lclc = extract_params(wrapper)
    np.save(os.path.join(args.out_dir, "params_additive.npy"), theta_lc)
    print(f"[save] additive params (L={L}, A=4) → {args.out_dir}/params_additive.npy")
    if theta_lclc is not None:
        np.save(os.path.join(args.out_dir, "params_pairwise.npy"), theta_lclc)
        print(f"[save] pairwise params → {args.out_dir}/params_pairwise.npy")

    # ── 6. Additive weight heatmap
    plot_weight_heatmap(theta_lc, wt_onehot,
                        os.path.join(args.out_dir, "additive_weights.png"))

    # ── 7. Package as SurrogateGT and save
    sgt = SurrogateGT(mavenn_model, seq_len=L)
    sgt.save(os.path.join(args.out_dir, "surrogate_gt.pkl"))

    # Quick sanity check: score the WT sequence
    wt_score = sgt.score(wt_onehot[np.newaxis])[0]
    print(f"\n[check] WT score = {wt_score:.6f}")

    print("\nDone. Load with:")
    print(f"  from scripts.train_surrogate_gt import SurrogateGT")
    print(f"  gt = SurrogateGT.load('{args.out_dir}/surrogate_gt.pkl')")
    print(f"  scores = gt.score(X_onehot)   # X: (N, {L}, 4)")


if __name__ == "__main__":
    main()
