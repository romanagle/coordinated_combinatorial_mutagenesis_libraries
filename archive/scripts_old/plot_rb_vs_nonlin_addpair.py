"""plot_rb_vs_nonlin_addpair.py

Side-by-side normalised histograms:
  left  — ResidualBind predictions
  right — Nonlinear additive+pairwise synthetic GT (fixed normalisation)

Both anchored at WT = 0, then each normalised to [0, 1] for display.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/home/nagle/final_version/residualbind')
sys.path.insert(0, '/home/nagle/final_version/squid-nn')
sys.path.append('/home/nagle/final_version/squid-manuscript/squid')

from seq_utils import rna_to_one_hot, remove_padding
from ground_truth import (
    init_additive_noWT, init_pairwise_potts_optionA, init_sigmoid_nonlin,
    compute_gt_scores_for_library_potts, additive_affinity_noWT,
    pairwise_potts_energy,
)
import residualbind as rb
import helper

SEQ        = "AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA"
EXPERIMENT = "RNCMPT00111"
SEED       = 42
LIB_SIZE   = 20_000
MUT_RATE   = 4
OUT        = "outputs/gt_comparison/rb_vs_nonlin_addpair.png"

os.makedirs(os.path.dirname(OUT), exist_ok=True)


def generate_library(wt_onehot, n_seqs, exact_mut_count, rng):
    L, wt_idx = wt_onehot.shape[0], np.argmax(wt_onehot, axis=1)
    parts = []
    for start in range(0, n_seqs, 50_000):
        nc    = min(50_000, n_seqs - start)
        pos   = np.argpartition(rng.random((nc, L)), exact_mut_count, axis=1)[:, :exact_mut_count]
        X     = np.tile(wt_onehot[None], (nc, 1, 1)).astype(np.float32)
        n_idx = np.repeat(np.arange(nc), exact_mut_count)
        p_idx = pos.ravel()
        rand_nuc = rng.integers(0, 3, size=nc * exact_mut_count)
        new_nucs = np.where(rand_nuc >= wt_idx[p_idx], rand_nuc + 1, rand_nuc)
        X[n_idx, p_idx, :]        = 0
        X[n_idx, p_idx, new_nucs] = 1
        parts.append(X)
    return np.concatenate(parts)


# ── Sequence setup
oh_seq            = rna_to_one_hot(SEQ.upper().replace("T", "U"))
wt_onehot, _      = remove_padding(oh_seq)
L                 = wt_onehot.shape[0]
exact_mut         = min(MUT_RATE, max(1, L // 2))

# ── Library
rng   = np.random.default_rng(SEED)
X_lib = generate_library(wt_onehot, LIB_SIZE, exact_mut, rng)
print(f"Library: {len(X_lib):,} seqs, L={L}, mut={exact_mut}")

# ── Synthetic GT params
rng_gt = np.random.default_rng(SEED)
W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
edges, J = init_pairwise_potts_optionA(
    rng_gt, wt_onehot, p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
)
rng_ref   = np.random.default_rng(SEED + 1)
X_ref     = generate_library(wt_onehot, 200_000, exact_mut, rng_ref)
s_add_ref = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
nk        = init_sigmoid_nonlin(s_add_ref)
s_pair_ref = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
nk["_norm_std_addpair"] = float(np.std(s_add_ref + s_pair_ref)) + 1e-8
del X_ref, s_add_ref, s_pair_ref
print(f"Sigmoid ref_std={nk['_norm_std']:.4f}  ref_std_addpair={nk['_norm_std_addpair']:.4f}")

gt_scores = compute_gt_scores_for_library_potts(
    X_lib, W_mut=W_mut, mut_map=mut_map, b0=b0,
    nonlin_name="sigmoid", nonlin_kwargs=nk, edges=edges, J=J,
)
nonlin_scores = gt_scores["nonlin_additive_pairwise"]

# ── ResidualBind
data_path    = Path.home() / 'residualbind' / 'data' / 'RNAcompete_2013' / 'rnacompete2013.h5'
save_path    = '/home/nagle/residualbind/weights/log_norm_seq'
weights_path = os.path.join(save_path, EXPERIMENT + '_weights.hdf5')
rbp_index    = helper.find_experiment_index(data_path, EXPERIMENT)
train, _, _  = helper.load_rnacompete_data(data_path, ss_type='seq',
                                            normalization='log_norm', rbp_index=rbp_index)
model = rb.ResidualBind(list(train['inputs'].shape)[1:], num_class=1, weights_path=weights_path)
model.load_weights()
rb_raw    = model.predict(X_lib, batch_size=512).ravel()
wt_score  = model.predict(wt_onehot[np.newaxis], batch_size=1).ravel()[0]
rb_scores = rb_raw - wt_score
print(f"ResidualBind WT={wt_score:.4f}  range=[{rb_scores.min():.4f}, {rb_scores.max():.4f}]  std={rb_scores.std():.4f}")
print(f"Nonlin Add+Pair      range=[{nonlin_scores.min():.4f}, {nonlin_scores.max():.4f}]  std={nonlin_scores.std():.4f}")


def norm01(s):
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return np.clip((s - lo) / max(hi - lo, 1e-8), 0, 1)


# ── Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

BINS = 60

for ax, scores, label, color in [
    (axes[0], rb_scores,      f"ResidualBind\n({EXPERIMENT})",        "#e41a1c"),
    (axes[1], nonlin_scores,  "Nonlin Add+Pair GT\n(synthetic)",      "#ff7f00"),
]:
    s_n = norm01(scores)
    ax.hist(s_n, bins=BINS, color=color, alpha=0.75, edgecolor="none", density=True)
    ax.axvline(norm01(np.array([0.0]))[0], color="black", lw=1.5, ls="--", label="WT (anchored)")
    ax.set_xlabel("Score (normalised to [0, 1])", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    raw_lo, raw_hi = np.percentile(scores, 1), np.percentile(scores, 99)
    ax.text(0.98, 0.97,
            f"std={scores.std():.3f}\nrange [{raw_lo:.3f}, {raw_hi:.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig.suptitle(f"20k library · {LIB_SIZE:,} seqs · {exact_mut} mutations · L={L}", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot] {OUT}")
