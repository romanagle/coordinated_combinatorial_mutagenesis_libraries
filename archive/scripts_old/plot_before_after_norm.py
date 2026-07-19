"""plot_before_after_norm.py

Three-panel comparison of the nonlin_additive_pairwise score distribution:
  left   — ResidualBind (reference biology)
  center — Nonlin Add+Pair GT, OLD normalisation (divide by std(s_add))
  right  — Nonlin Add+Pair GT, FIXED normalisation (divide by std(s_add + s_pair))
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
    additive_affinity_noWT, pairwise_potts_energy, apply_global_nonlin,
)
import residualbind as rb
import helper

SEQ        = "AAAAAAACCCCCAAAAAAUCGGCUGGACCGGGAAAAAAAAA"
EXPERIMENT = "RNCMPT00111"
SEED       = 42
LIB_SIZE   = 20_000
MUT_RATE   = 4
OUT        = "outputs/gt_comparison/before_after_norm.png"

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


def score_nonlin_addpair(s_addpair, ref_std, nk):
    """Apply sigmoid nonlinearity and anchor at WT=0."""
    s_n   = s_addpair / ref_std
    y     = apply_global_nonlin(s_n, "sigmoid", nk).reshape(-1)
    wt_nl = float(apply_global_nonlin(np.array([[0.0]]), "sigmoid", nk))
    return y - wt_nl


# ── Sequence
oh_seq       = rna_to_one_hot(SEQ.upper().replace("T", "U"))
wt_onehot, _ = remove_padding(oh_seq)
L            = wt_onehot.shape[0]
exact_mut    = min(MUT_RATE, max(1, L // 2))

# ── Library
rng   = np.random.default_rng(SEED)
X_lib = generate_library(wt_onehot, LIB_SIZE, exact_mut, rng)

# ── GT params
rng_gt = np.random.default_rng(SEED)
W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
edges, J = init_pairwise_potts_optionA(
    rng_gt, wt_onehot, p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10, wt_rowcol_zero=True,
)

# ── Reference library for calibration
rng_ref    = np.random.default_rng(SEED + 1)
X_ref      = generate_library(wt_onehot, 200_000, exact_mut, rng_ref)
s_add_ref  = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
s_pair_ref = pairwise_potts_energy(X_ref, edges, J, b=0.0).reshape(-1)
nk         = init_sigmoid_nonlin(s_add_ref)
ref_std_add     = nk["_norm_std"]                                     # OLD: std(s_add)
ref_std_addpair = float(np.std(s_add_ref + s_pair_ref)) + 1e-8       # NEW: std(s_addpair)
del X_ref, s_add_ref, s_pair_ref

print(f"ref_std_add={ref_std_add:.4f}  ref_std_addpair={ref_std_addpair:.4f}  "
      f"ratio={ref_std_addpair/ref_std_add:.1f}×")

# ── Score library: raw energies
s_add     = additive_affinity_noWT(X_lib, W_mut, mut_map, b=b0).reshape(-1)
s_pair    = pairwise_potts_energy(X_lib, edges, J, b=0.0).reshape(-1)
s_addpair = s_add + s_pair

# OLD normalisation: divide by std(s_add)
scores_old = score_nonlin_addpair(s_addpair, ref_std_add, nk)
# NEW normalisation: divide by std(s_add + s_pair)
scores_new = score_nonlin_addpair(s_addpair, ref_std_addpair, nk)

print(f"OLD  range=[{scores_old.min():.4f}, {scores_old.max():.4f}]  std={scores_old.std():.4f}")
print(f"NEW  range=[{scores_new.min():.4f}, {scores_new.max():.4f}]  std={scores_new.std():.4f}")

# ── ResidualBind
data_path    = Path.home() / 'residualbind' / 'data' / 'RNAcompete_2013' / 'rnacompete2013.h5'
save_path    = '/home/nagle/residualbind/weights/log_norm_seq'
weights_path = os.path.join(save_path, f'{EXPERIMENT}_weights.hdf5')
rbp_index    = helper.find_experiment_index(data_path, EXPERIMENT)
train, _, _  = helper.load_rnacompete_data(data_path, ss_type='seq',
                                            normalization='log_norm', rbp_index=rbp_index)
model     = rb.ResidualBind(list(train['inputs'].shape)[1:], num_class=1, weights_path=weights_path)
model.load_weights()
rb_raw    = model.predict(X_lib, batch_size=512).ravel()
wt_score  = model.predict(wt_onehot[np.newaxis], batch_size=1).ravel()[0]
rb_scores = rb_raw - wt_score
print(f"RB   range=[{rb_scores.min():.4f}, {rb_scores.max():.4f}]  std={rb_scores.std():.4f}")


def norm01(s):
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return np.clip((s - lo) / max(hi - lo, 1e-8), 0, 1), lo, hi


# ── Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

panels = [
    (axes[0], rb_scores,   f"ResidualBind\n({EXPERIMENT})",              "#e41a1c"),
    (axes[1], scores_old,  "Nonlin Add+Pair GT\nOLD: ÷ std(s_add)",      "#999999"),
    (axes[2], scores_new,  "Nonlin Add+Pair GT\nFIXED: ÷ std(s_add+pair)", "#ff7f00"),
]

for ax, scores, title, color in panels:
    s_n, lo, hi = norm01(scores)
    ax.hist(s_n, bins=60, color=color, alpha=0.80, edgecolor="none", density=True)
    wt_n = float(np.clip((0.0 - lo) / max(hi - lo, 1e-8), 0, 1))
    ax.axvline(wt_n, color="black", lw=1.5, ls="--", label="WT = 0")
    ax.set_xlabel("Score (normalised to [0, 1])", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.text(0.97, 0.97,
            f"std={scores.std():.3f}\n[{np.percentile(scores,1):.3f}, {np.percentile(scores,99):.3f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig.suptitle(f"20k library · {exact_mut} mutations · L={L}  |  effect of addpair normalisation fix",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot] {OUT}")
