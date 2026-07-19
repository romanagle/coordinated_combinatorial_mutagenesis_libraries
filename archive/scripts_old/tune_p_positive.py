"""tune_p_positive.py

Sweep p_positive values for init_pairwise_potts_optionA and overlay each
nonlin_additive_pairwise distribution against ResidualBind on a single plot.
Uses OLD normalisation (÷ std(s_add)) so the pile-up shape is preserved.

The goal: find the p_positive that best matches ResidualBind's distribution
shape — left-skewed bulk with a right tail extending above WT.
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
OUT        = "outputs/gt_comparison/tune_p_positive.png"

P_POSITIVE   = 0.15
Y_MAX        = 30
TAU          = 0.75
# norm_scale < 1.0: divide by less than ref_std_addpair → larger z-scores
# → peak further left AND more mass in the intermediate region
NORM_SCALES  = [1.0, 0.7, 0.5, 0.35, 0.25]

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


def score_nonlin_addpair(s_addpair, nk):
    """Score using the ref_std_addpair stored in nk (proper normalisation)."""
    ref_std = float(nk.get("_norm_std_addpair", nk["_norm_std"]))
    s_n     = s_addpair / ref_std
    y       = apply_global_nonlin(s_n, "sigmoid", nk).reshape(-1)
    wt_nl   = float(apply_global_nonlin(np.array([[0.0]]), "sigmoid", nk))
    return y - wt_nl


def norm01(s):
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return np.clip((s - lo) / max(hi - lo, 1e-8), 0, 1), lo, hi


# ── Sequence
oh_seq       = rna_to_one_hot(SEQ.upper().replace("T", "U"))
wt_onehot, _ = remove_padding(oh_seq)
L            = wt_onehot.shape[0]
exact_mut    = min(MUT_RATE, max(1, L // 2))

# ── Shared library (same sequences for all sweeps for fair comparison)
rng_lib = np.random.default_rng(SEED)
X_lib   = generate_library(wt_onehot, LIB_SIZE, exact_mut, rng_lib)

# ── Shared additive weights
rng_gt = np.random.default_rng(SEED)
W_mut, mut_map, b0 = init_additive_noWT(rng_gt, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)

# ── Calibrate sigmoid on reference library — compute ref std for BOTH energies
# We sweep p_positive, which changes s_pair and therefore std(s_addpair).
# Use p_positive=0 (all-negative) for the reference calibration so the sigmoid
# midpoint is consistent across the sweep; only the tail changes.
rng_ref    = np.random.default_rng(SEED + 1)
X_ref      = generate_library(wt_onehot, 200_000, exact_mut, rng_ref)
s_add_ref  = additive_affinity_noWT(X_ref, W_mut, mut_map, b=b0).reshape(-1)
nk         = init_sigmoid_nonlin(s_add_ref)
# Reference Potts (p_positive=0) for calibrating ref_std_addpair
rng_ref2   = np.random.default_rng(SEED)
_, _2, _3  = init_additive_noWT(rng_ref2, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
edges_ref, J_ref = init_pairwise_potts_optionA(
    rng_ref2, wt_onehot,
    p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10,
    wt_rowcol_zero=True, p_positive=0.0,
)
s_pair_ref = pairwise_potts_energy(X_ref, edges_ref, J_ref, b=0.0).reshape(-1)
ref_std_addpair = float(np.std(s_add_ref + s_pair_ref)) + 1e-8
nk["_norm_std_addpair"] = ref_std_addpair
# Place sigmoid inflection at z=0 (= WT).  With most mutants at z<0, the
# sigmoid naturally produces a left-skewed distribution (bulk in left tail,
# thin right tail from p_positive sequences that push past WT).
nk["sig_z0"] = 0.0
nk["y_max"]  = float(Y_MAX)
nk["y_min"]  = 0.0
del X_ref, s_add_ref, s_pair_ref, edges_ref, J_ref
print(f"ref_std_add={nk['_norm_std']:.4f}  ref_std_addpair={ref_std_addpair:.4f}  sig_z0={nk['sig_z0']}")

# Precompute additive scores on library (shared)
s_add = additive_affinity_noWT(X_lib, W_mut, mut_map, b=b0).reshape(-1)

# ── ResidualBind
data_path    = Path.home() / 'residualbind' / 'data' / 'RNAcompete_2013' / 'rnacompete2013.h5'
save_path    = '/home/nagle/residualbind/weights/log_norm_seq'
weights_path = os.path.join(save_path, f'{EXPERIMENT}_weights.hdf5')
rbp_index    = helper.find_experiment_index(data_path, EXPERIMENT)
train, _, _  = helper.load_rnacompete_data(data_path, ss_type='seq',
                                            normalization='log_norm', rbp_index=rbp_index)
rb_model  = rb.ResidualBind(list(train['inputs'].shape)[1:], num_class=1, weights_path=weights_path)
rb_model.load_weights()
rb_raw    = rb_model.predict(X_lib, batch_size=512).ravel()
wt_score  = rb_model.predict(wt_onehot[np.newaxis], batch_size=1).ravel()[0]
rb_scores = rb_raw - wt_score
rb_n, rb_lo, rb_hi = norm01(rb_scores)
pct_above_wt_rb = float(np.mean(rb_scores > 0) * 100)
print(f"ResidualBind: std={rb_scores.std():.3f}  "
      f"range=[{rb_scores.min():.3f}, {rb_scores.max():.3f}]  "
      f"above-WT={pct_above_wt_rb:.1f}%")

# ── Fixed Potts (p_positive=0.15)
rng_p = np.random.default_rng(SEED)
_, _, _ = init_additive_noWT(rng_p, wt_onehot, sigma=0.5, l1_w=0.1, bias=0.0)
edges, J = init_pairwise_potts_optionA(
    rng_p, wt_onehot,
    p_edge=0.70, df=5.0, lambda_J=2.0, p_rescue=0.10,
    wt_rowcol_zero=True, p_positive=P_POSITIVE,
)
s_pair    = pairwise_potts_energy(X_lib, edges, J, b=0.0).reshape(-1)
s_addpair = s_add + s_pair

# ── Sweep norm_scale
gt_results = []
nk["sig_tau"] = TAU
for norm_scale in NORM_SCALES:
    nk_t = dict(nk)
    nk_t["_norm_std_addpair"] = ref_std_addpair * norm_scale
    scores    = score_nonlin_addpair(s_addpair, nk_t)
    pct_above = float(np.mean(scores > 0) * 100)
    print(f"norm_scale={norm_scale:.2f}  std={scores.std():.3f}  "
          f"range=[{scores.min():.2f}, {scores.max():.2f}]  above-WT={pct_above:.1f}%")
    gt_results.append((norm_scale, scores, pct_above))

# ── Plot: one subplot per panel (ResidualBind + each norm_scale)
ncols = 1 + len(gt_results)
fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4), sharey=False)

BINS = 50

ax = axes[0]
ax.hist(rb_scores, bins=BINS, color="#e41a1c", alpha=0.80, density=True, edgecolor="none")
ax.axvline(0, color="black", lw=1.5, ls="--")
ax.set_title("ResidualBind", fontsize=10, fontweight="bold")
ax.set_xlabel("Score (WT = 0)", fontsize=9)
ax.set_ylabel("Density", fontsize=9)
ax.text(0.97, 0.97, f"above-WT\n{pct_above_wt_rb:.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.grid(True, alpha=0.25)

gt_colors = ["#999999", "#4393c3", "#2166ac", "#d6604d", "#b2182b"]

for ax, (norm_scale, scores, pct_above), color in zip(axes[1:], gt_results, gt_colors):
    ax.hist(scores, bins=BINS, color=color, alpha=0.80, density=True, edgecolor="none")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.set_title(f"norm_scale = {norm_scale:.2f}", fontsize=10)
    ax.set_xlabel("Score (WT = 0)", fontsize=9)
    ax.text(0.97, 0.97, f"above-WT\n{pct_above:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.grid(True, alpha=0.25)

fig.suptitle(f"Nonlin Add+Pair GT — sweeping norm_scale  (tau={TAU}, y_max={Y_MAX}, z0=0, "
             f"p_pos={P_POSITIVE}, 20k lib, 4 mut)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[plot] {OUT}")
