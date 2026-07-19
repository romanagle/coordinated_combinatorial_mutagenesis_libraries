"""
mRNA_RBP/plots/plot_groundtruth_vs_surrogate_varied.py

Two separate files, one per RBP (MSI1 and VTS1 are different proteins scored
on different native sequences -- never combined into one file):

  coefficients_groundtruth_vs_surrogate_msi1.png
  coefficients_groundtruth_vs_surrogate_vts1.png

"Deep squid" = a nonlinear additive+pairwise MAVE-NN surrogate trained on a
200K-sequence library with a uniform number of 3-10 mutations per sequence,
labeled by ResidualBind, with its additive+pairwise coefficients (alpha, J)
extracted directly from the fitted model
(coefs_varied_mutrate_nonlinear_additive_p_pairwise_<oracle>.npz, built by
train_surrogate_varied_mutrate.py / train_surrogate_vts1_mut10.py's varied-
mutrate sibling). This *is* the oracle/ground-truth side here -- it is not
re-derived by brute-force scanning, its own fitted coefficients are the
ground truth. See feedback_deep_squid_terminology memory.

Each file has two panels, sharing one colorbar scale:
  Left  -- deep squid's own additive+pairwise coefficients, as described
           above.
  Right -- a genuine downstream nonlinear additive+pairwise MAVE-NN surrogate
           *trained on* a different, randomly-mutated 20K-sequence library
           (fixed 10% mutation rate, not the uniform-3-to-10 library deep
           squid itself was trained on) *labeled by deep squid*
           (coefs_..._deepsquid_<rbp>.npz, built by
           train_deepsquid_surrogate_mut10.py) -- a surrogate of deep squid,
           not deep squid itself.

Usage:
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/plots/plot_groundtruth_vs_surrogate_varied.py
"""

import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

import mRNA_RBP.viz as viz
from coef_map_shared import right_panel_pairwise_vmax

CACHE_DIR = os.path.join(_HERE, "outputs", "notebook_plots")
OUT_DIR   = os.path.join(_HERE, "outputs", "notebook_plots", "coefficients_map")

MSI1_SEQ        = "AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA"
MSI1_STEM_PAIRS = [(8, 30), (9, 29), (10, 28), (11, 27), (12, 26), (13, 25), (14, 24), (15, 23)]
MSI1_MOTIF_POSITIONS = [17, 18, 19, 20, 21]

VTS1_SEQ        = "AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA"
VTS1_STEM_PAIRS = [(16, 21), (15, 22), (14, 23), (25, 31), (24, 32)]
VTS1_MOTIF_POSITIONS = [20, 21, 22, 23, 24]

CONFIGS = {
    "msi1": dict(
        seq=MSI1_SEQ, stem_pairs=MSI1_STEM_PAIRS, motif_positions=MSI1_MOTIF_POSITIONS,
        deepsquid_coef=os.path.join(
            _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
            "surrogate_coefs",
            "coefs_varied_mutrate_nonlinear_additive_p_pairwise_residualbind_ensemble.npz"),
        surrogate_coef=os.path.join(
            _HERE, "outputs_surrogate_varied_mutrate", "surrogate_coefs",
            "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_msi1.npz"),
        label="MSI1",
    ),
    "vts1": dict(
        seq=VTS1_SEQ, stem_pairs=VTS1_STEM_PAIRS, motif_positions=VTS1_MOTIF_POSITIONS,
        deepsquid_coef=os.path.join(
            _HERE, "outputs_vts1_residualbind", "surrogate_coefs",
            "coefs_varied_mutrate_nonlinear_additive_p_pairwise_vts1_residualbind.npz"),
        surrogate_coef=os.path.join(
            _HERE, "outputs_surrogate_varied_mutrate", "surrogate_coefs",
            "coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_vts1.npz"),
        label="VTS1",
    ),
}


def fig_to_array(fig, dpi=150) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    buf.close()
    return arr


def hstack_images(images: list, pad_value: int = 255) -> np.ndarray:
    h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < h:
            pad = np.full((h - img.shape[0], img.shape[1], 3), pad_value, dtype=img.dtype)
            img = np.vstack([img, pad])
        padded.append(img)
    return np.hstack(padded)


def full_beta(J: np.ndarray) -> dict:
    """beta[(i,j)] = J[i,:,j,:] (4,4) for every i<j position pair -- not just
    the declared stem pairs. Real learned coupling shows up plenty outside the
    stem too, so restricting to stem_pairs here would silently zero out most
    of the map."""
    L = J.shape[0]
    return {(i, j): J[i, :, j, :] for i in range(L) for j in range(i + 1, L)}


def run_one(name: str, cfg: dict):
    L = len(cfg["seq"])
    stem_pairs = cfg["stem_pairs"]
    motif_positions = cfg["motif_positions"]

    gt = np.load(cfg["deepsquid_coef"])
    gt_y_std = float(gt["y_std"]) if "y_std" in gt else 1.0
    alpha_gt = gt["alpha"].astype(np.float64) * gt_y_std      # (L, 4)
    J_gt     = gt["J"].astype(np.float64) * gt_y_std           # (L, 4, L, 4)
    beta_gt  = full_beta(J_gt)

    sur = np.load(cfg["surrogate_coef"])
    y_std = float(sur["y_std"]) if "y_std" in sur else 1.0
    alpha_sur = sur["alpha"].astype(np.float64) * y_std  # (L, 4)
    J_sur     = sur["J"].astype(np.float64) * y_std       # (L, 4, L, 4)
    beta_sur  = full_beta(J_sur)

    # Color scale is driven by the pairwise (beta) values only -- the additive
    # (alpha/SSM) values run much larger in magnitude and would otherwise wash
    # out the pairwise panel's diverging colormap. Use the abs-max-per-block
    # value -- the single number actually drawn per square -- not the raw
    # 16-entries-per-block values (those run ~5-10x smaller, which previously
    # made vmax too small and every square look near-saturated/dark).
    #
    # Left (deep squid) gets its own scale from its own coefficients. Right
    # (surrogate) is pinned to right_panel_pairwise_vmax() so this pairwise
    # map renders on the same scale as its counterpart in
    # coefficients_oracle_vs_surrogate_<rbp>.png -- MSI1 and VTS1 don't need
    # to match each other, just their own two "right panel" surrogates.
    gt_vals  = np.array([np.abs(M).max() for M in beta_gt.values()])
    gt_vmax  = max(float(np.nanpercentile(gt_vals, 95)), 1e-6)
    sur_vmax = right_panel_pairwise_vmax(name)

    fig_gt = viz.plot_coefficients(
        alpha_gt, beta_gt, L, stem_pairs=stem_pairs, motif_positions=motif_positions, vmax=gt_vmax)
    gt_rho = float(gt["rho_test"]) if "rho_test" in gt else float("nan")
    fig_gt.suptitle(f"{cfg['label']} deep squid\n"
                     "(nonlin additive+pairwise surrogate trained on 200K seqs, 3-10 muts, "
                     f"ResidualBind labels)  rho_test={gt_rho:+.3f}", fontsize=10, y=1.02)

    fig_sur = viz.plot_coefficients(
        alpha_sur, beta_sur, L, stem_pairs=stem_pairs, motif_positions=motif_positions, vmax=sur_vmax)
    rho = float(sur["rho_test"]) if "rho_test" in sur else float("nan")
    fig_sur.suptitle(f"{cfg['label']} nonlinear additive+pairwise surrogate\n"
                      f"(trained on 20K seqs, 10% mut rate, deep-squid labels)  "
                      f"rho_test={rho:+.3f}", fontsize=10, y=1.02)

    os.makedirs(OUT_DIR, exist_ok=True)
    combined = hstack_images([fig_to_array(fig_gt), fig_to_array(fig_sur)])
    out_path = os.path.join(OUT_DIR, f"coefficients_groundtruth_vs_surrogate_{name}.png")
    Image.fromarray(combined).save(out_path)

    plt.close(fig_gt)
    plt.close(fig_sur)
    print(f"Saved -> {out_path}")


def main():
    for name, cfg in CONFIGS.items():
        run_one(name, cfg)


if __name__ == "__main__":
    main()
