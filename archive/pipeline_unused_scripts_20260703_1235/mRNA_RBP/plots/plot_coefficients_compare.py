"""
mRNA_RBP/plot_coefficients_compare.py

Side-by-side additive + pairwise coefficient maps for two nonlinear
additive+pairwise surrogates trained against ResidualBind-ensemble labels:

  Left  -- surrogate trained on the VTS1 varied-mutation-rate library
           (mutation counts 3-10, 200K sequences)
  Right -- surrogate trained on the VTS1 fixed mut=10% / lib=20,000 random
           library

Both surrogates were fit with a full (dense, all-positions) pairwise gpmap,
so unlike plot_coefficients.py (which restricts the pairwise panel to the
synthetic ground truth's declared stem pairs), this script shows the FULL
L x L pairwise landscape each surrogate actually learned -- there's no
synthetic structural prior to restrict to when the labels come from the real
ResidualBind ensemble. The declared stem pairs (RNAfold MFE structure of
VTS1_SEQ, see VTS1_STEM_PAIRS below) are still boxed in green as a reference.

A single vmax is shared across both coefficient maps so the color scales are
directly comparable.

Usage:
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/plot_coefficients_compare.py
    /home/nagle/miniconda3/envs/squid/bin/python3.7 mRNA_RBP/plot_coefficients_compare.py --oracle vts1_residualbind
"""

import argparse
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

from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_MOTIF_POSITIONS, VTS1_SEQ, VTS1_STEM_PAIRS,
)
from mRNA_RBP.oracles import (
    RESIDUALBIND_ORACLE, default_output_base, normalize_oracle_name, primary_gt_key,
)
import mRNA_RBP.viz as viz

ORACLE_LABEL = {
    RESIDUALBIND_ORACLE: "ResidualBind surrogate (MSI1 ensemble)",
    "vts1_residualbind": "ResidualBind surrogate (real VTS1 / RNCMPT00111)",
}


def full_pairwise_dict(J: np.ndarray, L: int) -> dict:
    """All (i, j) i<j position pairs -> their (4,4) coupling block.

    Unlike the synthetic-GT convention (sparse dict restricted to declared
    stem pairs), this surrogate's pairwise gpmap is dense over every position
    pair, so we show all of it.
    """
    return {(i, j): J[i, :, j, :] for i in range(L) for j in range(i + 1, L)}


def fig_to_array(fig, dpi=150) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=RESIDUALBIND_ORACLE,
                        choices=[RESIDUALBIND_ORACLE, "vts1_residualbind"])
    parser.add_argument("--out_base", default=None,
                        help="Library/coef root. Defaults to outputs_<oracle>, but the "
                             "MSI1-on-VTS1_SEQ run was historically saved nested under "
                             "outputs_vts1_residualbind/outputs_residualbind_ensemble/")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    gt_key = primary_gt_key(oracle_name)
    out_base = args.out_base or default_output_base(_HERE, oracle_name)
    coef_dir = os.path.join(out_base, "surrogate_coefs")
    out_dir = os.path.join(out_base, "notebook_plots")
    varied_coef_path = os.path.join(
        coef_dir, f"coefs_varied_mutrate_nonlinear_additive_p_pairwise_{oracle_name}.npz")
    fixed_coef_path = os.path.join(
        coef_dir, f"coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_{oracle_name}.npz")
    fixed_lib_path = os.path.join(out_base, "instance_00", "vts1_mut10", "lib_20000.npz")
    label = ORACLE_LABEL[oracle_name]

    L = len(VTS1_SEQ)
    os.makedirs(out_dir, exist_ok=True)

    coefs_v = np.load(varied_coef_path)
    y_std_v = float(coefs_v["y_std"])
    alpha_v = coefs_v["alpha"] * y_std_v
    J_v     = coefs_v["J"] * y_std_v
    rho_v   = float(coefs_v["rho_test"]) if "rho_test" in coefs_v else float("nan")

    coefs_f = np.load(fixed_coef_path)
    y_std_f = float(np.load(fixed_lib_path)[f"scores_{gt_key}"].std())
    alpha_f = coefs_f["alpha"] * y_std_f
    J_f     = coefs_f["J"] * y_std_f
    rho_f   = float(coefs_f["rho_test"]) if "rho_test" in coefs_f else float("nan")

    beta_v = full_pairwise_dict(J_v, L)
    beta_f = full_pairwise_dict(J_f, L)

    all_vals = np.concatenate([
        alpha_v.ravel(), alpha_f.ravel(),
        np.array([M.ravel() for M in beta_v.values()]).ravel(),
        np.array([M.ravel() for M in beta_f.values()]).ravel(),
    ])
    vmax = max(float(np.abs(all_vals).max()), 1e-6)
    print(f"Shared vmax: {vmax:.4f}")

    fig_v = viz.plot_coefficients(
        alpha_v, beta_v, L, stem_pairs=VTS1_STEM_PAIRS, motif_positions=VTS1_MOTIF_POSITIONS, vmax=vmax)
    fig_v.suptitle(f"VTS1 varied mut-rate (3-10 mut, n=200K)\n{label}  rho_test={rho_v:+.3f}",
                   fontsize=10, y=1.02)

    fig_f = viz.plot_coefficients(
        alpha_f, beta_f, L, stem_pairs=VTS1_STEM_PAIRS, motif_positions=VTS1_MOTIF_POSITIONS, vmax=vmax)
    fig_f.suptitle(f"VTS1 fixed mut=10%, lib=20,000\n{label}  rho_test={rho_f:+.3f}",
                   fontsize=10, y=1.02)

    fig_v.savefig(os.path.join(out_dir, "coefficients_varied_mutrate.png"),
                  dpi=150, bbox_inches="tight")
    fig_f.savefig(os.path.join(out_dir, "coefficients_mut10_lib20000.png"),
                  dpi=150, bbox_inches="tight")

    combined = hstack_images([fig_to_array(fig_v), fig_to_array(fig_f)])
    out_path = os.path.join(out_dir, "coefficients_compare_vts1_varied_vs_mut10.png")
    Image.fromarray(combined).save(out_path)

    plt.close(fig_v)
    plt.close(fig_f)
    print(f"Saved -> {out_path}")
    print(f"Saved -> {os.path.join(out_dir, 'coefficients_varied_mutrate.png')}")
    print(f"Saved -> {os.path.join(out_dir, 'coefficients_mut10_lib20000.png')}")


if __name__ == "__main__":
    main()
