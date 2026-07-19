"""
mRNA_RBP/plots/coef_map_shared.py

Shared helper so the "right panel" (surrogate) pairwise coefficient maps in
the coefficients_map/ comparison plots use the same color scale *within* an
RBP, across the two different labeling oracles:

  coefficients_oracle_vs_surrogate_<rbp>.png       -- right = surrogate
      trained on the 20K/10%-mut-rate library, labeled by the raw
      ResidualBind ensemble oracle (plot_oracle_vs_surrogate.py)
  coefficients_groundtruth_vs_surrogate_<rbp>.png  -- right = surrogate
      trained on the *same* 20K/10%-mut-rate library, labeled by deep squid
      instead (plot_groundtruth_vs_surrogate_varied.py)

MSI1 and VTS1 are scaled independently -- they do not need to match each
other, only the two right-panel surrogates within the same RBP need to.
"""

import os

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RIGHT_PANEL_COEF_PATHS = {
    "msi1": [
        os.path.join(
            _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
            "surrogate_coefs",
            "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_residualbind_ensemble.npz"),
        os.path.join(
            _HERE, "outputs_surrogate_varied_mutrate", "surrogate_coefs",
            "coefs_k00_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_msi1.npz"),
    ],
    "vts1": [
        os.path.join(
            _HERE, "outputs_vts1_residualbind", "surrogate_coefs",
            "coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_vts1_residualbind.npz"),
        os.path.join(
            _HERE, "outputs_surrogate_varied_mutrate", "surrogate_coefs",
            "coefs_vts1_mut10_lib20000_nonlinear_additive_p_pairwise_deepsquid_vts1.npz"),
    ],
}


def _abs_max_per_block(J: np.ndarray) -> list:
    L = J.shape[0]
    return [float(np.abs(J[i, :, j, :]).max()) for i in range(L) for j in range(i + 1, L)]


def right_panel_pairwise_vmax(rbp: str, percentile: float = 95.0) -> float:
    """Shared pairwise (beta) color-scale vmax for both 20K/mut10 surrogates
    of the given RBP (ResidualBind-labeled and deep-squid-labeled), so their
    two 'right panel' pairwise maps render on an identical scale.

    Uses the abs-max-per-block value (the single number actually drawn per
    square) -- not the raw 16-entries-per-block values, which run smaller and
    would understate the true color range. A percentile rather than the raw
    max is used since real pairwise coefficients are nonzero almost
    everywhere and a handful of genuinely large pairs would otherwise wash
    out the rest.
    """
    rbp = rbp.lower()
    vals = []
    for path in RIGHT_PANEL_COEF_PATHS[rbp]:
        d = np.load(path)
        y_std = float(d["y_std"]) if "y_std" in d else 1.0
        J = d["J"].astype(np.float64) * y_std
        vals.extend(_abs_max_per_block(J))
    return max(float(np.percentile(np.abs(vals), percentile)), 1e-6)
