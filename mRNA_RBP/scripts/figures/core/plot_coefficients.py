"""
mRNA_RBP/scripts/figures/core/plot_coefficients.py

Generate:
  <out_dir>/coefficients_gt.png       – oracle/GT alpha + beta contact map
  <out_dir>/coefficients_surrogate.png – surrogate alpha + beta
  <out_dir>/coefficients_map/coefficients_oracle_vs_surrogate_<name>.png
      – the two above, side by side

Default condition: instance 0, mut_rate=10%, lib_size=20000, nl+pair surrogate.

Oracle-agnostic. For the synthetic ground truth, alpha/beta come from the
analytic MrnaRbpGroundTruth object (unchanged behavior). For any real
ResidualBind-backed oracle (MSI1, VTS1, HuR, ...), which has no analytic
ground-truth coefficients, alpha/beta are reconstructed from data the
standardized pipeline already produces: single-mutant deltas from
``ssm.npz`` (additive alpha) and the 16-combo-per-stem-pair grid from
``pairwise_lib.npz`` (pairwise beta), both relative to the cached WT score.

This intentionally covers only the "oracle_vs_surrogate" comparison (used by
Synthetic GT / ResidualBind oracle MSI1 / ResidualBind oracle VTS1 today).
It does NOT attempt the separate "deepSQUID groundtruth_vs_surrogate"
variant, which needs an extra non-standard training run (a surrogate
relabeled through deep squid itself) and per-RBP hardcoded shared color
scales -- see the archived coef_map_shared.py. That one stays a deliberate,
documented gap rather than a guess.

Usage:
    python mRNA_RBP/scripts/figures/core/plot_coefficients.py --oracle hur --wt_activity high
    python mRNA_RBP/scripts/figures/core/plot_coefficients.py --mut_rate 25 --lib_size 20000
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "external", "residualbind"))

from mRNA_RBP.src.gt_init import MrnaRbpGroundTruth
from mRNA_RBP.src.evaluate import make_ssm_deltas_from_scores
from mRNA_RBP.src.oracles import (
    MRNA_ORACLE, ORACLE_SHORT_NAME, default_output_base, normalize_oracle_name,
    primary_gt_key, sequence_config_for_oracle,
)
import mRNA_RBP.src.coef_metrics as coef_metrics
import mRNA_RBP.src.viz as viz

OUT_DIR_DEFAULT = os.path.join(_HERE, "outputs", "notebook_plots")


class _CoefView:
    """Minimal (alpha, beta) container -- everything viz.plot_gt_vs_surrogate needs."""
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


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


def _oracle_coef_view(inst_dir: str, score_key: str, stem_pairs: list) -> _CoefView:
    """Reconstruct (alpha, beta) for a real oracle from cached ssm.npz / pairwise_lib.npz."""
    ssm_path = os.path.join(inst_dir, "ssm.npz")
    d_ssm = np.load(ssm_path)
    wt_key = f"wt_score_{score_key}"
    wt_score = float(d_ssm[wt_key][0]) if wt_key in d_ssm.files else 0.0

    delta_W = make_ssm_deltas_from_scores(
        d_ssm["nuc_ids"], d_ssm[f"scores_{score_key}"], wt_idx=None, wt_activity=wt_score,
    )
    alpha = delta_W.T  # (4, L) -> (L, 4)

    beta = {}
    pw_path = os.path.join(inst_dir, "pairwise_lib.npz")
    if os.path.isfile(pw_path) and stem_pairs:
        d_pw = np.load(pw_path)
        pw_scores = d_pw[f"scores_{score_key}"].astype(np.float64) - wt_score
        for p, (i, j) in enumerate(stem_pairs):
            block = pw_scores[p * 16:(p + 1) * 16].reshape(4, 4)
            beta[(int(i), int(j))] = block

    return _CoefView(alpha, beta)


def build_gt_and_surrogate_views(oracle_name: str, instance: int, seq: str, stem_pairs: list,
                                 motif_positions: list, inst_dir: str, coef_path: str):
    """Load (gt_view, sur_alpha, sur_beta) for one condition.

    ``gt_view`` exposes ``.alpha`` (L,4) / ``.beta`` ({(i,j):(4,4)}) -- the
    analytic ``MrnaRbpGroundTruth`` for the synthetic GT, or a reconstructed
    ``_CoefView`` (from cached ssm.npz/pairwise_lib.npz) for any real oracle.
    """
    if not os.path.isfile(coef_path):
        raise FileNotFoundError(f"Coef file not found: {coef_path}")
    score_key = primary_gt_key(oracle_name)

    coefs  = np.load(coef_path)
    y_std  = float(coefs["y_std"]) if "y_std" in coefs else 1.0
    sur_alpha = coefs["alpha"] * y_std
    sur_beta  = {(int(i), int(j)): coefs["J"][i, :, j, :] * y_std
                 for (i, j) in stem_pairs}

    if oracle_name == MRNA_ORACLE:
        from mRNA_RBP.scripts.pipeline.generate_libraries import STEM_SIGMA
        gt = MrnaRbpGroundTruth(seq, stem_pairs, motif_positions,
                                seed=instance, stem_sigma=STEM_SIGMA)
    else:
        gt = _oracle_coef_view(inst_dir, score_key, stem_pairs)

    return gt, sur_alpha, sur_beta


def main(oracle_name: str = MRNA_ORACLE, wt_activity: str = "high",
        instance: int = 0, mut_rate: int = 10, lib_size: int = 20000,
        out_base: str = None, out_dir: str = None, save_svg: bool = False):
    oracle_name = normalize_oracle_name(oracle_name)
    score_key = primary_gt_key(oracle_name)
    out_base = out_base or default_output_base(_HERE, oracle_name, wt_activity)
    out_dir = out_dir or OUT_DIR_DEFAULT
    inst_dir = os.path.join(out_base, f"instance_{instance:02d}")
    seq, stem_pairs, motif_positions = sequence_config_for_oracle(oracle_name, wt_activity)
    if oracle_name != MRNA_ORACLE:
        # Prefer the per-instance edges actually used to build this
        # instance's libraries over the sequence_config_for_oracle value.
        # For fixed-structure oracles these coincide; for mrna_negative_control
        # topology is chosen randomly per instance, so sequence_config_for_oracle
        # returns [] and the real edges only exist in this cached file.
        gt_params_path = os.path.join(inst_dir, "gt_params.npz")
        if os.path.isfile(gt_params_path):
            d_params = np.load(gt_params_path)
            stem_pairs = [(int(i), int(j)) for (i, j) in d_params["edges"]]
    L = len(seq)

    coef_path = os.path.join(
        out_base, "surrogate_coefs",
        f"coefs_k{instance:02d}_mut{mut_rate:02d}_lib{lib_size}"
        f"_nonlinear_additive_p_pairwise_{score_key}.npz",
    )

    os.makedirs(out_dir, exist_ok=True)

    name_suffix = "synthetic" if oracle_name == MRNA_ORACLE else ORACLE_SHORT_NAME.get(oracle_name, oracle_name)

    gt, sur_alpha, sur_beta = build_gt_and_surrogate_views(
        oracle_name, instance, seq, stem_pairs, motif_positions, inst_dir, coef_path,
    )

    similarity = coef_metrics.compute_coefficient_similarity(
        gt.alpha, sur_alpha, gt.beta, sur_beta, stem_pairs,
    )
    sim_path = os.path.join(out_dir, f"coefficients_similarity_{name_suffix}.json")
    with open(sim_path, "w") as f:
        json.dump({
            "oracle": oracle_name, "wt_activity": wt_activity, "instance": instance,
            "mut_rate": mut_rate, "lib_size": lib_size, **similarity,
        }, f, indent=2)
    print(f"Mean cosine similarity (additive):                 {similarity['mean_cosine_similarity_additive']}")
    print(f"Mean cosine similarity (pairwise):                 {similarity['mean_cosine_similarity_pairwise']}")
    print(f"Mean cosine similarity (additive, sign-corrected): {similarity['mean_cosine_similarity_additive_sign_corrected']}")
    print(f"Mean cosine similarity (pairwise, sign-corrected): {similarity['mean_cosine_similarity_pairwise_sign_corrected']}")
    print(f"Saved → {sim_path}")

    fig_gt, fig_sur = viz.plot_gt_vs_surrogate(
        gt, sur_alpha, sur_beta, L, stem_pairs=stem_pairs, motif_positions=motif_positions,
    )

    fig_gt.savefig(os.path.join(out_dir, "coefficients_gt.png"),
                   dpi=150, bbox_inches="tight")
    if save_svg:
        fig_gt.savefig(os.path.join(out_dir, "coefficients_gt.svg"), bbox_inches="tight")
    print("Saved → coefficients_gt.png" + (" / .svg" if save_svg else ""))

    fig_sur.savefig(os.path.join(out_dir, "coefficients_surrogate.png"),
                    dpi=150, bbox_inches="tight")
    if save_svg:
        fig_sur.savefig(os.path.join(out_dir, "coefficients_surrogate.svg"), bbox_inches="tight")
    print("Saved → coefficients_surrogate.png" + (" / .svg" if save_svg else ""))

    map_dir = os.path.join(out_dir, "coefficients_map")
    os.makedirs(map_dir, exist_ok=True)
    combined = hstack_images([fig_to_array(fig_gt), fig_to_array(fig_sur)])
    label = "groundtruth" if oracle_name == MRNA_ORACLE else "oracle"
    combined_path = os.path.join(map_dir, f"coefficients_{label}_vs_surrogate_{name_suffix}.png")
    Image.fromarray(combined).save(combined_path)
    print(f"Saved → {combined_path}")

    plt.close(fig_gt)
    plt.close(fig_sur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=MRNA_ORACLE,
                        choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1",
                                 "hur", "residualbind_hur", "qki", "residualbind_qki",
                                 "twister", "twister_ribozyme", "deepsquid_hur", "deepsquid_vts1",
                                 "mrna_negative_control", "negative_control"])
    parser.add_argument("--wt_activity", choices=["high", "low"], default="high")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--mut_rate", type=int, default=10)
    parser.add_argument("--lib_size", type=int, default=20000)
    parser.add_argument("--out_base", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--svg", action="store_true",
                        help="Also save an .svg copy of each figure")
    args = parser.parse_args()
    main(oracle_name=args.oracle, wt_activity=args.wt_activity,
         instance=args.instance, mut_rate=args.mut_rate, lib_size=args.lib_size,
         out_base=args.out_base, out_dir=args.out_dir, save_svg=args.svg)
