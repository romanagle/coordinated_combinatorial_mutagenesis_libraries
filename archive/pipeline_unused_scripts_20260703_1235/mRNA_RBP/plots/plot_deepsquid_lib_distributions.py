"""
mRNA_RBP/plots/plot_deepsquid_lib_distributions.py

"Deep squid" version of rand_lib_dist.png / rand_lib_dist_<rbp>_oracle.png:
instead of the synthetic GT or the raw ResidualBind ensemble, this scores the
same random libraries with the nonlinear additive+pairwise MAVE-NN surrogate
that was trained on the 200K varied-mutation-rate (3-10 mut) library --
i.e. that surrogate used as an interpretable stand-in oracle
(oracles.SurrogateMAVENNOracle), per the user's naming convention:
"deep squid" = the SQUID surrogate trained on the 200K library, treated as an
oracle -- not the raw ResidualBind ensemble.

Reuses the exact same nuc_ids as the raw-oracle version so panels are
directly comparable sequence-for-sequence:
  MSI1 rand05/10/25 -- outputs_vts1_residualbind/outputs_residualbind_ensemble/
                        instance_00/mut{05,10,25}/lib_20000.npz
  VTS1 rand10        -- outputs_vts1_residualbind/instance_00/vts1_mut10/lib_20000.npz
  VTS1 rand05/25     -- cached nuc_ids in outputs/notebook_plots/cache_vts1_lib_scores.npz

Also builds the "deep squid" version of library_distributions_<rbp>_oracle.png
(activity-balanced / Pairwise / Exhaustive panels), reusing the exact same nuc_ids/labels
already cached by plot_oracle_lib_distributions.py:
  MSI1 activity-balanced/pairwise/type3 -- outputs/instance_00/{activity_balanced,pairwise_lib,type3}.npz
                                (canonical synthetic-GT instance; MSI1_SEQ ==
                                that canonical SEQ)
  VTS1 activity-balanced/pairwise/type3 -- cached nuc_ids/labels in cache_vts1_lib_scores.npz

Output (separate files per RBP):
  outputs/notebook_plots/rand_lib_dist_msi1_deepsquid.png
  outputs/notebook_plots/rand_lib_dist_vts1_deepsquid.png
  outputs/notebook_plots/library_distributions_msi1_deepsquid.png
  outputs/notebook_plots/library_distributions_vts1_deepsquid.png

Usage (squid env -- needs mavenn/tensorflow):
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/plots/plot_deepsquid_lib_distributions.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401 -- import before mavenn/tensorflow, see note below
import matplotlib.cm as cm
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.oracles import SurrogateMAVENNOracle  # noqa: E402
from mRNA_RBP.plots.plot_oracle_lib_distributions import (  # noqa: E402
    _stacked_hist, _plain_hist, shared_xlim, C_PAIR,
)
from mRNA_RBP.sequence_configs import MSI1_SEQ, MSI1_STEM_PAIRS, VTS1_SEQ, VTS1_STEM_PAIRS  # noqa: E402

MSI1_MODEL_DIR  = os.path.join(
    _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble",
    "surrogate_models", "varied_mutrate_nonlin_pairwise")
MSI1_RB_BASE    = os.path.join(
    _HERE, "outputs_vts1_residualbind", "outputs_residualbind_ensemble", "instance_00")

VTS1_MODEL_DIR  = os.path.join(
    _HERE, "outputs_vts1_residualbind", "surrogate_models", "varied_mutrate_nonlin_pairwise")
VTS1_LIB_BASE   = os.path.join(_HERE, "outputs_vts1_residualbind", "instance_00")
VTS1_CACHE      = os.path.join(_HERE, "outputs", "notebook_plots", "cache_vts1_lib_scores.npz")

SYNTH_INST = os.path.join(_HERE, "outputs", "instance_00")   # MSI1_SEQ == canonical SEQ

OUT_DIR = os.path.join(_HERE, "outputs", "notebook_plots")
BINS = 60


def _score(oracle: SurrogateMAVENNOracle, nuc_ids: np.ndarray) -> np.ndarray:
    x = np.eye(4, dtype=np.float32)[nuc_ids]
    return oracle.score_all(x)[oracle.score_key].astype(np.float64)


def build_msi1_data(oracle: SurrogateMAVENNOracle) -> dict:
    data = {}
    for mut_dir, label in [("mut05", "rand05"), ("mut10", "rand10"), ("mut25", "rand25")]:
        d = np.load(os.path.join(MSI1_RB_BASE, mut_dir, "lib_20000.npz"))
        nuc_ids = d["nuc_ids"]
        print(f"  MSI1 {label}: scoring n={len(nuc_ids):,} with deep squid surrogate")
        data[label] = {"scores": _score(oracle, nuc_ids), "n": len(nuc_ids)}

    t2_path = os.path.join(SYNTH_INST, "activity_balanced.npz")
    if not os.path.isfile(t2_path):
        t2_path = os.path.join(SYNTH_INST, "type2.npz")
    d_t2 = np.load(t2_path)
    print(f"  MSI1 activity-balanced: scoring n={len(d_t2['nuc_ids']):,} with deep squid surrogate")
    data["type2"] = {"scores": _score(oracle, d_t2["nuc_ids"]),
                      "labels": d_t2["rate_labels"].astype(int)}

    d_pw = np.load(os.path.join(SYNTH_INST, "pairwise_lib.npz"))
    print(f"  MSI1 pairwise: scoring n={len(d_pw['nuc_ids']):,} with deep squid surrogate")
    data["pairwise"] = {"scores": _score(oracle, d_pw["nuc_ids"])}

    d_t3 = np.load(os.path.join(SYNTH_INST, "type3.npz"))
    print(f"  MSI1 type3: scoring n={len(d_t3['nuc_ids']):,} with deep squid surrogate")
    data["type3"] = {"scores": _score(oracle, d_t3["nuc_ids"]),
                      "labels": d_t3["rate_labels"].astype(np.uint8)}

    return data


def build_vts1_data(oracle: SurrogateMAVENNOracle) -> dict:
    data = {}

    d_r10 = np.load(os.path.join(VTS1_LIB_BASE, "vts1_mut10", "lib_20000.npz"))
    nids_10 = d_r10["nuc_ids"]
    print(f"  VTS1 rand10: scoring n={len(nids_10):,} with deep squid surrogate")
    data["rand10"] = {"scores": _score(oracle, nids_10), "n": len(nids_10)}

    cache = np.load(VTS1_CACHE, allow_pickle=True)
    if "wt_seq" not in cache or str(cache["wt_seq"].item()) != VTS1_SEQ:
        raise RuntimeError(
            f"{VTS1_CACHE} is stale or not VTS1-sequence native. "
            "Regenerate it with plot_oracle_lib_distributions.py first."
        )
    for label in ("rand05", "rand25"):
        nids = cache[f"{label}_nids"].astype(np.int32)
        print(f"  VTS1 {label}: scoring n={len(nids):,} with deep squid surrogate")
        data[label] = {"scores": _score(oracle, nids), "n": len(nids)}

    nids_t2 = cache["type2_nids"].astype(np.int32)
    print(f"  VTS1 activity-balanced: scoring n={len(nids_t2):,} with deep squid surrogate")
    data["type2"] = {"scores": _score(oracle, nids_t2),
                      "labels": cache["type2_labels"].astype(np.int32)}

    nids_pw = cache["pairwise_nids"].astype(np.int32)
    print(f"  VTS1 pairwise: scoring n={len(nids_pw):,} with deep squid surrogate")
    data["pairwise"] = {"scores": _score(oracle, nids_pw)}

    nids_t3 = cache["type3_nids"].astype(np.int32)
    print(f"  VTS1 type3: scoring n={len(nids_t3):,} with deep squid surrogate")
    data["type3"] = {"scores": _score(oracle, nids_t3),
                      "labels": cache["type3_labels"].astype(np.int32)}

    return data


def plot_rand_lib_dist(data: dict, rbp_label: str, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (label, pct) in zip(axes, [("rand05", 5), ("rand10", 10), ("rand25", 25)]):
        scores = data[label]["scores"]
        n = data[label]["n"]
        ax.hist(scores, bins=BINS, density=True, color="#4C72B0", alpha=0.85)
        mean = float(np.mean(scores))
        ax.axvline(mean, color="black", linestyle="--", linewidth=1.3,
                   label=f"mean={mean:.3f}")
        ax.set_title(f"Random lib ({pct}% mut, n={n:,})", fontsize=10)
        ax.set_xlabel(f"Deep squid surrogate score (Δ activity, {rbp_label})")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_library_distributions(data: dict, rbp_label: str, out_path: str):
    xlabel = f"Deep squid surrogate score (Δ activity, {rbp_label})"
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.subplots_adjust(wspace=0.35)

    t2 = data["type2"]
    scores_t2, labels_t2 = t2["scores"], t2["labels"]
    mut_range_t2 = sorted(set(labels_t2.tolist()))
    cmap_t2 = cm.get_cmap("plasma_r", len(mut_range_t2) + 2)
    colors_t2 = {m: cmap_t2(i + 1) for i, m in enumerate(mut_range_t2)}
    _stacked_hist(axes[0], scores_t2, labels_t2, mut_range_t2, colors_t2,
                  title=f"Activity-balanced library  (n={len(scores_t2):,})", xlabel=xlabel)

    scores_pw = data["pairwise"]["scores"]
    _plain_hist(axes[1], scores_pw,
                title=f"Pairwise library  (n={len(scores_pw):,})",
                xlabel=xlabel, color=C_PAIR)

    t3 = data["type3"]
    scores_t3, labels_t3 = t3["scores"], t3["labels"]
    mut_range_t3 = sorted(set(labels_t3.tolist()))
    cmap_t3 = cm.get_cmap("viridis", len(mut_range_t3) + 2)
    colors_t3 = {m: cmap_t3(i + 1) for i, m in enumerate(mut_range_t3)}
    _stacked_hist(axes[2], scores_t3, labels_t3, mut_range_t3, colors_t3,
                  title=f"Exhaustive library  (n={len(scores_t3):,})", xlabel=xlabel)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== MSI1 deep squid surrogate ===")
    msi1_oracle = SurrogateMAVENNOracle(seq=MSI1_SEQ, stem_pairs=MSI1_STEM_PAIRS,
                                        model_dir=MSI1_MODEL_DIR)
    msi1 = build_msi1_data(msi1_oracle)
    plot_rand_lib_dist(msi1, "MSI1", os.path.join(OUT_DIR, "rand_lib_dist_msi1_deepsquid.png"))
    plot_library_distributions(msi1, "MSI1",
        os.path.join(OUT_DIR, "library_distributions_msi1_deepsquid.png"))

    print("=== VTS1 deep squid surrogate ===")
    vts1_oracle = SurrogateMAVENNOracle(seq=VTS1_SEQ, stem_pairs=VTS1_STEM_PAIRS,
                                        model_dir=VTS1_MODEL_DIR)
    vts1 = build_vts1_data(vts1_oracle)
    plot_rand_lib_dist(vts1, "VTS1", os.path.join(OUT_DIR, "rand_lib_dist_vts1_deepsquid.png"))
    plot_library_distributions(vts1, "VTS1",
        os.path.join(OUT_DIR, "library_distributions_vts1_deepsquid.png"))


if __name__ == "__main__":
    main()
