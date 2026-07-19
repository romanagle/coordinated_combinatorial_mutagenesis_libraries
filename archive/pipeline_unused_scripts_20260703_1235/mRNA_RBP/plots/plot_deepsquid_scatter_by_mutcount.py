"""
mRNA_RBP/plots/plot_deepsquid_scatter_by_mutcount.py

Same experiment as plot_oracle_scatter_by_mutcount.py, but the training/eval
labels come from the "deep squid" surrogate -- the nonlinear additive+pairwise
MAVE-NN model trained on the 200K varied-mutation-rate library, treated as an
oracle (oracles.SurrogateMAVENNOracle) -- instead of the raw ResidualBind
ensemble.

Two output figures (one per RBP), each with 3 panels (5%, 10%, 25% mut rate):
  - Blue:    random holdout from training library (deep-squid-labeled)
  - Colored: activity-balanced library, colored by mutation count

Reuses the exact same nuc_ids as the raw-oracle scatter version (random libs
from the pre-generated ResidualBind training pools, activity-balanced from the fixed
20,000-sequence cache) so panels are directly comparable sequence-for-sequence.

Usage (squid env -- needs mavenn/tensorflow):
    /home/nagle/miniconda3/envs/squid/bin/python3.7 \
        mRNA_RBP/plots/plot_deepsquid_scatter_by_mutcount.py
"""

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "mRNA_RBP"))

from mRNA_RBP.oracles import SurrogateMAVENNOracle
from mRNA_RBP.generate_libraries import activity_balanced_path
from mRNA_RBP.plots.plot_oracle_scatter_by_mutcount import (
    MUT_RATES, run_one, make_figure, _OUT,
)
from mRNA_RBP.plots.plot_deepsquid_lib_distributions import (
    MSI1_SEQ, MSI1_STEM_PAIRS, MSI1_MODEL_DIR, MSI1_RB_BASE,
    VTS1_SEQ, VTS1_STEM_PAIRS, VTS1_MODEL_DIR, VTS1_LIB_BASE, VTS1_CACHE,
    SYNTH_INST, _score,
)

# plot_deepsquid_lib_distributions builds these paths with os.path.join (str);
# this module uses pathlib's "/" operator, so coerce them to Path here.
MSI1_RB_BASE  = Path(MSI1_RB_BASE)
VTS1_LIB_BASE = Path(VTS1_LIB_BASE)
VTS1_CACHE    = Path(VTS1_CACHE)
SYNTH_INST    = Path(SYNTH_INST)

SCORE_LABEL = "Deep squid surrogate score"
TITLE_LABEL = "deep squid oracle labels"


def run_msi1():
    print("\n=== MSI1 deep squid ===")
    oracle = SurrogateMAVENNOracle(seq=MSI1_SEQ, stem_pairs=MSI1_STEM_PAIRS,
                                   model_dir=MSI1_MODEL_DIR)

    d_t2 = np.load(activity_balanced_path(str(SYNTH_INST)))
    type2_nids   = d_t2["nuc_ids"].astype(np.int32)
    type2_labels = d_t2["rate_labels"].astype(int)
    print(f"  scoring activity-balanced (n={len(type2_nids):,}) with deep squid …")
    type2_oracle = _score(oracle, type2_nids)

    panel_data, labels_list = [], []
    for pct in MUT_RATES:
        d = np.load(MSI1_RB_BASE / f"mut{pct:02d}" / "lib_20000.npz")
        nuc_ids = d["nuc_ids"].astype(np.int32)
        print(f"  scoring mut{pct:02d}% random lib (n={len(nuc_ids):,}) with deep squid …")
        oracle_scores = _score(oracle, nuc_ids)
        data = run_one(nuc_ids, oracle_scores, type2_nids, type2_oracle,
                       f"MSI1 mut{pct:02d}%")
        panel_data.append(data)
        labels_list.append(type2_labels)

    make_figure(panel_data, labels_list, "MSI1",
                _OUT / "scatter_by_mutcount_msi1_deepsquid.png",
                score_label=SCORE_LABEL, title_label=TITLE_LABEL)


def run_vts1():
    print("\n=== VTS1 deep squid ===")
    oracle = SurrogateMAVENNOracle(seq=VTS1_SEQ, stem_pairs=VTS1_STEM_PAIRS,
                                   model_dir=VTS1_MODEL_DIR)

    vts1_cache   = np.load(VTS1_CACHE)
    if "wt_seq" not in vts1_cache or str(vts1_cache["wt_seq"].item()) != VTS1_SEQ:
        raise RuntimeError(
            f"{VTS1_CACHE} is stale or not VTS1-sequence native. "
            "Regenerate it with plot_oracle_lib_distributions.py first."
        )
    type2_nids   = vts1_cache["type2_nids"].astype(np.int32)
    type2_labels = vts1_cache["type2_labels"].astype(int)
    print(f"  scoring activity-balanced (n={len(type2_nids):,}) with deep squid …")
    type2_oracle = _score(oracle, type2_nids)

    d_r10 = np.load(VTS1_LIB_BASE / "vts1_mut10" / "lib_20000.npz")
    vts1_nids = {
        5:  vts1_cache["rand05_nids"].astype(np.int32),
        10: d_r10["nuc_ids"].astype(np.int32),
        25: vts1_cache["rand25_nids"].astype(np.int32),
    }

    panel_data, labels_list = [], []
    for pct in MUT_RATES:
        nuc_ids = vts1_nids[pct]
        print(f"  scoring mut{pct:02d}% random lib (n={len(nuc_ids):,}) with deep squid …")
        oracle_scores = _score(oracle, nuc_ids)
        data = run_one(nuc_ids, oracle_scores, type2_nids, type2_oracle,
                       f"VTS1 mut{pct:02d}%")
        panel_data.append(data)
        labels_list.append(type2_labels)

    make_figure(panel_data, labels_list, "VTS1",
                _OUT / "scatter_by_mutcount_vts1_deepsquid.png",
                score_label=SCORE_LABEL, title_label=TITLE_LABEL)


def main():
    run_msi1()
    run_vts1()
    print("\nDone.")


if __name__ == "__main__":
    main()
