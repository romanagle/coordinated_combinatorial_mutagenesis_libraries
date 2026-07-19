"""Precompute ResidualBind ensemble scores for the mRNA Type3 comparison.

Run with the RBP Torch environment, not the SQUID environment, e.g.:

    /home/nagle/miniconda3/envs/toehold_gpu/bin/python mRNA_RBP/experiments/score_residualbind_type3.py
    /home/nagle/miniconda3/envs/toehold_gpu/bin/python mRNA_RBP/experiments/score_residualbind_type3.py --oracle residualbind_vts1

Scores the canonical MSI1 mRNA_RBP library with the MSI1 ensemble, or native
VTS1-sequence libraries with the VTS1 ensemble. Do not conflate the two: they
are independently trained ensembles (see oracles.RESIDUALBIND_MSI1_ORACLE /
_VTS1_ORACLE).

The output cache is consumed by ``bar_surrogate_models_type3.py --score_cache``
and ``plots/plot_coefficients_residualbind.py --score_cache``.
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA,
    ACTIVITY_BALANCED_CANDIDATE_N,
    ACTIVITY_BALANCED_MUT_COUNTS,
    ACTIVITY_BALANCED_TARGET_N,
    activity_balanced_path,
)
from mRNA_RBP.ground_truth import uniformize_by_histogram
from mRNA_RBP.oracles import (
    RESIDUALBIND_MSI1_ORACLE,
    build_oracle,
    normalize_oracle_name,
    primary_gt_key,
)
from mRNA_RBP.sequence_configs import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
    generate_pairwise_nuc_ids,
    generate_type3_nuc_ids,
    sample_exact_mutants,
)

INSTANCE = 0
MUT_PCT = 10
LIB_SIZE = 20_000

def _load_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)["nuc_ids"]


def _score_ids(oracle, score_key, nuc_ids):
    x = np.eye(4, dtype=np.float32)[nuc_ids]
    return oracle.score_all(x)[score_key].astype(np.float32)


def _build_vts1_activity_balanced(oracle, score_key):
    per_count, rem = divmod(ACTIVITY_BALANCED_CANDIDATE_N,
                            len(ACTIVITY_BALANCED_MUT_COUNTS))
    all_ids = []
    for i, mut_count in enumerate(ACTIVITY_BALANCED_MUT_COUNTS):
        target_n = per_count + (1 if i < rem else 0)
        ids = sample_exact_mutants(
            VTS1_SEQ, mut_count, target_n, seed=20_000 + mut_count
        )
        all_ids.append(ids)
        print(f"[build] activity-balanced {mut_count}-mut candidates n={len(ids):,}",
              flush=True)
    candidates = np.concatenate(all_ids, axis=0)
    candidate_scores = _score_ids(oracle, score_key, candidates)
    _, keep = uniformize_by_histogram(
        candidate_scores, X=None, n_bins=200, clip_lo=1, clip_hi=99,
        target_n=ACTIVITY_BALANCED_TARGET_N, seed=20_600,
    )
    selected = candidates[keep]
    counts = (selected != np.array([{"A": 0, "C": 1, "G": 2, "U": 3}[c]
                                    for c in VTS1_SEQ], dtype=np.uint8)).sum(axis=1)
    vals, cnts = np.unique(counts, return_counts=True)
    print("[build] activity-balanced selected n="
          f"{len(selected):,}: " +
          "  ".join(f"{m}mut:{c}" for m, c in zip(vals, cnts)),
          flush=True)
    return selected


def _ssm_delta(oracle, score_key):
    wt_oh = oracle.wt_one_hot()
    wt_idx = np.argmax(wt_oh, axis=1).astype(np.uint8)
    seqs, positions, nucs = [], [], []
    for pos in range(len(wt_idx)):
        for nuc in range(4):
            seq = wt_idx.copy()
            seq[pos] = nuc
            seqs.append(seq)
            positions.append(pos)
            nucs.append(nuc)
    nuc_ids = np.stack(seqs).astype(np.uint8)
    scores = _score_ids(oracle, score_key, nuc_ids)
    delta = np.zeros((4, len(wt_idx)), dtype=np.float32)
    for idx, (pos, nuc) in enumerate(zip(positions, nucs)):
        delta[nuc, pos] = scores[idx]
    return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Defaults to outputs/residualbind_type3_scores_instance00.npz "
                             "(MSI1) or _vts1_instance00.npz (VTS1)")
    parser.add_argument("--oracle", default=RESIDUALBIND_MSI1_ORACLE,
                        choices=["residualbind_ensemble", "residualbind_msi1", "msi1",
                                 "vts1", "residualbind_vts1", "vts1_residualbind"])
    parser.add_argument("--residualbind_dir", default=None)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    score_key = primary_gt_key(oracle_name)
    is_msi1 = oracle_name == RESIDUALBIND_MSI1_ORACLE
    suffix = "_vts1" if not is_msi1 else ""
    out = args.out or os.path.join(
        _HERE, "outputs", f"residualbind_type3_scores{suffix}_instance00.npz")

    if is_msi1:
        seq = SEQ
        stem_pairs = STEM_PAIRS
        motif_positions = MOTIF_POSITIONS
        inst_dir = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}")
        paths = {
            "train": os.path.join(inst_dir, f"mut{MUT_PCT:02d}", f"lib_{LIB_SIZE}.npz"),
            "type2": activity_balanced_path(inst_dir),
            "pairwise": os.path.join(inst_dir, "pairwise_lib.npz"),
            "type3": os.path.join(inst_dir, "type3.npz"),
        }
        ids = {name: _load_ids(path) for name, path in paths.items()}
    else:
        seq = VTS1_SEQ
        stem_pairs = VTS1_STEM_PAIRS
        motif_positions = VTS1_MOTIF_POSITIONS
        ids = {
            "train": sample_exact_mutants(
                VTS1_SEQ, mut_count=4, target_n=LIB_SIZE, seed=10_000 + LIB_SIZE
            ),
            "pairwise": generate_pairwise_nuc_ids(VTS1_SEQ, VTS1_STEM_PAIRS),
            "type3": generate_type3_nuc_ids(VTS1_SEQ)[0],
        }

    oracle = build_oracle(
        oracle_name,
        seq=seq,
        stem_pairs=stem_pairs,
        motif_positions=motif_positions,
        seed=INSTANCE,
        stem_sigma=STEM_SIGMA,
        residualbind_dir=args.residualbind_dir,
    )

    if not is_msi1:
        ids["type2"] = _build_vts1_activity_balanced(oracle, score_key)

    scores = {}
    for name, nuc_ids in ids.items():
        print(f"[score] {name:8s} n={len(nuc_ids):,}", flush=True)
        scores[f"scores_{name}"] = _score_ids(oracle, score_key, nuc_ids)

    print("[score] ssm deltas", flush=True)
    scores["ssm_delta"] = _ssm_delta(oracle, score_key)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        wt_seq=np.array(seq),
        stem_pairs=np.asarray(stem_pairs, dtype=np.int32),
        motif_positions=np.asarray(motif_positions, dtype=np.int32),
        **{f"nuc_ids_{name}": nuc_ids for name, nuc_ids in ids.items()},
        **scores,
    )
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
