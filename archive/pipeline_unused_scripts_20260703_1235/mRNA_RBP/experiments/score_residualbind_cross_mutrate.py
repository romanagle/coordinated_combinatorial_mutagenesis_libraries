"""mRNA_RBP/score_residualbind_cross_mutrate.py

Step A of the ResidualBind cross-mutation-rate experiment. Scores the
canonical MSI1 nucleotide libraries with the MSI1 ensemble, or VTS1-specific
exact-count mutant libraries with the VTS1 ensemble, for instance 0 across all
3 mutation rates x 3 library sizes (9 files).

Run with a torch+CUDA-capable environment, e.g. toehold_gpu (NOT the squid
environment -- squid's torch install can't load CUDA on this machine):

    /home/nagle/miniconda3/envs/toehold_gpu/bin/python \
        mRNA_RBP/experiments/score_residualbind_cross_mutrate.py --oracle vts1

Output mirrors the synthetic-GT lib npz schema (nuc_ids + one score key)
under the oracle's default_output_base() convention:

    mRNA_RBP/outputs_<oracle>/instance_00/mut{pct:02d}/lib_{n}.npz
        nuc_ids            (N, L) uint8
        wt_seq             scalar string
        scores_<gt_key>    (N,) float32

Consumed by mRNA_RBP/experiments/cross_mutrate_residualbind.py (run in the
squid env).
"""

import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    OUT_BASE, SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA,
    MUT_RATES_PCT, LIB_SIZES, mut_count_for,
)
from mRNA_RBP.oracles import (
    RESIDUALBIND_MSI1_ORACLE,
    build_oracle, default_output_base, normalize_oracle_name, primary_gt_key,
)
from mRNA_RBP.sequence_configs import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
    sample_exact_mutants,
)

INSTANCE = 0


def _score_nuc_ids(oracle, nuc_ids, gt_key, chunk=4096):
    parts = []
    for start in range(0, len(nuc_ids), chunk):
        ids = nuc_ids[start:start + chunk]
        x = np.eye(4, dtype=np.float32)[ids]
        parts.append(oracle.score_all(x)[gt_key].astype(np.float32))
    return np.concatenate(parts)


def _saved_seq_matches(path: str, seq: str) -> bool:
    if not os.path.exists(path):
        return False
    d = np.load(path, allow_pickle=True)
    if "wt_seq" not in d:
        return False
    return str(d["wt_seq"].item()) == seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True,
                        help="residualbind_msi1 / msi1 or residualbind_vts1 / vts1")
    parser.add_argument("--residualbind_dir", default=None)
    parser.add_argument("--out_base", default=None,
                        help="Defaults to outputs_<oracle>/ next to mRNA_RBP/")
    parser.add_argument("--mut_rates", nargs="+", type=int, default=MUT_RATES_PCT)
    parser.add_argument("--lib_sizes", nargs="+", type=int, default=LIB_SIZES)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    gt_key = primary_gt_key(oracle_name)
    out_base = args.out_base or default_output_base(_HERE, oracle_name)
    is_msi1 = oracle_name == RESIDUALBIND_MSI1_ORACLE
    seq = SEQ if is_msi1 else VTS1_SEQ
    stem_pairs = STEM_PAIRS if is_msi1 else VTS1_STEM_PAIRS
    motif_positions = MOTIF_POSITIONS if is_msi1 else VTS1_MOTIF_POSITIONS

    print(f"[load] building {oracle_name} ensemble oracle...", flush=True)
    oracle = build_oracle(
        oracle_name,
        seq=seq,
        stem_pairs=stem_pairs,
        motif_positions=motif_positions,
        seed=INSTANCE,
        stem_sigma=STEM_SIGMA,
        residualbind_dir=args.residualbind_dir,
    )

    for pct in args.mut_rates:
        for n_lib in args.lib_sizes:
            dst_dir = os.path.join(out_base, f"instance_{INSTANCE:02d}", f"mut{pct:02d}")
            dst_path = os.path.join(dst_dir, f"lib_{n_lib}.npz")
            if _saved_seq_matches(dst_path, seq):
                print(f"  [skip] already scored {dst_path}")
                continue

            if is_msi1:
                src_path = os.path.join(OUT_BASE, f"instance_{INSTANCE:02d}",
                                        f"mut{pct:02d}", f"lib_{n_lib}.npz")
                if not os.path.exists(src_path):
                    print(f"  [skip] missing {src_path}")
                    continue
                nuc_ids = np.load(src_path)["nuc_ids"]
            else:
                mc = mut_count_for(pct, len(seq))
                nuc_ids = sample_exact_mutants(
                    seq, mc, n_lib, seed=INSTANCE * 10_000 + pct * 100 + n_lib
                )

            print(f"  [score] mut{pct:02d} lib_{n_lib}  n={len(nuc_ids):,}", flush=True)
            scores = _score_nuc_ids(oracle, nuc_ids, gt_key)

            os.makedirs(dst_dir, exist_ok=True)
            np.savez_compressed(dst_path, nuc_ids=nuc_ids,
                                wt_seq=np.array(seq),
                                stem_pairs=np.asarray(stem_pairs, dtype=np.int32),
                                motif_positions=np.asarray(motif_positions, dtype=np.int32),
                                **{f"scores_{gt_key}": scores})
            print(f"    -> saved {dst_path}  "
                  f"(mean={scores.mean():+.4f}  std={scores.std():.4f})", flush=True)

    print(f"\n[done] {oracle_name} scoring complete.")


if __name__ == "__main__":
    main()
