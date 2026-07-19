"""mRNA_RBP/experiments/score_ssm_cross_mutrate.py

Generates the single-site mutagenesis (SSM) library -- all L*3 single-mutant
sequences -- and scores it with a real ResidualBind ensemble oracle (MSI1 or
VTS1, selected via --oracle). This gives plot_cross_mutrate.py's heatmap SSM
row a true oracle single-mutant baseline (additive-only reconstruction of the
real ResidualBind readout), instead of falling back to a per-surrogate
alpha-only reconstruction from the trained MAVE-NN coefficients.

Run with a torch+CUDA-capable environment (same as score_residualbind_cross_mutrate.py):

    /home/nagle/miniconda3/envs/toehold_gpu/bin/python \
        mRNA_RBP/experiments/score_ssm_cross_mutrate.py --oracle vts1

Writes: mRNA_RBP/outputs_<oracle>/instance_00/ssm.npz
    nuc_ids            (3L, L) uint8
    scores_<gt_key>    (3L,) float32
"""

import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import (
    SEQ, STEM_PAIRS, MOTIF_POSITIONS, STEM_SIGMA, generate_ssm,
)
from mRNA_RBP.oracles import (
    RESIDUALBIND_MSI1_ORACLE,
    build_oracle, default_output_base, normalize_oracle_name, primary_gt_key,
)
from mRNA_RBP.sequence_configs import (
    VTS1_MOTIF_POSITIONS,
    VTS1_SEQ,
    VTS1_STEM_PAIRS,
)
from mRNA_RBP.seq_utils import rna_to_one_hot

INSTANCE = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True,
                        help="residualbind_msi1 / msi1 or residualbind_vts1 / vts1")
    parser.add_argument("--residualbind_dir", default=None)
    parser.add_argument("--out_base", default=None,
                        help="Defaults to outputs_<oracle>/ next to mRNA_RBP/")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    gt_key = primary_gt_key(oracle_name)
    out_base = args.out_base or default_output_base(_HERE, oracle_name)
    is_msi1 = oracle_name == RESIDUALBIND_MSI1_ORACLE
    seq = SEQ if is_msi1 else VTS1_SEQ
    stem_pairs = STEM_PAIRS if is_msi1 else VTS1_STEM_PAIRS
    motif_positions = MOTIF_POSITIONS if is_msi1 else VTS1_MOTIF_POSITIONS

    dst_dir = os.path.join(out_base, f"instance_{INSTANCE:02d}")
    dst_path = os.path.join(dst_dir, "ssm.npz")
    if os.path.exists(dst_path):
        d = np.load(dst_path, allow_pickle=True)
        if "wt_seq" in d and str(d["wt_seq"].item()) == seq:
            print(f"[skip] already scored {dst_path}")
            return

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

    wt_onehot = rna_to_one_hot(seq)
    nuc_ids = generate_ssm(wt_onehot)
    x = np.eye(4, dtype=np.float32)[nuc_ids]
    print(f"[score] SSM library  n={len(nuc_ids)}", flush=True)
    scores = oracle.score_all(x)[gt_key].astype(np.float32)

    os.makedirs(dst_dir, exist_ok=True)
    np.savez_compressed(
        dst_path,
        nuc_ids=nuc_ids,
        wt_seq=np.array(seq),
        stem_pairs=np.asarray(stem_pairs, dtype=np.int32),
        motif_positions=np.asarray(motif_positions, dtype=np.int32),
        **{f"scores_{gt_key}": scores},
    )
    print(f"  -> saved {dst_path}  (mean={scores.mean():+.4f}  std={scores.std():.4f})")


if __name__ == "__main__":
    main()
