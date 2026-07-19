"""mRNA_RBP/score_residualbind_vts1_mut10.py

Scores the EXISTING VTS1-GCUGG fixed mut=10%/lib=20,000 library (generated
by generate_vts1_mut10_library.py) with a real ResidualBind model instead of
the synthetic ground truth. Supports two oracles (see --oracle): the MSI1
ensemble (default) or the real VTS1 model (RNCMPT00111 -- see oracles.py).

Run with the environment matching the chosen oracle:
    --oracle residualbind_ensemble (default): RBP Torch env
        /home/nagle/final_version/rbp/.venv/bin/python \
            mRNA_RBP/score_residualbind_vts1_mut10.py
    --oracle vts1_residualbind: residbind conda env (TF2.18/Keras3)
        /home/nagle/miniconda3/envs/residbind/bin/python \
            mRNA_RBP/score_residualbind_vts1_mut10.py --oracle vts1_residualbind

Reads:  mRNA_RBP/outputs/instance_00/vts1_mut10/lib_20000.npz
Writes: mRNA_RBP/outputs_<oracle>/instance_00/vts1_mut10/lib_20000.npz
    nuc_ids          (N, L) uint8   -- passthrough
    scores_<gt_key>  (N,)   float32
"""

import argparse
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.generate_libraries import STEM_SIGMA
from mRNA_RBP.generate_varied_mutrate_library import (
    VTS1_MOTIF_POSITIONS, VTS1_SEQ, VTS1_STEM_PAIRS,
)
from mRNA_RBP.oracles import (
    RESIDUALBIND_ORACLE, build_oracle, default_output_base,
    normalize_oracle_name, primary_gt_key,
)

INSTANCE = 0
SRC_PATH = os.path.join(_HERE, "outputs", f"instance_{INSTANCE:02d}",
                        "vts1_mut10", "lib_20000.npz")


def _score_nuc_ids(oracle, gt_key, nuc_ids, chunk=4096):
    parts = []
    for start in range(0, len(nuc_ids), chunk):
        ids = nuc_ids[start:start + chunk]
        x = np.eye(4, dtype=np.float32)[ids]
        parts.append(oracle.score_all(x)[gt_key].astype(np.float32))
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=RESIDUALBIND_ORACLE,
                        choices=[RESIDUALBIND_ORACLE, "vts1_residualbind"])
    parser.add_argument("--residualbind_dir", default=None,
                        help="Checkpoint dir (MSI1 ensemble) or weights path (VTS1)")
    parser.add_argument("--out_base", default=None,
                        help="Defaults to outputs_<oracle>/ next to mRNA_RBP/")
    parser.add_argument("--src_path", default=SRC_PATH)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing scored lib_20000.npz")
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    gt_key = primary_gt_key(oracle_name)

    if not os.path.exists(args.src_path):
        raise FileNotFoundError(
            f"{args.src_path} not found -- run generate_vts1_mut10_library.py first"
        )

    out_base = args.out_base or default_output_base(_HERE, oracle_name)
    dst_path = os.path.join(out_base, f"instance_{INSTANCE:02d}",
                            "vts1_mut10", "lib_20000.npz")
    if os.path.exists(dst_path) and not args.overwrite:
        print(f"[skip] already scored {dst_path}")
        return

    print(f"[load] building {oracle_name} oracle...", flush=True)
    oracle = build_oracle(
        oracle_name,
        seq=VTS1_SEQ,
        stem_pairs=VTS1_STEM_PAIRS,
        motif_positions=VTS1_MOTIF_POSITIONS,
        seed=INSTANCE,
        stem_sigma=STEM_SIGMA,
        residualbind_dir=args.residualbind_dir,
    )

    d = np.load(args.src_path)
    nuc_ids = d["nuc_ids"]
    print(f"[score] vts1_mut10 lib  n={len(nuc_ids):,}", flush=True)
    scores = _score_nuc_ids(oracle, gt_key, nuc_ids)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    np.savez_compressed(dst_path, nuc_ids=nuc_ids,
                        **{f"scores_{gt_key}": scores})
    print(f"  -> saved {dst_path}  "
          f"(mean={scores.mean():+.4f}  std={scores.std():.4f}  "
          f"min={scores.min():+.4f}  max={scores.max():+.4f})", flush=True)


if __name__ == "__main__":
    main()
