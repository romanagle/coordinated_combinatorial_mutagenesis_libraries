"""
mRNA_RBP/scripts/pipeline/generate_model_comparison_score_cache.py

Builds the score-cache npz consumed by
mRNA_RBP/scripts/figures/core/bar_surrogate_models_type3.py --score_cache, purely from
already-cached standardized pipeline artifacts (ssm.npz, activity_balanced.npz,
pairwise_lib.npz, type3.npz, mut{pct}/lib_{N}.npz) -- no live oracle scoring,
no torch needed. Oracle-agnostic; works for any registered oracle including
the synthetic GT.

Usage:
    python mRNA_RBP/scripts/pipeline/generate_model_comparison_score_cache.py --oracle hur --wt_activity high
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.src.evaluate import make_ssm_deltas_from_scores
from mRNA_RBP.src.oracles import (
    MRNA_ORACLE, default_output_base, normalize_oracle_name, primary_gt_key,
)

INSTANCE = 0
MUT_PCT = 10
LIB_SIZE = 20_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", default=MRNA_ORACLE,
                        choices=[MRNA_ORACLE, "mrna", "residualbind", "residualbind_ensemble",
                                 "residualbind_msi1", "vts1", "residualbind_vts1",
                                 "hur", "residualbind_hur", "qki", "residualbind_qki",
                                 "twister", "twister_ribozyme", "deepsquid_hur", "deepsquid_vts1",
                                 "mrna_negative_control", "negative_control"])
    parser.add_argument("--wt_activity", choices=["high", "low"], default="high")
    parser.add_argument("--out_base", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    oracle_name = normalize_oracle_name(args.oracle)
    score_key = primary_gt_key(oracle_name)
    out_base = args.out_base or default_output_base(_HERE, oracle_name, args.wt_activity)
    inst_dir = os.path.join(out_base, f"instance_{INSTANCE:02d}")
    out_path = args.out or os.path.join(out_base, "model_comparison_score_cache.npz")

    ssm_path = os.path.join(inst_dir, "ssm.npz")
    d_ssm = np.load(ssm_path)
    wt_key = f"wt_score_{score_key}"
    wt_score = float(d_ssm[wt_key][0]) if wt_key in d_ssm.files else 0.0
    ssm_delta = make_ssm_deltas_from_scores(
        d_ssm["nuc_ids"], d_ssm[f"scores_{score_key}"], wt_idx=None, wt_activity=wt_score,
    )

    eval_paths = {
        "type2": os.path.join(inst_dir, "activity_balanced.npz"),
        "pairwise": os.path.join(inst_dir, "pairwise_lib.npz"),
        "type3": os.path.join(inst_dir, "type3.npz"),
    }
    if not os.path.isfile(eval_paths["type2"]):
        legacy = os.path.join(inst_dir, "type2.npz")
        if os.path.isfile(legacy):
            eval_paths["type2"] = legacy

    payload = {"ssm_delta": ssm_delta}
    for name, path in eval_paths.items():
        d = np.load(path)
        payload[f"nuc_ids_{name}"] = d["nuc_ids"].astype(np.uint8)
        payload[f"scores_{name}"] = (d[f"scores_{score_key}"].astype(np.float64) - wt_score)

    train_path = os.path.join(inst_dir, f"mut{MUT_PCT:02d}", f"lib_{LIB_SIZE}.npz")
    d_train = np.load(train_path)
    payload["nuc_ids_train"] = d_train["nuc_ids"][:LIB_SIZE].astype(np.uint8)
    payload["scores_train"] = (
        d_train[f"scores_{score_key}"][:LIB_SIZE].astype(np.float64) - wt_score
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved score cache -> {out_path}")


if __name__ == "__main__":
    main()
