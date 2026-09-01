"""
mRNA_RBP/scripts/pipeline/generate_random_region_score_cache.py

Builds the <name>_natural_random_library_scores[_<wt_activity>].npz cache
consumed by mRNA_RBP/scripts/figures/core/plot_residualbind_vts1_rand_region_distributions.py
-- the WT-relative (delta, WT=0) scores come purely from the standard
mut05/10/25 random libraries the pipeline already generates
(instance_00/mut{pct}/lib_20000.npz); no live oracle scoring needed for those.

For real ResidualBind-backed oracles (MSI1/VTS1/HuR/QKI) this also records
the WT sequence's *raw* (non-anchored) ResidualBind prediction
(``wt_raw_score``) and derives per-sequence raw scores
(``rand{pct}_raw_scores`` = delta + wt_raw_score) for the companion raw-score
figure -- getting that one scalar requires building the live oracle, so for
those oracles this script now needs the torch (toehold_gpu) env. The
synthetic GT and Twister oracles have no raw ResidualBind score and are
unaffected -- still no torch needed for those.

The deep-squid surrogates (deepsquid_hur/deepsquid_vts1) are trained purely
on WT-relative deltas distilled from their real ResidualBind counterpart, so
they never learn a raw prediction scale of their own -- there's no
"deepsquid raw score" to compute. Instead, for these, wt_raw_score is taken
from the *real* oracle it was distilled from (see REAL_ORACLE_FOR_DEEPSQUID)
and the deep-squid delta is reinterpreted on that real oracle's raw scale --
an approximation (the surrogate approximates the real oracle's deltas, not
guaranteed to hit them exactly), not a scale the surrogate itself learned.

Oracle-agnostic; works for any registered oracle including the synthetic GT.

Usage:
    python mRNA_RBP/scripts/pipeline/generate_random_region_score_cache.py --oracle hur --wt_activity high
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_HERE, ".."))

from mRNA_RBP.src.oracles import (
    MRNA_ORACLE, ORACLE_SHORT_NAME, RESIDUALBIND_MSI1_ORACLE,
    RESIDUALBIND_VTS1_ORACLE, RESIDUALBIND_HUR_ORACLE, RESIDUALBIND_QKI_ORACLE,
    DEEPSQUID_HUR_ORACLE, DEEPSQUID_VTS1_ORACLE,
    build_oracle, default_output_base, normalize_oracle_name, primary_gt_key,
    sequence_config_for_oracle,
)

INSTANCE = 0
MUT_RATES_PCT = (5, 10, 25)

# Oracles with a real ResidualBind ensemble behind them -- the only ones for
# which "raw ResidualBind score" is a meaningful, cacheable quantity.
RAW_SCORE_ORACLES = (
    RESIDUALBIND_MSI1_ORACLE, RESIDUALBIND_VTS1_ORACLE,
    RESIDUALBIND_HUR_ORACLE, RESIDUALBIND_QKI_ORACLE,
)

# Deep-squid surrogates borrow their WT raw anchor from the real oracle they
# were distilled from (see module docstring) rather than having their own.
REAL_ORACLE_FOR_DEEPSQUID = {
    DEEPSQUID_HUR_ORACLE: RESIDUALBIND_HUR_ORACLE,
    DEEPSQUID_VTS1_ORACLE: RESIDUALBIND_VTS1_ORACLE,
}


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
    short = ORACLE_SHORT_NAME.get(oracle_name, oracle_name)
    out_base = args.out_base or default_output_base(_HERE, oracle_name, args.wt_activity)
    inst_dir = os.path.join(out_base, f"instance_{INSTANCE:02d}")

    seq, stem_pairs, motif_positions = sequence_config_for_oracle(oracle_name, args.wt_activity)

    suffix = "" if args.wt_activity == "high" else f"_{args.wt_activity}_wt"
    out_path = args.out or os.path.join(
        out_base, f"{short}_natural_random_library_scores{suffix}.npz")

    payload = {
        "wt_seq": seq,
        "stem_pairs": np.asarray(stem_pairs, dtype=np.int32),
        "motif_positions": np.asarray(motif_positions, dtype=np.int32),
    }

    wt_raw = None
    raw_source_oracle = oracle_name if oracle_name in RAW_SCORE_ORACLES else \
        REAL_ORACLE_FOR_DEEPSQUID.get(oracle_name)
    if raw_source_oracle is not None:
        oracle = build_oracle(
            raw_source_oracle, seq=seq, stem_pairs=stem_pairs, motif_positions=motif_positions,
            seed=0, stem_sigma=3.0, wt_activity=args.wt_activity,
        )
        wt_raw = oracle.wt_raw_score
        payload["wt_raw_score"] = np.float32(wt_raw)
        if raw_source_oracle != oracle_name:
            payload["wt_raw_score_source_oracle"] = np.array([raw_source_oracle])
        print(f"  wt_raw_score: {wt_raw:.4f}"
              + (f"  (from {raw_source_oracle})" if raw_source_oracle != oracle_name else ""))

    for pct in MUT_RATES_PCT:
        lib_path = os.path.join(inst_dir, f"mut{pct:02d}", "lib_20000.npz")
        d = np.load(lib_path)
        nuc_ids = d["nuc_ids"].astype(np.uint8)
        delta_scores = d[f"scores_{score_key}"].astype(np.float32)
        prefix = f"rand{pct:02d}"
        payload[f"{prefix}_nids"] = nuc_ids
        payload[f"{prefix}_delta_scores"] = delta_scores
        payload[f"{prefix}_mut_count"] = np.int32(pct)
        if wt_raw is not None:
            payload[f"{prefix}_raw_scores"] = (delta_scores + wt_raw).astype(np.float32)
        print(f"  {prefix}: n={len(nuc_ids):,}  "
              f"delta ∈ [{delta_scores.min():.4f}, {delta_scores.max():.4f}]")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
