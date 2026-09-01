"""Generate a VTS1 random-library cache around a non-overlapping natural WT.

The WT is an RNAcompete-2013 RNCMPT00111 probe whose GCUGG motif is fully
unpaired in its RNAfold MFE structure. Scores come from the VTS1 ResidualBind
ensemble, matching the real-oracle HuR cache used by the paired ridgelines.
"""

from pathlib import Path
import sys
import itertools

import numpy as np


ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from mRNA_RBP.scripts.pipeline.generate_libraries import sample_unique_mutants
from mRNA_RBP.src.oracles import build_oracle


WT_SEQ = "AAAAAAGACGAGAGCGACACCGGCUGGCCCGACGGAAAAAA"
STEM_PAIRS = [(21, 32), (20, 33), (19, 34)]
MOTIF_POSITIONS = [22, 23, 24, 25, 26]
RATES = ((5, 2, 7_380), (10, 4, 20_000), (25, 10, 20_000))
OUT = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "distribution_pair_figure" / "vts1_replacement_random_library_scores.npz"


def all_double_mutants(wt: np.ndarray) -> np.ndarray:
    """Return the complete C(41, 2) * 3^2 double-mutant library."""
    wt_ids = wt.argmax(axis=1).astype(np.uint8)
    rows = []
    for i, j in itertools.combinations(range(len(wt_ids)), 2):
        alt_i = [base for base in range(4) if base != wt_ids[i]]
        alt_j = [base for base in range(4) if base != wt_ids[j]]
        for base_i, base_j in itertools.product(alt_i, alt_j):
            row = wt_ids.copy()
            row[i], row[j] = base_i, base_j
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def main() -> None:
    oracle = build_oracle(
        "vts1",
        seq=WT_SEQ,
        stem_pairs=STEM_PAIRS,
        motif_positions=MOTIF_POSITIONS,
        seed=0,
        stem_sigma=3.0,
        wt_activity="high",
    )
    wt = oracle.wt_one_hot()
    payload = {
        "wt_seq": np.array(WT_SEQ),
        "stem_pairs": np.asarray(STEM_PAIRS, dtype=np.int32),
        "motif_positions": np.asarray(MOTIF_POSITIONS, dtype=np.int32),
        "wt_raw_score": np.float32(oracle.wt_raw_score),
        "source": np.array("RNAcompete-2013 RNCMPT00111 test probe 107676"),
    }

    for rate_index, (pct, mutation_count, target_n) in enumerate(RATES):
        rng = np.random.default_rng(200 + rate_index * 10)
        nuc_ids = (
            all_double_mutants(wt)
            if mutation_count == 2
            else sample_unique_mutants(wt, mutation_count, target_n, rng)
        )
        scores = oracle.score_all(np.eye(4, dtype=np.float32)[nuc_ids])[
            "vts1_residualbind"
        ].astype(np.float32)
        prefix = f"rand{pct:02d}"
        payload[f"{prefix}_nids"] = nuc_ids.astype(np.uint8)
        payload[f"{prefix}_delta_scores"] = scores
        print(
            f"{prefix}: n={len(nuc_ids):,}, mutations={mutation_count}, "
            f"delta=[{scores.min():.4f}, {scores.max():.4f}]"
        )

    np.savez_compressed(OUT, **payload)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
