# ResidualBind oracle HuR

Figures are in `figures/`; copied cache/model/weight artifacts are in `libraries_used_for_figures/`. Built end-to-end by `mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle hur_residualbind`. The manifest records every file included.

- Figures: `27`
- Cached artifacts: `102`

## WT sequences and structures

High/low natural probes were selected from the RNAcompete-2013 measurement
distribution for HuR (RNCMPT00112) by `rbp/scripts/select_natural_wt.py`
(motif = `AUUUA`, the canonical ARE core): high = highest measured intensity
among exactly-one-motif probes within mean + 2 s.d.; low = lowest measured
intensity within mean - 2 s.d. Stem pairs are user-supplied RNAFold
dot-bracket structures, parsed to zero-based pairs. Defined in
`mRNA_RBP/src/sequence_configs.py`.

| | High activity | Low activity |
|---|---|---|
| Sequence | `AAGGGGUACACAUCAACGACAAUUUAGCGUAAACUUUGUAA` | `AAAAGACAGGAACUGGGCUCGUCAUAGGAACGCUAUUUAAA` |
| Dot-bracket | `((((..(((...................)))..))))....` | `....(((((........)).)))((((.....)))).....` |
| Stem pairs (0-based) | (0,36),(1,35),(2,34),(3,33),(6,30),(7,29),(8,28) | (4,22),(5,21),(6,20),(7,18),(8,17),(23,35),(24,34),(25,33),(26,32) |
| Motif (`AUUUA`) positions | 21-25 | 34-38 |
| Measured RNAcompete intensity | `5.571758` (z = +2.00) | `-2.952760` (z = -1.29) |
| Raw ResidualBind ensemble prediction | `1.871476` | `-0.121115` |
| Selection distribution | n=241,311 finite measurements, mean=0.396960, sd=2.590554 | (same distribution) |

Note: in the low-activity probe, the AUUUA motif (34-38) partially overlaps
stem pairs (24,34) and (23,35) — i.e. RNAFold predicts part of the ARE motif
is base-paired (structurally occluded), plausibly explaining why it was
selected as the *low*-activity construct (HuR binds single-stranded AU-rich
elements).

As with all oracles in this pipeline, the oracle score itself is
WT-anchored (`score = raw_prediction(x) - raw_prediction(WT)`), so each WT
sequence scores exactly 0 regardless of its raw ResidualBind prediction
above.

## Coefficient similarity (GT/oracle vs surrogate)

Mean cosine similarity between the additive (alpha) and pairwise (beta) weight matrices, at instance 0 / mut_rate 10% / lib_size 20000, computed from cached artifacts by `mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py` (see `coefficient_similarity.json`). Cosine similarity is scale-invariant but not sign-invariant, and MAVE-NN's GE nonlinearity has a real gauge freedom `(alpha, J, b) -> (-alpha, -J, -b)` that reproduces identical predictions -- the *sign-corrected* columns apply the single global sign (to both alpha and beta jointly) that maximizes agreement, so a surrogate that matched up to this legitimate flip isn't scored as if it learned the opposite direction.

| condition | cos sim (additive) | cos sim (pairwise) | sign-corrected (additive) | sign-corrected (pairwise) | sign flipped? |
|---|---|---|---|---|---|
| `hur_high` | 0.6105 | 0.5730 | 0.6105 | 0.5730 | — |
| `hur_low` | 0.9078 | 0.2356 | 0.9078 | 0.2356 | — |
