# deepSQUID HuR

Figures are in `figures/`; copied cache/model/weight artifacts are in `libraries_used_for_figures/`. Built end-to-end by `mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle deepsquid_hur`. The manifest records every file included.

- Figures: `27`
- Cached artifacts: `101`

## Coefficient similarity (GT/oracle vs surrogate)

Mean cosine similarity between the additive (alpha) and pairwise (beta) weight matrices, at instance 0 / mut_rate 10% / lib_size 20000, computed from cached artifacts by `mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py` (see `coefficient_similarity.json`). Cosine similarity is scale-invariant but not sign-invariant, and MAVE-NN's GE nonlinearity has a real gauge freedom `(alpha, J, b) -> (-alpha, -J, -b)` that reproduces identical predictions -- the *sign-corrected* columns apply the single global sign (to both alpha and beta jointly) that maximizes agreement, so a surrogate that matched up to this legitimate flip isn't scored as if it learned the opposite direction.

| condition | cos sim (additive) | cos sim (pairwise) | sign-corrected (additive) | sign-corrected (pairwise) | sign flipped? |
|---|---|---|---|---|---|
| `deepsquid_hur_high` | 0.7270 | 0.5589 | 0.7270 | 0.5589 | — |
| `deepsquid_hur_low` | 0.8766 | 0.1973 | 0.8766 | 0.1973 | — |
