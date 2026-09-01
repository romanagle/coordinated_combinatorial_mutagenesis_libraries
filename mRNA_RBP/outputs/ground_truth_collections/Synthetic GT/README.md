# Synthetic GT

Figures are in `figures/`; copied cache/model/weight artifacts are in `libraries_used_for_figures/`. Synthetic source libraries remain in `mRNA_RBP/outputs/instance_*` and are copied into this collection as cached artifacts. The manifest records every file included in the curated collection.

- Figures copied in manifest: `12`
- Cached artifacts copied in manifest: `680`
- Missing source files: `0`

## Coefficient similarity (GT/oracle vs surrogate)

Mean cosine similarity between the additive (alpha) and pairwise (beta) weight matrices, at instance 0 / mut_rate 10% / lib_size 20000, computed from cached artifacts by `mRNA_RBP/scripts/maintenance/compute_coefficient_similarity.py` (see `coefficient_similarity.json`). Cosine similarity is scale-invariant but not sign-invariant, and MAVE-NN's GE nonlinearity has a real gauge freedom `(alpha, J, b) -> (-alpha, -J, -b)` that reproduces identical predictions -- the *sign-corrected* columns apply the single global sign (to both alpha and beta jointly) that maximizes agreement, so a surrogate that matched up to this legitimate flip isn't scored as if it learned the opposite direction.

| condition | cos sim (additive) | cos sim (pairwise) | sign-corrected (additive) | sign-corrected (pairwise) | sign flipped? |
|---|---|---|---|---|---|
| `synthetic` | -0.2569 | 0.9253 | -0.2569 | 0.9253 | — |
