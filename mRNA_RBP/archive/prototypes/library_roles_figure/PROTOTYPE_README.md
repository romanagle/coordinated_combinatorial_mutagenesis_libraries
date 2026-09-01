# PROTOTYPE — predictor and evaluation-library roles

Throwaway static alternatives for separating predictor/training sources from
evaluation libraries. They are not manuscript-ready figures.

Regenerate all PNGs from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/library_roles_figure/generate_prototypes.py
```

## Corrected semantics

- The SSM additive baseline is a predictor derived from single-substitution measurements.
- The random-trained predictor is a nonlinear additive-plus-pairwise surrogate trained on a 20K library at 10% mutation rate.
- Activity-balanced, targeted pairwise, and saturated additive-plus-pairwise are evaluation libraries.
- Targeted pairwise mutates only positions in annotated base-pairing regions.
- Saturated additive-plus-pairwise is not a training library in the current pipeline.
- The SSM predictor was not evaluated on the random holdout in these result files; prototypes mark it as unavailable.
- Saturated evaluation values remain provisional because the current libraries contain four triple mutants.

All metrics come from the Synthetic GT and high-WT deepSQUID VTS1/HuR
library-size result JSONs. No ResidualBind-oracle result is used.

## Variants

- A: predictor-by-evaluation matrix; most explicit separation of the two axes.
- B: random holdout, activity-balanced, and targeted-pairwise evaluations for the random-trained model; saturated evaluation was removed while its purpose is reconsidered.
- C: role-first schematic with a compact metric summary; most explanatory but densest.

## Current decision

- Variant B is preferred.
- Do not include saturated additive-plus-pairwise in Variant B until its purpose is revisited.
- Treat the high targeted-pairwise VTS1 result cautiously pending a redesigned library that increases paired-stem mutation order from 2 through mutation of the complete stem.
