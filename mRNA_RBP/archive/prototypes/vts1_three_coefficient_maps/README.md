# VTS1 three-way pairwise coefficient comparison

Side-by-side comparison of the VTS1-high pairwise maps for:

1. the nonlinear additive + pairwise SQUID surrogate trained on the
   ResidualBind VTS1 oracle at 10% mutation and a 20,000-sequence library;
2. the equivalent surrogate trained on the deepSQUID VTS1 oracle; and
3. epistasis reconstructed directly from the exhaustive single/double-mutant
   deepSQUID saturation library.

The first two panels contain fitted latent `J` coefficients. The third is an
observed activity-scale epistasis contrast, `double - single_i - single_j + WT`,
because the saturation library is a measurement set rather than a fitted model.
Each panel therefore has its own symmetric color scale.

Run from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/vts1_three_coefficient_maps/make_figure.py
```
