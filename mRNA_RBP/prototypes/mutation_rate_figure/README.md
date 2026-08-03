# THROWAWAY PROTOTYPE — mutation-rate figure

Question: how should the same cross-mutation-rate results be presented across
Synthetic GT, deepSQUID VTS1 (high-WT), and deepSQUID HuR (high-WT), while
making high-train-rate → low-test-rate comparisons easy to see?

Regenerate all three PNG variants from the existing result JSON files:

```bash
python mRNA_RBP/prototypes/mutation_rate_figure/render.py
```

All panels use the `nonlinear additive + pairwise` surrogate at library size
20,000. The mixed-rate training condition is not present in these result JSON
files; prototypes that reserve room for it label it as unavailable and contain
no fabricated value.

This directory is disposable. It does not alter canonical plotting code or
outputs.
