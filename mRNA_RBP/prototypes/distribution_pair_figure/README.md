# PROTOTYPE — paired random-library distributions

Throwaway static figure prototypes comparing the deepSQUID VTS1 and HuR
high-WT random-library score distributions. Canonical plotting code and outputs
are intentionally untouched.

Regenerate all three PNGs from the existing score caches:

```bash
python mRNA_RBP/prototypes/distribution_pair_figure/prototype_distribution_pair.py
```

The figures use cached WT-relative scores and cached sequence/region metadata.
No values are simulated. Variant A omits the “neither” class; variants B and C
show the complete random library without assigning region-specific effects.

## Decision

- Preferred layout: Variant A.
- Blocker: VTS1 confounds motif and stem effects because its motif lies within the stem.
- Follow-up: Rebuild this comparison after selecting a landscape with non-overlapping motif and stem regions.
