# THROWAWAY PROTOTYPE — central library-size figure

Question: how should the central library-size result combine Synthetic GT,
deepSQUID VTS1 high-WT, and deepSQUID HuR high-WT while emphasizing random
versus uniform evaluation?

This directory is deliberately isolated from canonical figure code. It reads
the existing JSON caches and writes three alternative static PNG layouts.

Regenerate all variants from `/home/nagle/final_version`:

```bash
python mRNA_RBP/scripts/figures/prototypes/library_size_figure/make_variants.py
```

Variants:

- A: conventional three-panel small multiples, best for continuity with the
  existing figures.
- B: paired slope/gap display, best for foregrounding disagreement between
  evaluation regimes.
- C: compact heatmap table, best for rapid cross-system and cross-size lookup.

Data caveat: Synthetic GT contains ten values per condition, so variant A
shows mean ± SD there. The two deepSQUID high-WT caches currently contain one
value per condition; their markers are shown without uncertainty. All variants
use the nonlinear additive + pairwise surrogate at 10% mutation rate and only
the `rand` and `type2` evaluation keys (`type2` is labeled uniform evaluation).

## Decision

- Selected: Variant A, the three-panel small-multiples design.
- Follow-up: Regenerate after VTS1 and HuR each have ten initializations so all panels can show uncertainty.
