# THROWAWAY PROTOTYPE — matched prediction scatter pair

Design question: how should one matched HuR/VTS1 scatter pair show the contrast in
surrogate agreement while keeping the actual Spearman values prominent?

All variants use cached **high-WT** predictions from the nonlinear additive +
pairwise surrogate trained on the **10% mutation-rate, 20,000-sequence** library.
They use the activity-balanced evaluation set because that is the cached evaluation
on which HuR and VTS1 visibly differ. The evaluation set contains several mutation-
count groups; their striations remain unresolved and are not interpreted here.

Regenerate all PNGs from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/scatter_pair_figure/make_prototypes.py
```

Outputs:

- `variant_a_clean_pair.png` — minimal matched scatter pair
- `variant_b_mutcount_context.png` — mutation-count groups shown in a restrained palette
- `variant_c_density_pair.png` — full-data density rendering with compact rho headers

This directory is disposable prototype work. It does not modify canonical figure
code or outputs.

## Decision

- Main text: Variant A, activity-balanced evaluation only.
- Supplement/exploratory: Variant B.
- Promotion gate: Move Variant B to the main text only if the striation analysis explains the Hamming-distance-dependent prediction bias.
