# random_lib_mutagenesis

## Objective

- Probe sequence-function relationships by perturbing sequences and measuring functional change.
- Compare site saturation mutagenesis (SSM) against random mutagenesis for recovering that relationship.
- SSM captures additive effects but misses higher-order interactions.
- Random mutagenesis samples multi-position effects but may be inefficient for brittle sequences or sequences with strong higher-order interactions.

## Current status

- A draft manuscript structure with embedded figures and refined interpretation notes is available at `mRNA_RBP/NARRATIVE.md`.
- Narrative snapshot: [[Artifacts/random_lib_mutagenesis/2026-08-03_narrative_map]].
- Hole review is paused at Section 3's targeted-pairwise library design.
- Preferred library-size layout is the three-landscape triptych.
- Preferred evaluation-library layout is Variant B without saturated additive-plus-pairwise.

## Active blockers

- Mutation-rate figure design is unresolved; revisit after rejecting [directional matrices](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_a_directional_matrices.png), [direction-first comparison](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_b_direction_first.png), and [mixed-rate reservation](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_c_mixed_rate_reservation.png).

## Important decisions

- deepSQUID results are primary; ResidualBind is retained only to validate deepSQUID as a replacement oracle.
- Active landscapes are Synthetic GT, high-WT VTS1, and high-WT HuR, with room to add examples.
- HuR is the sequence-motif negative control for genuine landscape recovery.
- Targeted pairwise currently probes only annotated base-pairing regions.
- Saturated additive-plus-pairwise currently serves only as an evaluation library.
- Each manuscript figure should carry one unique message or coherent message set.

## Durable research memory

- Hypothesis: in brittle sequences, random mutations often disrupt folding/base-pairing, enriching libraries for nonfunctional variants and obscuring the true sequence-function relationship.
- Hypothesis: high predictive accuracy on held-out variants can overestimate recovery of the underlying sequence-function relationship.

## Open research questions

- Should a surrogate be trained on the saturated additive-plus-pairwise library, which currently serves only as an evaluation library?
- What unique purpose should saturated additive-plus-pairwise evaluation serve in the manuscript?
- Why is targeted-pairwise recovery unexpectedly high for VTS1?
- Does targeted-pairwise recovery persist when mutation order increases through the complete stem?

## Next milestone

- Resume hole review by deciding whether targeted-pairwise mutation orders should be stratified or pooled.

## Future directions

## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
