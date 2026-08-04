# random_lib_mutagenesis

## Objective

- Probe sequence-function relationships by perturbing sequences and measuring functional change.
- Compare site saturation mutagenesis (SSM) against random mutagenesis for recovering that relationship.
- SSM captures additive effects but misses higher-order interactions.
- Random mutagenesis samples multi-position effects but may be inefficient for brittle sequences or sequences with strong higher-order interactions.

## Current status

- A draft manuscript structure with embedded figures and refined interpretation notes is available at `mRNA_RBP/NARRATIVE.md`.
- Narrative snapshot: [[Artifacts/random_lib_mutagenesis/2026-08-03_narrative_map]].
- Results now contain three parts: Synthetic GT proof of concept, biological application, and failure-mode diagnosis.
- Discussion and Conclusion are intentionally empty until themes mature.
- Supplement tracks raw Spearman values and the conditional low-WT comparison.
- Hole review is paused at Part III's targeted-pairwise and mutation-dependent failure modes.
- Preferred library-size layout is the HuR/VTS1 biological pair without SSM overlay.
- An accepted placeholder table summarizes mean Spearman ρ ± SD and additive/pairwise coefficient cosine similarity.
- Preferred evaluation-library layout is Variant B without saturated additive-plus-pairwise.

## Active blockers

- Mutation-rate figure design is unresolved; revisit after rejecting [directional matrices](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_a_directional_matrices.png), [direction-first comparison](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_b_direction_first.png), and [mixed-rate reservation](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_c_mixed_rate_reservation.png).
- Part I: generate the Synthetic GT negative control and replace both simulated placeholders.
- Part I: simplify the model-misspecification comparison without weakening its claim.
- Part II: select a landscape with separable motif and stem regions.
- Part II: define the score-loss threshold for nonfunctional VTS1 variants.
- Part II: rerun library-size comparisons across 10 initializations and populate the accepted summary table.
- Part III: redesign targeted-pairwise evaluation across mutation orders through the complete stem.
- Part III: determine whether redesigned targeted-pairwise VTS1 recovery remains high.
- Part III: define saturated additive-plus-pairwise evaluation's unique manuscript purpose.
- Part III: add mixed-rate training and quantify cross-landscape generalization deficits.
- Part III: explain mutation-count striations and report mutation-specific Spearman values.
- Higher-order recovery: measure signal missed by saturated singles and doubles, especially in VTS1.
- Coefficient recovery: resolve Synthetic GT pairwise cosine similarity before using coefficient agreement.

## Important decisions

- The manuscript uses a three-part Results progression: proof of concept, biological replication, then failure-mode diagnosis.
- Synthetic GT model misspecification belongs in Part I; biological failure attribution belongs in Part III.
- deepSQUID results are primary; ResidualBind is retained only to validate deepSQUID as a replacement oracle.
- Active landscapes are Synthetic GT, high-WT VTS1, and high-WT HuR, with room to add examples.
- HuR is the sequence-motif negative control for genuine landscape recovery.
- Targeted pairwise currently probes only annotated base-pairing regions.
- Saturated additive-plus-pairwise currently serves only as an evaluation library.
- Part II opens with paired biological ridgelines, followed by the HuR/VTS1 library-size comparison.
- SSM baselines stay off the library-size plot and belong in the separate summary table.
- The summary table reports mean Spearman ρ ± SD plus additive and pairwise cosine similarity.
- Each manuscript figure should carry one unique message or coherent message set.
- Low-WT distributions are parked for the supplement unless they show stronger mutation-rate-dependent skew.
- Full coefficient maps are cut; cosine similarity is the preferred summary.
- Repeated scatterplots are cut; one matched HuR/VTS1 pair is sufficient.
- Twister and MSI1 are archived and excluded from the active example set.

## Durable research memory

- Hypothesis: in brittle sequences, random mutations often disrupt folding/base-pairing, enriching libraries for nonfunctional variants and obscuring the true sequence-function relationship.
- Hypothesis: high predictive accuracy on held-out variants can overestimate recovery of the underlying sequence-function relationship.
- Hypothesis: random-holdout accuracy can conceal an underspecified surrogate functional form.
- Current uniform-evaluation SSM baselines are:

| Landscape | SSM Spearman ρ |
| --- | ---: |
| HuR | 0.865 |
| VTS1 | 0.728 |

## Open research questions

- Should a surrogate be trained on the saturated additive-plus-pairwise library, which currently serves only as an evaluation library?
- What unique purpose should saturated additive-plus-pairwise evaluation serve in the manuscript?
- Why is targeted-pairwise recovery unexpectedly high for VTS1?
- Does targeted-pairwise recovery persist when mutation order increases through the complete stem?

## Next milestone

- Resume Part III by refining targeted-pairwise and mutation-dependent failure-mode figures.

## Future directions

- Reconsider low-WT distributions if they establish stronger mutation-rate-dependent skew.

## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
