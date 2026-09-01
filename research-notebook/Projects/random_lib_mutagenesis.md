# random_lib_mutagenesis

## Objective

- Probe sequence-function relationships by perturbing sequences and measuring functional change.
- Compare site saturation mutagenesis (SSM) against random mutagenesis for recovering that relationship.
- SSM captures additive effects but misses higher-order interactions.
- Random mutagenesis samples multi-position effects but may be inefficient for brittle sequences or sequences with strong higher-order interactions.

## Current status

- Current narrative map: [[Artifacts/random_lib_mutagenesis/2026-08-05_narrative_map]].
- Previous snapshot: [[Artifacts/random_lib_mutagenesis/2026-08-04_narrative_map]].
- The manuscript is being restructured as a cautionary methods paper about concealed landscape complexity.
- A 29-point registry supports reusable claims, evidence, qualifications, and holes.
- Four complete narrative variants reorder the evidence around biological surprise, workflow factors, propagation, or stress testing.
- Results currently contain Synthetic GT proof, biological replication, and failure-mode evidence.
- Discussion and Conclusion are intentionally empty until themes mature.
- Supplement tracks raw Spearman values and the conditional low-WT comparison.
- Hole review is paused at Part III's targeted-pairwise and mutation-dependent failure modes.
- Preferred library-size layout is the HuR/VTS1 biological pair without SSM overlay.
- An accepted placeholder table summarizes mean Spearman ρ ± SD and additive/pairwise coefficient cosine similarity.
- Preferred evaluation-library layout is Variant B without saturated additive-plus-pairwise.
- Part III, Section 2 now uses standardized activity-score residual violins instead of paired scatterplots.
- Synthetic GT control figures now use real instance-00 libraries and predictions; full-scale replication remains pending.
- The redesigned motif-only negative control reaches activity-balanced ρ = 0.527 despite random-holdout ρ = 1.000.
- Sequence-only VTS1 UMAP and Hamming-neighborhood prototypes test whether residual errors are locally coherent.
- Prediction-bin entropy prototypes compare VTS1 and HuR across 5%, 10%, and 25% mutation-rate training.
- An actual-activity mutation map tests VTS1 motif retention across activity strata.
- Four-GFP and SMN1 datasets are candidate higher-order biological landscapes.
- Dataset assessments: [[Artifacts/random_lib_mutagenesis/2026-08-21_protein_combinatorial_datasets|protein candidates]] and [[Artifacts/random_lib_mutagenesis/2026-08-21_sm_n1_dataset_assessment|SMN1]].
- Coefficient-map prototypes compare VTS1, HuR, and Synthetic GT representations.
- A separate prototype tests triple-mutant probes beyond saturated single/double coverage.
- Structured Synthetic GT now has its varied-mutation deepSQUID coefficient cache; negative-control caches remain missing.

## Active blockers

- Select additional biological landscapes and define finite-landscape integration.
- Determine whether GFP orthologs provide sufficient replication beyond one protein family.
- Mutation-rate figure design is unresolved; revisit after rejecting [directional matrices](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_a_directional_matrices.png), [direction-first comparison](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_b_direction_first.png), and [mixed-rate reservation](../../mRNA_RBP/prototypes/mutation_rate_figure/variant_c_mixed_rate_reservation.png).
- Part I: replicate the redesigned motif-only Synthetic GT negative control across 10 instances.
- Part I: simplify the model-misspecification comparison without weakening its claim.
- Part II: select a landscape with separable motif and stem regions.
- Part II: define the score-loss threshold for nonfunctional VTS1 variants.
- Part II: rerun library-size comparisons across 10 initializations and populate the accepted summary table.
- Part III: redesign targeted-pairwise evaluation across mutation orders through the complete stem.
- Part III: determine whether redesigned targeted-pairwise VTS1 recovery remains high.
- Part III: define saturated additive-plus-pairwise evaluation's unique manuscript purpose.
- Part III: add mixed-rate training and quantify cross-landscape generalization deficits.
- Part III: interpret overprediction and underprediction before versus after four mutations.
- Part III: determine whether activity strata contain matched motif-intact and motif-disrupted mechanisms.
- Motif placement: hold compatible motif count fixed before attributing high scores to position.
- Higher-order recovery: measure signal missed by saturated singles and doubles, especially in VTS1.
- Coefficient recovery: resolve Synthetic GT pairwise cosine similarity before using coefficient agreement.
- Synthetic GT coefficient comparison: train the missing negative-control fixed-rate and varied-mutation caches.

## Important decisions

- The manuscript warns about hidden complexity rather than prescribing optimal experimental choices.
- Mutation scheme, library size, evaluation distribution, and surrogate form all belong in the main argument.
- Multiple biological datasets should test whether the caution generalizes across landscapes.
- Narrative variants remain alternatives until one ordering is selected.
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
- Production saturated additive-plus-pairwise libraries contain only exhaustive singles and doubles.
- Triple-mutant nonlinearity probes remain a separate exploratory extension.
- Paired mutation-count scatterplots are replaced by standardized residual violins.
- The Part III residual figure uses the 10% random-trained surrogate, approximately four mutations per sequence.
- Twister and MSI1 are archived and excluded from the active example set.
- Synthetic GT negative control keeps the structured control's sequence backbone, score interface, and sigmoid, but uses motif-only additive effects.
- The control uses motif scale 3.0, background scale 0.10, and no pairwise interactions.
- VTS1 pipelines use explicit high/low natural sequence configurations for ResidualBind and deepSQUID oracles.
- The high-WT VTS1 sequence is `AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA`.
- Scatterplots use equal deterministic display samples while statistics retain all predictions.
- Activity balancing is interpreted as phenotype coverage, not balanced mechanistic sequence-space coverage.
- Balanced teacher–student motif-placement controls used 3,700 sequences per landscape with equal start coverage.
- Held-out teacher–student correlations were 0.942 for VTS1 (GCAGG) and 0.975 for HuR (AUUUA).
- Retained positional activity structure argues against simple motif-count coverage as the sole explanation.
- Compatible `GCNGG` multiplicity contributes to high VTS1 teacher scores but does not fully explain positional structure.
- Teacher–student controls inherit biases from the original deepSQUID teacher and do not establish biological significance.

## Durable research memory

- Measured finite landscapes support direct subsampling but cannot score arbitrary generated variants.
- The Somermeyer collection contains four GFP ortholog landscapes with quantitative cellular brightness.
- cgreGFP and avGFP have sharp epistatic peaks; amacGFP and ppluGFP2 have flatter peaks.
- GFP protein genotypes use native-reference mutation strings and separate wild-type FASTA records.
- SMN1 contains 30,732 measured eight-nucleotide genotypes from 32,768 valid combinations.
- SMN1 variants span mutation orders zero through eight with reference `CAGUAAGU` normalized to `y = 100`.
- SMN1 measures splice-site activity, not folded RNA structural constraint.
- VTS1 motif-copy analysis: [[Artifacts/random_lib_mutagenesis/2026-08-21_vts1_motif_copy_number]].
- Compatible motif count correlates with VTS1 teacher score at `r = 0.488`.

| `GCNGG` copies | Sequences | Mean teacher score | Top-activity sequences |
| ---: | ---: | ---: | ---: |
| 1 | 1,687 | -0.516 | 4.5% |
| 2 | 1,899 | +0.391 | 13.7% |
| 3 | 113 | +1.111 | 29.2% |
| 4 | 1 | +2.117 | 100.0% |

- Exact `GCAGG` copy count is more weakly associated with VTS1 activity at `r = 0.149`.
- HuR `AUUUA` copy count has only a weak association with teacher activity at `r = 0.096`.
- Hypothesis: in brittle sequences, random mutations often disrupt folding/base-pairing, enriching libraries for nonfunctional variants and obscuring the true sequence-function relationship.
- Hypothesis: high predictive accuracy on held-out variants can overestimate recovery of the underlying sequence-function relationship.
- Hypothesis: random-holdout accuracy can conceal an underspecified surrogate functional form.
- Saturated additive-plus-pairwise evaluation contains 123 singles and 7,380 doubles for a 41-nt sequence.
- Coefficient-map prototypes: [[Artifacts/random_lib_mutagenesis/2026-08-26_coefficient_maps]].
- Saturated-nonlinearity prototype: [[Artifacts/random_lib_mutagenesis/2026-08-26_saturated_nonlinearity_library]].
- Standardized residual is `(prediction - truth) / SD(truth)`; negative values indicate underprediction.
- VTS1 15-mutation residuals span both directions, so signed averages conceal severe errors.
- Accepted Part III residual figure: [[Artifacts/random_lib_mutagenesis/2026-08-05_activity_balanced_residual_violins]].
- Pipeline scripts read pairwise structure from `gt.stem_pairs` or cached `gt_params.npz["edges"]`, not static sequence configuration.
- Fixed-four-mutation training does not calibrate predictions across mutation orders; see [[Artifacts/random_lib_mutagenesis/2026-08-09_synthetic_gt_negative_control_differential]].
- The prior unstructured random-edge negative control failed its intended contrast and was replaced by the motif-only design.
- Negative-control recovery by evaluation regime is:

| Synthetic GT | Random holdout ρ | Activity-balanced ρ |
| --- | ---: | ---: |
| Structured positive control | 0.989 | 0.315 |
| Unstructured negative control | 1.000 | 0.379 |

- Redesigned negative-control recovery is:

| Landscape | Random holdout ρ | Activity-balanced ρ |
| --- | ---: | ---: |
| Structured Synthetic GT | 0.990 | 0.315 |
| Motif-only negative control | 1.000 | 0.527 |
| HuR | — | 0.941 |

- Negative-control recovery within each activity-balanced mutation count is:

| Mutations | n | Within-group ρ | Median prediction error |
| ---: | ---: | ---: | ---: |
| 3 | 5,865 | 0.999976 | -0.0847 |
| 5 | 5,349 | 0.999971 | +0.0871 |
| 7 | 6,421 | 0.999832 | +0.2394 |
| 15 | 2,365 | 0.996841 | +0.6518 |
- VTS1 residuals are locally coherent within every exact mutation-count shell:

| Mutations | Sequences | Median neighbor mismatches | Observed/null residual difference | Permutation p |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 7,524 | 2 | 0.684 | 0.001 |
| 5 | 5,763 | 5 | 0.810 | 0.001 |
| 7 | 4,340 | 7 | 0.814 | 0.001 |
| 15 | 2,373 | 15 | 0.747 | 0.001 |

- The neighborhood test uses ten nearest Hamming neighbors and 999 within-shell residual permutations.
- Archived exploratory outputs: [[Artifacts/random_lib_mutagenesis/2026-08-12_residual_sequence_umap/vts1_high_residual_sequence_umap.png|VTS1 residual UMAP]] and [[Artifacts/random_lib_mutagenesis/2026-08-12_prediction_entropy/vts1_high_prediction_bin_entropy.png|VTS1 entropy map]].
- VTS1 `GCUGG` retention varies strongly with actual activity; see [[Artifacts/random_lib_mutagenesis/2026-08-13_vts1_activity_mutation_map|activity mutation map]].

| Activity stratum | Median activity | Motif intact | Matched null | Mean mutations |
| --- | ---: | ---: | ---: | ---: |
| Lowest bin | -2.780 | 5.1% | 35.1% | 8.622 |
| Middle bin | -0.607 | 85.2% | 52.5% | 5.134 |
| Highest bin | 1.350 | 71.0% | 42.3% | 7.208 |

- Broad activity coverage can coexist with motif preservation and sparse alternative mechanisms.
- Median standardized residuals for the 10% random-trained surrogate are:

| Landscape | 3 mutations | 5 mutations | 7 mutations | 15 mutations |
| --- | ---: | ---: | ---: | ---: |
| HuR | -0.189 | +0.117 | +0.320 | +0.334 |
| VTS1 | -0.553 | +0.458 | +1.078 | +0.401 |

- Current uniform-evaluation SSM baselines are:

| Landscape | SSM Spearman ρ |
| --- | ---: |
| HuR | 0.865 |
| VTS1 | 0.728 |

## Open research questions

- Do the four GFP orthologs provide enough biological replication for a methods-heavy paper?
- How should finite measured landscapes be integrated without introducing a learned reference oracle?
- Should a surrogate be trained on the saturated additive-plus-pairwise library, which currently serves only as an evaluation library?
- What unique purpose should saturated additive-plus-pairwise evaluation serve in the manuscript?
- Why is targeted-pairwise recovery unexpectedly high for VTS1?
- Does targeted-pairwise recovery persist when mutation order increases through the complete stem?
- Does VTS1 cover matched motif-intact and motif-disrupted solutions producing the same activity?

## Next milestone

- Prototype finite-landscape library-size and mutation-profile experiments on SMN1.

## Future directions

- Reconsider low-WT distributions if they establish stronger mutation-rate-dependent skew.

## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
| Sequence-specific recognition of RNA hairpins by the SAM domain of Vts1p | Aviv, Lin, Ben-Ari, Smibert & Sicheri | https://www.nature.com/articles/nsmb1053 | 2026-08-05 | Nat Struct Mol Biol (2006) paper describing the structural basis of Vts1p SAM domain hairpin recognition; companion paper to PDB 2F8K. |
| Heterogeneity of the GFP fitness landscape and data-driven protein design | Gonzalez Somermeyer et al. | https://doi.org/10.7554/eLife.75842 | 2026-08-21 | Assessed four higher-order GFP brightness landscapes as protein candidates for the surrogate pipeline. |
| Learning sequence-function relationships with scalable, interpretable Gaussian processes | Zhou et al. | https://doi.org/10.1101/2025.08.15.670613 | 2026-08-21 | Identified SMN1 as a nearly exhaustive higher-order splice-site landscape suitable for direct subsampling. |

## Structures

| PDB ID | Title | Author | Link | Date researched | Notes |
| --- | --- | --- | --- | --- | --- |
| 2F8K | Sequence-specific recognition of RNA hairpins by the SAM domain of Vts1 | Aviv, Lin, Ben-Ari, Smibert & Sicheri | https://www.rcsb.org/structure/2F8K | 2026-08-05 | VTS1 SAM domain bound to an RNA hairpin; candidate VTS1 structure for the manuscript. |
| 6GD2 | Structure of HuR RRM3 in complex with RNA | Pabis & Sattler | https://www.rcsb.org/structure/6GD2 | 2026-08-05 | HuR RRM3 bound to RNA; candidate HuR structure for the manuscript. |
