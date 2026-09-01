# Random-library mutagenesis narrative map

## Question

- How do four factors (mutation scheme, library size, evaluation distribution, and surrogate-model form) shape when random-holdout evaluation overstates recovery of RNA sequence–function landscapes?

- Part I: Random libraries sample skewed regions of RNA activity landscapes.
- Part II: Synthetic GT establishes the evaluation bias under controlled conditions.
- Part III: Biological RNA landscapes reproduce the evaluation bias.
- Part III subpoint: The evaluation gap persists across tested training-library sizes.
- Part IV: Random-holdout accuracy can conceal surrogate-model misspecification.

## Methods

### Synthetic ground-truth landscape

- A single fixed 41-nucleotide RNA sequence defines the landscape.
- The function combines additive weights (α, L × 4) and pairwise couplings at stem pairs, followed by a global sigmoid nonlinearity.
- At non-stem positions, non-WT nucleotides receive additive weights drawn from N(0, σ).
- Stem positions have zero additive contribution, and pairwise couplings penalize all non-compensatory mutations.
- The WT always scores 0, and all mutations score ≤ 0 by construction.
- Five instances use different random weight draws, with different α magnitudes but the same structure.

![Synthetic ground-truth additive weights and stem-pair couplings](<figures/synthetic_gt_coefficients_only.png>)

- A motif-only negative control shares the same sequence and motif position as the structured GT, but has only the additive motif effect — no other privileged additive regions and no pairwise coupling — so it serves as a sequence-only sequence-to-function relationship.
- This is not fully biologically realistic; whether to revisit the negative-control setup is still open.

![Structured positive control versus motif-only negative control: exact additive and pairwise coefficient matrices](<figures/both_synthetic_gt_additive_pairwise_matrices.png>)

### Experimental design

- The experiment crosses 5 synthetic ground-truth instances, 3 mutation rates (5%, 10%, and 25%), 3 training-library sizes (200, 2,000, and 20,000 variants), and 4 surrogate functional forms.
- The four surrogate functional forms are additive, additive plus pairwise, nonlinear additive, and nonlinear additive plus pairwise.

### Other methods

- deepSQUID VTS1 provides a biological structure-dependent landscape with a high-activity wild type and recognizes a GCUGG motif.
- deepSQUID HuR provides a sequence-motif negative control with a high-activity wild type and recognizes an AUUUA motif.

![Placeholder for VTS1 and HuR high-activity-sequence secondary structures](<figures/vts1_hur_secondary_structure_placeholder.png>)

- Random libraries vary size and mutation rate and train nonlinear additive-plus-pairwise surrogates.
- Activity-balanced evaluation measures recovery outside the random library's skewed activity distribution.
- Targeted-pairwise evaluation currently contains double mutants restricted to annotated base-pairing regions.
- SSM supplies an additive-only baseline for learning higher-order sequence-function relationships.
- Oracle validation: held-out Spearman ρ between deepSQUID and ResidualBind is 0.966 (VTS1, n=39,991) and 0.973 (HuR, n=40,358), supporting deepSQUID as the primary oracle in subsequent experiments.

## Results

### Part I. Synthetic ground-truth proof of concept

#### 1. Skewed random-library distributions can inflate random-holdout accuracy

- At a provisionally selected 10% mutation rate, the random training and random test activity distributions are expected to show similar skew, whereas the uniform test library should cover the activity distribution more evenly.
- The matched skew of the random training and test libraries is expected to produce high random-holdout accuracy without probing the entire sequence–function landscape.
- Uniform evaluation should therefore assess recovery of the underlying sequence–function relationship more accurately than a random holdout drawn from the training distribution.

![Positive- and negative-control Synthetic GT additive and pairwise coefficients alongside activity distributions for model-development, held-out random-test, and activity-balanced evaluation libraries](<figures/synthetic_gt_activity_distributions.png>)

#### 2. Random-holdout accuracy can conceal poor landscape-wide recovery

- Both Synthetic GTs use fixed-10% (four-mutation), 20K training libraries and produce nearly perfect random-library holdout correlations (structured positive control: Spearman ρ = 0.990; motif-only negative control: ρ = 1.000).
- Both use the standard activity-balanced construction: 200K candidate sequences split across 3, 5, 7, and 15 mutations, followed by histogram uniformization to 20K sequences under each control's own ground-truth activity.
- With the smallest tested near-neutral background that supports a complete evaluation library (effect scale 0.10 versus 3.0 at motif positions), the motif-dominated negative control reaches random-holdout ρ = 1.000 and activity-balanced ρ = 0.527, still below HuR (ρ = 0.941).
- This small nonzero background avoids the score-bin sparsity of exactly neutral and 0.05-scale backgrounds, allowing the standard histogram-uniformization procedure to retain the full 20K activity-balanced target.

![Actual positive- and negative-control Synthetic GT predictions on matched random-library holdout and activity-balanced evaluation sets](<figures/synthetic_gt_scatter_actual.png>)

#### 3. Random-holdout accuracy can mask model misspecification

- High random-holdout accuracy can conceal that the surrogate functional form is less complex than the ground-truth sequence–function relationship requires.
- Evaluation outside the random library's skewed distribution is needed to reveal whether a simpler functional form has genuinely recovered the ground-truth function.

![Synthetic GT model-misspecification comparison](<outputs/ground_truth_collections/Synthetic GT/figures/model_comparison/model_comparison_bar_type3.png>)

### Part II. Biological application

#### 1. Biological activity distributions resemble the Synthetic GT prediction

- The biological random-library activity distributions show the same type of skew posited in the Synthetic GT proof of concept.
- Increasing mutation count produces progressively skewed random-library activity distributions.
- The VTS1 high-WT sequence was replaced with a natural RNCMPT00111 probe whose GCUGG motif is fully unpaired in its own RNAfold MFE structure (zero overlap with its 3-pair stem), resolving the earlier motif/stem confound; raw ResidualBind WT activity also increased from 4.55 to 8.83.
- Libraries, the deepSQUID VTS1 surrogate (held-out test ρ = 0.960), and this distribution figure were regenerated under the new sequence; other downstream VTS1 figures remain to be regenerated (see Holes).

![VTS1 and HuR region-conditioned activity distributions at 10% mutation rate](<figures/simplified_a_10pct_focus.png>)

#### 2. Biological landscapes recreate accuracy inflation across library sizes

- The accuracy inflation predicted by the Synthetic GT proof of concept is reproduced in the biological landscapes.
- Smaller training libraries reduce recovery in both biological landscapes.
- At 20K variants, VTS1 appears nearly perfectly recovered on random variants despite remaining far from recovered under activity-balanced evaluation.
- At 20K variants, the ΔSpearman correlation coefficient between random-holdout and uniform-evaluation performance is 0.05 for HuR and 0.32 for VTS1.
- HuR is the negative control: random and activity-balanced evaluation are both high at 20K, indicating genuine recovery for a sequence-motif landscape.
- The contrast suggests that held-out accuracy is reliable for HuR but inflated for VTS1.
- The summary table will test whether predictive-accuracy inflation is accompanied by disagreement between the learned and ground-truth additive and pairwise coefficients.

![Biological library-size comparison](<figures/variant_f_biological_pair.png>)

![Placeholder for biological recovery and coefficient-agreement summary](<figures/hur_vts1_summary_table_placeholder.png>)

### Part III. Failure modes underlying the recovery gap

#### 1. Targeted-pairwise evaluation tests recovery of secondary-structure interactions

- A surrogate trained on the same 20K random library scores differently depending on which held-out sequence distribution evaluates it.
- The targeted-pairwise library tests whether the surrogate understands pairwise interactions within secondary structure.
- Random-holdout evaluation remains high for all three landscapes, whereas activity-balanced evaluation separates HuR from Synthetic GT and VTS1.
- Current targeted-pairwise VTS1 recovery is unexpectedly high and is not yet trusted.
- Saturated additive-plus-pairwise evaluation is omitted until its unique purpose is clarified.

![Preferred evaluation-library comparison prototype](<figures/variant_b_mutation_order.png>)

#### 2. Mutation count and mutation rate locate overprediction and underprediction

- Models trained at high mutation rates do not necessarily generalize to seemingly simpler, lower-mutation sequences.
- The below-diagonal failure is strongest for simpler Synthetic GT surrogates and weaker for VTS1.
- The effect disappears for HuR, provisionally linking the mismatch to landscape complexity.
- The SSM row and the quantitative size of the cross-landscape difference remain unresolved.
- HuR predictions remain close to the diagonal, whereas VTS1 predictions depart strongly from it.
- VTS1 can underpredict lower-Hamming-distance variants and overpredict higher-Hamming-distance variants.
- Mutation-count striations are reproducible visual structure but do not yet have a mechanistic interpretation.
- Standardized activity-score residual distributions replace the paired scatterplots as the mutation-count-resolved view of overprediction and underprediction.

![HuR and VTS1 standardized activity-score residual distributions by mutation count](<figures/variant_o_standardized_residual_violins.png>)

![Synthetic GT mutation-rate transfer](<outputs/ground_truth_collections/Synthetic GT/figures/mutation_rate_sweep/synthetic_gt_cross_mutrate_heatmap.png>)

![VTS1 mutation-rate transfer](<outputs/ground_truth_collections/deepSQUID VTS1/figures/mutation_rate_sweep/deepsquid_vts1_cross_mutrate_heatmap_high.png>)

![HuR mutation-rate transfer](<outputs/ground_truth_collections/deepSQUID HuR/figures/mutation_rate_sweep/deepsquid_hur_cross_mutrate_heatmap_high.png>)

## Discussion

## Conclusion

## Holes

- Biological generalization: determine whether and where SMN1 or the GFP ortholog landscapes supply supporting evidence without replacing Synthetic GT, HuR, or VTS1 as the main examples, then repeat the core library-size, mutation-rate, and evaluation-library comparisons on any dataset selected.
- Part I, Section 1: generate the synthetic ground-truth negative control and test whether, provisionally at 10% mutation rate, its random training and test activity distributions share the expected skew relative to uniform evaluation.
- Part I, Section 2: replace the simulated expectation with the matched negative-control and existing Synthetic GT scatterplots and observed Spearman correlation coefficients.
- Part I, Section 3: determine which model-form and evaluation-library comparisons are necessary to demonstrate hidden misspecification without overloading the figure.
- Part II, Section 1: define what score loss constitutes a nonfunctional VTS1 variant.
- VTS1 sequence swap: every VTS1-dependent figure downstream of the deepSQUID surrogate (Part II, Section 2 library-size comparison; Part III, Section 1 targeted-pairwise/library-roles; Part III, Section 2 residual violins and mutation-rate transfer heatmap) was generated under the old, motif-overlapping WT and needs regeneration under the new sequence.
- Part II, Section 2: rerun all library-size comparisons across 10 initializations and regenerate the selected triptych.
- Part II, Section 2: populate the HuR/VTS1 summary table with mean Spearman ρ ± SD and additive- and pairwise-coefficient cosine similarities across initializations.
- Part III, Section 1: redesign targeted pairwise to test mutation orders n = 2, 3, 4, and onward through mutation of the complete stem.
- Part III, Section 1: determine whether targeted-pairwise VTS1 recovery remains high after the redesign.
- Part III, Section 1: define the unique purpose of saturated additive-plus-pairwise evaluation before restoring it to any figure.
- Part III, Section 2: add the missing mixed-mutation-rate training row.
- Part III, Section 2: quantify the below-diagonal generalization deficit across landscapes and test its relationship to complexity.
- Part III, Section 2: interpret or remove the SSM row.
- Part III, Section 2: explain the mutation-count striations and report mutation-specific Spearman values.
- Higher-order recovery: measure what sequence-function signal saturated singles and doubles cannot capture, especially in VTS1.
- Coefficient recovery: resolve the unexpectedly high Synthetic GT pairwise cosine similarity before using coefficient agreement.

## Supplement

### VTS1 activity and motif coverage

- Broad activity coverage can coexist with motif preservation and sparse alternative mechanisms in VTS1.

![VTS1 activity mutation map](<figures/vts1_activity_mutation_map.png>)

### Raw Spearman correlation coefficients

- Report the raw Spearman correlation coefficients for every library size, mutation rate, surrogate form, and initialization.

### Low-wild-type activity distributions

- Candidate supplement analysis: retain only if it establishes stronger mutation-rate-dependent skew than the high-wild-type controls.

## Revisit

- Full coefficient maps are cut; cosine similarity is the preferred coefficient-agreement summary.
- Repeated scatterplots across every mutation rate are cut; one matched HuR/VTS1 pair is sufficient.
- ResidualBind-oracle pipeline results are cut except for validating deepSQUID as a replacement oracle.
- Twister and MSI1 are archived and excluded from the active example set.
- [deepSQUID versus ResidualBind held-out agreement](<outputs/ground_truth_collections/deepsquid_vs_real_oracle_heldout_test_bar.png>) is cut; two values don't warrant a bar chart, so the oracle-validation claim is now stated directly in Methods.
