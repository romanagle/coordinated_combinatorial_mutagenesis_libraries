# Random-library mutagenesis narrative map

## Question

- Random mutagenesis samples multi-position variants, while library size and mutation rate determine sequence-space coverage.
- Brittle landscapes may enrich random libraries for nonfunctional variants and obscure the underlying sequence-function relationship.
- Held-out accuracy may overstate recovery when training and test variants share the same biased distribution.
- How do random-library size, mutation rate, and sequence-function landscape properties affect recovery of the underlying sequence-function relationship?

## Methods

### Synthetic ground-truth landscape

- A single fixed 41-nucleotide RNA sequence defines the landscape.
- The function combines additive weights (α, L × 4) and pairwise couplings at stem pairs, followed by a global sigmoid nonlinearity.
- At non-stem positions, non-WT nucleotides receive additive weights drawn from N(0, σ).
- Stem positions have zero additive contribution, and pairwise couplings penalize all non-compensatory mutations.
- The WT always scores 0, and all mutations score ≤ 0 by construction.
- Five instances use different random weight draws, with different α magnitudes but the same structure.

![[Artifacts/random_lib_mutagenesis/synthetic_gt_coefficients_only.png]]

### Experimental design

- The experiment crosses 5 synthetic ground-truth instances, 3 mutation rates (5%, 10%, and 25%), 3 training-library sizes (200, 2,000, and 20,000 variants), and 4 surrogate functional forms.
- The four surrogate functional forms are additive, additive plus pairwise, nonlinear additive, and nonlinear additive plus pairwise.

### Other methods

- deepSQUID VTS1 provides a biological structure-dependent landscape with a high-activity wild type and recognizes a GCUGG motif.
- deepSQUID HuR provides a sequence-motif negative control with a high-activity wild type and recognizes an AUUUA motif.

![[Artifacts/random_lib_mutagenesis/vts1_hur_secondary_structure_placeholder.png]]

- Random libraries vary size and mutation rate and train nonlinear additive-plus-pairwise surrogates.
- Activity-balanced evaluation measures recovery outside the random library's skewed activity distribution.
- Targeted-pairwise evaluation currently contains double mutants restricted to annotated base-pairing regions.
- SSM supplies an additive-only baseline for learning higher-order sequence-function relationships.
- Oracle validation: held-out agreement between deepSQUID and ResidualBind supports using deepSQUID as the primary oracle in subsequent experiments.

![[Artifacts/random_lib_mutagenesis/deepsquid_vs_real_oracle_heldout_test_bar.png]]

## Results

### Part I. Synthetic ground-truth proof of concept

#### 1. Skewed random-library distributions can inflate random-holdout accuracy

- At a provisionally selected 10% mutation rate, the random training and random test activity distributions are expected to show similar skew, whereas the uniform test library should cover the activity distribution more evenly.
- The matched skew of the random training and test libraries is expected to produce high random-holdout accuracy without probing the entire sequence–function landscape.
- Uniform evaluation should therefore assess recovery of the underlying sequence–function relationship more accurately than a random holdout drawn from the training distribution.

![[Artifacts/random_lib_mutagenesis/synthetic_gt_distribution_placeholder.png]]

#### 2. Random-holdout accuracy can conceal poor landscape-wide recovery

- In the expected negative control, random-library and uniform-holdout predictions both follow the diagonal and produce high Spearman correlation coefficients.
- In the existing Synthetic GT, random-library holdout predictions are expected to follow the diagonal only within the densely sampled lower-left region, producing a high Spearman correlation coefficient over a narrow activity range.
- Uniform-holdout sequences span the activity grid, while their predictions are expected to depart from the diagonal and produce a low Spearman correlation coefficient.
- This contrast would show that high random-holdout accuracy can reflect recovery of a skewed region rather than understanding of the full sequence–function landscape.

![[Artifacts/random_lib_mutagenesis/synthetic_gt_scatter_expectation.png]]

#### 3. Random-holdout accuracy can mask model misspecification

- High random-holdout accuracy can conceal that the surrogate functional form is less complex than the ground-truth sequence–function relationship requires.
- Evaluation outside the random library's skewed distribution is needed to reveal whether a simpler functional form has genuinely recovered the ground-truth function.

![[Artifacts/random_lib_mutagenesis/model_comparison_bar_type3.png]]

### Part II. Biological application

#### 1. Biological activity distributions resemble the Synthetic GT prediction

- The biological random-library activity distributions show the same type of skew posited in the Synthetic GT proof of concept.
- Increasing mutation count produces progressively skewed random-library activity distributions.
- The current VTS1 example confounds motif disruption with stem disruption because its motif lies inside the stem.
- A new landscape with non-overlapping motif and stem regions is required before assigning the mechanism.

![[Artifacts/random_lib_mutagenesis/variant_a_paired_ridgelines.png]]

#### 2. Biological landscapes recreate accuracy inflation across library sizes

- The accuracy inflation predicted by the Synthetic GT proof of concept is reproduced in the biological landscapes.
- Smaller training libraries reduce recovery in both biological landscapes.
- At 20K variants, VTS1 appears nearly perfectly recovered on random variants despite remaining far from recovered under activity-balanced evaluation.
- At 20K variants, the ΔSpearman correlation coefficient between random-holdout and uniform-evaluation performance is 0.05 for HuR and 0.32 for VTS1.
- HuR is the negative control: random and activity-balanced evaluation are both high at 20K, indicating genuine recovery for a sequence-motif landscape.
- The contrast suggests that held-out accuracy is reliable for HuR but inflated for VTS1.
- The summary table will test whether predictive-accuracy inflation is accompanied by disagreement between the learned and ground-truth additive and pairwise coefficients.

![[Artifacts/random_lib_mutagenesis/variant_f_biological_pair.png]]

![[Artifacts/random_lib_mutagenesis/hur_vts1_summary_table_placeholder.png]]

### Part III. Failure modes underlying the recovery gap

#### 1. Targeted-pairwise evaluation tests recovery of secondary-structure interactions

- A surrogate trained on the same 20K random library scores differently depending on which held-out sequence distribution evaluates it.
- The targeted-pairwise library tests whether the surrogate understands pairwise interactions within secondary structure.
- Random-holdout evaluation remains high for all three landscapes, whereas activity-balanced evaluation separates HuR from Synthetic GT and VTS1.
- Current targeted-pairwise VTS1 recovery is unexpectedly high and is not yet trusted.
- Saturated additive-plus-pairwise evaluation is omitted until its unique purpose is clarified.

![[Artifacts/random_lib_mutagenesis/variant_b_mutation_order.png]]

#### 2. Mutation count and mutation rate locate overprediction and underprediction

- Models trained at high mutation rates do not necessarily generalize to seemingly simpler, lower-mutation sequences.
- The below-diagonal failure is strongest for simpler Synthetic GT surrogates and weaker for VTS1.
- The effect disappears for HuR, provisionally linking the mismatch to landscape complexity.
- The SSM row and the quantitative size of the cross-landscape difference remain unresolved.
- HuR predictions remain close to the diagonal, whereas VTS1 predictions depart strongly from it.
- VTS1 can underpredict lower-Hamming-distance variants and overpredict higher-Hamming-distance variants.
- Mutation-count striations are reproducible visual structure but do not yet have a mechanistic interpretation.
- Standardized activity-score residual distributions replace the paired scatterplots as the mutation-count-resolved view of overprediction and underprediction.

![[Artifacts/random_lib_mutagenesis/2026-08-05_activity_balanced_residual_violins.png]]

![[Artifacts/random_lib_mutagenesis/synthetic_gt_cross_mutrate_heatmap.png]]

![[Artifacts/random_lib_mutagenesis/deepsquid_vts1_cross_mutrate_heatmap_high.png]]

![[Artifacts/random_lib_mutagenesis/deepsquid_hur_cross_mutrate_heatmap_high.png]]

## Discussion

## Conclusion

## Holes

- Part I, Section 1: generate the synthetic ground-truth negative control and test whether, provisionally at 10% mutation rate, its random training and test activity distributions share the expected skew relative to uniform evaluation.
- Part I, Section 2: replace the simulated expectation with the matched negative-control and existing Synthetic GT scatterplots and observed Spearman correlation coefficients.
- Part I, Section 3: determine which model-form and evaluation-library comparisons are necessary to demonstrate hidden misspecification without overloading the figure.
- Part II, Section 1: select and run a landscape with separable motif and stem regions.
- Part II, Section 1: define what score loss constitutes a nonfunctional VTS1 variant.
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

### Raw Spearman correlation coefficients

- Report the raw Spearman correlation coefficients for every library size, mutation rate, and initialization.

### Low-wild-type activity distributions

- Candidate supplement analysis: retain only if it establishes stronger mutation-rate-dependent skew than the high-wild-type controls.

## Revisit

- Full coefficient maps are cut; cosine similarity is the preferred coefficient-agreement summary.
- Repeated scatterplots across every mutation rate are cut; one matched HuR/VTS1 pair is sufficient.
- ResidualBind-oracle pipeline results are cut except for validating deepSQUID as a replacement oracle.
- Twister and MSI1 are archived and excluded from the active example set.
