# Random-library mutagenesis narrative map

## Question

- Random mutagenesis samples multi-position variants, while library size and mutation rate determine sequence-space coverage.
- Brittle landscapes may enrich random libraries for nonfunctional variants and obscure the underlying sequence-function relationship.
- Held-out accuracy may overstate recovery when training and test variants share the same biased distribution.
- How do random-library size, mutation rate, and sequence-function landscape properties affect recovery of the underlying sequence-function relationship?

## Methods

- Synthetic GT provides a constructed nonlinear additive-plus-pairwise landscape with known ground truth.
- deepSQUID VTS1 provides a biological structure-dependent landscape with a high-activity wild type.
- deepSQUID HuR provides a sequence-motif negative control with a high-activity wild type.
- Random libraries vary size and mutation rate and train nonlinear additive-plus-pairwise surrogates.
- Activity-balanced evaluation measures recovery outside the random library's skewed activity distribution.
- Targeted-pairwise evaluation currently contains double mutants restricted to annotated base-pairing regions.
- SSM supplies an additive-only baseline for learning higher-order sequence-function relationships.

## Results

### 1. deepSQUID can stand in for the biological oracle

- Held-out agreement between deepSQUID and ResidualBind supports using deepSQUID as the primary oracle in subsequent experiments.

![deepSQUID versus ResidualBind held-out agreement](<outputs/ground_truth_collections/deepsquid_vs_real_oracle_heldout_test_bar.png>)

### 2. Random-library test accuracy can overstate landscape recovery

- Smaller training libraries reduce recovery across every landscape.
- Synthetic GT and VTS1 retain high random-holdout accuracy while activity-balanced evaluation reveals substantially weaker landscape recovery.
- At 20K variants, VTS1 appears nearly perfectly recovered on random variants despite remaining far from recovered under activity-balanced evaluation.
- HuR is the negative control: random and activity-balanced evaluation are both high at 20K, indicating genuine recovery for a sequence-motif landscape.
- The contrast suggests that held-out accuracy is reliable for HuR but inflated for the more complex Synthetic GT and VTS1 landscapes.

![Selected library-size comparison prototype](<prototypes/library_size_figure/outputs/variant_a_triptych.png>)

### 3. Evaluation-library choice changes the apparent success of the same surrogate

- A surrogate trained on the same 20K random library scores differently depending on which held-out sequence distribution evaluates it.
- Random-holdout evaluation remains high for all three landscapes, whereas activity-balanced evaluation separates HuR from Synthetic GT and VTS1.
- Current targeted-pairwise VTS1 recovery is unexpectedly high and is not yet trusted.
- Saturated additive-plus-pairwise evaluation is omitted until its unique purpose is clarified.

![Preferred evaluation-library comparison prototype](<prototypes/library_roles_figure/variant_b_mutation_order.png>)

### 4. Mutation-rate mismatch exposes incomplete train-test generalization

- Models trained at high mutation rates do not necessarily generalize to seemingly simpler, lower-mutation sequences.
- The below-diagonal failure is strongest for simpler Synthetic GT surrogates and weaker for VTS1.
- The effect disappears for HuR, provisionally linking the mismatch to landscape complexity.
- The SSM row and the quantitative size of the cross-landscape difference remain unresolved.

![Synthetic GT mutation-rate transfer](<outputs/ground_truth_collections/Synthetic GT/figures/mutation_rate_sweep/synthetic_gt_cross_mutrate_heatmap.png>)

![VTS1 mutation-rate transfer](<outputs/ground_truth_collections/deepSQUID VTS1/figures/mutation_rate_sweep/deepsquid_vts1_cross_mutrate_heatmap_high.png>)

![HuR mutation-rate transfer](<outputs/ground_truth_collections/deepSQUID HuR/figures/mutation_rate_sweep/deepsquid_hur_cross_mutrate_heatmap_high.png>)

### 5. Prediction errors depend systematically on mutation count

- HuR predictions remain close to the diagonal, whereas VTS1 predictions depart strongly from it.
- VTS1 can underpredict lower-Hamming-distance variants and overpredict higher-Hamming-distance variants.
- Mutation-count striations are reproducible visual structure but do not yet have a mechanistic interpretation.
- The paired scatterplots should appear once; mutation-specific Spearman values carry the quantitative comparison elsewhere.

![Preferred HuR and VTS1 scatterplot pair](<prototypes/scatter_pair_figure/variant_a_clean_pair.png>)

### 6. Random-library activity distributions may explain biased recovery

- Increasing mutation count produces progressively skewed random-library activity distributions.
- The current VTS1 example confounds motif disruption with stem disruption because its motif lies inside the stem.
- A new landscape with non-overlapping motif and stem regions is required before assigning the mechanism.

![Blocked paired-distribution prototype](<prototypes/distribution_pair_figure/variant_a_paired_ridgelines.png>)

## Holes

- Section 2: rerun all library-size comparisons across 10 initializations and regenerate the selected triptych.
- Section 3: redesign targeted pairwise to test mutation orders n = 2, 3, 4, and onward through mutation of the complete stem.
- Section 3: determine whether targeted-pairwise VTS1 recovery remains high after the redesign.
- Section 3: define the unique purpose of saturated additive-plus-pairwise evaluation before restoring it to any figure.
- Section 4: add the missing mixed-mutation-rate training row.
- Section 4: quantify the below-diagonal generalization deficit across landscapes and test its relationship to complexity.
- Section 4: interpret or remove the SSM row.
- Section 5: explain the mutation-count striations and report mutation-specific Spearman values.
- Section 6: select and run a landscape with separable motif and stem regions.
- Section 6: define what score loss constitutes a nonfunctional VTS1 variant.
- Higher-order recovery: measure what sequence-function signal saturated singles and doubles cannot capture, especially in VTS1.
- Coefficient recovery: resolve the unexpectedly high Synthetic GT pairwise cosine similarity before using coefficient agreement.

## Revisit

- Low-WT activity distributions are parked for the supplement unless they establish stronger mutation-rate-dependent skew than high-WT controls.
- Full coefficient maps are cut; cosine similarity is the preferred coefficient-agreement summary.
- Repeated scatterplots across every mutation rate are cut; one matched HuR/VTS1 pair is sufficient.
- ResidualBind-oracle pipeline results are cut except for validating deepSQUID as a replacement oracle.
- Twister and MSI1 are archived and excluded from the active example set.
