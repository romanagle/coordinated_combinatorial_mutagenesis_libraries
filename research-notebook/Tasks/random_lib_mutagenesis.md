# [[random_lib_mutagenesis]]

## Now

- [ ] Select manuscript-ready coefficient-map comparisons from the August 26 prototypes.
- [ ] Prototype finite-landscape library-size and mutation-profile experiments on SMN1.
- [ ] Test cgreGFP as the sharply constrained GFP landscape against amacGFP and ppluGFP2.
- [ ] Decide how many unrelated biological systems are needed to support generality.
- [ ] Update the Part III, Section 2 interpretation for the [[../../mRNA_RBP/prototypes/activity_balanced_failure_bars/outputs/variant_o_standardized_residual_violins.png|accepted residual violin figure]], focusing on overprediction and underprediction before versus after four mutations (approximately the 10% mutation rate).

## Next

- [ ] Train the missing negative-control fixed-rate and varied-mutation deepSQUID coefficient caches.
- [ ] Refine the accepted Part I model-misspecification subpoint and decide which comparisons to retain in the [[Artifacts/random_lib_mutagenesis/model_comparison_bar_type3.png|two-panel Spearman/RMSE figure]].
- [ ] Create a Methods figure showing the workflow from ResidualBind → deepSQUID → surrogate-model training → evaluation libraries.
- [ ] Rerun full pipeline across 10 initializations; regenerate Synthetic GT, VTS1, and HuR high-WT library-size figures.
- [ ] Build an activity × motif-status × mutation-count occupancy map for VTS1 and identify sparse mechanistic cells.
- [ ] Populate the accepted HuR/VTS1 summary table with mean Spearman ρ ± SD and additive- and pairwise-coefficient cosine similarities across initializations.
- [ ] Replicate the redesigned motif-only synthetic ground-truth negative control across 10 instances.
- [ ] Add the missing mixed-mutation-rate training row to mutation-rate heatmaps.
- [ ] Redesign random-library inflation comparison across three landscapes and mutation rates; remove excluded rows.
- [ ] Test whether increasing mutation rate produces a more strongly skewed functional-score distribution for high-activity WT sequences than for low-activity WT sequences; decide whether the low-WT comparison supports a supplemental figure.
- [ ] Quantify cross-landscape train-test generalization differences and test their relationship to landscape complexity.
- [ ] Measure higher-order sequence-function signal missed by saturated additive-plus-pairwise mutagenesis, especially in VTS1.
- [ ] Select a landscape with non-overlapping motif and stem regions for distribution analysis.
- [ ] Revisit the purpose of the saturated additive-plus-pairwise evaluation library before assigning it a manuscript figure role.
- [ ] Redesign targeted-pairwise mutagenesis to include variants with all paired stem positions mutated at increasing orders (n = 2, 3, 4, ... through the full stem); rerun and investigate whether the current high VTS1 recovery is trustworthy.
- [ ] Audit scatterplots for unequal random-holdout versus activity-balanced point counts; standardize deterministic display sampling while retaining full datasets for reported statistics.

## Waiting

- [ ]

## Ideas

-

## Completed recently

- Redesigned the Synthetic GT negative control as motif-only additive effects with a 0.10 near-neutral background and no pairwise edges.
- Routed VTS1 ResidualBind and deepSQUID pipelines through explicit high/low natural sequence configurations.
- Built a per-position VTS1 activity mutation map with a mutation-count-matched motif-retention null; `GCUGG` retention is 5.1% / 85.2% / 71.0% across low/mid/high activity bins.
- Provided and incorporated the existing secondary-structure renderings of the high-activity VTS1 and HuR sequences.
- Found VTS1 and HuR crystal structures for inclusion in the manuscript.
- Removed >2-mutation variants from saturated additive-plus-pairwise libraries; reran all three landscapes.
