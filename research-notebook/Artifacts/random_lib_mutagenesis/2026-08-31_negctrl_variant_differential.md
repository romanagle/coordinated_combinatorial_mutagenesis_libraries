# Negative-control variant differential

## Question

Which structural choices in a negative-control synthetic ground-truth (GT) design -- motif
concentration, weak non-structural pairwise coupling, the sigmoid nonlinearity, or background
signal magnitude -- change how well a surrogate recovers it on a standard 10% mutation-rate /
20,000-sequence random-holdout split?

## Setup

- Instance: `00` (seed 0), matching the "instance 00" convention used elsewhere in the pipeline.
- All five variants share the same sequence backbone and declared motif positions as
  `mrna_negative_control` (`MSI1_SEQ`, `MSI1_MOTIF_POSITIONS`).
- Surrogate: nonlinear additive + pairwise (same MAVE-NN config and training routine as
  `lib_size_spearman.py`).
- Training library: 20,000 sequences at 10% mutation rate (~4 mutations of 41 nt), generated
  fresh per variant (no cached pipeline libraries involved).
- Metric: Spearman rho on MAVE-NN's internal random holdout split (`rho_rand`) -- the same
  quantity `lib_size_spearman.py` reports for the mut10%/lib20000 condition.
- Script: `mRNA_RBP/scripts/experiments/negctrl_variant_differential.py` (one-off, not part of
  the registered oracle pipeline -- does not modify `gt_init.py`).

## The differential

| Variant | Category | Design | If-true / if-false |
| --- | --- | --- | --- |
| V0 motif-only (current design) | Domain | Registered `mrna_negative_control`: motif_sigma=3.0 at 5 motif positions, bg_sigma=0.10 elsewhere, no pairwise | If true: near-ceiling rho_rand. If false: motif concentration alone depresses rho_rand. |
| V1 null / no motif | Sampling-coverage | Same bg_sigma=0.10 everywhere, no privileged region | If true: rho_rand matches V0 (concentration doesn't matter). If false: removing the motif changes rho_rand. |
| V2 motif + leaky pairwise | Domain (mechanism boundary) | V0 plus weak (sigma=0.3), sparse (p_edge=0.15) WC-compatible pairwise edges scattered across the whole sequence -- not stem-restricted; recreates the original failed negative control's topology layered on a motif-only background | If true: even weak non-structural pairwise coupling depresses rho_rand relative to V0. If false: a pairwise-capable surrogate absorbs it and rho_rand stays high. |
| V3 motif, linear (no nonlinearity) | Statistical artifact / methodological | V0 without the sigmoid squashing (raw additive score) | If true: removing the nonlinearity changes rho_rand relative to V0. If false: rho_rand is unchanged, ruling out the nonlinearity as a random-holdout factor. |
| V4 diffuse, moderate background | Sampling-coverage | No privileged region, bg_sigma=0.5 (the structured GT's own background scale; 5x V1's) | If true: rho_rand tracks total additive signal magnitude, not concentration. If false: rho_rand matches V0/V1 regardless of background scale. |

## Coefficient maps

![[2026-08-31_negctrl_variant_differential_coefficients.png]]

Individual full-detail figures (per-variant additive + pairwise panels) and the raw results:
[negctrl_variant_differential outputs](../../../mRNA_RBP/prototypes/negctrl_variant_differential/outputs/).

## 10% / 20,000-sequence random-holdout Spearman rho

| Variant | n_train | rho_rand |
| --- | ---: | ---: |
| V0 motif-only (current design) | 19,974 | 0.9998 |
| V1 null / no motif | 19,974 | 1.0000 |
| V2 motif + leaky pairwise | 19,974 | 0.9991 |
| V3 motif, linear (no nonlinearity) | 19,974 | 0.9985 |
| V4 diffuse, moderate background | 19,974 | 1.0000 |

All five span only 0.9985-1.0000 (range of 0.0015).

## Activity-score distributions (variants x mutation rates)

Random-library GT score histograms, 5 variants x 3 mutation rates (5%, 10%, 25%; lib_size
20,000, or fewer when the exact-mutation-count sequence space is smaller than 20,000 -- e.g.
6,838 unique 2-mutant sequences at 5%). GT scoring only, no surrogate training.

![[2026-08-31_negctrl_variant_activity_distributions_5x3.png]]

| Variant | Shape |
| --- | --- |
| V0 motif-only (current design) | Sharp WT-adjacent spike plus discrete lower clusters (sigmoid saturation around how many of the 5 motif positions are hit) |
| V1 null / no motif | Smooth, unimodal, narrow; shifts left and widens only mildly as mutation rate increases |
| V2 motif + leaky pairwise | Visually near-identical to V0 -- the weak leaky pairwise doesn't perturb the marginal score distribution |
| V3 motif, linear (no nonlinearity) | Heavy left tail, no saturation ceiling -- unsquashed motif hits produce large-magnitude outliers, unlike every nonlinear variant |
| V4 diffuse, moderate background | Smooth, unimodal, wider than V1 (5x background scale) but still no discrete clusters -- no privileged region means no saturating subpopulation |

The nonlinearity, not the motif itself, is what produces V0/V2's multi-modal/clustered shape: V1
and V4 (both nonlinear, no motif) stay smooth, while V3 (motif, no nonlinearity) loses the
saturation ceiling and becomes heavy-tailed instead of clustered. This is a visible, qualitative
difference between the variants that `rho_rand` does not surface -- worth carrying into the
activity-balanced follow-up in Next steps below.

## Tested differential

| Hypothesis | Executed check | Verdict | Evidence |
| --- | --- | --- | --- |
| V0 motif-only recovers near-ceiling | Trained nonlinear additive+pairwise surrogate on 20K seqs @ 10% mut rate | Confirmed | rho_rand = 0.9998 |
| V1 vs V0: motif concentration doesn't matter for random-holdout recovery | Same training/eval with motif removed (uniform bg_sigma=0.10) | Confirmed | rho_rand = 1.0000 vs 0.9998 (delta = +0.0002) |
| V4 vs V0/V1: background magnitude doesn't materially change random-holdout recovery | Same training/eval, no motif, bg_sigma=0.5 (5x V1) | Confirmed | rho_rand = 1.0000, indistinguishable from V1 despite 5x stronger diffuse signal |
| V2 vs V0: weak leaky pairwise depresses random-holdout recovery | Same training/eval, motif-only + 24 weak WC-compatible non-stem edges | Ruled out (as a practically meaningful effect) | rho_rand = 0.9991 vs 0.9998 -- lowest-but-one of the five, but the 0.0007 gap is far smaller than the activity-balanced gaps seen for the same design axis in the 2026-08-09 differential (0.315 vs 0.379 vs 0.527) |
| V3 vs V0: removing the sigmoid nonlinearity changes random-holdout recovery | Same additive structure, raw (unsquashed) score | Ruled out (as a practically meaningful effect) | rho_rand = 0.9985 vs 0.9998 -- lowest of the five, but still 0.0013 above V2's is-the-difference-real threshold; not distinguishable from noise at this scale |

## Verdict

- None of the four design axes tested (motif concentration, background magnitude, weak
  non-stem-restricted pairwise coupling, sigmoid nonlinearity) produces a practically
  meaningful difference in 10%/20,000-sequence random-holdout Spearman rho -- every variant
  sits in [0.9985, 1.0000].
- This reaffirms the 2026-08-09 finding
  ([[2026-08-09_synthetic_gt_negative_control_differential]]): rho_rand on a fixed-mutation-rate
  random holdout is not a discriminating metric for negative-control design choices. A
  pairwise-capable nonlinear surrogate interpolates any of these designs almost perfectly
  within one Hamming shell, regardless of whether the underlying GT has a concentrated motif,
  diffuse background, weak leaky pairwise coupling, or no nonlinearity at all.
- V2 (leaky pairwise) and V3 (no nonlinearity) show the two largest -- still tiny -- reductions
  relative to V0, making them the more promising candidates to re-test under the
  activity-balanced (mutation-count-extrapolation) evaluation, where the 2026-08-09 differential
  showed real separation (0.315-0.527) that random-holdout rho could not see.

## Next steps

- [ ] Re-run V0-V4 through `generate_type1_activity_balanced`-style evaluation (3/5/7/15-mutation
      activity-balanced library) rather than random holdout, to see whether these design axes
      separate the way motif-only vs. the original unstructured control did on 2026-08-09.
- [ ] If a design axis does separate under activity-balanced evaluation, consider it as a
      candidate replacement or companion for the registered `mrna_negative_control` oracle.

## Archived artifacts

- ![[2026-08-31_negctrl_variant_differential_coefficients.png]]
- ![[2026-08-31_negctrl_variant_activity_distributions_5x3.png]]
- [Results JSON](../../../mRNA_RBP/prototypes/negctrl_variant_differential/outputs/negctrl_variant_differential_results.json)
- [Experiment script](../../../mRNA_RBP/scripts/experiments/negctrl_variant_differential.py)
