# VTS1 activity-balanced mutation map

## Question

- Does broad activity coverage also provide mechanistically diverse VTS1 sequences within each activity stratum?

## Method

- Sorted 20,000 high-WT VTS1 activity-balanced variants by actual deepSQUID activity.
- Divided variants into 20 equal-count bins of 1,000 sequences.
- Calculated per-position mutation frequency relative to the 41-nt WT sequence.
- Measured complete `GCUGG` retention at WT positions 21–25.
- Matched the null to each bin's mixture of 3-, 5-, 7-, and 15-mutation sequences.

## Result

![[Artifacts/random_lib_mutagenesis/2026-08-13_vts1_activity_mutation_map/vts1_activity_mutation_map.png]]

- Low activity strongly associates with `GCUGG` disruption.
- Central activity bins preserve `GCUGG` substantially more than expected from mutation count alone.
- Activity balancing covers phenotype space without guaranteeing balanced mechanistic coverage.

| Activity bin | n | Median activity | Motif intact | Matched null | Mean mutations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1,000 | -2.780 | 5.1% | 35.1% | 8.622 |
| 2 | 1,000 | -2.563 | 22.3% | 39.7% | 7.544 |
| 3 | 1,000 | -2.346 | 44.3% | 40.7% | 7.312 |
| 4 | 1,000 | -2.128 | 61.7% | 44.4% | 6.584 |
| 5 | 1,000 | -1.911 | 67.5% | 46.3% | 6.250 |
| 6 | 1,000 | -1.693 | 74.3% | 49.5% | 5.654 |
| 7 | 1,000 | -1.476 | 78.2% | 50.3% | 5.516 |
| 8 | 1,000 | -1.258 | 83.5% | 50.6% | 5.454 |
| 9 | 1,000 | -1.041 | 83.5% | 51.9% | 5.210 |
| 10 | 1,000 | -0.824 | 84.2% | 52.3% | 5.198 |
| 11 | 1,000 | -0.607 | 85.2% | 52.5% | 5.134 |
| 12 | 1,000 | -0.389 | 85.3% | 53.4% | 5.030 |
| 13 | 1,000 | -0.172 | 88.1% | 54.0% | 4.872 |
| 14 | 1,000 | 0.045 | 87.6% | 54.0% | 4.904 |
| 15 | 1,000 | 0.263 | 86.8% | 54.8% | 4.858 |
| 16 | 1,000 | 0.480 | 84.3% | 53.4% | 5.084 |
| 17 | 1,000 | 0.698 | 85.4% | 53.2% | 5.076 |
| 18 | 1,000 | 0.915 | 81.0% | 50.5% | 5.574 |
| 19 | 1,000 | 1.132 | 77.4% | 47.1% | 6.278 |
| 20 | 1,000 | 1.350 | 71.0% | 42.3% | 7.208 |

## Interpretation limits

- Equal activity-bin sizes arise by quantile construction, not equal representation in the candidate pool.
- Motif disruption alone does not define a mechanism; matched substitutions and structural contexts remain unmeasured.
- The archived CSV preserves unrounded values for all plotted bins.

## Files

- [[Artifacts/random_lib_mutagenesis/2026-08-13_vts1_activity_mutation_map/vts1_activity_motif_summary.csv|Per-bin motif summary CSV]]
