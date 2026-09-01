# Synthetic GT negative-control differential

## Question

Why does the activity-balanced negative-control scatter form four separated curves despite near-perfect random-holdout recovery?

## Setup

- Instance: `00`.
- Surrogate: nonlinear additive plus pairwise.
- Training library: 20,000 fixed four-mutation sequences (10% of 41 nt).
- Evaluation library: 20,000 activity-balanced sequences with 3, 5, 7, or 15 mutations.
- Display samples are deterministic and equal-sized; statistics use all predictions.

## Evaluation performance

| Synthetic GT | Random-holdout Spearman ρ | Activity-balanced Spearman ρ |
| --- | ---: | ---: |
| Structured positive control | 0.989 | 0.315 |
| Unstructured negative control | 1.000 | 0.379 |

## Negative-control performance by mutation count

| Mutations | n | Within-group Spearman ρ | Median prediction error | 5th–95th percentile error |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 5,865 | 0.999976 | -0.0847 | [-0.0902, -0.0707] |
| 5 | 5,349 | 0.999971 | +0.0871 | [+0.0628, +0.0923] |
| 7 | 6,421 | 0.999832 | +0.2394 | [+0.1515, +0.2624] |
| 15 | 2,365 | 0.996841 | +0.6518 | [+0.3316, +0.7162] |

Removing each mutation-count group's median residual raises the combined Spearman ρ from 0.379 to 0.996.

## Tested differential

| Hypothesis | Executed check | Verdict | Evidence |
| --- | --- | --- | --- |
| Mutation-count bands | Recomputed performance within each mutation count. | Confirmed | Four groups occur at 3, 5, 7, and 15 mutations; each has ρ > 0.996. |
| Cross-count calibration failure | Removed each group's median residual. | Confirmed | Combined ρ increases from 0.379 to 0.996. |
| Pairwise interactions create the bands | Measured cached pairwise contributions. | Mostly ruled out | Pairwise contribution is zero for 95% of sequences and nonzero for 4.38%. |
| Display subsampling creates the bands | Compared the frozen 20,000-point data with the deterministic display sample. | Ruled out | Full data contain the same four mutation-count groups. |
| Ordinary poor fitting | Evaluated the fixed-four-mutation random holdout. | Ruled out | Random-holdout ρ = 0.999984. |

## Verdict

- Fixed-four-mutation training supports interpolation on one Hamming shell but not calibration across mutation orders.
- The activity-balanced library tests mutation-count extrapolation as well as activity coverage.
- The current unstructured Synthetic GT does not validate the intended negative-control contrast.

## Archived artifacts

- ![[2026-08-09_synthetic_gt_activity_distributions.png]]
- ![[2026-08-09_synthetic_gt_scatter_actual.png]]
- [[2026-08-09_synthetic_gt_control_predictions.npz|Frozen prediction data]]
