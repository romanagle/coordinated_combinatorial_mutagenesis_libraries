# Cross-mutation-rate bias versus random training-library size

All comparisons use the nonlinear additive + pairwise surrogate trained at a
25% mutation rate. Values are Spearman |rho|. The 200K condition was evaluated
on the same deterministic lower-rate pool samples used by the existing
cross-mutation-rate experiment.

| Landscape | Training size | Matched 25% test | 5% test | 10% test |
| --- | ---: | ---: | ---: | ---: |
| Synthetic GT (mean, n=10) | 2K | 0.404 | 0.558 | 0.497 |
| Synthetic GT (mean, n=10) | 20K | 0.828 | 0.781 | 0.758 |
| VTS1 high-WT (n=1) | 2K | 0.672 | 0.834 | 0.773 |
| VTS1 high-WT (n=1) | 20K | 0.999 | 0.848 | 0.892 |
| VTS1 high-WT (n=1) | 200K | 0.977 | 0.780 | 0.845 |
| HuR high-WT (n=1) | 2K | 0.759 | 0.827 | 0.846 |
| HuR high-WT (n=1) | 20K | 1.000 | 0.915 | 0.957 |
| HuR high-WT (n=1) | 200K | 1.000 | 0.879 | 0.939 |

## Interpretation

- Reducing the library from 20K to 2K lowers absolute cross-rate performance
  in every landscape and comparison. It primarily causes broad underfitting,
  so the matched-minus-cross gap is not itself a useful bias measure at 2K.
- Increasing from 20K to 200K does not remove the biological low-rate deficit.
  Both VTS1 and HuR are worse on the 5% and 10% tests at 200K than at 20K.
- The stable HuR 200K fit has matched-test rho 0.9996, so its remaining deficit
  cannot be attributed to general underfitting. The effect therefore appears
  distributional rather than a simple lack-of-data effect.
- VTS1's 200K matched-test rho is 0.977, below its 20K value of 0.999; treat
  the exact VTS1 200K effect size cautiously. It nevertheless provides no
  evidence that more random high-rate data rescues lower-rate transfer.
- A unique 200K 5%-rate library is impossible for this 41-position system:
  the exact-two-mutant space contains only 7,380 sequences. This does not
  affect the tested 25%-training to lower-rate-transfer question.
- The ten-instance synthetic 200K sweep was not run because the CPU-only fits
  require roughly 10--15 minutes per condition. Its complete 2K/20K comparison
  is included above.
