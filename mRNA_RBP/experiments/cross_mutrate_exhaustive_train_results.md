# Exhaustive single/double training across mutation rates

The nonlinear additive + pairwise surrogate was trained on all 7,503 single
and double mutants, then evaluated on deterministic 4,000-sequence samples
from the 5%, 10%, and 25% mutation-rate pools.

| Landscape | Instances | 5% test | 10% test | 25% test |
| --- | ---: | ---: | ---: | ---: |
| Synthetic GT | 10 | 0.835 ± 0.196 | 0.794 ± 0.223 | 0.680 ± 0.237 |
| deepSQUID VTS1 high-WT | 1 | 0.993 | 0.972 | 0.778 |
| deepSQUID HuR high-WT | 1 | 0.991 | 0.969 | 0.831 |

Values are Spearman |rho|; ± values are sample standard deviations across
synthetic instances.

## Interpretation

For VTS1, exhaustive single/double training removes the original low-rate
failure: the 5% test correlation increases from 0.848 for 25%-random training
to 0.993, and the 10% test correlation increases from 0.892 to 0.972. HuR has
the same qualitative pattern. However, both exhaustive-trained biological
surrogates lose accuracy at 25%, showing that complete low-order coverage does
not teach the higher-order sequence landscape.

The synthetic result is heterogeneous across the ten ground-truth instances.
Its mean low-rate performance improves only modestly relative to 25%-random
training (5%: 0.835 versus 0.781; 10%: 0.794 versus 0.758) and remains below
the rate-matched random-trained models. Thus the directional low-rate bias
mostly disappears in the biological examples, but exhaustive singles and
doubles are not a universal replacement for distribution-matched training.

Only the nonlinear additive + pairwise surrogate is used for this and future
cross-mutation-rate analyses.
