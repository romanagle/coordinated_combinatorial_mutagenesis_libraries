# VTS1 motif-copy-number check

## Question

- Do high VTS1 teacher scores arise because tiled sequences contain multiple compatible motifs?

## Data and method

- Source: `vts1_balanced_teacher_student/model/mave_df.csv`.
- Library: 3,700 randomized backgrounds with one forced `GCAGG` placement per sequence.
- Motif count: overlapping matches to `GCNGG`, where `N ∈ {A,C,G,U}`.
- Activity: standardized deepSQUID teacher label stored as `y`.
- Top activity: upper 10% of all teacher labels.

## VTS1 results

| `GCNGG` copies | Sequences | Mean teacher score | Median teacher score | Top-activity sequences |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1,687 | -0.516 | -0.863 | 4.5% |
| 2 | 1,899 | +0.391 | +0.239 | 13.7% |
| 3 | 113 | +1.111 | +1.326 | 29.2% |
| 4 | 1 | +2.117 | +2.117 | 100.0% |

- Pearson correlation between compatible motif count and teacher score is `r = 0.488`.
- Exact `GCAGG` copy count has a weaker correlation with teacher score: `r = 0.149`.
- Additional compatible `GCNGG` occurrences are associated with higher VTS1 teacher activity.
- Motif multiplicity can therefore inflate some high-placement scores.
- The association does not establish causality because motif count covaries with background sequence and placement.
- Retained positional structure after balanced placement means motif count is not the sole explanation.

## HuR comparison

| `AUUUA` copies | Sequences | Mean teacher score | Median teacher score | Top-activity sequences |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,554 | -0.065 | -0.235 | 7.5% |
| 2 | 1,143 | +0.148 | -0.130 | 15.7% |
| 3 | 3 | -0.606 | -0.744 | 0.0% |

- Pearson correlation between `AUUUA` count and HuR teacher score is `r = 0.096`.
- The motif-copy association is much stronger for VTS1 than for HuR.

## Interpretation

- VTS1 motif-copy number is a real confound in the balanced placement library.
- It may explain part of the highest predicted activity, but not the complete position-dependent pattern.
- A controlled comparison should hold compatible motif count fixed while varying placement.
