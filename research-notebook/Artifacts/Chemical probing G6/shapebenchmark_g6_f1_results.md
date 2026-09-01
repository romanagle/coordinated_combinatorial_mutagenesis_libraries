# ShapeKnots 24-RNA G6 benchmark

## Method

- Each RNA used its supplied CT reference, independent SHAPE track, and full sequence.
- Reactivities were normalized by each RNA's positive 90th percentile.
- The in-sample likelihood model pooled all 24 normalized, reference-labeled tracks.
- Four one-sided CT annotations were repaired only when the stated partner was otherwise unpaired.
- F1 values compare base pairs against the supplied reference structure.

## Aggregate result

| Method | Mean F1 |
| --- | ---: |
| TornadoFold G6 without probing | 0.510 |
| Python G6 CYK with SHAPE | 0.476 |
| Mean guided-minus-unguided change | -0.034 |

| Outcome | RNAs |
| --- | ---: |
| Improved | 6 |
| Unchanged | 3 |
| Worsened | 15 |

## Per-RNA results

| RNA | Unguided G6 F1 | SHAPE-guided G6 F1 | Delta F1 | ViennaRNA MFE F1 |
| --- | ---: | ---: | ---: | ---: |
| 5' domain of 16S rRNA, E. coli | 0.441 | 0.627 | +0.186 | 0.579 |
| 5' domain of 16S rRNA, H. volcanii | 0.774 | 0.720 | -0.055 | 0.832 |
| 5' domain of 23S rRNA, E. coli | 0.669 | 0.692 | +0.022 | 0.686 |
| 5S rRNA, E. coli | 0.298 | 0.298 | 0.000 | 0.260 |
| Adenine riboswitch, V. vulnificus | 1.000 | 0.930 | -0.070 | 1.000 |
| Fluoride riboswitch, P. syringae | 0.000 | 0.263 | +0.263 | 0.333 |
| Group I Intron, T. thermophila | 0.743 | 0.741 | -0.003 | 0.642 |
| Group I intron, Azoarcus sp. | 0.496 | 0.328 | -0.168 | 0.512 |
| Group II intron, O. iheyensis | 0.533 | 0.426 | -0.108 | 0.675 |
| HIV-1 5' pseudoknot domain | 0.188 | 0.189 | +0.000 | 0.283 |
| Hepatitis C virus IRES domain | 0.372 | 0.321 | -0.051 | 0.299 |
| Lysine riboswitch, T. maritime | 0.410 | 0.400 | -0.010 | 0.370 |
| M-Box riboswitch, B. subtilis | 0.809 | 0.839 | +0.030 | 0.894 |
| P546 domain, bI3 group I intron | 0.833 | 0.769 | -0.064 | 0.692 |
| Pre-Q1 riboswitch, B. subtilis | 0.000 | 0.000 | 0.000 | 0.000 |
| RNase P, B. subtilis | 0.445 | 0.336 | -0.109 | 0.479 |
| SAM I riboswitch, T. tengcongensis | 0.685 | 0.471 | -0.214 | 0.630 |
| SARS corona virus pseudoknot | 0.356 | 0.308 | -0.048 | 0.340 |
| Signal recognition particle RNA, human | 0.222 | 0.180 | -0.042 | 0.113 |
| TPP riboswitch, E. coli | 0.718 | 0.591 | -0.127 | 0.810 |
| Telomerase pseudoknot, human | 0.000 | 0.000 | 0.000 | 0.000 |
| cyclic-di-GMP riboswitch, V. cholerae | 0.346 | 0.600 | +0.254 | 0.862 |
| tRNA(asp), yeast | 0.895 | 0.667 | -0.228 | 0.500 |
| tRNA(phe), E. coli | 1.000 | 0.723 | -0.277 | 1.000 |

## Interpretation limit

- The likelihood model is in-sample because each evaluated RNA contributes to its pooled histogram.
