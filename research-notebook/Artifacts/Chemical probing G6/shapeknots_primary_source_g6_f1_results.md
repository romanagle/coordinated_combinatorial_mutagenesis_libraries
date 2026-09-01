# ShapeKnots 24-RNA G6 benchmark — primary-source replication

## Method

- Independently re-derived the same 24 RNAs directly from the Hajdin et al. 2013 ShapeKnots
  supplementary data (`3rdparty/shapeknots/` in `stanti/shapebenchmark`), rather than the
  pre-extracted `.fa`/`.shape`/`.ct` triples used for the first ShapeKnots run.
- New converter `prepare_shapeknots.py` parses RNAstructure `.seq`-format sequences, reuses the
  existing pseudoknot-aware CT-to-dot-bracket conversion, and extracts SHAPE reactivities from
  `ShapeKnots_SNRNASM.xlsx`'s `datamatrix` sheet (one column per RNA).
- Header-to-file mapping was verified by cross-checking every sequence/CT length against its
  xlsx column's data range; 23/24 matched exactly, and the 24th (H. volcanii 16S) has one extra
  trailing xlsx value beyond the modeled sequence, consistent with raw probing traces including
  flanking nucleotides trimmed from the folded domain.
- `-999`/blank xlsx cells are written out as `NA`, matching the existing missing-value convention.
- Same G6 pipeline, same in-sample pooled likelihood model, same F1-only scoring as the first run.

## Aggregate result

| Method | Mean F1 |
| --- | ---: |
| TornadoFold G6 without probing | 0.506 |
| Python G6 CYK with SHAPE | 0.482 |
| Mean guided-minus-unguided change | -0.024 |

| Outcome | RNAs |
| --- | ---: |
| Improved | 7 |
| Unchanged | 2 |
| Worsened | 15 |

Closely matches the first ShapeKnots run (0.510 / 0.476 / -0.034, 6/3/15), which used the
pre-extracted `benchmarkdata/` triples. Per-RNA F1 values agree to within ~0.02 across both
independently-parsed data sources, cross-validating both preparation pipelines.

## Per-RNA results

| RNA | Unguided G6 F1 | SHAPE-guided G6 F1 | Delta F1 | ViennaRNA MFE F1 |
| --- | ---: | ---: | ---: | ---: |
| 5' domain of 16S rRNA, E. coli | 0.441 | 0.632 | +0.191 | 0.579 |
| 5' domain of 16S rRNA, H. volcanii | 0.774 | 0.730 | -0.044 | 0.832 |
| 5' domain of 23S rRNA, E. coli | 0.669 | 0.684 | +0.015 | 0.686 |
| 5S rRNA, E. coli | 0.298 | 0.294 | -0.005 | 0.260 |
| Adenine riboswitch, V. vulnificus | 1.000 | 0.930 | -0.070 | 1.000 |
| Fluoride riboswitch, P. syringae | 0.000 | 0.263 | +0.263 | 0.333 |
| Group I Intron, T. thermophila | 0.743 | 0.738 | -0.005 | 0.642 |
| Group I intron, Azoarcus sp. | 0.496 | 0.328 | -0.168 | 0.512 |
| Group II intron, O. iheyensis | 0.533 | 0.426 | -0.108 | 0.675 |
| HIV-1 5' pseudoknot domain | 0.188 | 0.207 | +0.018 | 0.283 |
| Hepatitis C virus IRES domain | 0.372 | 0.485 | +0.113 | 0.299 |
| Lysine riboswitch, T. maritime | 0.410 | 0.397 | -0.014 | 0.370 |
| M-Box riboswitch, B. subtilis | 0.809 | 0.851 | +0.043 | 0.894 |
| P546 domain, bI3 group I intron | 0.741 | 0.667 | -0.074 | 0.692 |
| Pre-Q1 riboswitch, B. subtilis | 0.000 | 0.000 | 0.000 | 0.000 |
| RNase P, B. subtilis | 0.445 | 0.335 | -0.111 | 0.479 |
| SAM I riboswitch, T. tengcongensis | 0.685 | 0.464 | -0.221 | 0.630 |
| SARS corona virus pseudoknot | 0.356 | 0.308 | -0.048 | 0.340 |
| Signal recognition particle RNA, human | 0.222 | 0.180 | -0.042 | 0.113 |
| TPP riboswitch, E. coli | 0.718 | 0.653 | -0.065 | 0.810 |
| Telomerase pseudoknot, human | 0.000 | 0.000 | 0.000 | 0.000 |
| cyclic-di-GMP riboswitch, V. cholerae | 0.346 | 0.600 | +0.254 | 0.862 |
| tRNA(asp), yeast | 0.895 | 0.667 | -0.228 | 0.500 |
| tRNA(phe), E. coli | 1.000 | 0.723 | -0.277 | 1.000 |

## Figures

![[Artifacts/Chemical probing G6/shapeknots_primary_source_g6_f1_scatter.png]]

![[Artifacts/Chemical probing G6/shapeknots_primary_source_g6_f1_by_family_scatter.png]]

![[Artifacts/Chemical probing G6/shapeknots_primary_source_g6_f1_binned_delta.png]]

![[Artifacts/Chemical probing G6/shapeknots_primary_source_shape_paired_unpaired.png]]

## Environment note

- Must run with `G6_chemical_probing/.venv/bin/python3`, not the system `python3`: the system
  interpreter has a real pip-installed `tornado` (web framework) that shadows the pipeline's
  local `rscape_v2.6.4/python/d-SCFG/grammars/tornado` namespace package, breaking
  `compare_g6.py`'s `from tornado.tornado_fold import grmfold_stats_parse` import.

## Interpretation limit

- Same in-sample likelihood caveat as the first run: each evaluated RNA contributes to its own
  pooled histogram.
- `prepare_shapeknots.py` is written but not yet committed/pushed to `G6_chemical_probing`.
