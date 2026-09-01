# RNA-FM and RiNALMo UMAP Split Analysis

Date: 2026-08-03

## Observation

- RNA-FM masked UMAP shows separated isotype regions.
- A broad split separates type-II/long tRNAs from short tRNAs.
- A second RNA-FM split separates `Thr/fMet/His/Ala/Val`-rich regions from `Glu/Cys/Gly/Gln`-rich regions.
- RiNALMo shows the broad isotype/type-II structure too.

## Checks Run

- Sampled masked RNA-FM and RiNALMo embedding rows.
- Mapped masked sequences to `Anticodon_predicted_isotype`, domain, phylum, order, and sequence length.
- Computed high-dimensional cosine kNN structure without relying on UMAP coordinates.
- Compared observed same-label kNN rates against random baselines.
- Fit diagonal GMMs on PCA-reduced RNA-FM embeddings for selected short isotypes.
- Selected GMM size by BIC across tested `k=1..12`.

## High-Dimensional kNN Results

| Check | RNA-FM | RiNALMo | Random baseline |
| --- | ---: | ---: | ---: |
| Same isotype among kNN | 0.656 | 0.730 | 0.076 / 0.073 |
| Same type-II status among kNN | 0.949 | 0.961 | 0.585 / 0.590 |
| Same length bin among kNN | 0.942 | 0.944 | 0.582 |
| Same phylum among kNN | 0.227 | 0.260 | 0.091 / 0.078 |
| Same order among kNN | 0.046 | 0.053 | 0.008 / 0.007 |

## Interpretation

- The broad split is present in both FMs.
- The broad split is not a UMAP-only artifact.
- Type-II status and sequence length explain more of the broad split than phylogeny.
- Phylogeny contributes local structure but is weaker than isotype.
- Embedding nearest neighbors are sequence-similar, but sequence identity alone does not explain every isotype island.

## RNA-FM Short-Isotype GMM

Focused isotypes:

- `Thr`
- `fMet`
- `His`
- `Ala`
- `Val`
- `Glu`
- `Cys`
- `Gly`
- `Gln`

Sample:

- RNA-FM masked embeddings.
- 30,000 focused rows.
- PCA-reduced to 20 dimensions.
- Diagonal covariance GMM.
- BIC evaluated for `k=1..12`.

## GMM Model Selection

| k | BIC |
| ---: | ---: |
| 1 | 1703139.8 |
| 2 | 1687691.8 |
| 3 | 1672381.3 |
| 4 | 1659736.1 |
| 5 | 1650080.8 |
| 6 | 1639952.7 |
| 7 | 1633351.3 |
| 8 | 1623288.2 |
| 9 | 1617712.8 |
| 10 | 1605902.4 |
| 11 | 1603839.2 |
| 12 | 1592073.4 |

Result:

- Best tested model: `k=12`.
- BIC did not support a simple two-cluster Gaussian model.
- The split is a higher-level organization over multiple components.

## RNA-FM GMM Cluster Families

`Thr/fMet/His/Ala/Val`-rich components:

- Cluster 2: `His/Ala/Gly/Val`.
- Cluster 3: `Val/Ala`.
- Cluster 4: mostly `Thr`.
- Cluster 6: `Val/Thr`.
- Cluster 12: almost pure `fMet`.

`Glu/Cys/Gly/Gln`-rich components:

- Cluster 1: mostly `Gln`, with `Glu/Gly/His/Thr` mixed.
- Cluster 7: mostly `Glu`.
- Cluster 9: mostly `Gly`.
- Cluster 10: `Gln/Cys`.
- Cluster 11: mostly `Cys`.

Bridge components:

- Cluster 5: mixed across named isotypes, with higher `>80nt` fraction.
- Cluster 8: mixed, with extra `Thr`.

## Sequence Similarity Check

- Embedding nearest-neighbor masked-sequence identity was higher than random.
- RNA-FM nearest-neighbor identity: 0.703 versus 0.359 random.
- RiNALMo nearest-neighbor identity: 0.642 versus 0.347 random.
- RNA-FM cluster 12 had high within-cluster identity: 0.83.
- RNA-FM clusters 2, 3, 4, and 10 had within-cluster identities near 0.62-0.68.
- RNA-FM cluster 5 had low within-cluster identity: 0.25.

## Verdict

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| Broad RNA-FM split is also present in RiNALMo | Confirmed | RiNALMo same-isotype and same type-II kNN rates exceed random baselines. |
| Broad split is mostly type-II/length structure | Confirmed | Same type-II and same length-bin kNN rates are about 0.94-0.96. |
| Phylogeny explains the split | Partly supported | Same phylum/order are enriched but weaker than isotype/type-II. |
| RNA-FM short-isotype split is two Gaussian clusters | Ruled out in tested range | BIC improved through `k=12`. |
| RNA-FM short-isotype split reflects multiple sequence-similar components | Supported | Components align with isotype families and show elevated within-cluster sequence identity. |

## Immediate Follow-Up

- Project RNA-FM GMM assignments onto the original UMAP.
- Inspect sequence motifs separating the component families.
- Compare clusters against structural features beyond length.
