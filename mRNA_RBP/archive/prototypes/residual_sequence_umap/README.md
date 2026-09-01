# Residual sequence UMAP prototype

This prototype embeds the high-WT VTS1 activity-balanced evaluation library
using sequence information only. Each exact mutation-count shell (3, 5, 7,
or 15 mutations from WT) receives an independent UMAP based on categorical
Hamming distance. Point color is the residual from the nonlinear
additive-plus-pairwise surrogate trained on the 20,000-sequence, 10%-mutation
random library.

The accompanying permutation test compares the mean absolute residual
difference among each sequence's 10 nearest Hamming neighbors with the same
quantity after shuffling residuals within that mutation-count shell. A ratio
below one indicates locally coherent residuals.

Install the added dependency and run:

```bash
python -m pip install -r mRNA_RBP/prototypes/residual_sequence_umap/requirements.txt
python mRNA_RBP/scripts/figures/prototypes/residual_sequence_umap/make_prototype.py
python mRNA_RBP/scripts/figures/prototypes/residual_sequence_umap/make_neighbor_graph.py
python mRNA_RBP/scripts/figures/prototypes/residual_sequence_umap/make_representative_neighborhoods.py
```

UMAP is exploratory and can exaggerate visual separation. Interpret apparent
clusters only alongside the neighborhood test.

`make_neighbor_graph.py` adds an adaptive nearest-neighbor overlay and creates
one explanatory neighborhood for each mutation count. The example center is
selected as the sequence whose eight nearest neighbors share the most exact
substitutions with it; it illustrates what proximity means and is not assigned
a cluster label. The three-mutation overview additionally labels density-based
satellite clusters by their dominant exact mutation and exports the assignments
to `vts1_high_3mut_cluster_labels.csv`. Residuals do not enter clustering.

`make_representative_neighborhoods.py` replaces the unreadable global graph
with three non-overlapping local examples per mutation shell. The examples
span underprediction, near-zero error, and overprediction. Beside each graph,
a mutation-frequency panel identifies exact substitutions shared by all or by
a subset of neighborhood members.
