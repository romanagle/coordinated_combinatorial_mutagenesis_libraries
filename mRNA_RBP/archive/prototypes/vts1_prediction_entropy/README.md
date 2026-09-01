# High-WT prediction-bin entropy prototype

This is a throwaway figure prototype answering one question: after sorting the
20,000 activity-balanced VTS1-high sequences by the final prediction of the
nonlinear additive-plus-pairwise surrogate trained on the 10%-mutation, 20K
random library, how does nucleotide diversity vary across 30 equal-count bins?

The prototype now supports the matched HuR-high library under the same design.
The heatmap reports Shannon entropy at each of 41 positions on its theoretical
0–2 bit scale. Bins run from the lowest predictions at the top to the highest
at the bottom. The side panel reports each bin's median prediction. Its red WT
line is the pipeline's WT-anchored activity reference at zero; the cached
prediction artifact does not retain a separately evaluated surrogate WT value.

Run in one command from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/vts1_prediction_entropy/make_prototype.py
python mRNA_RBP/scripts/figures/prototypes/vts1_prediction_entropy/make_prototype.py --target hur-high
python mRNA_RBP/scripts/figures/prototypes/vts1_prediction_entropy/make_prototype.py --mutation-rate 5
python mRNA_RBP/scripts/figures/prototypes/vts1_prediction_entropy/make_prototype.py --mutation-rate 25
```

The command writes the PNG and a CSV containing every sequence-row assignment
to `outputs/`.
