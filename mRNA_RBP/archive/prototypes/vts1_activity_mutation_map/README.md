# VTS1 activity mutation map

This figure tests whether the `GCUGG` motif remains fixed despite broad actual
activity coverage in the high-WT VTS1 activity-balanced library. It divides the
20,000 library sequences into 20 equal-count bins using the cached deepSQUID
VTS1 oracle score—not the surrogate prediction—and plots the fraction mutated
at every sequence position.

The side panel shows the fraction retaining the complete WT `GCUGG` at
positions 21–25. Its null curve is the expected intact fraction if each exact
mutation count were placed uniformly among the 41 positions; it therefore
accounts for the differing mixtures of 3-, 5-, 7-, and 15-mutation sequences
across activity bins.

Run from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/vts1_activity_mutation_map/make_figure.py
```

The command writes a PNG and the plotted per-bin motif summary to `outputs/`.
