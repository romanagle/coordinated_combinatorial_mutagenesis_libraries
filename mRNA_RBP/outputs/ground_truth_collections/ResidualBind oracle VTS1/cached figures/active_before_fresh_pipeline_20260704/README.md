# ResidualBind VTS1 figures

Flat figure set matching `../Synthetic GT/figures`, excluding coefficient maps.

Regenerate the high/low random-region figures with:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python mRNA_RBP/plots/plot_residualbind_vts1_rand_region_distributions.py --wt both
```

Regenerate the library-distribution and pairwise-heatmap figures with:

```bash
MPLCONFIGDIR=/tmp/mpl-cache python mRNA_RBP/plots/plot_residualbind_vts1_collection_figures.py
```

The staged model/rho/scatter/cross-mutrate figures are VTS1 versions copied from `../cached figures/`.

The ad hoc summary plots generated during cleanup were moved to `../cached outputs/ad_hoc_figures_20260704/`.
