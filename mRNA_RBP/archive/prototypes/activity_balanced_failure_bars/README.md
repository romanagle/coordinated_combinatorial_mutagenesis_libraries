# PROTOTYPE — activity-balanced failure decomposition

Question: which mutation-count strata cause the random-trained nonlinear
additive+pairwise surrogate to fail on the HuR and VTS1 activity-balanced
evaluation libraries?

This is throwaway figure code. It reads the existing 20k-training prediction
caches and produces three vertically stacked HuR/VTS1 bar-layout variants.

```bash
python mRNA_RBP/scripts/figures/prototypes/activity_balanced_failure_bars/make_prototypes.py
```

- Variant A: grouped bars retain the 5%, 10%, and 25% training regimes.
- Variant B: mean bars emphasize mutation order; dots retain the underlying
  training-regime values.
- Variant C: horizontal mean bars make the mutation-count rows literal and
  annotate the training-regime range.
- Variant D: grouped bars show global percentile-rank RMSE. All 20,000
  sequences are ranked together before errors are split by mutation count, so
  both within- and between-mutation-count ranking failures are retained.
- Variants E–G restrict the figure to the 10% random-training surrogate:
  bars, lollipops, and a connected error profile, respectively.
- Variants H–K put HuR and VTS1 in one plotting area and treat mutation count
  as categorical (equal spacing): overlaid profiles, dumbbells, paired bars,
  and a two-row heatmap.
- Variant L extends the preferred paired-bar layout with a diverging signed
  percentile-rank-bias panel. Positive means over-ranked; negative means
  under-ranked.
- Variants M–N replace rank residuals with standardized activity-score
  residuals, `(prediction - truth) / SD(truth)`: paired box distributions and
  mean ± SD points. These show direction and dispersion in one panel.
- Variant O shows the same standardized residual distributions as paired
  violins with median markers.
