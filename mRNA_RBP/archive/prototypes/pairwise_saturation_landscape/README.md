# Pairwise saturation landscape prototype

Question: how should a 41-position × 4-nucleotide pairwise saturated
mutagenesis library be visualized? Three static variants use the actual
deepSQUID VTS1-high saturated library (123 singles + 7,380 doubles).

- `variant_a_upper_triangle_dots.png`: one dot per measured variant; activity
  is color. Singles occupy the diagonal.
- `variant_b_symmetric_heatmap.png`: the same measurements mirrored into a
  dense matrix for easier pattern recognition.
- `variant_c_3d_activity.png`: activity is height and color in a 3D view.

Generate all variants with one command from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/pairwise_saturation_landscape/make_prototypes.py
```

This is throwaway figure-design code, not a production plotting module.

## Interactive Plotly version

`interactive_activity_landscape.html` is self-contained and offers tabs for
the 2D mutation map and rotatable 3D activity landscape. Hovering shows the
mutation identities, activity, and complete variant sequence.

Regenerate it after installing Plotly:

```bash
python mRNA_RBP/scripts/figures/prototypes/pairwise_saturation_landscape/make_interactive.py
```
