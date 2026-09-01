# PROTOTYPE — saturated library with nonlinearity probes

This prototype restores the original four seed-42 triple mutants as a
separate extension of the exhaustive single-plus-double library. It does not
change the production `type3.npz` library.

For the 41-nt high-WT deepSQUID VTS1 sequence, the prototype contains:

- 123 single mutants
- 7,380 double mutants
- 4 triple mutants
- 7,507 unique sequences total

Regenerate the library and figures from the repository root with the `squid`
environment:

```bash
MPLCONFIGDIR=/tmp/mplconfig conda run -n squid \
  python mRNA_RBP/scripts/figures/prototypes/saturated_nonlinearity_library/make_prototype.py
```

Everything produced by the prototype is written to `outputs/` beside this
README and generator.
