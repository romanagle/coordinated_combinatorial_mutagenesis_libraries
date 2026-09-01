# VTS1 activity motif histogram

This exploratory histogram uses all 20,000 sequences from the high-WT VTS1
activity-balanced library, including sequences with zero motif mutations. It
counts mutations at the four constrained positions in `GCNGG` (one-based
positions 21, 22, 24, and 25); the central U at position 23 is treated as
unconstrained `N` and ignored. The x-axis is the ResidualBind activity score,
the y-axis is probability density, and each equal-width bar is colored by the
mean number of mutated `GCNGG` positions among sequences in that activity bin.

Run from the repository root:

```bash
python mRNA_RBP/scripts/figures/prototypes/vts1_activity_motif_histogram/make_figure.py
```

The command writes a PNG and the exact per-bin values used in the plot to
`outputs/`. Use `--bins` to change the default 50-bin resolution.

To make a nucleotide-frequency logo for all sequences with a ResidualBind
activity score greater than zero, including sequences with zero motif mutations:

```bash
python mRNA_RBP/scripts/figures/prototypes/vts1_activity_motif_histogram/make_sequence_logo.py
```

This writes the logo and its underlying per-position nucleotide frequencies to
`outputs/`.

To identify mutations that support high activity when the constrained motif is
disrupted, run counterfactual WT-reversion attribution in the `squid`
environment:

```bash
LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64 \
  conda run -n squid python \
  mRNA_RBP/scripts/figures/prototypes/vts1_activity_motif_histogram/make_rescue_attribution.py
```

For each motif-disrupted sequence in the rightmost histogram bin, the script
reverts every observed mutation individually and in pairs, rescores with the
same saved deepSQUID VTS1 model, and writes attribution CSVs and maps under
`outputs/rescue_attribution/`. A positive single-reversion attribution means
that reverting the mutation lowers predicted activity, so the observed
mutation supports the rescuing effect in that sequence background.
