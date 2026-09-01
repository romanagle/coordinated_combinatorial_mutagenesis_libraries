# PROTOTYPE — paired random-library distributions

Throwaway static figure prototypes comparing VTS1 and HuR ResidualBind
random-library score distributions. VTS1 uses an original RNAcompete-2013
probe with a fully unpaired GCUGG motif and a separate three-pair stem.

Regenerate all three PNGs from the existing score caches:

```bash
/home/nagle/miniconda3/envs/toehold_gpu/bin/python3 \
  mRNA_RBP/scripts/figures/prototypes/distribution_pair_figure/generate_replacement_vts1_cache.py
python mRNA_RBP/scripts/figures/prototypes/distribution_pair_figure/prototype_distribution_pair.py
```

The figures use cached WT-relative scores and cached sequence/region metadata.
No values are simulated. Variant A omits the “neither” class; variants B and C
show the complete random library without assigning region-specific effects.

Variant A uses mutually exclusive motif-only and stem-only classes. Sequences
that mutate both regions are omitted, as are sequences that mutate neither.

- VTS1 high-WT (canonical as of 2026-08-31): `AAAAAAGACGAGAGCGACACCGGCUGGCCCGACGGAAAAAA`
- RNAfold structure: `...................(((..........)))......`
- Stem pairs: `(21,32), (20,33), (19,34)`; motif positions: `22–26`.
