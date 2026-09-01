# [[YbeY]]

## Now

- [ ] Rerun `plot_phng_summary_dots_proportional_noncanon.py` against the redo TSV and compare figures.
- [ ] Address Yanqing's comment on Fig S14 (Bipolaricaulota uS11 C-terminus) in the SI doc: increase the figure's font size, report how many genomes are represented in the logos, and simplify the "red box marks the four-residue PHNG-equivalent window" sentence — e.g. just say these bacteria lack it.
- [ ] Reply to Jamie confirming the scatter is not log2-scaled (area is affine in n with a min_size=100 floor).

## Next

- [ ] For each sequence, locate the codon corresponding to position 118 (H, appears well conserved). Take n nucleotides up- and downstream of it and calculate nt frequencies for each position relative to 118.

## Waiting

## Ideas

## Completed recently

- Investigated Jamie's PHNG scatter comment: confirmed dot sizes are not log2-scaled; traced compressed non-canonical dots to the additive min_size=100 area floor.
- Sent Kate the NG/GG/DG count breakdown TSV.
- Regenerated hierarchical uS11 motif counts through genus.
- Confirmed hierarchical uS11 motif counts against the species-level source TSV.
- Decided the gap-included PHNG logos replace the valid-only panels.
- Sent Kate the gap-included Bipolaricaulota logo.
- Decided how to handle Bipolarcaulota.
- Added total species-per-clade column to motif counts TSV.
- Sent genus-level motif counts TSV to Kate.
- Sent Fap7 and YbeY FASTAs to Kate.
- Confirmed available YbeY and Fap7 FASTAs.
