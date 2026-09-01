
## Objective

- Analyze YbeY conservation and its relationship to uS11 TIGR03632 motifs.

## Current status

- uS11 residues 107-110 are extracted from the TIGR03632 HMM match-state alignment.
- Species-level YbeY/uS11 tables preserve the full taxonomy row universe.
- uS11 and YbeY full aligned FASTAs are available in the Roma workspace.
- Exact final YbeY and Fap7 FASTA subsets are available in the Roma workspace.
- Redo uS11 motif counts include hierarchical clade rows through genus.
- Gap-included PHNG logos exist for all seven selected clades.
- Gap-included Bipolaricaulota logo uses all 179 uS11 rows as the denominator.
- NG/GG/DG suffix-count breakdown is generated from redo uS11 residues.
- Manuscript reviewer comments from Yanqing (Fig S14) and Jamie (scatter dot scaling) are open.
- Scatter dot area is linear in n with an additive floor, so rare non-canonical variants are drawn too large.

## Active blockers

- None recorded.

## Important decisions

- Gap-included PHNG logos replace the valid-only panels for all seven clades.
- Interpret uS11 positions 107-110 as TIGR03632 match-state positions.
- Remove HMMER insert-state characters before indexing TIGR03632 positions.
- Use the Fap7 initial-tree 0.98-gap-trimmed subset for methods-matching final aligned hits.

## Durable research memory

- NG/DG/GG suffix counts use uS11 residues 109-110, not exact PHNG/PHDG/PHGG motifs.
- Suffix-count TSV: `/data/roma/ybey/results/us11_suffix_counts_by_genus_redo_us11_2026-07-30.tsv`.
- Suffix-count generator: `/data/roma/ybey/write_us11_suffix_counts_by_genus_redo_us11_2026_07_30.py`.
- HMMER uppercase residues and `-` gaps correspond to model match states.
- HMMER lowercase residues and `.` characters are insert-state material.
- `match_state_alignment_length` is 117 because TIGR03632 has 117 match states.
- `match_state_alignment_length` is not a per-protein amino-acid length.
- `has_residues_107_110` is the useful completeness flag for the motif positions.
- Current full YbeY aligned FASTA: `/data/roma/ybey/individual/bac120_r232_reps_TIGR00043.faa`.
- Current uS11 aligned FASTA: `/data/roma/ybey/us11_aligned.faa`.
- Final YbeY exact-hit FASTAs are in `/data/roma/ybey/ybey_search/ybey_homologs_final_fastas/`.
- Final Fap7 exact-hit FASTAs are in `/data/roma/ybey/fap7_search/fap7_homologs_final_fastas/`.
- Methods-matching Fap7 aligned subset: `/data/roma/ybey/fap7_search/fap7_homologs_final_fastas/fap7_homologs_final_hits_aligned_initial_tree_0.98_gap_thresh.afa`.
- Hierarchical redo motif counts: `/data/roma/ybey/results/us11_motif_counts_by_genus_redo_us11_2026-07-30.tsv`.
- Fap7 methods trimming drops columns with >98% gaps, keeping columns with >=2% occupancy.
- uS11 logo code drops positions with >50% gaps, keeping positions with >=50% occupancy.
- Gap-included PHNG logos preserve gap-heavy positions as reduced stack height.
- Gap-included Bipolaricaulota PNG: `Artifacts/YbeY/phng_logo_gap_included_p_Bipolaricaulota.png`.
- Gap-included all-clade SVG artifacts: `Artifacts/YbeY/phng_logo_gap_included_*.svg`.
- Gap-included all-clade SVG workspace directory: `/data/roma/ybey/redo_us11_alignment_2026-07-30/phng_logos_gap_included_svg/`.
- Gap-included clade summary TSV: `/data/roma/ybey/redo_us11_alignment_2026-07-30/phng_logos_valid_107_110/phng_clade_summary_gap_included.tsv`.
- Gap-included all-clade generator: `/data/roma/ybey/redo_us11_alignment_2026-07-30/plot_bipolaricaulota_phng_logo_gap_included.py`.
- YbeY final aligned hits are a row subset of an existing TIGR00043 alignment, not a new alignment.
- Current PHNG/YbeY scatter: `/data/roma/ybey/redo_us11_alignment_2026-07-30/phng_summary_dots_original_redo_valid_denoms.png`.
- Its generator is `plot_phng_summary_dots_original_redo.py`, reading `us11_positions_107_110.tsv`.
- Dot size is `s = 100 + 1700 * (n / n_max)` passed to `scatter(s=...)`, which matplotlib treats as point area.
- The scaling is affine in n, never logarithmic; log10 and log2 give identical ratios and compress far harder.
- The additive floor outweighs the data term below n ≈ 9,838, so small-n dots are drawn too large.
- `110 not G` (n=111) is 98.9% floor and carries essentially no count information.
- `plot_phng_summary_dots_proportional_noncanon.py` sizes non-canonical dots proportionally and pins canonical dots to a fixed reference, but still reads the pre-redo FASTA input.
- Canonical dots span roughly 1,500x, so they cannot share a proportional area scale with the non-canonical dots.

| Dot | n | drawn area | floor share |
| --- | ---: | ---: | ---: |
| 107 P | 154,765 | 1673.1 | 6.0% |
| 107 not P | 12,806 | 230.2 | 43.4% |
| 108 H | 165,795 | 1785.2 | 5.6% |
| 108 not H | 1,751 | 117.8 | 84.9% |
| 109 N | 164,132 | 1768.3 | 5.7% |
| 109 not N | 3,307 | 133.6 | 74.8% |
| 110 G | 167,253 | 1800.0 | 5.6% |
| 110 not G | 111 | 101.1 | 98.9% |

| Sizing scheme | area ratio at 108 (H vs ≠H) | diameter ratio |
| --- | ---: | ---: |
| current (floor + linear area) | 15.2x | 3.9x |
| log10 or log2 | 1.6x | 1.3x |
| radius-linear | 24.9x | 5.0x |
| strict area ∝ n | 94.7x | 9.7x |

| Genus-level suffix bucket | Old exact motif | New suffix bucket | Change | Percent change |
| --- | ---: | ---: | ---: | ---: |
| NG | 150,846 | 164,060 | +13,214 | +8.8% |
| DG | 580 | 590 | +10 | +1.7% |
| GG | 1,648 | 2,468 | +820 | +49.8% |
| Total | 153,074 | 167,118 | +14,044 | +9.2% |

| Added non-PH motif | Added counts |
| --- | ---: |
| AHNG | 11,485 |
| PFNG | 1,187 |
| AHGG | 554 |
| PYNG | 142 |
| AFNG | 123 |
| THNG | 120 |
| QFGG | 96 |
| PFGG | 69 |

| Redo uS11 metric | Total |
| --- | ---: |
| Clade rows through genus | 43,800 |
| Leaf genus rows | 34,834 |
| Total species in clades | 189,801 |
| uS11 hits | 168,840 |
| Full-length motifs 107-110 | 167,350 |
| PHNG | 150,846 |
| PHDG | 580 |
| PHGG | 1,648 |
| Other full-length motifs | 14,276 |
| Missing or invalid 107-110 | 1,490 |

| Hierarchical TSV validation | Result |
| --- | ---: |
| Expected clade rows | 43,800 |
| Output clade rows | 43,800 |
| Missing rows | 0 |
| Count mismatches | 0 |
| Arithmetic invariant violations | 0 |

| Gap-included Bipolaricaulota logo metric | Value |
| --- | ---: |
| uS11 rows in denominator | 179 |
| Position 107 non-gap occupancy | 99.4% |
| Position 108 non-gap occupancy | 99.4% |
| Position 109 non-gap occupancy | 53.1% |
| Position 110 non-gap occupancy | 24.0% |

| Gap-included PHNG clade | n uS11 | 107 occ. | 108 occ. | 109 occ. | 110 occ. | YbeY | Fap7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p__Bipolaricaulota | 179 | 99.4% | 99.4% | 53.1% | 24.0% | 0.0% | 0.0% |
| p__Patescibacteriota | 10,408 | 99.3% | 99.3% | 99.3% | 99.3% | 69.8% | 0.0% |
| c__Gracilibacteria | 866 | 99.2% | 99.2% | 99.2% | 99.2% | 49.5% | 0.0% |
| c__Saccharimonadia | 1,773 | 99.4% | 99.4% | 99.4% | 99.4% | 5.9% | 0.0% |
| c__Dojkabacteria | 216 | 99.1% | 99.1% | 99.1% | 99.1% | 63.0% | 0.0% |
| p__UBP14 | 25 | 100.0% | 100.0% | 100.0% | 100.0% | 12.0% | 0.0% |
| o__Desulfurobacteriales | 29 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 86.2% |

## Open research questions

- None recorded.

## Next milestone

- Resolve the Yanqing and Jamie manuscript comments, including a correctly scaled PHNG/YbeY scatter.

## Future directions

- Consider renaming `match_state_alignment_length` if it continues to cause confusion.

## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
