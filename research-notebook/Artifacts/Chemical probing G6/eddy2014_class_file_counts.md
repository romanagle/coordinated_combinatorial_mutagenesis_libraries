# Eddy 2014 class-file counts — `1st`/`2nd` are pooled splits, not 16S/23S

_Analysis date: 2026-08-02 — [[Chemical probing G6]]_

Test: do the Eddy 2014 `1st` and `2nd` SHAPE class files correspond to 16S (SSU) and 23S (LSU)?
Method: compare the value counts in each class file against the helix-end position counts
derived from the 16S and 23S pairing arrays.

## Counts

| Class | 16S (by RNA) | 23S (by RNA) | `1st` file | `2nd` file | `1st` vs `2nd` |
| --- | --- | --- | --- | --- | --- |
| helix-end (pairing) | 432 | 791 | 577 | 576 | differ by 1 |
| unpaired | — | — | 828 | 828 | equal |
| stacked | — | — | 689 | 689 | equal |

## Reading

- If `1st = 16S` and `2nd = 23S`, the helix-end row would read 432/791. It reads 577/576.
- 577/576 is an even split of the pooled total, not an RNA-specific one.
- The two file counts also sum to 1,153, not the 1,223 helix-end positions across 16S + 23S.
- All six class files (3 classes × `1st`/`2nd`) are numerically sorted and divided equally or
  within one value.

**Conclusion:** the `1st`/`2nd` suffix marks an equal-sized sorted split of pooled values, not a
per-RNA dataset. Treat Eddy 2014 pooled SHAPE values as nonpositional data; do not attach them
to 16S or 23S coordinates.

## Limits

- 16S/23S per-RNA counts for the unpaired and stacked classes were not computed — the pairing
  class alone is sufficient to reject the hypothesis.
- Lengths cannot prove *random* allocation before sorting. That needs the original
  `shape_16S.dat` / `shape_23S.dat` arrays, which are unavailable.
- Open question: were pooled class values randomly divided before being sorted into halves?
