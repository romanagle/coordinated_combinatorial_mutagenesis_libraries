# ResidualBind oracle VTS1

Figures are in `figures/`; copied cache/model/weight artifacts are in `cached_libraries/`. The source pipeline workspace for this GT is contained in `pipeline_workspace/`. The manifest records every file included in the curated collection.

- Figures copied in manifest: `11`
- Cached artifacts copied in manifest: `131`
- Missing source files: `0`

## WT Sequence Contexts

- High-WT random-library figure: `rand_lib_dist_vts1_oracle_region_classes.png`
  uses `AAGGACACUAAGUACAGGUUGCUGGCACAGGGCGCUCAUAA`.
- Low-WT random-library figure: `rand_lib_dist_vts1_oracle_region_classes_low_wt.png`
  uses `AAAAGAUGGCUAUGCGACCCGCUGGAACUAGUAAGUGAAAA`.
- The completed pipeline/activity-balanced library uses the high-WT sequence.

## Activity-Balanced Library

The canonical evaluation library is `activity_balanced.npz`; this collection
does not use or write `type2.npz`.

Initialization recipe:

- Sample exact-mutant candidate pools at 3, 5, 7, and 15 mutations from the
  selected WT sequence.
- Use 200,000 total candidates split evenly across those mutation counts.
- Deduplicate candidates globally.
- Score candidates with the ResidualBind VTS1 oracle score.
- Histogram-uniformize in score space with 200 equal-width bins, percentile
  clipping `[1, 99]`, seed `k*10000 + 600`, and target cap 20,000 sequences.
- The final count can be below 20,000 if nonempty score bins are sparse.
