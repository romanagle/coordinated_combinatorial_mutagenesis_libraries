
_Updated: August 14, 2026 (13:02)_

## Objective

Build a reproducible RNA-structure benchmarking workflow that combines SHAPE and DMS reactivity data with deposited or Rfam covariance-model-derived reference structures, then uses those data to compare secondary-structure prediction methods such as ViennaRNA and G6 CYK.

## Current status


- The RDAT corpus has been split and sorted by reagent. From 1,024 original files, reagent splitting produced 1,257 files, including 357 SHAPE and 149 DMS single-molecule files used by the histogram pipeline.

- Collision-suffix replicates were merged into 341 SHAPE and 135 DMS sequence groups.

- After structure-availability and signal-quality filtering, the dataset contains 243 SHAPE and 108 DMS sequences (351 total).

- Rfam covariance-model projection fills reference-structure gaps for 96 SHAPE groups and 23 DMS groups that lack a deposited structure.

- The `cmscan` -> `cmalign` -> WUSS projection -> window-slicing workflow is documented and validated. On six short, well-characterized RNA families, ViennaRNA and G6 CYK reach roughly 70-100% F1, supporting the validity of the projection and scoring pipeline.

- IRES benchmarking is complete at the family-summary level. G6 CYK outperforms ViennaRNA on four of six non-IRES families, while both methods perform substantially worse on long IRES RNAs.

- Portable histogram exports and directly scoreable RMDB/Eddy 2014 manifests have been generated. The RMDB scoring export contains 298 entries that passed cached-hit, signal, and parse checks.

- Human RNAmap benchmarking contains 2,502 paired TornadoFold G6 comparisons. Adding DMS improves 770 sequences, worsens 462, and leaves 1,270 exactly unchanged. Mean delta F1 is positive for G6-only baseline bins below 0.4 and negative for bins from 0.4 through 1.0.

- The Human RNAmap plotting stage is packaged as a portable workflow with per-hit, family-level, and binned-delta example outputs.

- The portable workflow is source-managed at `romanagle/G6_chemical_probing`; `main` uses consolidated `scripts/plot_g6.py` as of commit `82b1146`.

- The 71-record Human RNAmap IRES end-to-end run is prepared for Cannon; 13 folds are complete and cached.

- The cluster bundle rebases cache paths automatically and supports bounded parallel fold workers.

- Eddy 2014 `1st` and `2nd` files are equal-sized sorted splits, not separate 16S and 23S datasets.

- Watts 2009 HIV-1 data now has a validated 9,173-nt FASTA and an Rfam-derived structure projection. The top hit is RRE/RF00036 at nucleotides 7255–7591 with E-value `3.6e-118`.

- HIV-1 G6 benchmarking now covers all 11 inclusion-threshold `cmscan` hits (not just RRE), each folded and scored independently against its own Rfam-projected reference. Corpus-guided SHAPE improves F1 on several hits (RRE +0.212, HIV_PBS +0.091) and hurts on others (HIV_FE −0.186, HIV-1_DIS −0.083); 7 of 11 hits already score F1=1.0 under every method. A parallel HIV-only-guided run (chem-hist built from these same 11 hits) mostly hurts performance rather than inflating it, since that histogram is small (554/342 samples) and dominated by RRE.

- R-scape 2.6.4 is now built natively on Cannon: `~/rivaslab2026/rscape_v2.6.4_linux/` is a real x86_64 ELF build (config.log host `x86_64-pc-linux-gnu` on `holy8a24102`), symlinked into `humanrnamap_g6_cluster_bundle/rscape_v2.6.4`. The earlier macOS arm64 build and its redundant source tarball were removed.

- A 2,344-record smartSHAPE macrophage workbook was evaluated and rejected as a benchmark source because no fragment produced an inclusion-qualified Rfam hit.

- The 24-RNA ShapeKnots benchmark now runs end-to-end with supplied CT references and SHAPE tracks.

| ShapeKnots G6 measure | Result |
| --- | ---: |
| Unguided mean F1 | 0.510 |
| SHAPE-guided mean F1 | 0.476 |
| Mean delta F1 | -0.034 |
| Improved / unchanged / worsened | 6 / 3 / 15 |

[[Artifacts/Chemical probing G6/shapebenchmark_g6_f1_results|Full 24-RNA results and method notes]]

- A second, independently-parsed ShapeKnots run (`prepare_shapeknots.py`, reading sequences/CT/reactivities directly from the Hajdin et al. 2013 primary source instead of the pre-extracted triples) reproduces the first run's result to within ~0.02 F1 per RNA, cross-validating both pipelines.

| ShapeKnots G6 measure (primary-source replication) | Result |
| --- | ---: |
| Unguided mean F1 | 0.506 |
| SHAPE-guided mean F1 | 0.482 |
| Mean delta F1 | -0.024 |
| Improved / unchanged / worsened | 7 / 2 / 15 |

[[Artifacts/Chemical probing G6/shapeknots_primary_source_g6_f1_results|Primary-source replication: full 24-RNA results and method notes]]

- `scripts/plot_g6.py` and `run_reactivity_g6_pipeline.py` now only support F1 for `--metric`/`--plot-metrics`; sensitivity and PPV plots are no longer generated (they were unused). Pushed to `G6_chemical_probing` as commits `f55d407` and `72c4d73`.

- ViennaRNA RNAfold was benchmarked with and without native Deigan SHAPE pseudoenergies across all 24 ShapeKnots RNAs.

| RNAfold condition | Mean F1 | Improved | Unchanged | Worsened |
| --- | ---: | ---: | ---: | ---: |
| Without SHAPE | 0.533 | — | — | — |
| With SHAPE | 0.642 | 16 | 6 | 2 |

- The matching ShapeKnots analysis is complete (`run_shapeknots_shape_benchmark.py`, 8-way concurrent) across all 24 RNAs, closely matching RNAfold's own SHAPE gain.

| ShapeKnots condition | Mean F1 | Improved | Unchanged | Worsened |
| --- | ---: | ---: | ---: | ---: |
| Without SHAPE | 0.543 | — | — | — |
| With SHAPE | 0.652 | 14 | 6 | 4 |

- `G6_chemical_probing/` is now reorganized so each benchmark lives under its own `datasets/<Name>/` directory (`Deigan_2009`, `HIV-1`, `IRES`, `RMDB`, `viennaRNASHAPE`). The 24-RNA ShapeKnots/RNAfold benchmark set (formerly `data/shapebenchmark/`) is now `datasets/viennaRNASHAPE/`, tracked directly in the repo instead of as a nested clone of `stanti/shapebenchmark`. `prepare_shapeknots.py` and `run_shapeknots_shape_benchmark.py` are pushed to `origin/main`.

- `Deigan_2009`'s 5-record input table has only 3 completed folds; `deigan_16s` and `deigan_23s` were never prepared or folded and need to run on the cluster.

| Deigan_2009 record | Length (nt) | Status |
| --- | ---: | --- |
| deigan_trna | 75 | complete |
| deigan_hcv_domain2 | 95 | complete |
| deigan_p546 | 155 | complete |
| deigan_16s | 1,542 | pending (cluster) |
| deigan_23s | 2,904 | pending (cluster) |

## Active blockers

- Reference structure is unavailable for 66 SHAPE groups and 11 DMS groups, so those groups cannot be included in structure-conditioned evaluation without another reference source.

- Signal-quality filtering removes an additional 32 SHAPE and 16 DMS groups.

- Long IRES RNAs, especially the IRES_Picorna family, remain difficult for both ViennaRNA and G6 CYK. Pseudoknots and sequence lengths of roughly 500-900 nt limit performance and complicate interpretation.

- Eddy 2014 pooled outputs lack positions, and unavailable 16S/23S SHAPE arrays prevent exact reconstruction.

- No separate daily-note files were found in the workspace, so recent day-by-day progress cannot be linked independently.

- `run_cmscan_batch()` chooses the most frequent family rather than the best-first significant hit; repeated fragments can override the true top hit.

- The HIV-only SHAPE histogram (built from the same 11 hits it would guide folding for) is too small and RRE-dominated to trust as fold guidance; treat its results as an upper-bound/circularity check only, not a fair comparison.

- Several in vivo SHAPE-map papers cited by the 2026 review exposed protocols but no reusable datasets.

- The current pooled in-sample SHAPE likelihood model reduces mean F1 on the 24-RNA benchmark.

- This project's `rivaslab2026/data/shapebenchmark/` working directory is not a Git repo and has no lock, so concurrent agent sessions can collide by independently launching the same script into the same output paths -- this happened once (16:17-16:35) and was caught by inspecting `ps aux` for duplicate ShapeKnots processes before it corrupted any file. Check for an already-running job before resuming similar batch work. Note: this directory is now `G6_chemical_probing/datasets/viennaRNASHAPE/` and is git-tracked, which should make concurrent-write collisions easier to detect via `git status`.

- The Deigan_2009 16S/23S cluster run hasn't been submitted yet -- interactive Cannon SSH needs Duo/password auth that can't be completed from an agent session; requires the user to run `sbatch` themselves.

  

## Important decisions

  

- Use only `shape/single_molecule` and `dms/single_molecule` RDAT inputs for the histogram dataset.

- Merge files that differ only by collision suffix before applying structure and signal filters.

- Prefer deposited RDAT structures when present; otherwise use a structure projected from the best Rfam covariance-model alignment.

- Represent residues in query-specific insertion columns as unknown (`?`) and remove consensus pairs whose partner is deleted or falls outside the selected sequence window.

- Keep only wild type for mutate-and-map experiments, average replicate scores, discard rows with zero reads, and trim flanking sequence.

- Require at least 10 positive measurements and a 90th-percentile signal of at least 0.05.

- Preserve Eddy 2014 pooled SHAPE values as nonpositional data rather than incorrectly attaching them to rRNA coordinates.

- Reject `1st = 16S` and `2nd = 23S`: helix-end counts are 432/791 by RNA but 577/576 by file.

| Class | 16S (by RNA) | 23S (by RNA) | `1st` file | `2nd` file |
| --- | --- | --- | --- | --- |
| helix-end (pairing) | 432 | 791 | 577 | 576 |
| unpaired | — | — | 828 | 828 |
| stacked | — | — | 689 | 689 |

  All six class files are numerically sorted and split equally or within one value —
  an equal-sized pooled split, not a per-RNA dataset.
  [[Artifacts/Chemical probing G6/eddy2014_class_file_counts|Full table and reasoning]]

- Interpret low IRES F1 as a property of the difficult target class, not as evidence that the projection/scoring pipeline is invalid, because the same pipeline performs well on shorter control families.

- Summarize Human RNAmap delta F1 with ten baseline-F1 bins. Use violin distributions with full observed ranges and labeled arithmetic means because the 50.8% exact-zero mass makes Tukey outlier classification misleading.

- Select the first significant `cmscan` row for single-sequence Rfam projection, not the most frequently recurring family.

- For genome-scale sequences, `cmalign` each hit's own extracted window against its family CM, not the whole genome -- whole-genome alignment lets repeated elements (e.g. `mir-TAR` at both HIV LTRs) get stitched into one bogus alignment.

- Before scoring a prediction against a multi-bracket-type (WUSS) reference with `easel compstruct`, normalize the reference to plain `()` notation first, matching `compare_g6.py`'s internal handling -- otherwise `easel` rejects it as a "bad trusted structure".

- Score HIV-1 hits independently against two chem-hist sources (RMDB corpus, HIV-only) in separate output trees rather than merging, so the circular HIV-only numbers can't be mistaken for a fair comparison.

- For small hit counts (N=11), skip the scatter/family-scatter/binned-delta plot bundle in favor of one per-hit bar chart ordered by ViennaRNA MFE performance -- the original plots assume hundreds of hits per family.

- Run the long Human RNAmap IRES fold stage on Cannon with four workers, 64 GB, and per-record cache reuse.

- Transfer R-scape source rather than macOS binaries; rebuild natively on the cluster.

- Keep the Human RNAmap plotting workflow consolidated in `scripts/plot_g6.py` rather than separate plotting entry points.

- Use `romanagle/G6_chemical_probing` as the canonical Git remote for the portable workflow.

- Fail fast on bad records instead of soft-excluding them: `run_reactivity_g6_pipeline.py` and `plot_g6.py` used to catch per-record errors (missing input, invalid data, no Rfam hit, <10 positive measurements, failed fold, missing/invalid comparison CSV), mark the record excluded/fold_failed, and continue, writing a `preparation_exclusions.csv`/`<prefix>_exclusions.csv` report. That pattern is removed: any such error now raises immediately and stops the whole run. Also removed the `fixtures/` directory, which only existed to document the exclusions-CSV format.

- Do not pursue the smartSHAPE macrophage workbook or submit it to Cannon; its short fragments provide no usable Rfam consensus references.

- Use supplied ShapeKnots CT structures as independent labels and preserve crossing pairs with multiple dot-bracket alphabets.

- Repair one-sided CT pairs only when the stated partner is otherwise unpaired; four annotations required this repair.

- Restrict `plot_g6.py`/`run_reactivity_g6_pipeline.py` to F1-only plotting (drop sensitivity/PPV graphs) and validate `--plot-metrics` at startup so an unsupported value fails immediately instead of after prepare/histogram/fold finish.

- Reuse the pseudoknot-aware CT-to-dot-bracket conversion from `prepare_shapebenchmark.py` rather than duplicating it when adding `prepare_shapeknots.py` for the primary-source ShapeKnots data.

- Track `datasets/viennaRNASHAPE/` directly in `G6_chemical_probing` rather than as a nested git clone, since its origin (`stanti/shapebenchmark`) is a fork the user doesn't control.

- Use `G6_chemical_probing`'s existing GitHub remote (`romanagle/G6_chemical_probing`) to sync repo organization for the cluster run, rather than building a one-off transfer tarball like the prior Human RNAmap IRES cluster run.

- Run this pipeline with `G6_chemical_probing/.venv/bin/python3`, not the system `python3`: the system interpreter's installed `tornado` package shadows the pipeline's local `tornado.tornado_fold` namespace module and breaks `compare_g6.py`.

- Compare RNAfold with versus without SHAPE using ViennaRNA's Deigan implementation and defaults `m=1.8`, `b=-0.6`.

- Compare ShapeKnots with versus without SHAPE independently, preserving pseudoknot-aware prediction in both conditions.

- Flatten ShapeKnots' predicted pseudoknot layer (the `<>` bracket type from `ct2dot`) to unpaired before scoring with `easel compstruct`: `easel` rejects any structure using a second bracket type as a "bad test structure", and none of the 24 reference structures in this dataset encode crossing pairs either (verified across all 24), so flattening keeps ShapeKnots on the same pseudoknot-blind basis as every other method in this benchmark.

- Fold ShapeKnots records up to 8-way concurrently (longest sequences scheduled first), not sequentially: single-threaded ShapeKnots costs ~8 minutes on the largest (530-nt) RNA, versus RNAfold's near-instant MFE fold.

| smartSHAPE Rfam audit measure | Count |
| --- | ---: |
| Workbook records | 2,344 |
| Unique sequences | 1,803 |
| Reported weak matches | 144 |
| Hits with E-value ≤ 0.001 | 0 |
| Usable Rfam consensus structures | 0 |

| Human RNAmap IRES run state | Records |
| --- | ---: |
| Prepared | 71 |
| Complete and cached | 13 |
| Remaining | 58 |


## Next milestone

Compare SHAPE effects across RNAfold, ShapeKnots, and G6 now that all three are complete on the 24-RNA benchmark.

  
Complete and validate the 71-record Human RNAmap IRES run on Cannon, then perform the focused IRES_Picorna error analysis.

Run the candidate prediction/scoring methods on the exported, directly scoreable RMDB dataset; report paired/unpaired and structure-prediction performance separately for SHAPE and DMS; and stratify results by reference source (deposited versus Rfam-derived), RNA family, and sequence length. Document how pseudoknots are handled by each method and metric.

For the Human RNAmap comparison, test whether the apparent baseline-dependent DMS effect persists after stratifying by RNA family and sequence length.

For Eddy 2014, obtain the original SHAPE arrays and reconstruct the six pooled class splits exactly.


## References

- [Rfam structure-projection workflow and validation](cmalign_projection_workflow.md)

- [RDAT processing statistics and filtering decisions](RDAT%20processing%20stats%20%28SHAPE%20DMS%20pipeline%29%20e51c261c398b4c4bb845c11124e0220a.md)

- [Histogram export documentation](../exports/histogram_data/README.md)

- [Scoring-data export documentation](../exports/scoring_data/README.md)

- [G6 cross-check documentation](../scripts/tornado_crosscheck/README.md)

- [Human RNAmap binned delta statistics](../Artifacts/Chemical%20probing%20G6/humanrnamap_g6_f1_binned_delta.csv)

- [[Artifacts/Chemical probing G6/eddy2014_class_file_counts|Eddy 2014 class-file counts (1st/2nd vs. 16S/23S)]]

- [[Artifacts/Chemical probing G6/HUMANRNAMAP_G6_PLOTTING_WORKFLOW|Human RNAmap G6 plotting workflow]]

- [HIV-1 SHAPE histogram builder](../../rivaslab2026/scripts/histograms/build_hiv_shape_histogram.py)

- [HIV-1 G6 fold/score pipeline (11 hits, corpus + HIV-only chem-hist)](../../rivaslab2026/scripts/pipelines/run_hiv_g6_pipeline.py)

- [HIV-1 per-hit delta-by-MFE bar chart](../../rivaslab2026/scripts/plotting/plot_hiv_g6_delta_by_mfe.py)

- [Human RNAmap IRES cluster handoff](../../rivaslab2026/humanrnamap_g6_workflow/CLUSTER.md)

- [Human RNAmap G6 workflow repository](https://github.com/romanagle/G6_chemical_probing)

- [[Artifacts/Chemical probing G6/shapebenchmark_g6_f1_results|ShapeKnots 24-RNA G6 benchmark]]


## Important Links

| Link | Description | Date added |
| --- | --- | --- |
| [github.com/stanti/shapebenchmark](https://github.com/stanti/shapebenchmark) | Source repo for the 24-RNA SHAPE benchmark set (Deigan/Zarringhalam/Washietl methods); originally cloned into `data/shapebenchmark/`, now tracked directly at `G6_chemical_probing/datasets/viennaRNASHAPE/`. Underlies the ViennaRNA `RNAfold --shape` paper (Lorenz et al.) benchmark. | 2026-08-12 |


## Related daily notes


No files explicitly identified as daily notes are currently present in the workspace. The two project notes below contain the closest dated or progress-oriented record:

- June 16, 2026: [RDAT processing statistics](RDAT%20processing%20stats%20%28SHAPE%20DMS%20pipeline%29%20e51c261c398b4c4bb845c11124e0220a.md)

- Undated: [Rfam projection workflow and validation results](cmalign_projection_workflow.md)


## Literature Tracker

| Title | Author | Link | Date researched | Summary |
| --- | --- | --- | --- | --- |
| Architecture and secondary structure of an entire HIV-1 RNA genome | Watts et al. | [nature.com/articles/nature08237](https://www.nature.com/articles/nature08237) | 2026-08-02 | Source of the HIV-1 SHAPE reactivity data converted into a validated 9,173-nt FASTA and used as the basis for HIV-1 Rfam projection and G6 benchmarking. |
| Nano-DMS-MaP allows isoform-specific RNA structure determination | Smyth et al. | [nature.com/articles/s41592-023-01862-7](https://www.nature.com/articles/s41592-023-01862-7#Sec24) | 2026-08-05 | Nature Methods paper on long-read, isoform-specific DMS-MaP probing; relevant to the DMS reactivity side of the SHAPE/DMS benchmarking corpus. |
| Evaluating the accuracy of SHAPE-directed RNA secondary structure predictions | Heitsch et al. | [academic.oup.com/nar/article/41/5/2807/2414458](https://academic.oup.com/nar/article/41/5/2807/2414458) | 2026-08-05 | NAR 2013 paper using a stochastic ternary model (unpaired/helix-end/stacked) fit to SHAPE data on E. coli 16S/23S rRNA to quantify how much SHAPE-directed soft constraints actually improve NNTM secondary-structure prediction accuracy; directly relevant to benchmarking how much SHAPE/DMS reactivity improves ViennaRNA and G6 CYK predictions in this project. |
| Computational Analysis of Conserved RNA Secondary Structure in Transcriptomes and Genomes | Eddy | [annualreviews.org/content/journals/10.1146/annurev-biophys-051013-022950](https://www.annualreviews.org/content/journals/10.1146/annurev-biophys-051013-022950) | 2026-08-06 | Annual Review of Biophysics 2014 review by Sean Eddy unifying approaches for incorporating chemical/enzymatic structure-probing data into computational RNA secondary-structure prediction under a probabilistic inference framework; this is the source of the "Eddy 2014" 16S/23S class-file dataset already used for the 1st/2nd pooled-split analysis in this project. |
| ViennaRNA Package installation guide | ViennaRNA Package authors | [viennarna.readthedocs.io](https://viennarna.readthedocs.io/en/latest/install.html) | 2026-08-07 | Used to select the supported Bioconda installation route for the Cannon environment. |
| Recent Advances in Chemical Probing Strategies for RNA Structure Determination In Vivo | Yarshova, Zhao & Kwok | [doi.org/10.1002/chem.202503291](https://doi.org/10.1002/chem.202503291) | 2026-08-11 | Chemistry – A European Journal (2026) review of in vivo chemical probing strategies for RNA structure determination; full text is paywalled, so summary is based on title/metadata only. |
| An ultra low-input method for global RNA structure probing uncovers Regnase-1-mediated regulation in macrophages | Piao et al. | [sciencedirect.com/science/article/pii/S2667325821003113](https://www.sciencedirect.com/science/article/pii/S2667325821003113) | 2026-08-11 | Its supplementary smartSHAPE workbook was rejected because 2,344 short fragments yielded no pipeline-qualified Rfam consensus references. |
| SHAPE directed RNA folding | Lorenz et al. | [doi.org/10.1093/bioinformatics/btv523](https://doi.org/10.1093/bioinformatics/btv523) | 2026-08-11 | ViennaRNA 2.2+ `RNAfold --shape` paper; its 24-RNA benchmark set (Deigan/Zarringhalam/Washietl methods compared) is the source of the `stanti/shapebenchmark` repo cloned into `data/shapebenchmark/`. |
| Accurate SHAPE-directed RNA Secondary Structure Modeling, Including Pseudoknots | Hajdin et al. | [doi.org/10.1073/pnas.1219988110](https://doi.org/10.1073/pnas.1219988110) | 2026-08-11 | ShapeKnots paper; its supplementary `.sequence`/`.ct`/xlsx dataset (`3rdparty/shapeknots/`) is the primary source independently re-parsed by the new `prepare_shapeknots.py` converter. |
