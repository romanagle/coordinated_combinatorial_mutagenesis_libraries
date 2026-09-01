# Non-destructive cleanup manifest

## Purpose

- Classify the current workspace before moving or deleting anything.
- Preserve all existing user changes.
- Separate manuscript authority from generated evidence and exploratory work.

## Classification labels

- `ACTIVE` — directly supports the repository-current manuscript.
- `SOURCE` — canonical code or configuration.
- `GENERATED` — reproducible outputs or caches; not a navigation source.
- `CANDIDATE` — plausible manuscript material without a settled role.
- `SUPERSEDED` — rejected layout, outdated initialization, or stale narrative state.
- `ARCHIVED` — intentionally excluded from active work.
- `UNKNOWN` — requires user review before any move or deletion.

## Top-level classification

| Path | Class | Current role | Proposed action |
| --- | --- | --- | --- |
| `NARRATIVE.md` | ACTIVE | Repository-current narrative map | Keep at project root. |
| `PROJECT_MAP.md` | ACTIVE | Navigation and authority map | Keep at project root. |
| `src/` | SOURCE | Reusable ground truths, oracle routing, evaluation, sequence, and visualization modules | Keep importable scientific code here. |
| `scripts/` | SOURCE + CANDIDATE | All executable pipeline, figure, maintenance, and exploratory commands | Navigate by the categorized subdirectories. |
| `experiments/` | CANDIDATE evidence | Result notes only; executable experiment code moved to `scripts/experiments/` | Review notes against active or revisit sections. |
| `prototypes/` | GENERATED (transient) | Sparse, transient figure-making workspace only; executable code lives in `scripts/figures/prototypes/` | Keep empty except for figures currently under active iteration; promote or archive on selection. |
| `figures/` | ACTIVE | Flat store of narrative-selected images produced by prototype scripts, one per `NARRATIVE.md` embed | Keep in sync with `NARRATIVE.md`; remove a file only when its embed is cut. |
| `archive/prototypes/` | ARCHIVED | Former `prototypes/<name>/` directories, in full, after their selected figure (if any) was promoted to `figures/` | Review by dated batch; do not use as evidence. |
| `outputs/` | ACTIVE + GENERATED | Canonical collections plus legacy synthetic working files | Navigate through `ground_truth_collections/`; migrate remaining legacy run files separately. |
| `runs/` | GENERATED | Canonical location for active working runs | Keep out of manuscript navigation. |
| `models/` | SOURCE | Canonical fitted models loaded by oracle code | Retain and document promotion after retraining. |
| `archive/inactive_runs/` | ARCHIVED | Recoverable material removed from active navigation | Review by dated batch; do not use as evidence. |

The naming contract and per-directory status are documented in
[`docs/OUTPUT_DIRECTORIES.md`](docs/OUTPUT_DIRECTORIES.md). In short, `outputs/ground_truth_collections/`
is the curated evidence store; generated working state now lives under `runs/`.
| `research/` | UNKNOWN | Research material | Review contents before classification. |
| `data/` | SOURCE | Input data | Retain; document provenance separately. |
| `docs/` | UNKNOWN | Project documentation | Merge useful guidance into `PROJECT_MAP.md` or retain intentionally. |
| `__pycache__/` | GENERATED | Python bytecode | Safe deletion candidate after worktree review. |

## Active selected figure paths

| Manuscript role | Selected path | Class |
| --- | --- | --- |
| Synthetic GT definition | `figures/synthetic_gt_coefficients_only.png` | ACTIVE |
| Synthetic GT vs. negative control comparison | `figures/both_synthetic_gt_additive_pairwise_matrices.png` | ACTIVE |
| Biological structures | `figures/vts1_hur_secondary_structure_placeholder.png` | ACTIVE placeholder |
| Oracle validation | *(cut; stated as text in Methods — see Revisit in `NARRATIVE.md`)* | — |
| Synthetic distributions | `figures/synthetic_gt_activity_distributions.png` | ACTIVE |
| Synthetic recovery | `figures/synthetic_gt_scatter_actual.png` | ACTIVE |
| Model misspecification | `outputs/ground_truth_collections/Synthetic GT/figures/model_comparison/model_comparison_bar_type3.png` | ACTIVE provisional |
| Biological distributions | `figures/simplified_a_10pct_focus.png` | ACTIVE |
| Library-size recovery | `figures/variant_f_biological_pair.png` | ACTIVE provisional |
| Biological recovery/coefficient summary | `figures/hur_vts1_summary_table_placeholder.png` | ACTIVE placeholder |
| Evaluation-library comparison | `figures/variant_b_mutation_order.png` | ACTIVE |
| Mutation-resolved errors | `figures/variant_o_standardized_residual_violins.png` | ACTIVE |
| VTS1 activity/motif coverage (supplement) | `figures/vts1_activity_mutation_map.png` | ACTIVE |

## Prototype directory triage

`prototypes/` was reorganized into a sparse, transient-only workspace (2026-08-31, Batch 13 below):
every directory's selected figure (if it had one) was promoted to `figures/`, and the entire former
directory — selected-figure-bearing or CANDIDATE — was moved intact to `archive/prototypes/<name>/`.
`prototypes/` is currently empty; new work should only ever pass through it on its way to `figures/`
or `archive/prototypes/`, never accumulate there.

## Output directory triage

| Path | Size | Class | Proposed disposition |
| --- | ---: | --- | --- |
| `outputs/` | 998 MB | GENERATED + UNKNOWN | Inventory curated collections separately from raw instances and caches. |
| `runs/synthetic_negative_control/` | 76 MB | GENERATED active | Retain until the redesigned control is replicated and curated. |
| `runs/deepsquid/vts1/high/` | 47 MB | GENERATED active | Retain for active VTS1 results. |
| `runs/deepsquid/hur/high/` | 47 MB | GENERATED active | Retain for active HuR results. |
| `runs/residualbind/vts1/high/` | 17 MB before model extraction | GENERATED validation | Retain required validation/training provenance. |
| `runs/residualbind/hur/high/` | 53 MB before model extraction | GENERATED validation | Retain required validation/training provenance. |
| `models/residualbind/` | 4 WT-specific models | SOURCE | Canonical runtime dependency; retained outside working runs. |
| `archive/inactive_runs/low_wt/` | About 155 MB before model extraction | ARCHIVED | Recoverable low-WT runs; not part of active navigation. |

## Immediate safe cleanup candidates

- Rejected figure variants only after selected outputs and generating scripts are verified.
- `_figure_staging/` directories after confirming selected figures are curated elsewhere.
- Superseded narrative variants after the Google Doc and repository narrative are reconciled.
- Low-WT output trees only after the supplement decision is explicit.

## Completed cleanup batches

### Batch 1 — Python bytecode caches

- Completed: 2026-08-31.
- Removed: 14 `__pycache__/` directories under `mRNA_RBP/`.
- Reduction: approximately 1.2 MB and 14 generated directories.
- Recovery: rerun the relevant Python modules.
- Verification: no `__pycache__/` directories remain under `mRNA_RBP/`.

### Batch 2 — Rejected prototype renderings

- Completed: 2026-08-31.
- Removed: 33 rejected, superseded, or unselected PNG/HTML renderings from seven prototype families.
- Reduction: approximately 4.1 MB and 33 competing presentation variants.
- Retained: all generating scripts, underlying data, and narrative-selected figures.
- Retained pending reconciliation: both biological distribution selections named by competing narrative files.
- Recovery: rerun the retained prototype scripts; tracked historical variants also remain in Git history.
- Verification: every currently selected prototype path remains present.

### Batch 3 — Duplicate archived figure tree

- Completed: 2026-08-31.
- Removed: `archive/finalgraphabrcms/`.
- Retained copy: `archive/outputs/finalgraphabrcms/`.
- Reduction: approximately 395 MB.
- Recovery: copy the retained duplicate back to its former path.
- Verification: both trees contained the same 1,334 relative files with no differing file contents; only empty directory entries differed.

### Batch 4 — Archived W&B runs

- Completed: 2026-08-31.
- Removed: `archive/wandb/`, containing 682 local run directories and two sweep directories.
- Reduction: approximately 454 MB.
- Repository references: none.
- Recovery: remote W&B synchronization or an external backup only; the removed tree was untracked.
- Verification: `archive/wandb/` no longer exists.

### Batch 5 — Unused narratives and prototype directories

- Completed: 2026-08-31.
- Removed narratives: `NARRATIVES.md`, `POINTS.md`, and four unused files under `narrative_variants/`.
- Retained narratives: `NARRATIVE.md` and `narrative_variants/evaluation_bias_figure_inventory.md`.
- Removed prototypes: `mutation_rate_figure/`, `synthetic_gt_three_maps/`, and `vts1_activity_motif_tiling/`.
- Reduction: approximately 2.2 MB, six narrative files, and three complete prototype trees.
- Recovery: unavailable for untracked files except through external backups; historical conclusions remain in the research notebook.
- Verification: all removed prototype trees were absent from both retained narrative documents.

### Batch 6 — Retired notebook and inactive ground-truth collections

- Completed: 2026-08-31.
- Removed: `notebooks/reproduce_gt_pipeline.ipynb` and the resulting empty `notebooks/`; the authoritative replacement is `scripts/reproduce_gt_pipeline.py`.
- Removed: both MSI1 ground-truth collections and `outputs_residualbind_ensemble/`.
- Removed: `ResidualBind oracle VTS1_legacy_backup/`; the active VTS1 collections remain.
- Removed: the unused empty `data/` directory.
- Reduction: approximately 10.2 MB and seven obsolete paths.
- Recovery: tracked historical content remains in Git history; untracked reorganized content requires an external backup.
- Verification: the collection manifest and README now list only Synthetic GT, VTS1, and HuR collections.

### Batch 7 — External repository grouping

- Completed: 2026-08-31.
- Moved: `rbp_benchmark/` to `external/rbp_benchmark/`.
- Classification: independent clean clone of `MasayukiNagai/rbp_benchmark`; not used by the active pipeline.
- Retained: nested Git history and remote configuration.
- Project boundary: `rbp/` remains project-owned code at the workspace root.
- Follow-up completed: `residualbind/`, `squid-nn/`, and `squid-manuscript/` moved under `external/` after active path migration.

### Batch 8 — Active external dependency migration

- Completed: 2026-08-31.
- Moved: `residualbind/`, `squid-nn/`, and `squid-manuscript/` into `external/`.
- Updated: active `mRNA_RBP/` and `rbp/` dependency paths.
- Preserved: nested Git histories and all pre-existing local modifications.
- Verification: Python compilation, dependency imports, nested remotes, and absence of old root paths.

### Batch 9 — Output workspace consolidation

- Completed: 2026-08-31.
- Moved: five active root-level `outputs_*` workspaces into `runs/` by oracle family, target, and WT state.
- Quarantined: four inactive low-WT workspaces under `archive/inactive_runs/low_wt/`; nothing was deleted.
- Promoted: HuR and VTS1 high- and low-WT MAVE-NN models into `models/residualbind/`.
- Updated: centralized output routing, runtime model lookup, active prototype paths, curated provenance paths, CLI help, and workspace documentation.
- Recovery: follow `archive/inactive_runs/low_wt/README.md`; active moves can be reversed using the mapping in Git's rename view or this document's output triage table.
- Verification: oracle imports and routing, all four retained model paths, required active libraries, JSON syntax, absence of live hard-coded old paths, and `git diff --check`.

### Batch 10 — Script and inactive-run naming cleanup

- Completed: 2026-08-31.
- Moved: 13 executable Python commands and `run_gt_pipeline.sh` from the `mRNA_RBP/` root into `mRNA_RBP/scripts/`.
- Retained at root: nine importable scientific modules plus the project navigation documents.
- Renamed: `quarantine/2026-08-31_low_wt_runs/` to the literal `archive/inactive_runs/low_wt/`.
- Updated: Python imports, subprocess commands, shell pipeline commands, documentation, and provenance paths.
- Verification: all 14 Python files parse, standard-environment entry points reach `--help`, project-root calculations remain `mRNA_RBP/`, and the shell pipeline passes `bash -n`.
- Environment note: `lib_size_spearman.py --help` reaches the existing SQUID environment but that environment currently fails while importing SciPy because its `libstdc++` lacks `GLIBCXX_3.4.29`; this is unrelated to the path migration.

### Batch 11 — Complete source/script separation

- Completed: 2026-08-31.
- Moved: eight reusable scientific modules from the project root into `src/`.
- Moved: pipeline commands into `scripts/pipeline/`, maintenance utilities into `scripts/maintenance/`, plot builders into `scripts/figures/core/`, prototype generators into `scripts/figures/prototypes/<figure>/`, and exploratory commands into `scripts/experiments/`.
- Retained: prototype figures, caches, and documentation in `prototypes/<figure>/`; executable code now writes back to those same asset directories.
- Updated: package imports, subprocess commands, shell commands, project-root calculations, prototype output roots, narrative documentation, and workspace maps.
- Result: no Python scripts remain scattered across `prototypes/`, `plots/`, or `experiments/`; the former `plots/` directory was removed.

### Batch 12 — Figure inventory retired into `NARRATIVE.md`

- Completed: 2026-08-31.
- Removed: `narrative_variants/evaluation_bias_figure_inventory.md` (and the now-empty `narrative_variants/` directory), which had drifted stale — it still described the discarded "unstructured negative control" (activity-balanced ρ = 0.379) while `NARRATIVE.md` already carried the motif-only redesign (ρ = 0.527).
- Folded into `NARRATIVE.md`: a new Supplement subsection for the VTS1 activity/motif-coverage figure (previously only listed as "supported" in the inventory, not yet in `NARRATIVE.md`); named SMN1/GFP explicitly in the existing biological-generalization Hole; added "surrogate form" to the raw-Spearman Supplement bullet's axis list.
- Dropped as stale or already covered: the inventory's stale negative-control numbers, its unselected figure choices (already superseded by `NARRATIVE.md`'s own selections), and every "Explicit holes" entry that duplicated an existing `NARRATIVE.md` Hole or belonged in the project task file instead.
- Left out pending user review: the VTS1 residual-sequence-UMAP and prediction-bin-entropy/motif-histogram explorations — the inventory listed these as still-open, but the research notebook already files the UMAP under "archived exploratory outputs," so promoting any of them into `NARRATIVE.md` was left to the user rather than assumed.
- Updated: `PROJECT_MAP.md` (removed the figure-inventory pointer, authority-order entry, and the now-resolved synchronization-gap notes).
- Result: `NARRATIVE.md` is the single manuscript-content document; no more parallel figure inventory to keep in sync.

### Batch 13 — `prototypes/` emptied into `figures/` and `archive/prototypes/`

- Completed: 2026-08-31.
- Created: `figures/` (flat, images only) and `archive/prototypes/`.
- Promoted: the 10 images `NARRATIVE.md` currently embeds from `prototypes/<name>/` moved to `figures/<file>.png`; `NARRATIVE.md`'s 10 corresponding links updated in place. (The former "Oracle validation" figure was cut from `NARRATIVE.md` separately, earlier the same day — its claim is now stated as text in Methods, so it was never part of this batch's promotion set.)
- Archived whole, unmodified: the remaining contents of those same 10 directories (rejected sibling variants, READMEs, npz/csv caches) plus all 10 directories that had no `NARRATIVE.md` reference at all (`hur_three_coefficient_maps/`, `oracle_validation_scorecard/`, `pairwise_saturation_landscape/`, `residual_sequence_umap/`, `saturated_nonlinearity_library/`, `scatter_pair_figure/`, `synthetic_gt_coefficient_maps/`, `vts1_activity_motif_histogram/`, `vts1_prediction_entropy/`, `vts1_three_coefficient_maps/`) — each moved intact to `archive/prototypes/<name>/`, at the user's explicit direction to archive every currently-unreferenced prototype directory, settled-candidate or not.
- Retained: generating code, already living separately at `scripts/figures/prototypes/<name>/` per Batch 11 — untouched, so every archived figure remains regenerable by rerunning its script.
- Result: `prototypes/` is empty. Going forward it exists only for work actively being iterated on; a directory leaves it either by promoting its selected image to `figures/` (rest of directory to `archive/prototypes/`) or, if fully rejected, by moving wholesale to `archive/prototypes/`.
- Recovery: every moved file is untracked or plain-modified in Git (no commits made); `git status`/`git diff` against the pre-batch worktree recovers exact former paths, and each archived directory keeps its generating script runnable in `scripts/figures/prototypes/<name>/` for full regeneration.
- Verification: all 10 `figures/` paths embedded in `NARRATIVE.md` resolve on disk; no `prototypes/` path remains referenced anywhere in `NARRATIVE.md`.

## Blockers before moving or deleting

- The Google Doc has not been compared with repository narratives.
- The worktree already contains extensive modifications and deletions.
- Several generated collections are being reorganized in the uncommitted worktree.
- The redesigned negative control and VTS1 sequence changes are not yet committed in the code repository.
- Active figure provenance is still split between `figures/` (prototype-script output) and `outputs/ground_truth_collections/` (pipeline output); no single evidence store yet covers both.
- `ResidualBind oracle VTS1/manifest.json` contains 116 pre-existing source references to raw/staging artifacts that are no longer present; its curated destination files need a separate provenance reconciliation.

## Proposed cleanup sequence

1. Export the Google Doc to Markdown or place a copy in the workspace.
2. Reconcile it with `NARRATIVE.md`.
3. ~~Select one canonical file for narrative state and one for figure inventory.~~ Done in Batch 12: `NARRATIVE.md` is the single canonical file; the separate figure inventory was retired.
4. Trace every selected figure to its generating script and required input files.
5. Move rejected variants into one clearly named dated archive directory.
6. Consolidate active generated evidence into a small curated-results tree.
7. Remove reproducible caches only after regeneration commands are documented and tested.
8. Commit the existing code/output reorganization before starting physical cleanup.
