# Random-library mutagenesis project map

## Start here

- Manuscript prose is managed externally in Google Docs and is not available in this repository.
- Repository-current narrative: [`NARRATIVE.md`](NARRATIVE.md).
- Figure inventory: [`narrative_variants/evaluation_bias_figure_inventory.md`](narrative_variants/evaluation_bias_figure_inventory.md).
- Cleanup classification: [`CLEANUP_MANIFEST.md`](CLEANUP_MANIFEST.md).
- Output-directory guide: [`docs/OUTPUT_DIRECTORIES.md`](docs/OUTPUT_DIRECTORIES.md).
- Research status and decisions: [`../research-notebook/Projects/random_lib_mutagenesis.md`](../research-notebook/Projects/random_lib_mutagenesis.md).
- Canonical research tasks: [`../research-notebook/Tasks/random_lib_mutagenesis.md`](../research-notebook/Tasks/random_lib_mutagenesis.md).

## Authority order

1. Google Doc — manuscript-current wording and ordering; external and not verified here.
2. `NARRATIVE.md` — repository-current claims, figure selections, holes, and exclusions.
3. `evaluation_bias_figure_inventory.md` — broadest figure index, but scientifically stale in places.

## Known synchronization gaps

- The figure inventory still describes the discarded unstructured negative control with activity-balanced `ρ = 0.379`.
- `NARRATIVE.md` uses the motif-only negative control with activity-balanced `ρ = 0.527`.
- The inventory selects `distribution_pair_figure/variant_a_paired_ridgelines.png`.
- `NARRATIVE.md` selects `distribution_pair_figure/simplified_a_10pct_focus.png`.
- Google Doc agreement cannot be assessed until it is exported or pasted into the workspace.

## Repository-current manuscript shape

### Methods

- Structured Synthetic GT definition.
- VTS1 and HuR biological landscape definitions.
- Random-library training and activity-balanced evaluation.
- deepSQUID validation against ResidualBind.

### Results Part I — Synthetic proof of concept

- Random libraries and activity-balanced libraries cover different activity distributions.
- Random-holdout accuracy can coexist with weak activity-balanced recovery.
- Evaluation outside random-library skew can reveal model misspecification.

### Results Part II — Biological application

- HuR and VTS1 activity distributions provide biological sampling context.
- Library-size comparisons test biological accuracy inflation.
- Matched activity-balanced scatterplots contrast HuR and VTS1 recovery.
- Standardized residual violins localize errors by mutation count.

### Supplement

- VTS1 activity/motif coverage.
- Raw Spearman values.
- Low-WT distributions only if their role becomes clear.

### Revisit

- Mutation-rate transfer.
- Evaluation-library comparisons involving targeted-pairwise variants.
- Full coefficient maps and coefficient-recovery summaries.
- Saturated-library and higher-order probes.
- VTS1 sequence-neighborhood and motif-mechanism explorations.

## Active landscapes

- Structured Synthetic GT.
- Motif-only Synthetic GT negative control.
- high-WT deepSQUID VTS1.
- high-WT deepSQUID HuR.

## Not active in the main argument

- Twister ribozyme.
- ResidualBind result sets beyond deepSQUID validation.
- Low-WT landscapes unless retained for a specific supplement claim.
- SMN1 and GFP until their manuscript roles are explicitly selected.

## Canonical code areas

- Reusable scientific modules: `src/`.
- Oracle definitions and routing: `src/oracles.py`.
- Sequence and structure configurations: `src/sequence_configs.py`.
- Synthetic ground truths: `src/gt_init.py`.
- Pipeline commands: `scripts/pipeline/`.
- Maintenance commands: `scripts/maintenance/`.
- Core and prototype figure generators: `scripts/figures/`.
- Exploratory commands: `scripts/experiments/`.
- Prototype figures and supporting assets: `prototypes/` (no executable code).
- Curated active figures and supporting artifacts: `outputs/ground_truth_collections/`.
- Generated run directories and their status: `docs/OUTPUT_DIRECTORIES.md`; do not browse `outputs_*` for final figures.
- External reference repositories: `../external/`; project-owned oracle training remains in `../rbp/`.

## Working rule

- A figure is manuscript-active only when the repository-current narrative or Google Doc explicitly selects it.
- A prototype without a selected narrative role remains provisional even when its analysis is valid.
- Generated output directories are evidence stores, not navigation entry points.
- Do not delete or consolidate files until the cleanup manifest is reviewed.
