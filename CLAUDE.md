# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

This repository contains SQUID/MAVE-NN surrogate-model benchmarking
experiments. The active `mRNA_RBP/` pipeline is **oracle-agnostic and
multi-oracle**: the same set of stages (library generation, surrogate
training, eval, plotting) runs against any registered "oracle" — a
black-box scoring function over one-hot mRNA sequences. Registered oracles
(see `mRNA_RBP/src/oracles.py`):

| Oracle name | What it is |
|---|---|
| `mrna_ground_truth` (`mrna`) | Synthetic, manually specified GT: noWT additive effects + stem-pair coupling + sigmoid nonlinearity (see below) |
| `mrna_negative_control` (`negative_control`, `negctrl`) | Motif-only counterpart to `mrna_ground_truth`: same sequence backbone and declared motif positions (privileged additive region, HuR-like single-motif mechanism), near-neutral background elsewhere, **no pairwise interactions at all** (`stem_pairs=[]`), same sigmoid form/constants — see `MrnaNegativeControlGroundTruth` below. |
| `residualbind_ensemble` (`msi1`) | Real ResidualBind ensemble oracle for MSI1, fixed WT |
| `vts1_residualbind` (`vts1`) | Real ResidualBind ensemble oracle for VTS1/RNCMPT00111, high/low natural-probe WT variants |
| `hur_residualbind` (`hur`) | Real ResidualBind ensemble oracle for HuR/ELAVL1, high/low WT variants |
| `qki_residualbind` (`qki`) | Real ResidualBind ensemble oracle for QKI, high/low WT variants |
| `surrogate_varied_mutrate` | Generic "deep squid" layer-2 surrogate (a MAVE-NN model trained on 200K sequences) used *as* an oracle itself |
| `twister_ribozyme` (`twister`) | Twister ribozyme — no live black-box oracle exists; the "deep squid" surrogate trained directly on real Kobori & Yokobayashi (2016) RA measurements is the only scorable oracle |

MSI1 and VTS1 are two **distinct, independently-trained** ResidualBind
ensembles — never conflate them. `oracles.normalize_oracle_name` maps
user-facing aliases to canonical names; `oracles.build_oracle` constructs
the actual oracle object; `oracles.sequence_config_for_oracle` resolves
`(seq, stem_pairs, motif_positions)` per oracle (and per `wt_activity` for
VTS1/HuR/QKI); `oracles.default_output_base` picks the output root
(`outputs/` for the synthetic GT, `outputs_twister_ribozyme/` for Twister,
`outputs_<oracle>[_<wt_activity>]/` for everything else);
`oracles.oracle_gt_keys` / `oracles.primary_gt_key` give the score-dict
key(s) each oracle produces and which one is canonical.

"Deep squid" = a MAVE-NN surrogate trained on a 200K-sequence
varied-mutation-rate library and then reloaded as a standalone stand-in
oracle (`SurrogateMAVENNOracle`), distilled from a real black-box
ResidualBind ensemble. Each real oracle gets its own independently-trained
deep-squid surrogate, keyed `deepsquid_<short_name>` — never a shared
implicit default.

All active project code lives under `mRNA_RBP/src/` and `mRNA_RBP/scripts/`.
Old exploratory scripts live in `archive/scripts_old/`. The repository-level
`scripts/reproduce_gt_pipeline.py` remains a separate reproducibility entry point.

## Python environment

Two conda envs are both genuinely required and unavoidable on this machine
(the `squid` env's torch install fails to import — `libcublas.so.11:
undefined symbol` — so live oracle scoring can never happen there):

```bash
# toehold_gpu: working torch+CUDA. Used to build/score real ResidualBind
# ensemble oracles (any stage that scores raw sequences through a live oracle).
TORCH_PY=/home/nagle/miniconda3/envs/toehold_gpu/bin/python3

# squid: working tensorflow/mavenn, python3.7. Used to train GE/MAVE-NN
# surrogates and for all plotting. Cannot score real ResidualBind oracles
# live — for those it reads scores cached by the toehold_gpu stage instead.
SQUID_PY=/home/nagle/miniconda3/envs/squid/bin/python3.7
export LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
```

All scripts use `sys.path.insert` manually — there is no install step.

## Running the pipeline

Run from `/home/nagle/final_version`.

```bash
# Full regeneration for one oracle (resumable — skips already-generated files;
# replaces the old, archived master_job.sh, which only handled the synthetic GT).
bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle mrna
bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle hur  --wt_activity high [--n_instances 1]
bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle vts1 --wt_activity low

# Single command that also trains/plots/assembles a curated collection
# (matching the "Synthetic GT" layout: figures/<category>/... +
# libraries_used_for_figures/ + manifest.json + README.md) for any oracle,
# running both high/low WT contexts automatically where applicable:
python mRNA_RBP/scripts/pipeline/build_ground_truth_collection.py --oracle vts1 --n_instances 1
```

`run_gt_pipeline.sh` stages (each individually resumable):

| Stage | Env | Script |
|---|---|---|
| 1 | toehold_gpu | `generate_libraries.py` — mut05/10/25 pools, SSM, eval libs |
| 2 | toehold_gpu | `generate_varied_mutrate_library.py` — 200K deep-squid pool |
| 3 | squid | `lib_size_spearman.py` — surrogate coefs + rho cache |
| 4 | squid | `train_surrogate_varied_mutrate.py` — deep-squid surrogate model |
| 5 | squid | `cross_mutrate_eval.py` — cross-mutation-rate generalization JSON |
| 6 | squid | `generate_scatter_by_mutcount_predictions.py` — scatter-by-mutcount artifact |

Step-by-step / individual scripts also take `--oracle NAME [--wt_activity high|low] [--n_instances N]`:

```bash
python mRNA_RBP/scripts/pipeline/generate_libraries.py --oracle mrna --n_instances 1
python mRNA_RBP/scripts/pipeline/lib_size_spearman.py  --oracle mrna --n_instances 1

# Individual plot scripts (after lib_size_spearman.py has run)
python mRNA_RBP/scripts/figures/core/plot_library_distributions.py
python mRNA_RBP/scripts/figures/core/plot_coefficients.py
python mRNA_RBP/scripts/figures/core/plot_scatter_by_mutcount.py
python mRNA_RBP/scripts/figures/core/plot_model_comparison_bar.py
```

`scripts/reproduce_gt_pipeline.py` is a separate, standalone reproducibility
wrapper (not part of `run_gt_pipeline.sh`): given a ground-truth preset
(`synthetic_gt`, `residualbind_vts1`, ...) it records the GT inputs,
optionally reruns supported plot scripts, and stages the resulting figures
into one directory with a manifest — a single reproducible entry point over
the historically-scattered fixed plot output locations.

## Module map (`mRNA_RBP/`)

**Core modules and pipeline commands** (`src/` modules are imported; `scripts/pipeline/` files are executed):

| File | Role |
|------|------|
| `gt_init.py` | `MrnaRbpGroundTruth` (synthetic ground-truth oracle) + `MrnaNegativeControlGroundTruth` (motif-only negative control) |
| `oracles.py` | Oracle registry: `normalize_oracle_name`, `build_oracle`, `sequence_config_for_oracle`, `default_output_base`, `oracle_gt_keys`/`primary_gt_key`, `ResidualBindEnsembleOracle`, `SurrogateMAVENNOracle` |
| `sequence_configs.py` | Per-oracle `(seq, stem_pairs, motif_positions)` definitions (MSI1 fixed; VTS1/HuR/QKI/Twister high/low or fixed configs) + `generate_type3_nuc_ids` |
| `generate_libraries.py` | Generates 2M-sequence pools, subsamples to lib sizes, builds eval libs (pairwise, type3, activity-balanced) — oracle-agnostic |
| `generate_varied_mutrate_library.py` | Layer-2 "deep squid" pool generator: one pool per real oracle spanning mutation counts 3–10 |
| `lib_size_spearman.py` | Trains all 4 surrogate configs, records Spearman ρ, saves coefs |
| `train_surrogate.py` | Standalone surrogate trainer for the nonlinear additive+pairwise config |
| `train_surrogate_varied_mutrate.py` | Trains the "deep squid" layer-2 surrogate on the varied-mutrate pool, labeled by real oracle scores |
| `train_twister_deepsquid.py` | "Deep squid" trainer for Twister — trained directly on real RA measurements (no live black-box oracle exists) |
| `parse_twister_data.py` | Parses the Kobori & Yokobayashi (2016) Twister ribozyme RA dataset into the pipeline's native pool format |
| `evaluate.py` | Structured eval datasets: additive (non-stem k-mutants), pairwise (stem 4×4 grids), SSM |
| `viz.py` | Coefficient plots mirroring the demo notebook style |
| `cross_mutrate_eval.py` | Cross-mut-rate Spearman eval (reconstructs phi from saved coefs, no retraining) |
| `build_ground_truth_collection.py` | One-command: generate libraries, train surrogates, produce figures, assemble a curated `ground_truth_collections/` entry for any oracle |
| `coef_metrics.py` | `compute_coefficient_similarity` — mean cosine similarity between GT/oracle-view and surrogate additive (alpha, per-position) and pairwise (beta, per-stem-pair) weights; used by `scripts/figures/core/plot_coefficients.py` and `scripts/maintenance/compute_coefficient_similarity.py` |
| `compute_coefficient_similarity.py` | Backfills `coefficient_similarity.json` + a README/manifest section for already-built `ground_truth_collections/` entries, purely from cached artifacts (no retraining, no PNGs touched) |
| `generate_model_comparison_score_cache.py` | Builds the score-cache npz for `scripts/figures/core/bar_surrogate_models_type3.py`, purely from cached artifacts (no live oracle) |
| `generate_random_region_score_cache.py` | Builds the natural-random-library score cache for `scripts/figures/core/plot_residualbind_vts1_rand_region_distributions.py`, from existing mut05/10/25 libs (no live oracle) |
| `generate_scatter_by_mutcount_predictions.py` | Oracle-agnostic scatter-by-mutcount prediction artifact; generalizes the VTS1-only script below |
| `generate_residualbind_vts1_scatter_predictions.py` | Older VTS1-hardcoded predecessor of the above; reads from a curated collection copy, not the live pipeline |
| `master_job.sh` | **Removed/archived** — replaced by `run_gt_pipeline.sh` |
| `run_gt_pipeline.sh` | Orchestrates the full 6-stage pipeline for one oracle across both conda envs |

Shared utilities:
- `mRNA_RBP/src/ground_truth.py` — `additive_affinity_noWT`, `pairwise_potts_energy`, `apply_global_nonlin`, `init_sigmoid_nonlin`, `soft_threshold`, `uniformize_by_histogram`
- `mRNA_RBP/src/seq_utils.py` — `rna_to_one_hot`

**`mRNA_RBP/scripts/figures/core/`** — 12 standalone plot scripts + `provenance.py` (shared helper,
stamps figures with a `Library: fresh|cached|unknown | made YYYY-MM-DD`
label, not a plot itself): `plot_coefficients.py`, `plot_cross_mutrate.py`,
`plot_library_distributions.py`, `plot_model_comparison_bar.py`,
`plot_scatter_by_mutcount.py`, `plot_rho_vs_libsize_type3.py`,
`bar_surrogate_models_type3.py`, `plot_synthetic_rand_region_distributions.py`,
`plot_residualbind_vts1_collection_figures.py`,
`plot_residualbind_vts1_rand_region_distributions.py`,
`plot_residualbind_vts1_result_figures.py`,
`plot_residualbind_vts1_scatter_by_mutcount.py`. All write to
`outputs/notebook_plots/` (or into a collection's `figures/` dir when run as
part of `build_ground_truth_collection.py`).

**`mRNA_RBP/scripts/experiments/`** — exploratory experiment commands. Result
notes remain under `mRNA_RBP/experiments/`.

## Ground truth: `MrnaRbpGroundTruth`

This section describes the synthetic `mrna_ground_truth` oracle
specifically (also the canonical MSI1-style sequence/structure, reused by
the real MSI1 oracle — see `sequence_configs.MSI1_SEQ` etc.). Other real
oracles (VTS1/HuR/QKI) use their own natural-probe sequences via
`sequence_config_for_oracle`.

The canonical sequence and structure (defined in `generate_libraries.py`,
mirrored in `sequence_configs.py` as `MSI1_SEQ`/`MSI1_STEM_PAIRS`/`MSI1_MOTIF_POSITIONS`):

```python
SEQ          = 'AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA'  # L=41
STEM_PAIRS   = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
MOTIF_POS    = [17, 18, 19, 20, 21]   # UGCAUG Nova motif in the hairpin loop
```

`MrnaRbpGroundTruth(seq, stem_pairs, motif_positions, seed=k, stem_sigma=3.0)` — instance `k` uses `seed=k`, giving 10 independently sampled weight sets with identical structure.

Key design choices:
- **noWT parameterization**: WT nucleotide at every position has weight 0 by construction; all mutation scores are ≤ 0, WT = 0 exactly
- **Additive weights at non-stem positions only**: stem positions (8–15, 23–30) have `W_mut = 0` — zero additive contribution. Non-stem positions get additive weights: motif positions (17–21) drawn from `N(0, motif_sigma=3.0)`, all other non-stem positions from `N(0, bg_sigma=0.5)`.
- **Pairwise coupling at stem positions only**: non-stem J entries are all zero. Each stem pair (i,j) gets a 4×4 block drawn from `−|N(0, stem_sigma=3.0)|`, then zeroed at: (a) the WT row, (b) the WT column, (c) all WC-compatible pairs (AU, CG, GC, UA in ACGU order). Remaining non-WT non-WC entries are ≤ 0 — compensatory mutations score 0, non-compensatory mutations score negatively.
- **Nonlinearity**: `y = a·σ(b·latent + c) + d − wt_raw` where `wt_raw = a·σ(c)+d`, anchoring WT to exactly 0

Real ResidualBind oracles (`ResidualBindEnsembleOracle`) and deep-squid
surrogates (`SurrogateMAVENNOracle`) follow the same WT-anchored convention
(`score = raw_prediction(x) − raw_prediction(WT)`, so WT = 0 exactly),
even though they have no explicit additive/pairwise parameterization.

### Scoring interface

```python
gt.score_all(x)   # (N,L,4) → dict with 4 keys, all ≤ 0, WT = 0     (mrna_ground_truth)
gt(x)             # (N,L,4) → (N,) nonlin_additive_pairwise scores only
gt.alpha          # (L,4) additive weight matrix
gt.beta           # {(i,j): (4,4)} pairwise blocks for declared stem pairs
gt.edges          # (E,2) stem pair indices
```

Real/surrogate oracles implement the same `wt_one_hot()` / `score_all(x)` /
`__call__(x)` interface but `score_all` returns a single-key dict (see
`oracles.oracle_gt_keys`), not the 4-key synthetic-GT dict.

### Negative control: `MrnaNegativeControlGroundTruth`

Motif-only counterpart to `MrnaRbpGroundTruth` (oracle name
`mrna_negative_control`), also in `gt_init.py`. Same `MSI1_SEQ` backbone,
same declared motif positions (`MSI1_MOTIF_POSITIONS`, shared with the
structured GT), same 4-key `score_all` interface and WT-anchored
`≤0`/`WT=0` convention, same sigmoid form and constants (`a=1, b=1, c=1.5,
d=0`) — but:
- **Motif-only additive, near-neutral background**: motif positions draw
  from `-HalfNormal(0, motif_sigma=3.0)` — the same scale as the structured
  GT's motif — while every other position draws from a much weaker
  `-HalfNormal(0, bg_sigma=0.10)`. Both use the same L1(0.03) soft
  threshold. This is designed to mirror a real single-motif mechanism (e.g.
  HuR) rather than a null/unstructured control: the contrast with the
  structured GT is the *absence of stem-pair coupling*, not the absence of
  any privileged region.
- **No pairwise interactions at all**: `oracles.sequence_config_for_oracle`
  fixes `stem_pairs=[]` for this oracle (unlike the earlier design,
  `motif_positions` *is* declared at the config level — shared with
  `mrna_ground_truth`). `gt.edges` is always an empty `(0, 2)` array and
  `gt.beta` is always `{}` — nothing pairwise is sampled, per-instance or
  otherwise. An earlier design instead sampled a sparse, per-instance-random
  pairwise topology via `init_wc_pairwise_sparse`/`p_edge`/`edge_seed`; those
  helpers remain in `gt_init.py` for `MrnaRbpGroundTruth`'s own use but are
  no longer called by `MrnaNegativeControlGroundTruth`. Because there are no
  edges, pairwise-dependent artifacts (`pairwise_lib.npz` contents, pairwise
  coefficient cosine similarity) are trivially empty for this oracle — that
  is expected, not a bug.

Code that needs an instance's edges (`generate_pairwise_lib`, `evaluate.py`,
`train_surrogate.py`, `viz.py`) reads `gt.stem_pairs` directly, never the
`sequence_config_for_oracle` value — for this oracle that is always `[]`, so
no special-casing is needed. `plots/plot_coefficients.py` and
`compute_coefficient_similarity.py` read edges back out of the cached
`gt_params.npz["edges"]` for any non-`mrna_ground_truth` oracle; for this
oracle that array is always empty.

## Output structure

`src.oracles.default_output_base` picks the working root per oracle: legacy
synthetic GT remains under `outputs/`, while biological runs use
`runs/<oracle-family>/<target>/<wt_activity>/`. High/low WT variants never
share or clobber the same `instance_00` tree. The working layout is:

```
mRNA_RBP/runs/<oracle-family>/<target>/<wt_activity>/
  instance_00/
    gt_params.npz            # alpha, edges, W_mut, mut_map, J, seed (synthetic GT only)
    ssm.npz                  # 3L single-site mutant scores
    pairwise_lib.npz         # 16 × n_stem_pairs combos (all 4×4 for each stem)
    type1_lib{N}.npz         # mixed-rate eval lib, target_n = 20% of lib size N
    activity_balanced.npz    # canonical uniformized eval lib (see below); type2.npz
                              # is a read-only legacy alias for older runs that
                              # never wrote activity_balanced.npz — new code never
                              # creates type2.npz
    type3.npz                # exhaustive single + double mutants + a few random triples
    varied_mutrate/
      pool.npz                # deep-squid pool: mutation counts 3-10, scored by the live oracle
    mut05/
      pool_2M.npz             # 2M sequences at 5% mut rate + GT/oracle scores
      lib_200.npz  lib_2000.npz  lib_20000.npz
    mut10/ ...
    mut25/ ...
  surrogate_coefs/            # npz per (instance, mut_rate, lib_size, surrogate_config)
  lib_size_spearman_results.json  # Spearman ρ cache (resumable)
  notebook_plots/             # figures written by ad hoc plot-script runs
  ground_truth_collections/   # curated, FROZEN figure/artifact snapshots — see below
```

`outputs/ground_truth_collections/<name>/` (e.g. `Synthetic GT`,
`ResidualBind oracle VTS1`, `ResidualBind oracle MSI1`, `deepSQUID VTS1`,
`deepSQUID MSI1`, `deepSQUID Twister`) holds **curated, frozen** collections
assembled by `build_ground_truth_collection.py`, separate from the live
`outputs[_<oracle>]/` working tree: `figures/<category>/...` for the final
plots, `libraries_used_for_figures/` for the exact libraries/coefs/results
JSON/oracle weights that produced them, `manifest.json` listing every copied
file, and `README.md` summarizing figure/artifact counts and (for
oracles with a high/low WT variant) which sequence each figure used. A
top-level `ground_truth_collections/README.md` indexes all collections.
Figures generated by these plot scripts are stamped (via `plots/provenance.py`)
with a small `Library: fresh|cached|unknown | made YYYY-MM-DD` label.

Each collection also carries `coefficient_similarity.json` (and a
"Coefficient similarity" README section + `manifest.json` key): mean cosine
similarity between the GT/oracle-view and surrogate additive/pairwise
weights (`coef_metrics.compute_coefficient_similarity`), at instance 0 /
mut_rate 10% / lib_size 20000. Written going forward by
`plots/plot_coefficients.py` as part of `build_ground_truth_collection.py`;
backfilled for existing collections by `compute_coefficient_similarity.py`.

## Workspace hygiene

This repo has repeatedly accumulated clutter (duplicate top-level docs,
near-duplicate scripts, multiple abandoned figure versions). Applies to any
agent working in this repo (Claude Code, Codex, etc.):

- **Don't create a new script if an existing one already does this.** Before
  adding a script, check `mRNA_RBP/scripts/pipeline/`, `scripts/figures/core/`,
  `scripts/maintenance/`, and `mRNA_RBP/src/` for something that already
  covers it, and extend/parameterize that instead of adding a near-duplicate
  (e.g. `oracles.py` is oracle-agnostic by design specifically so per-oracle
  scripts never need to be forked). One-off exploratory scripts belong in
  `scripts/experiments/` (with result notes in `experiments/`, per the module
  map above) — not scattered at the repo root or copied into a new location
  per experiment. Delete a one-off once its result is captured; don't leave
  it in the tree as dead weight.
- **No duplicate figure versions.** One canonical file per figure: regenerate
  and overwrite it in place rather than saving `_v2`/`_final`/`_new`-suffixed
  siblings, and don't emit the same figure in multiple formats (png + pdf +
  svg) unless a specific downstream consumer actually needs the second
  format. Variant iteration (`variant_a`, `variant_b`, ...) is only for
  active exploration inside `mRNA_RBP/prototypes/<name>/outputs/` — once a
  variant is accepted into `NARRATIVE.md`, the rejected variants and any
  intermediate copies should be cleaned up, not left sitting alongside it.
- **Put each figure where it's supposed to live**, per Output structure
  above: exploratory/prototype figures → `prototypes/<name>/outputs/`;
  ad hoc pipeline plots → `outputs/notebook_plots/`; finalized manuscript
  evidence → `outputs/ground_truth_collections/<name>/figures/<category>/`,
  produced via `build_ground_truth_collection.py`, not hand-copied. Never
  leave a finished figure at the repo root, in the working directory a
  script happened to run from, or in an untracked scratch path.
- Before adding any new top-level file or directory, check whether it
  belongs inside an existing one (`docs/`, `experiments/`, `prototypes/`,
  a specific `outputs/` subtree) rather than sitting loose at `mRNA_RBP/` or
  repo root.

## Surrogate configs

Four MAVE-NN configs trained per condition in `lib_size_spearman.py`:

| Name | gpmap | linearity | noise |
|------|-------|-----------|-------|
| `additive` | additive | linear | Gaussian |
| `additive + pairwise` | pairwise | linear | Gaussian |
| `nonlinear additive` | additive | nonlinear | SkewedT |
| `nonlinear additive + pairwise` | pairwise | nonlinear | Gaussian |

## Eval library types

- **Random** — uniform random subsample of the training pool (same mut rate)
- **Type1** (`type1_lib{N}.npz`) — uniformized across all 3 mut rates, target_n = 20% of training lib size N
- **Activity-balanced** (`activity_balanced.npz`, canonical eval lib) — built by
  `generate_type1_activity_balanced` in `generate_libraries.py`:
  1. Draw unique exact-mutant candidates at 3, 5, 7, and 15 mutations from WT.
  2. `ACTIVITY_BALANCED_CANDIDATE_N` = 200,000 total candidates, split evenly
     across those 4 mutation counts (50K each, `ACTIVITY_BALANCED_MUT_COUNTS`).
  3. Deduplicate globally.
  4. Score with the oracle's primary GT key (`oracles.primary_gt_key`).
  5. Histogram-uniformize in score space: 200 equal-width bins, percentile
     clipping `[1, 99]`, seed `k*10_000 + 600`.
  6. Cap at `ACTIVITY_BALANCED_TARGET_N` = 20,000 sequences (can be smaller
     if bins are sparse).
  `type2.npz` is a **deprecated legacy alias** for this same file — old runs
  that predate the rename read/wrote `type2.npz`; new code only ever
  produces `activity_balanced.npz` (`activity_balanced_path()` falls back to
  `type2.npz` read-only if the canonical file is missing).
- **Type3** (`type3.npz`) — deterministic, instance-independent: exhaustive
  single mutants (3L) + exhaustive double mutants (9·C(L,2)) + 4 random
  triple mutants, `rate_labels` records each row's mutation count.
- **Pairwise** (`pairwise_lib.npz`) — all 16 nuc combos per stem pair, other positions at WT (deterministic)

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues (romanagle/coordinated_combinatorial_mutagenesis_libraries), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
