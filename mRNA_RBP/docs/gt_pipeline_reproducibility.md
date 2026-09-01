# Ground-Truth Figure Pipeline Reproducibility

This document records the current figure pipeline and the inputs needed to
reproduce it for a new ground truth. The source-of-truth entry point is:

- Script: `scripts/reproduce_gt_pipeline.py`

The retired cache-first notebook has been removed. Use the script above for all
staging, regeneration, and manifest generation.

The script stages all final figures into one directory and writes a
`pipeline_manifest.json` with the sequence, RNAFold structure, parsed stem
pairs, motif positions, source files, and commands.

The default workflow is cache-first. Named collections can also run explicit
artifact-generation steps before plotting steps. Plotting scripts should read
frozen libraries, JSON results, or saved prediction artifacts; training should
not happen inside plotting.

## SSM Baseline Definition

The saturated single-site mutagenesis (SSM) baseline is an additive model built
from single-mutant effects relative to the selected WT sequence.

For each position `i` and alternate nucleotide `b`, the SSM table stores:

```text
delta[i,b] = score(WT with position i changed to b) - score(WT)
```

For a multi-mutant sequence, the SSM prediction is:

```text
score_hat(sequence) = score(WT) + sum(delta[i, observed_base_i])
```

WT entries in the table are exactly zero. Synthetic GT scores are constructed
with `score(WT) = 0`, so this reduces to a sum of single-mutant scores. Raw
ResidualBind-style oracle scores need the explicit WT subtraction; otherwise the
WT intercept is counted once for every mutated position. New `ssm.npz` caches
therefore include `wt_score_<score_key>` metadata, and all SSM consumers convert
cached single-mutant scores to deltas before predicting.

## Required Inputs

Every run should specify:

- `ground_truth`: one of `synthetic`, `residualbind`, or `deepsquid`.
- `sequence`: RNA sequence in A/C/G/U. DNA `T` is normalized to `U`.
- `structure`: RNAFold dot-bracket string of the same length as the sequence.
- `motif_positions`: optional zero-based motif positions, either comma-separated
  (`6,7,8`) or ranges (`17-21`).

The dot-bracket structure is parsed into zero-based stem-pair indices. For the
current synthetic MSI1-style sequence:

```text
sequence  = AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA
structure = ........((((((((.......))))))))..........
stem pairs = [(8,30),(9,29),(10,28),(11,27),(12,26),(13,25),(14,24),(15,23)]
motif positions = 17-21
```

## Current Figure Set

The pipeline stages these canonical outputs:

| Pipeline step | Current file |
| --- | --- |
| Ground-truth coefficient analysis | `mRNA_RBP/outputs/notebook_plots/coefficients_map/coefficients_oracle_vs_surrogate_msi1.png` |
| Random library | synthetic: `mRNA_RBP/outputs/notebook_plots/rand_lib_dist_synthetic_high_wt.png`, `mRNA_RBP/outputs/notebook_plots/rand_lib_dist_synthetic_low_wt.png`; non-synthetic: integrated cached figures in `mRNA_RBP/outputs/ground_truth_collections/<GT>/figures/` |
| Evaluation library distributions | `mRNA_RBP/outputs/notebook_plots/library_distributions.png` |
| Scatter across mutation rate | `mRNA_RBP/outputs/notebook_plots/scatter_by_mutcount.png` |
| Library-size trend | `mRNA_RBP/outputs/notebook_plots/rho_vs_libsize_type3.png` |
| Spearman/RMSE model misspecification | `mRNA_RBP/outputs/notebook_plots/model_comparison_bar_type3.png` |
| Cross-mutation-rate heatmap | `mRNA_RBP/outputs/notebook_plots/synthetic_gt_cross_mutrate_heatmap.png` |

## Script Usage

Use a named collection when one exists. This records the known
sequence/structure/motif metadata and stages active figures into
`mRNA_RBP/outputs/reproducible_pipeline/<collection>/figures`.

Stage Synthetic GT:

```bash
python scripts/reproduce_gt_pipeline.py --collection synthetic_gt --require-all
```

Stage ResidualBind VTS1:

```bash
python scripts/reproduce_gt_pipeline.py --collection residualbind_vts1 --require-all
```

Regenerate ResidualBind VTS1 artifacts once, then regenerate plots and stage:

```bash
MPLCONFIGDIR=/tmp/mplconfig \
XDG_CACHE_HOME=/tmp/xdg-cache \
LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64 \
/home/nagle/miniconda3/envs/squid/bin/python3.7 scripts/reproduce_gt_pipeline.py \
  --collection residualbind_vts1 \
  --regenerate-artifacts \
  --regenerate-plots \
  --require-all
```

Regenerate ResidualBind VTS1 plots only from existing artifacts:

```bash
python scripts/reproduce_gt_pipeline.py \
  --collection residualbind_vts1 \
  --regenerate-plots \
  --require-all
```

Stage the current outputs into one directory:

```bash
python scripts/reproduce_gt_pipeline.py \
  --ground-truth synthetic \
  --sequence AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA \
  --structure '........((((((((.......))))))))..........' \
  --motif-positions 17-21 \
  --out-dir mRNA_RBP/outputs/reproducible_pipeline/synthetic_gt
```

Use `--require-all` to fail if any canonical figure is missing:

```bash
python scripts/reproduce_gt_pipeline.py \
  --ground-truth synthetic \
  --sequence AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA \
  --structure '........((((((((.......))))))))..........' \
  --motif-positions 17-21 \
  --out-dir mRNA_RBP/outputs/reproducible_pipeline/synthetic_gt \
  --require-all
```

Regenerate the supported synthetic plots before staging. This is intentionally
opt-in because the scatter and model-comparison cells retrain surrogate models:

```bash
conda run -n squid env \
  LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64 \
  MPLCONFIGDIR=/tmp/mplconfig \
  XDG_CACHE_HOME=/tmp/xdg-cache \
  PYTHONPATH=. \
  python scripts/reproduce_gt_pipeline.py \
    --ground-truth synthetic \
    --sequence AAAAAAAAGCGCAUGCUUGCAUGGCAUGCGCAAAAAAAAAA \
    --structure '........((((((((.......))))))))..........' \
    --motif-positions 17-21 \
    --out-dir mRNA_RBP/outputs/reproducible_pipeline/synthetic_gt \
    --regenerate \
    --require-all
```

`--regenerate` currently calls the existing synthetic-cache plotting scripts:

```text
mRNA_RBP/scripts/pipeline/lib_size_spearman.py --out_json mRNA_RBP/outputs/lib_size_spearman_results_type3.json --recompute_saturated --saturated_only --gt_keys additive additive_pairwise nonlin_additive nonlin_additive_pairwise
mRNA_RBP/scripts/figures/core/plot_library_distributions.py
mRNA_RBP/scripts/figures/core/plot_synthetic_rand_region_distributions.py
mRNA_RBP/scripts/figures/core/plot_scatter_by_mutcount.py
mRNA_RBP/scripts/figures/core/plot_rho_vs_libsize_type3.py
mRNA_RBP/scripts/figures/core/bar_surrogate_models_type3.py
mRNA_RBP/scripts/figures/core/plot_cross_mutrate.py --out_prefix synthetic_gt_
```

For `residualbind` or `deepsquid`, first generate the matching score caches and
surrogate coefficient files under the same schema, then run the script without
`--regenerate` to stage the outputs. If a coefficient map, plot directory, or
random-library directory has a nonstandard location, pass:

```bash
--coefficient-figure path/to/coefficient_map.png
--mrna-plot-dir path/to/mRNA_RBP/notebook_plots
--random-library-dir path/to/random_library_figures
```

## How Each Figure Is Made

1. Ground-truth coefficient analysis compares an oracle single/double-mutant
   coefficient map against the nonlinear additive + pairwise surrogate
   coefficients trained on a 20k 10% mutation library.

2. Synthetic random library plots are generated by
   `mRNA_RBP/scripts/figures/core/plot_synthetic_rand_region_distributions.py`. They reuse the
   cached 5%, 10%, and 25% synthetic random libraries, split sequences by whether
   mutations hit the annotated stem, motif, or neither, and emit exactly two
   figures: high-WT activity and low-WT activity. ResidualBind/deepsquid runs can
   still stage the `rbp` random-library figures.

3. Evaluation library distributions compare the activity-balanced library,
   targeted pairwise library, and saturated additive + pairwise library. The
   synthetic plot recomputes scores from the current GT object instead of
   trusting stale score arrays in older `.npz` files.

   The activity-balanced library is initialized in a fixed way. For each
   instance, candidate sequences are sampled as exact mutants of the selected
   WT sequence at mutation counts 3, 5, 7, and 15, with
   `ACTIVITY_BALANCED_CANDIDATE_N = 200000` split evenly across those four
   counts. Candidates are globally deduplicated, scored by the selected oracle
   primary score, then histogram-uniformized in score space using 200 equal-width
   bins, percentile clipping `[1, 99]`, and seed `k*10000 + 600`. The output is
   capped at `ACTIVITY_BALANCED_TARGET_N = 20000`; the final count can be lower
   if the nonempty score bins are sparse. The canonical filename is
   `activity_balanced.npz`.

4. Scatter-by-mutation-count reads a saved prediction artifact. For
   ResidualBind VTS1, `mRNA_RBP/scripts/pipeline/generate_residualbind_vts1_scatter_predictions.py`
   trains the nonlinear additive + pairwise surrogate at 5%, 10%, and 25% once
   and writes `scatter_by_mutcount_predictions.npz`; the plotting script reads
   that file and does not train.

5. Library-size trend reads the type3 Spearman cache and plots random holdout,
   activity-balanced, targeted pairwise, and saturated additive + pairwise
   performance across 200, 2K, and 20K training sizes.

6. Model misspecification retrains/evaluates SSM, additive, additive + pairwise,
   nonlinear additive, and nonlinear additive + pairwise models on the same
   synthetic GT and shows Spearman on the top row and RMSE on the bottom row.

7. Cross-mutation-rate heatmap reads `cross_mutrate_results.json` and shows
   train-rate by test-rate generalization at 20K library size.

## Porting To A New Ground Truth

To make the pipeline fully meaningful for a new GT, produce the same artifacts:

- Random libraries and scores for 5%, 10%, and 25%.
- Activity-balanced evaluation library.
- Targeted pairwise evaluation library from the RNAFold stem pairs.
- Saturated additive + pairwise/type3 evaluation library.
- Surrogate coefficients for the nonlinear additive + pairwise model.
- Cross-mutation-rate result JSON.

The runner will record the new sequence/structure inputs and collect the plots;
the plotting scripts should be pointed at the corresponding cache/output root
or adapted to the new oracle label if the cache key differs.
