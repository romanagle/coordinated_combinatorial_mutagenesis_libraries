# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Benchmarks SQUID/MAVE-NN surrogate models against a synthetic mRNA-RBP ground truth. The ground truth is a manually-specified stem-loop RNA sequence scored by a noWT additive + pairwise Potts energy model passed through a sigmoid nonlinearity. Everything in `scripts/` is support code for this; the active pipeline lives entirely in `mRNA_RBP/`.

## Python environment

```bash
/home/nagle/miniconda3/envs/squid/bin/python3.7
export LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
```

All scripts use `sys.path.insert` manually — there is no install step.

## Running the pipeline

Run from `/home/nagle/final_version`:

```bash
# Full regeneration (wipes stale data, reruns everything for instance 0)
bash mRNA_RBP/master_job.sh

# Step-by-step (all resumable — skips already-generated files)
python mRNA_RBP/generate_libraries.py --n_instances 1
python mRNA_RBP/lib_size_spearman.py  --n_instances 1

# Individual plot scripts (after lib_size_spearman.py has run)
python mRNA_RBP/plot_library_distributions.py
python mRNA_RBP/plot_coefficients.py
python mRNA_RBP/plot_pairwise_heatmaps.py
python mRNA_RBP/plot_scatter_grid.py --eval_type type2
python mRNA_RBP/plot_rho_vs_libsize.py
python mRNA_RBP/bar_surrogate_models.py
```

## Module map (`mRNA_RBP/`)

| File | Role |
|------|------|
| `gt_init.py` | `MrnaRbpGroundTruth` class — the ground truth oracle |
| `generate_libraries.py` | Generates 2M-sequence pools, subsamples to lib sizes, builds eval libs |
| `lib_size_spearman.py` | Trains all 4 surrogate configs, records Spearman ρ, saves coefs |
| `train_surrogate.py` | Standalone surrogate trainer for the nonlinear additive+pairwise config |
| `evaluate.py` | Structured eval datasets: additive (non-stem k-mutants), pairwise (stem 4×4 grids), SSM |
| `viz.py` | Coefficient plots mirroring the demo notebook style |
| `master_job.sh` | Orchestrates the full pipeline end-to-end |

Shared utilities from `scripts/` that `mRNA_RBP/` imports:
- `scripts/ground_truth.py` — `additive_affinity_noWT`, `pairwise_potts_energy`, `apply_global_nonlin`, `init_sigmoid_nonlin`, `soft_threshold`, `uniformize_by_histogram`
- `scripts/seq_utils.py` — `rna_to_one_hot`

## Ground truth: `MrnaRbpGroundTruth`

The canonical sequence and structure (defined in `generate_libraries.py`):

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

### Scoring interface

```python
gt.score_all(x)   # (N,L,4) → dict with 4 keys, all ≤ 0, WT = 0
gt(x)             # (N,L,4) → (N,) nonlin_additive_pairwise scores only
gt.alpha          # (L,4) additive weight matrix
gt.beta           # {(i,j): (4,4)} pairwise blocks for declared stem pairs
gt.edges          # (E,2) stem pair indices
```

## Output structure

```
mRNA_RBP/outputs/
  instance_00/
    gt_params.npz           # alpha, edges, W_mut, mut_map, J, seed
    ssm.npz                 # 3L single-site mutant scores
    pairwise_lib.npz        # 16 × n_stem_pairs combos (all 4×4 for each stem)
    type1_lib{N}.npz        # uniformized eval lib scaled to 20% of lib size N
    type2.npz               # 20K-sequence mixed-rate (2/3/4-mut) uniformized eval lib
    mut05/
      pool_2M.npz           # 2M sequences at 5% mut rate + all 4 GT scores
      lib_200.npz  lib_2000.npz  lib_20000.npz
    mut10/ ...
    mut25/ ...
  surrogate_coefs/          # npz per (instance, mut_rate, lib_size, surrogate_config)
  lib_size_spearman_results.json  # Spearman ρ cache (resumable)
  notebook_plots/           # all output figures
```

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
- **Type1** — uniformized across all 3 mut rates, target_n = 20% of training lib size N
- **Type2** — uniformized 20K lib mixing 2/3/4-mut sequences (shared across lib sizes)
- **Pairwise** — all 16 nuc combos per stem pair, other positions at WT (deterministic)
