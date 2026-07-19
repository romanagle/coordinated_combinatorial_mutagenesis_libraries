#!/bin/bash
# master_job.sh — full regeneration pipeline
# Run from /home/nagle/final_version
# stem_sigma changed 1.5 → 3.0 (near-full compensatory rescue)
set -euo pipefail

PYTHON=/home/nagle/miniconda3/envs/squid/bin/python3.7
export LD_LIBRARY_PATH=/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd /home/nagle/final_version

# ── Step 1: wipe stale data ──────────────────────────────────────────────────
log "=== Deleting old libraries and coefs (instance 0 only) ==="
find mRNA_RBP/outputs/instance_00/ -name "*.npz" -exec rm -f {} \; 2>/dev/null || true
find mRNA_RBP/outputs/instance_00/ -name "mut*"  -type d -exec rm -rf {} \; 2>/dev/null || true
rm -f  mRNA_RBP/outputs/lib_size_spearman_results.json
rm -f  mRNA_RBP/outputs/cross_mutrate_results.json
rm -rf mRNA_RBP/outputs/surrogate_coefs/
log "Clean done."

# ── Step 2: generate all libraries ──────────────────────────────────────────
log "=== Generating libraries (instance 0 only) ==="
$PYTHON mRNA_RBP/generate_libraries.py --n_instances 1
log "Libraries done."

# ── Step 3: train surrogates + eval (the long step) ─────────────────────────
log "=== Training surrogates via lib_size_spearman (instance 0 only) ==="
$PYTHON mRNA_RBP/lib_size_spearman.py --n_instances 1
log "Surrogates done."

# ── Step 4: notebook_plots figures ──────────────────────────────────────────
log "=== Plotting: library_distributions ==="
$PYTHON mRNA_RBP/plot_library_distributions.py

log "=== Plotting: coefficients (GT + surrogate) ==="
$PYTHON mRNA_RBP/plot_coefficients.py

log "=== Plotting: pairwise heatmaps ==="
$PYTHON mRNA_RBP/plot_pairwise_heatmaps.py

log "=== Plotting: scatter_grid activity-balanced ==="
$PYTHON mRNA_RBP/plot_scatter_grid.py --eval_type type2

log "=== Plotting: rho vs libsize (rand / activity-balanced / pairwise) ==="
$PYTHON mRNA_RBP/plot_rho_vs_libsize.py

log "=== Cross-mutation-rate eval (no retraining, post-hoc) ==="
$PYTHON mRNA_RBP/cross_mutrate_eval.py --n_instances 1

log "=== Plotting: cross_mutrate (generalization gap) ==="
$PYTHON mRNA_RBP/plot_cross_mutrate.py

log "=== Plotting: preliminary_spearman ==="
$PYTHON mRNA_RBP/make_preliminary_spearman.py

log "=== Plotting: bar_surrogate_models (activity-balanced + pairwise) ==="
$PYTHON mRNA_RBP/bar_surrogate_models.py

log "=== ALL DONE ==="
ls -lh mRNA_RBP/outputs/notebook_plots/
