#!/bin/bash
# mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh — standardized, oracle-agnostic ground-truth
# pipeline runner. Replaces the old synthetic-only, since-archived
# master_job.sh: takes any registered oracle (mrna, residualbind_msi1,
# vts1_residualbind, hur_residualbind, ...) and runs every stage needed to
# produce a complete artifact set (raw mut-rate libraries, SSM baseline,
# eval libraries, surrogate coefs, AND the layer-2 "deep squid"
# varied-mutrate surrogate) -- the same complete set MSI1 has, without any
# bespoke per-RBP script.
#
# Two conda envs are required and genuinely unavoidable on this machine
# (verified, not assumed):
#   - toehold_gpu: working torch+CUDA, used to build/score real ResidualBind
#     ensemble oracles. Its torch is required for any stage that scores raw
#     sequences through a live oracle.
#   - squid: working tensorflow/mavenn, used to train the GE/MAVE-NN
#     surrogates. Its torch install fails to import at all
#     (libcublas.so.11: undefined symbol) -- so live oracle scoring can
#     never happen in this env. lib_size_spearman.py and
#     train_surrogate_varied_mutrate.py already account for this: for real
#     ResidualBind oracles they read scores cached by the toehold_gpu stage
#     instead of calling build_oracle() themselves.
#
# Usage:
#   bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle hur [--wt_activity high] [--n_instances 1]
#   bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle vts1 --wt_activity low
#   bash mRNA_RBP/scripts/pipeline/run_gt_pipeline.sh --oracle mrna
#
# Stages (each individually resumable/idempotent -- skips already-generated
# files, same as running the underlying scripts by hand):
#   1. [toehold_gpu] generate_libraries.py        -- mut05/10/25 pools, SSM, eval libs
#   2. [toehold_gpu] generate_varied_mutrate_library.py -- 200K deep-squid pool
#   3. [squid]       lib_size_spearman.py          -- surrogate coefs + rho cache
#   4. [squid]       train_surrogate_varied_mutrate.py  -- deep-squid surrogate model
#   5. [squid]       cross_mutrate_eval.py          -- cross-mutation-rate generalization JSON
#   6. [squid]       generate_scatter_by_mutcount_predictions.py -- scatter-by-mutcount artifact

set -euo pipefail

ORACLE=""
WT_ACTIVITY="high"
N_INSTANCES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --oracle) ORACLE="$2"; shift 2 ;;
    --wt_activity) WT_ACTIVITY="$2"; shift 2 ;;
    --n_instances) N_INSTANCES="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$ORACLE" ]]; then
  echo "Usage: run_gt_pipeline.sh --oracle NAME [--wt_activity high|low] [--n_instances N]" >&2
  exit 1
fi

REPO=/home/nagle/final_version
cd "$REPO"

TORCH_PY=/home/nagle/miniconda3/envs/toehold_gpu/bin/python3
SQUID_PY=/home/nagle/miniconda3/envs/squid/bin/python3.7
SQUID_LD_LIBRARY_PATH="/home/nagle/miniconda3/envs/squid/lib:/usr/local/cuda-11.2/lib64"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== [toehold_gpu] generate_libraries.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
"$TORCH_PY" mRNA_RBP/scripts/pipeline/generate_libraries.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY" --n_instances "$N_INSTANCES"

log "=== [toehold_gpu] generate_varied_mutrate_library.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
"$TORCH_PY" mRNA_RBP/scripts/pipeline/generate_varied_mutrate_library.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY"

log "=== [squid] lib_size_spearman.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
LD_LIBRARY_PATH="$SQUID_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
  "$SQUID_PY" mRNA_RBP/scripts/pipeline/lib_size_spearman.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY" --n_instances "$N_INSTANCES"

log "=== [squid] train_surrogate_varied_mutrate.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
LD_LIBRARY_PATH="$SQUID_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
  "$SQUID_PY" mRNA_RBP/scripts/pipeline/train_surrogate_varied_mutrate.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY"

log "=== [squid] cross_mutrate_eval.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
LD_LIBRARY_PATH="$SQUID_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
  "$SQUID_PY" mRNA_RBP/scripts/pipeline/cross_mutrate_eval.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY" --n_instances "$N_INSTANCES"

log "=== [squid] generate_scatter_by_mutcount_predictions.py --oracle $ORACLE --wt_activity $WT_ACTIVITY ==="
LD_LIBRARY_PATH="$SQUID_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" \
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp/xdg-cache \
  "$SQUID_PY" mRNA_RBP/scripts/pipeline/generate_scatter_by_mutcount_predictions.py --oracle "$ORACLE" --wt_activity "$WT_ACTIVITY" --force

log "=== ALL DONE  oracle=$ORACLE  wt_activity=$WT_ACTIVITY ==="
