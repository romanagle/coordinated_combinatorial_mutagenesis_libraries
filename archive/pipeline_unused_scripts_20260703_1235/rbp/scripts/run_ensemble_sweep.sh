#!/usr/bin/env bash
# Train the 10-member ResidualBind ensemble across 4 GPUs by splitting the 9 RBPs round-robin,
# then assemble summary.json and regenerate the comparison figure.
#
#   bash scripts/run_ensemble_sweep.sh                 # validation only
#   EVAL_TEST=1 bash scripts/run_ensemble_sweep.sh     # also evaluate held-out test
#
cd "$(dirname "$0")/.."   # -> repo root

PY=.venv/bin/python
H5=/home/shared/data/rnacompete/rnacompete2009.h5
CFG=configs/residualbind_ens10.json
OUT=results/residualbind_ens10
TEST_FLAG=""; [ "${EVAL_TEST:-0}" = "1" ] && TEST_FLAG="--eval_test"
mkdir -p "$OUT"

# 9 RBPs split round-robin across GPUs 0-3 (3,2,2,2).
GPU_RBPS=( "VTS1 RBM4 YB1" "Fusip SF2" "HuR SLM2" "PTB U1A" )

pids=()
for g in 0 1 2 3; do
  echo "GPU$g -> ${GPU_RBPS[$g]}"
  CUDA_VISIBLE_DEVICES=$g $PY scripts/train_ensemble.py \
    --config "$CFG" --h5_path "$H5" --output_dir "$OUT" --no_summary $TEST_FLAG \
    --rbps ${GPU_RBPS[$g]} > "$OUT/train_gpu$g.log" 2>&1 &
  pids+=($!)
done

# Wait for every GPU job (tolerate individual failures so we still assemble what finished).
for pid in "${pids[@]}"; do wait "$pid" || echo "warning: pid $pid exited non-zero"; done

echo "Members trained; assembling summary + figure"
$PY scripts/build_summary.py --results_dir "$OUT"
$PY scripts/plot_performance.py --results_dir results --split valid \
  --models residualbind residualbind_ens10 agft_trunk agft_bin16 agft_bin8 agft_bin4 \
  || echo "(plot skipped — some columns may be missing)"
echo "DONE"
