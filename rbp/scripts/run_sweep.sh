#!/usr/bin/env bash
# Launch the RNAcompete benchmark across the 4 GPUs, one run per GPU, in the background.
#
# Validation-only by default (no test peeking during exploration). For the FINAL numbers on the
# chosen config(s), re-run that config with EVAL_TEST=1.
#
#   bash scripts/run_sweep.sh                 # exploration: best_val_pearson
#   EVAL_TEST=1 bash scripts/run_sweep.sh     # also evaluate the held-out test split
#
set -euo pipefail
cd "$(dirname "$0")/.."   # -> repo root (rbp_benchmark)

PY=.venv/bin/python
H5=/home/shared/data/rnacompete/rnacompete2009.h5
W=/home/shared/models/alphagenome_pytorch/model_all_folds.safetensors
RESULTS=results
TEST_FLAG=""; [ "${EVAL_TEST:-0}" = "1" ] && TEST_FLAG="--eval_test"

launch() {  # gpu  label  config  [extra args...]
  local gpu=$1 label=$2 config=$3; shift 3
  mkdir -p "$RESULTS/$label"
  echo "GPU$gpu -> $label  ($config)"
  CUDA_VISIBLE_DEVICES=$gpu nohup $PY scripts/train.py \
    --config "$config" --h5_path "$H5" --all_rbps --output_dir "$RESULTS/$label" \
    $TEST_FLAG "$@" > "$RESULTS/$label/train.log" 2>&1 &
}

# ResidualBind is trained separately (see results/residualbind); this sweep is the AlphaGenome
# resolution variants, one per GPU.
launch 0 agft_trunk  configs/alphagenome_trunk.json  --pretrained_weights "$W"
launch 1 agft_bin16  configs/alphagenome_bin16.json  --pretrained_weights "$W"
launch 2 agft_bin8   configs/alphagenome_bin8.json   --pretrained_weights "$W"
launch 3 agft_bin4   configs/alphagenome_bin4.json   --pretrained_weights "$W"

echo
echo "Launched 4 runs in the background. Monitor with:"
echo "  tail -f $RESULTS/*/train.log"
echo "  nvidia-smi"
echo "When done, compare on validation:"
echo "  $PY scripts/compare.py --results_dir $RESULTS --metric best_val_pearson"
