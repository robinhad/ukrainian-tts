#!/usr/bin/env bash
# One isolated arm; training length and evaluation cadence come from metadata.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside SLURM}"
: "${ARM_ROOT:?Supply the prepared experiment arm directory}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source training/activate.sh
export UKTTS_ESPEAK_DATA_HASH_FILE=training/quality_runs/phonetic_transfer/frontend/ESPEAK_NG_DATA_HASH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True UKTTS_CUDA_CACHE_INTERVAL=10
read -r TARGET_STEPS EVALUATION_INTERVAL < <(python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["additional_steps"], r["evaluation_interval"])' "$ARM_ROOT/experiment.json")
python -m training.quality.supervise --output "$ARM_ROOT/runtime" --interval 60 \
    --minimum-available-gib 24 -- \
    python -m training.quality.wandb_metrics --events "$ARM_ROOT/train/tensorboard" \
    --state "$ARM_ROOT/wandb/state.json" --project ukrainian-tts \
    --name "${WANDB_RUN_NAME:-jets-rhotic-ab-${ARM_ROOT##*/}}" -- \
    python -m training.transfer.rhotic_ab run --root "$ARM_ROOT" &
SUPERVISOR_PID=$!
python -m training.transfer.monitor --pid "$SUPERVISOR_PID" \
    --log "$ARM_ROOT/runtime/command.log" --steps "$TARGET_STEPS" --iterations-per-epoch "$EVALUATION_INTERVAL" \
    --output "$ARM_ROOT/progress.jsonl" &
MONITOR_PID=$!
trap 'kill "$SUPERVISOR_PID" "$MONITOR_PID" 2>/dev/null || true' TERM INT EXIT
STATUS=0
wait "$SUPERVISOR_PID" || STATUS=$?
wait "$MONITOR_PID" || true
trap - TERM INT EXIT
exit "$STATUS"
