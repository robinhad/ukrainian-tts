#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside SLURM}"
: "${VITS_ROOT:?Supply the prepared VITS training directory}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
source training/activate.sh
export UKTTS_ESPEAK_DATA_HASH_FILE=training/quality_runs/phonetic_transfer/frontend/ESPEAK_NG_DATA_HASH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True UKTTS_CUDA_CACHE_INTERVAL=10
TARGET_STEPS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["steps"])' "$VITS_ROOT/experiment.json")
python -m training.quality.supervise --output "$VITS_ROOT/runtime" --interval 60 \
    --minimum-available-gib 24 -- \
    python -m training.quality.wandb_metrics --events "$VITS_ROOT/train_model/tensorboard" \
    --state "$VITS_ROOT/wandb/state.json" --project ukrainian-tts \
    --name "vits-libritts-ukrainian-multispeaker-50k" -- \
    python -m training.transfer.vits_run --root "$VITS_ROOT" &
SUPERVISOR_PID=$!
python -m training.transfer.monitor --pid "$SUPERVISOR_PID" \
    --log "$VITS_ROOT/runtime/command.log" --steps "$TARGET_STEPS" \
    --output "$VITS_ROOT/progress.jsonl" &
MONITOR_PID=$!
trap 'kill "$SUPERVISOR_PID" "$MONITOR_PID" 2>/dev/null || true' TERM INT EXIT
STATUS=0
wait "$SUPERVISOR_PID" || STATUS=$?
wait "$MONITOR_PID" || true
trap - TERM INT EXIT
exit "$STATUS"
