#!/usr/bin/env bash
# Full-model phonetic transfer, with a fresh optimizer and numeric-only W&B.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM GPU allocation}"
# sbatch executes a spooled copy, so BASH_SOURCE is not the repository path.
ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}/training"
cd "$ROOT/.."
source "$ROOT/activate.sh"
TRANSFER_ROOT=${TRANSFER_ROOT:-training/quality_runs/phonetic_transfer}
TTS_EXP=${TTS_EXP:-training/exp/phonetic_transfer_20k}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-8}
export UKTTS_ESPEAK_DATA_HASH_FILE="$TRANSFER_ROOT/frontend/ESPEAK_NG_DATA_HASH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
export UKTTS_CUDA_CACHE_INTERVAL=10
for split in train dev; do
    if [[ ! -f "$TRANSFER_ROOT/$split/metadata.json" ]]; then
        python -m training.transfer.prepare --bundle "$TRANSFER_ROOT/converted" \
            --pairs "training/data/quality_v12/manifests/quality_v12_$split.parquet" \
            --output "$TRANSFER_ROOT/$split"
    fi
done
python -m training.quality.supervise \
    --output "$TRANSFER_ROOT/run_${STEPS}" --interval 60 --minimum-available-gib 24 \
    -- python -m training.quality.wandb_metrics \
    --events "$TTS_EXP/tensorboard" --state "$TRANSFER_ROOT/wandb_${STEPS}/state.json" \
    --project ukrainian-tts --name "jets-english-ukrainian-phonetic-${STEPS}" \
    -- python -m training.transfer.train --bundle "$TRANSFER_ROOT/converted" \
    --train "$TRANSFER_ROOT/train" --valid "$TRANSFER_ROOT/dev" \
    --output "$TTS_EXP" --steps "$STEPS" --batch-size "$BATCH_SIZE" --external-metrics &
SUPERVISOR_PID=$!
python -m training.transfer.monitor --pid "$SUPERVISOR_PID" \
    --log "$TRANSFER_ROOT/run_${STEPS}/command.log" --steps "$STEPS" \
    --output "$TRANSFER_ROOT/run_${STEPS}/progress.jsonl" &
MONITOR_PID=$!
trap 'kill "$SUPERVISOR_PID" "$MONITOR_PID" 2>/dev/null || true' TERM INT EXIT
STATUS=0
wait "$SUPERVISOR_PID" || STATUS=$?
wait "$MONITOR_PID" || true
trap - TERM INT EXIT
exit "$STATUS"
