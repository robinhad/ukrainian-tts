#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
: "${SLURM_JOB_ID:?Use a SLURM GPU allocation}"
source "${ROOT}/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export QUALITY_RUN_DIR="${ROOT}/quality_runs/v11"
mkdir -p "$QUALITY_RUN_DIR"
python -m training.scripts.select_quality_v11 \
  --workers "${PROCESS_MODEL_WORKERS:-2}" --chunk-size "${PROCESS_CHUNK_SIZE:-128}"
bash "${ROOT}/scripts/prepare_quality_v11.sh"
export TRAIN_SET=quality_v11_train VALID_SET=quality_v11_dev TEST_SETS=quality_v11_eval
export MANIFEST_DIR="${ROOT}/data/quality_v11/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v11" EXP_DIR="${ROOT}/exp_quality_v11"
export TTS_EXP="${EXP_DIR}/tts_jets_quality_v11_50k"
export QUALITY_ROOT="${QUALITY_RUN_DIR}/checkpoints"
export MILESTONES="25k 50k" STEPS=50000
export UKTTS_CUDA_CACHE_INTERVAL=10
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m training.quality.supervise --output "$QUALITY_RUN_DIR/training" \
  --interval 60 --minimum-available-gib 24 --activity-log "$TTS_EXP/train.log" \
  -- bash "${ROOT}/scripts/run_quality_v10_training.sh"
