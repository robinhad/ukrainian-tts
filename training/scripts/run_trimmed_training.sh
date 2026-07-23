#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export DATASET_NAME=full_trimmed
export EXPERIMENT_NAME=tts_jets_uk_24k_trimmed
export DUMP_NAME=dump_full_trimmed
export EXP_NAME=exp_full_trimmed
export BATCH_BINS=${BATCH_BINS:-4000000}
export NUM_WORKERS=${NUM_WORKERS:-8}
export CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-false}
exec "${ROOT}/scripts/run_full_training.sh" "$@"
