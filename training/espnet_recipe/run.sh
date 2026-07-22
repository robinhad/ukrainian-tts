#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RECIPE_ROOT="${ROOT}/espnet_recipe"
if [[ ! -f "${ROOT}/activate.sh" ]]; then
    echo "Run training/scripts/bootstrap_env.sh first" >&2
    exit 1
fi
source "${ROOT}/activate.sh"
cd "${RECIPE_ROOT}"

export MANIFEST_DIR="${MANIFEST_DIR:-${ROOT}/data/manifests}"
GPU_COUNT="${GPU_COUNT:-1}"
TRAIN_SET="${TRAIN_SET:-smoke_train}"
VALID_SET="${VALID_SET:-smoke_dev}"
TEST_SETS="${TEST_SETS:-smoke_eval}"
export DATA_SETS="${DATA_SETS:-${TRAIN_SET} ${VALID_SET} ${TEST_SETS}}"
DUMP_DIR="${DUMP_DIR:-${ROOT}/dump}"
EXP_DIR="${EXP_DIR:-${ROOT}/exp}"

exec "${ESPNET_ROOT}/egs2/TEMPLATE/tts1/tts.sh" \
    --lang uk \
    --ngpu "${GPU_COUNT}" \
    --nj 8 \
    --inference_nj 1 \
    --gpu_inference true \
    --fs 24000 \
    --n_fft 1024 \
    --n_shift 256 \
    --win_length null \
    --fmin 0 \
    --fmax null \
    --audio_format wav \
    --feats_type raw \
    --tts_task gan_tts \
    --token_type phn \
    --g2p espeak_ng_ukrainian \
    --cleaner none \
    --train_set "${TRAIN_SET}" \
    --valid_set "${VALID_SET}" \
    --test_sets "${TEST_SETS}" \
    --srctexts "${RECIPE_ROOT}/data/${TRAIN_SET}/text" \
    --min_wav_duration 0.1 \
    --max_wav_duration 20 \
    --train_config "${RECIPE_ROOT}/conf/tuning/train_jets_uk_24k.yaml" \
    --inference_config "${RECIPE_ROOT}/conf/tuning/decode_jets.yaml" \
    --dumpdir "${DUMP_DIR}" \
    --expdir "${EXP_DIR}" \
    "$@"
