#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}
export CUDA_VISIBLE_DEVICES="${GPU_UUIDS}"
source "${ROOT}/activate.sh"

DATA_ROOT="${ROOT}/data/multispeaker_smoke"
DUMP_ROOT="${ROOT}/dump_multispeaker_smoke"
EXP_ROOT="${ROOT}/exp_multispeaker_smoke"
TTS_EXP="${EXP_ROOT}/tts_jets_uk_24k_multispeaker_smoke"
TRAIN_SET=multispeaker_smoke_train
VALID_SET=multispeaker_smoke_dev
TEST_SET=multispeaker_smoke_eval

run_logged() {
    local eta_minutes=$1
    shift
    python "${ROOT}/scripts/run_logged.py" \
        --log "${ROOT}/reports/commands.jsonl" \
        --eta-minutes "${eta_minutes}" -- "$@"
}

run_logged 2 python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --gpu-uuids "${GPU_UUIDS}" \
    --workspace "${ROOT}" \
    --output "${ROOT}/reports/resource_usage.jsonl"
run_logged 3 python -m pytest -c "${ROOT}/pytest.ini" "${ROOT}/tests"
run_logged 2 python "${ROOT}/scripts/fetch_legacy_speaker_embeddings.py" \
    --output "${ROOT}/data/multispeaker/source_metadata/spk_xvector_v6.ark" \
    --model-dir "${ROOT}/vendor/speechbrain-spkrec-ecapa-voxceleb" \
    --report "${ROOT}/reports/multispeaker_legacy_embeddings.json"
run_logged 10 python "${ROOT}/scripts/create_multispeaker_dataset.py" \
    --mode smoke \
    --cv-shards 4 \
    --output-root "${DATA_ROOT}"
run_logged 15 python "${ROOT}/scripts/prepare_audio.py" \
    --records "${DATA_ROOT}/source_records.jsonl" \
    --output-root "${DATA_ROOT}/processed_24k" \
    --output-records "${DATA_ROOT}/prepared_records.jsonl" \
    --trim-silence --trim-top-db 40 --trim-padding-ms 100
run_logged 5 python "${ROOT}/scripts/build_manifest.py" \
    --records "${DATA_ROOT}/prepared_records.jsonl" \
    --output-dir "${DATA_ROOT}/manifests" \
    --cache "${DATA_ROOT}/phonemes.sqlite3"
run_logged 5 python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/multispeaker_smoke_dataset.json"

export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET
export VALID_SET
export TEST_SETS="${TEST_SET}"
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SET}"
export DUMP_DIR="${DUMP_ROOT}"
export EXP_DIR="${EXP_ROOT}"
export GPU_COUNT=2

run_logged 45 "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 1 --stop_stage 6
run_logged 90 "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${TTS_EXP}" \
    --train_args \
        "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --use_amp false"
run_logged 15 "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "${TTS_EXP}" \
    --inference_model train.total_count.ave.pth

DECODE_DIR="${TTS_EXP}/decode_jets_train.total_count.ave/${TEST_SET}"
run_logged 3 python "${ROOT}/scripts/synthesize_eval.py" \
    --wav-dir "${DECODE_DIR}/wav" \
    --manifest "${DATA_ROOT}/manifests/${TEST_SET}.parquet" \
    --checkpoint "${TTS_EXP}/train.total_count.ave.pth" \
    --config "${TTS_EXP}/config.yaml" \
    --inference-log "${DECODE_DIR}/log/tts_inference.1.log" \
    --output "${ROOT}/reports/multispeaker_smoke_inference.json"
