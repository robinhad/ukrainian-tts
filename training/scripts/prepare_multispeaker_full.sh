#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ROOT}/data/multispeaker_full"
LADA_MANIFEST=${LADA_MANIFEST:-"${ROOT}/data/full_trimmed/manifests/all.parquet"}
GPU_UUIDS=${GPU_UUIDS:-"GPU-be591530-39fd-0b1c-50af-8c75548cb6b8,GPU-de1be084-ce05-942c-cb74-78e80652f184"}

if [[ ! -s "$LADA_MANIFEST" ]]; then
    echo "The Lada manifest does not exist: $LADA_MANIFEST" >&2
    exit 2
fi

source "${ROOT}/activate.sh"
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"

run_logged() {
    local eta_minutes=$1
    shift
    python "${ROOT}/scripts/run_logged.py" \
        --log "${ROOT}/reports/commands.jsonl" \
        --eta-minutes "$eta_minutes" -- "$@"
}

run_logged 2 python "${ROOT}/scripts/check_resources.py" \
    --mode full --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" \
    --output "${ROOT}/reports/resource_usage.jsonl"
run_logged 3 python "${ROOT}/scripts/fetch_legacy_speaker_embeddings.py" \
    --output "${ROOT}/data/multispeaker/source_metadata/spk_xvector_v6.ark" \
    --model-dir "${ROOT}/vendor/speechbrain-spkrec-ecapa-voxceleb" \
    --report "${ROOT}/reports/multispeaker_legacy_embeddings.json"

dataset_args=(
    --mode full
    --output-root "$DATA_ROOT"
    --lada-manifest "$LADA_MANIFEST"
)
if [[ -n "${DMYTRO_MANIFEST:-}" ]]; then
    dataset_args+=(--dmytro-manifest "$DMYTRO_MANIFEST")
fi
run_logged 120 python "${ROOT}/scripts/create_multispeaker_dataset.py" \
    "${dataset_args[@]}"
run_logged 240 python "${ROOT}/scripts/prepare_audio.py" \
    --records "${DATA_ROOT}/source_records.jsonl" \
    --output-root "${DATA_ROOT}/processed_24k" \
    --output-records "${DATA_ROOT}/prepared_records.jsonl" \
    --trim-silence --trim-top-db 40 --trim-padding-ms 100
run_logged 15 python "${ROOT}/scripts/build_manifest.py" \
    --records "${DATA_ROOT}/prepared_records.jsonl" \
    --output-dir "${DATA_ROOT}/manifests" \
    --cache "${DATA_ROOT}/phonemes.sqlite3"
run_logged 15 python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/multispeaker_full_dataset.json"
run_logged 10 python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --output "${ROOT}/reports/multispeaker_full_data_analysis.json"

export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET=multispeaker_train
export VALID_SET=multispeaker_dev
export TEST_SETS=multispeaker_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export DUMP_DIR="${ROOT}/dump_multispeaker_full"
export EXP_DIR="${ROOT}/exp_multispeaker_full"
export GPU_COUNT=2

run_logged 240 "${ROOT}/espnet_recipe/run_multispeaker.sh" \
    --stage 1 --stop_stage 6
