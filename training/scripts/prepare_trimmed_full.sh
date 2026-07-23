#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"

SOURCE_ROOT="${ROOT}/data/full"
TARGET_ROOT="${ROOT}/data/full_trimmed"
if [[ ! -f "${SOURCE_ROOT}/source_records.jsonl" ]]; then
    echo "Missing ${SOURCE_ROOT}/source_records.jsonl" >&2
    echo "Run create_full_dataset.py before this command." >&2
    exit 2
fi

run_logged() {
    local eta_minutes=$1
    shift
    python "${ROOT}/scripts/run_logged.py" \
        --log "${ROOT}/reports/commands.jsonl" --eta-minutes "${eta_minutes}" -- "$@"
}

run_logged 10 python "${ROOT}/scripts/prepare_audio.py" \
    --records "${SOURCE_ROOT}/source_records.jsonl" \
    --output-root "${TARGET_ROOT}/processed_24k" \
    --output-records "${TARGET_ROOT}/prepared_records.jsonl" \
    --trim-silence --trim-top-db 40 --trim-padding-ms 100
run_logged 5 python "${ROOT}/scripts/build_manifest.py" \
    --records "${TARGET_ROOT}/prepared_records.jsonl" \
    --output-dir "${TARGET_ROOT}/manifests" \
    --cache "${TARGET_ROOT}/phonemes.sqlite3"
run_logged 10 python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${TARGET_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/full_trimmed_dataset.json"
run_logged 5 python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${TARGET_ROOT}/manifests/all.parquet" \
    --output "${ROOT}/reports/full_trimmed_data_analysis.json"

export MANIFEST_DIR="${TARGET_ROOT}/manifests"
export TRAIN_SET=train
export VALID_SET=dev
export TEST_SETS=eval
export DATA_SETS="train dev eval"
export DUMP_DIR="${ROOT}/dump_full_trimmed"
export EXP_DIR="${ROOT}/exp_full_trimmed"
run_logged 45 "${ROOT}/espnet_recipe/run.sh" --stage 1 --stop_stage 6
