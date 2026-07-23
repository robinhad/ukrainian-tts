#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"
TARGET_ROOT="${ROOT}/data/smoke_trimmed"

run_logged() {
    local eta_minutes=$1
    shift
    python "${ROOT}/scripts/run_logged.py" \
        --log "${ROOT}/reports/commands.jsonl" --eta-minutes "${eta_minutes}" -- "$@"
}

run_logged 2 python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --workspace "${ROOT}" \
    --output "${ROOT}/reports/resource_usage.jsonl"
run_logged 3 python -m pytest -c "${ROOT}/pytest.ini" "${ROOT}/tests"
if [[ ! -f "${ROOT}/data/source_records.jsonl" ]]; then
    run_logged 10 python "${ROOT}/scripts/create_smoke_subset.py" --output-root "${ROOT}/data"
fi
run_logged 10 python "${ROOT}/scripts/prepare_audio.py" \
    --records "${ROOT}/data/source_records.jsonl" \
    --output-root "${TARGET_ROOT}/processed_24k" \
    --output-records "${TARGET_ROOT}/prepared_records.jsonl" \
    --trim-silence --trim-top-db 40 --trim-padding-ms 100
run_logged 5 python "${ROOT}/scripts/build_manifest.py" \
    --records "${TARGET_ROOT}/prepared_records.jsonl" \
    --output-dir "${TARGET_ROOT}/manifests" \
    --cache "${TARGET_ROOT}/phonemes.sqlite3"
run_logged 5 python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${TARGET_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/smoke_trimmed_dataset.json"

export MANIFEST_DIR="${TARGET_ROOT}/manifests"
export TRAIN_SET=smoke_train
export VALID_SET=smoke_dev
export TEST_SETS=smoke_eval
export DATA_SETS="smoke_train smoke_dev smoke_eval"
export DUMP_DIR="${ROOT}/dump_smoke_trimmed"
export EXP_DIR="${ROOT}/exp_smoke_trimmed"
SMOKE_EXP="${EXP_DIR}/tts_jets_uk_24k_trimmed_smoke"
run_logged 30 "${ROOT}/espnet_recipe/run.sh" --stage 1 --stop_stage 6
run_logged 90 "${ROOT}/espnet_recipe/run.sh" \
    --stage 7 --stop_stage 7 \
    --tts_exp "${SMOKE_EXP}" \
    --train_args "--max_epoch 1 --num_iters_per_epoch 100 --batch_bins 1000000 --use_amp false"
run_logged 15 "${ROOT}/espnet_recipe/run.sh" \
    --stage 8 --stop_stage 8 \
    --tts_exp "${SMOKE_EXP}" \
    --inference_model train.total_count.ave.pth
