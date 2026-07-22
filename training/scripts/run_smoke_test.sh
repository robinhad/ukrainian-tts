#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"
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
run_logged 10 python "${ROOT}/scripts/create_smoke_subset.py" --output-root "${ROOT}/data"
run_logged 10 python "${ROOT}/scripts/prepare_audio.py" \
    --records "${ROOT}/data/source_records.jsonl" \
    --output-root "${ROOT}/data/processed_24k" \
    --output-records "${ROOT}/data/prepared_records.jsonl"
run_logged 5 python "${ROOT}/scripts/build_manifest.py" \
    --records "${ROOT}/data/prepared_records.jsonl" \
    --output-dir "${ROOT}/data/manifests" \
    --cache "${ROOT}/data/phonemes.sqlite3"
run_logged 5 python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${ROOT}/data/manifests/all.parquet" \
    --report "${ROOT}/reports/smoke_dataset.json"
run_logged 30 "${ROOT}/espnet_recipe/run.sh" --stage 1 --stop_stage 6
run_logged 90 "${ROOT}/espnet_recipe/run.sh" \
    --stage 7 --stop_stage 7 --train_args "--max_epoch 1"
run_logged 15 "${ROOT}/espnet_recipe/run.sh" --stage 8 --stop_stage 8 \
    --train_args "--max_epoch 1" \
    --inference_model train.total_count.ave.pth
