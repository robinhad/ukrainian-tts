#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"
LOGGER=(python "${ROOT}/scripts/run_logged.py" --log "${ROOT}/reports/commands.jsonl" --)

"${LOGGER[@]}" python "${ROOT}/scripts/check_resources.py" \
    --mode smoke --require-torch --workspace "${ROOT}" \
    --output "${ROOT}/reports/resource_usage.jsonl"
"${LOGGER[@]}" python -m pytest -c "${ROOT}/pytest.ini" "${ROOT}/tests"
"${LOGGER[@]}" python "${ROOT}/scripts/create_smoke_subset.py" --output-root "${ROOT}/data"
"${LOGGER[@]}" python "${ROOT}/scripts/prepare_audio.py" \
    --records "${ROOT}/data/source_records.jsonl" \
    --output-root "${ROOT}/data/processed_24k" \
    --output-records "${ROOT}/data/prepared_records.jsonl"
"${LOGGER[@]}" python "${ROOT}/scripts/build_manifest.py" \
    --records "${ROOT}/data/prepared_records.jsonl" \
    --output-dir "${ROOT}/data/manifests" \
    --cache "${ROOT}/data/phonemes.sqlite3"
"${LOGGER[@]}" python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${ROOT}/data/manifests/all.parquet" \
    --report "${ROOT}/reports/smoke_dataset.json"
"${LOGGER[@]}" "${ROOT}/espnet_recipe/run.sh" --stage 1 --stop_stage 6
"${LOGGER[@]}" "${ROOT}/espnet_recipe/run.sh" \
    --stage 7 --stop_stage 7 --train_args "--max_epoch 1"
"${LOGGER[@]}" "${ROOT}/espnet_recipe/run.sh" --stage 8 --stop_stage 8
