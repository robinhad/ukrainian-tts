#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${ROOT}/.venv-audio-review/bin/python"
BASE="${ROOT}/data/expanded_v5_enhanced_novoa/manifests/all.parquet"
INPUT="${ROOT}/data/expanded_v5_enhanced_novoa/pre_enhancement_embedding_manifest.parquet"
OUTPUT="${ROOT}/data/expanded_v6_sidon_deess_novoa/audio_24k"
RESULT_ROOT="${ROOT}/reports/expanded_v6_sidon_deess_novoa_processing"
STATUS="${ROOT}/reports/expanded_v6_sidon_deess_novoa_preparation_status.jsonl"
MODEL_CACHE="${ROOT}/vendor/audio-enhancement-review-models/sidon"
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
if (( ${#GPUS[@]} != 2 )); then
    echo "This process requires exactly two GPU UUIDs." >&2
    exit 2
fi
free_gib=$(df -B1 --output=avail "$ROOT" | tail -1)
if (( free_gib < 30 * 1024 * 1024 * 1024 )); then
    echo "Free disk space is below 30 GiB." >&2
    exit 2
fi
mkdir -p "$OUTPUT" "$RESULT_ROOT"
results=("${RESULT_ROOT}/shard-0.jsonl" "${RESULT_ROOT}/shard-1.jsonl")
logs=("${RESULT_ROOT}/shard-0.log" "${RESULT_ROOT}/shard-1.log")

run_pass() {
    local attempt=$1
    pids=()
    for index in 0 1; do
        CUDA_VISIBLE_DEVICES="${GPUS[$index]}" "$PYTHON" "${ROOT}/scripts/preprocess_sidon_deess_audio.py" \
            --manifest "$BASE" --input-audio-manifest "$INPUT" --output-root "$OUTPUT" \
            --output-results "${results[$index]}" --model-cache "$MODEL_CACHE" \
            --device cuda:0 --num-shards 2 --shard-index "$index" --maximum-attempts 2 --resume \
            >>"${logs[$index]}" 2>&1 &
        pids+=("$!")
    done
    "$PYTHON" "${ROOT}/scripts/monitor_sidon_v6_preparation.py" \
        --results "${results[@]}" --worker-pids "${pids[@]}" --total 74156 \
        --workspace "$ROOT" --status "$STATUS" --interval-seconds 900 --minimum-free-gib 30 &
    monitor=$!
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    wait "$monitor" || failed=1
    echo "Sidon pass ${attempt} finished with status ${failed}."
    return "$failed"
}

run_pass 1 || run_pass 2
"$PYTHON" - "${results[@]}" <<'PY'
import json, sys
rows=[]
for path in sys.argv[1:]:
    rows.extend(json.loads(line) for line in open(path) if line.strip())
if len(rows) != 74156 or len({x['utterance_id'] for x in rows}) != 74156:
    raise SystemExit('The processing results do not cover 74,156 unique files.')
failed=[x for x in rows if x.get('processing_status') != 'ok']
if failed:
    raise SystemExit(f'Sidon processing still has {len(failed)} failed files.')
print('Sidon processing has 74,156 PASS files.')
PY
