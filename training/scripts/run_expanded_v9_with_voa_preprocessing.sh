#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=expanded_v9_cascade_with_voa
PYTHON="${ROOT}/.venv-audio-review/bin/python"
BASE="${ROOT}/data/expanded_v3/manifests/all.parquet"
INPUT="${ROOT}/data/expanded_v3/source_manifests/all.parquet"
OUTPUT="${ROOT}/data/${NAME}/audio_24k"
RESULT_ROOT="${ROOT}/reports/${NAME}_processing"
STATUS="${ROOT}/reports/${NAME}_preparation_status.jsonl"
CACHE="${ROOT}/vendor/audio-enhancement-review-models"
RNNOISE="${ROOT}/vendor/rnnoise-src/examples/rnnoise_demo"
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
EXPECTED=208867

IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
if (( ${#GPUS[@]} != 2 )); then
    echo "This process requires exactly two GPU UUIDs." >&2
    exit 2
fi
if (( WORKERS_PER_GPU < 1 || WORKERS_PER_GPU > 8 )); then
    echo "WORKERS_PER_GPU must be from 1 to 8." >&2
    exit 2
fi
if [[ ! -x "$RNNOISE" ]]; then
    echo "The pinned RNNoise binary is missing: ${RNNOISE}" >&2
    exit 2
fi
free_bytes=$(df -B1 --output=avail "$ROOT" | tail -1)
if (( free_bytes < 30 * 1024 * 1024 * 1024 )); then
    echo "Free disk space is below 30 GiB." >&2
    exit 2
fi

TOTAL_WORKERS=$((${#GPUS[@]} * WORKERS_PER_GPU))
mkdir -p "$OUTPUT" "$RESULT_ROOT"
results=()
logs=()
for ((index=0; index<TOTAL_WORKERS; index++)); do
    results+=("${RESULT_ROOT}/shard-${index}.jsonl")
    logs+=("${RESULT_ROOT}/shard-${index}.log")
done

run_pass() {
    local attempt=$1
    pids=()
    for ((index=0; index<TOTAL_WORKERS; index++)); do
        gpu_index=$((index % ${#GPUS[@]}))
        CUDA_VISIBLE_DEVICES="${GPUS[$gpu_index]}" "$PYTHON" \
            "${ROOT}/scripts/preprocess_training_cascade_audio.py" \
            --manifest "$BASE" --input-audio-manifest "$INPUT" \
            --output-root "$OUTPUT" --output-results "${results[$index]}" \
            --model-cache "$CACHE" --rnnoise-binary "$RNNOISE" \
            --profile-name "$NAME" \
            --device cuda:0 --num-shards "$TOTAL_WORKERS" --shard-index "$index" \
            --maximum-attempts 2 --resume >>"${logs[$index]}" 2>&1 &
        pids+=("$!")
    done
    "$ROOT/.venv/bin/python" "$ROOT/scripts/monitor_training_cascade_preparation.py" \
        --results "${results[@]}" --worker-pids "${pids[@]}" --total "$EXPECTED" \
        --workspace "$ROOT" --status "$STATUS" --interval-seconds 900 \
        --minimum-free-gib 30 &
    monitor=$!
    "$ROOT/.venv/bin/python" "$ROOT/scripts/monitor_expanded_v9_duration_progress.py" \
        --manifest "$BASE" --results "${results[@]}" \
        --worker-pids "${pids[@]}" --workspace "$ROOT" \
        --status "${ROOT}/reports/${NAME}_duration_status.jsonl" \
        --start-status "$STATUS" --interval-seconds 900 &
    duration_monitor=$!
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    wait "$monitor" || failed=1
    wait "$duration_monitor" || failed=1
    echo "Cascade pass ${attempt} finished with status ${failed}."
    return "$failed"
}

run_pass 1 || run_pass 2
"$ROOT/.venv/bin/python" - "$EXPECTED" "${results[@]}" <<'PY'
import json
import sys

expected = int(sys.argv[1])
rows = []
for path in sys.argv[2:]:
    rows.extend(json.loads(line) for line in open(path) if line.strip())
if len(rows) != expected or len({row["utterance_id"] for row in rows}) != expected:
    raise SystemExit("The cascade results do not cover the expected unique files.")
failed = [row for row in rows if row.get("processing_status") != "ok"]
if failed:
    raise SystemExit(f"The cascade still has {len(failed)} failed files.")
print(f"The cascade has {expected} PASS files.")
PY
