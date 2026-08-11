#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-"${ROOT}/eval/generated/per_dataset_train_audio_enhancement_review_v2"}
REPORT=${REPORT:-"${ROOT}/reports/per_dataset_train_audio_enhancement_review_v2.json"}
SELECTION="${OUTPUT}/selection.jsonl"
MODEL_CACHE=${MODEL_CACHE:-"${ROOT}/vendor/audio-enhancement-review-models/sidon"}
PYTHON="${ROOT}/.venv-audio-review/bin/python"
DATA_PYTHON="${ROOT}/.venv/bin/python"
RESOURCE_LOG="${ROOT}/reports/sidon_deess_listening_resources.csv"
MIN_FREE_GIB=${MIN_FREE_GIB:-30}
POSTPROCESS_CONFIG=${POSTPROCESS_CONFIG:-"${ROOT}/conf/audio_postprocess.yaml"}
FFMPEG_BINARY=${FFMPEG_BINARY:-auto}
runner_pid=$$
mapfile -t GPU_UUIDS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)

if (( ${#GPU_UUIDS[@]} < 2 )); then
    echo "This run requires two CUDA devices." >&2
    exit 2
fi

free_gib() {
    df --output=avail -B1G "$ROOT" | tail -1 | tr -d ' '
}

if (( $(free_gib) < MIN_FREE_GIB )); then
    echo "Free disk space is less than ${MIN_FREE_GIB} GiB." >&2
    exit 2
fi

monitor_resources() {
    echo "timestamp_kyiv,index,utilization_gpu_percent,memory_used_mib,power_draw_w,power_limit_w,temperature_c,free_disk_gib" >"$RESOURCE_LOG"
    while true; do
        now=$(TZ=Europe/Kyiv date --iso-8601=seconds)
        available=$(free_gib)
        nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used,power.draw,power.limit,temperature.gpu \
            --format=csv,noheader,nounits | sed "s/$/,${available}/; s/^/${now},/" >>"$RESOURCE_LOG" || true
        if (( available < MIN_FREE_GIB )); then
            echo "Free disk space is less than ${MIN_FREE_GIB} GiB." >&2
            kill -TERM "$runner_pid"
            exit 2
        fi
        sleep 15
    done
}

"$DATA_PYTHON" "${ROOT}/scripts/build_enhancement_backend_review.py" prepare \
    --output "$OUTPUT" --report "$REPORT" --count-per-dataset 10

monitor_resources &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

declare -a worker_pids
for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$shard]}" \
    "$PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
        --backend sidon_deess_only --selection "$SELECTION" --output "$OUTPUT" \
        --device cuda:0 --model-cache "$MODEL_CACHE" \
        --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
        --num-shards 2 --shard-index "$shard" \
        --summary "${OUTPUT}/backend_sidon_deess_only_shard_${shard}.json" \
        >"${ROOT}/reports/sidon_deess_listening_shard_${shard}.log" 2>&1 &
    worker_pids[$shard]=$!
done

status=0
set +e
for shard in 0 1; do
    wait "${worker_pids[$shard]}" || status=1
done
set -e
if (( status )); then
    echo "A Sidon and de-essing shard failed." >&2
    exit 1
fi

"$DATA_PYTHON" "${ROOT}/scripts/build_enhancement_backend_review.py" finalize \
    --output "$OUTPUT" --report "$REPORT"
