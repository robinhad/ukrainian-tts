#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-"${ROOT}/eval/generated/per_dataset_train_audio_enhancement_review_v2"}
REPORT=${REPORT:-"${ROOT}/reports/per_dataset_train_audio_enhancement_review_v2.json"}
SELECTION="${OUTPUT}/selection.jsonl"
MODEL_CACHE=${MODEL_CACHE:-"${ROOT}/vendor/audio-enhancement-review-models"}
REVIEW_PYTHON="${ROOT}/.venv-audio-review/bin/python"
TRAINING_PYTHON="${ROOT}/.venv/bin/python"
RESOURCE_LOG="${ROOT}/reports/enhancement_backend_review_resources.csv"
MIN_FREE_GIB=${MIN_FREE_GIB:-30}
POSTPROCESS_CONFIG=${POSTPROCESS_CONFIG:-"${ROOT}/conf/audio_postprocess.yaml"}
FFMPEG_BINARY=${FFMPEG_BINARY:-auto}
runner_pid=$$

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
        sleep 30
    done
}

monitor_resources &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

"$TRAINING_PYTHON" "${ROOT}/scripts/build_enhancement_backend_review.py" prepare \
    --output "$OUTPUT" --report "$REPORT" --count-per-dataset 10

"$TRAINING_PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
    --backend deepfilternet3 --selection "$SELECTION" --output "$OUTPUT" \
    --device cuda:0 --model-cache "${ROOT}/vendor/deepfilternet-cache" \
    --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
    >"${ROOT}/reports/enhancement_review_deepfilternet3.log" 2>&1 &
dfn_pid=$!

"$REVIEW_PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
    --backend resemble --selection "$SELECTION" --output "$OUTPUT" \
    --device cuda:1 --model-cache "${MODEL_CACHE}/resemble" \
    --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
    >"${ROOT}/reports/enhancement_review_resemble.log" 2>&1 &
resemble_pid=$!

set +e
wait "$dfn_pid"; dfn_status=$?
"$REVIEW_PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
    --backend sidon --selection "$SELECTION" --output "$OUTPUT" \
    --device cuda:0 --model-cache "${MODEL_CACHE}/sidon" \
    --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
    >"${ROOT}/reports/enhancement_review_sidon.log" 2>&1
sidon_status=$?
"$REVIEW_PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
    --backend sidon_deess_only --selection "$SELECTION" --output "$OUTPUT" \
    --device cuda:0 --model-cache "${MODEL_CACHE}/sidon" \
    --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
    >"${ROOT}/reports/enhancement_review_sidon_deess_only.log" 2>&1
sidon_deess_status=$?
set -e

set +e
wait "$resemble_pid"; resemble_status=$?
"$REVIEW_PYTHON" "${ROOT}/scripts/run_enhancement_review_backend.py" \
    --backend mossformer2 --selection "$SELECTION" --output "$OUTPUT" \
    --device cuda:1 --model-cache "${MODEL_CACHE}/mossformer2" \
    --postprocess-config "$POSTPROCESS_CONFIG" --ffmpeg "$FFMPEG_BINARY" \
    >"${ROOT}/reports/enhancement_review_mossformer2.log" 2>&1
moss_status=$?
set -e

if ((
    dfn_status || sidon_status || sidon_deess_status
    || resemble_status || moss_status
)); then
    echo "One or more enhancement backends failed." >&2
    echo "dfn=${dfn_status} sidon=${sidon_status} sidon_deess=${sidon_deess_status}" >&2
    echo "resemble=${resemble_status} moss=${moss_status}" >&2
    exit 1
fi

"$TRAINING_PYTHON" "${ROOT}/scripts/build_enhancement_backend_review.py" finalize \
    --output "$OUTPUT" --report "$REPORT"
