#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REGISTRY="${ROOT}/conf/expanded_v3_sources.yaml"
DATA_ROOT=${DATA_ROOT:-"${ROOT}/data/expanded_v3"}
BATCH_SHARDS=${VOA_BATCH_SHARDS:-10}
TOTAL_SHARDS=${VOA_TOTAL_SHARDS:-195}
START_SHARD=${VOA_START_SHARD:-0}
GPU_IDS_CSV=${VOA_GPU_IDS:-0,1}
SOURCE_ROOT="${DATA_ROOT}/sources/voa_ukr_user_grant"
PSEUDO_ROOT="${DATA_ROOT}/sources/pseudo_uk"
OUTPUT_RECORDS="${PSEUDO_ROOT}/records.jsonl"
MODEL_CACHE="${ROOT}/vendor/nemo-cache"
LOG_ROOT="${ROOT}/logs/expanded-v3/voa"
PYTHON="${ROOT}/.venv/bin/python"
export PYTHONPATH="$(cd "${ROOT}/.." && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${ROOT}/.venv-nemo/bin/python" ]]; then
    "${ROOT}/scripts/bootstrap_nemo_env.sh"
fi
mkdir -p "$SOURCE_ROOT/batches" "$SOURCE_ROOT/markers" "$PSEUDO_ROOT/audio" "$LOG_ROOT"
if (( START_SHARD == 0 )) && [[ -s "${SOURCE_ROOT}/markers/000-000.json" ]]; then
    START_SHARD=1
fi

IFS=',' read -r -a GPU_IDS <<<"$GPU_IDS_CSV"
WORKER_COUNT=${#GPU_IDS[@]}
if (( WORKER_COUNT < 1 )); then
    echo "VOA_GPU_IDS must contain at least one GPU identifier." >&2
    exit 1
fi

run_worker() {
    local worker_index=$1
    local gpu_id=${GPU_IDS[$worker_index]}
    local worker_records="${PSEUDO_ROOT}/records-gpu${gpu_id}.jsonl"
    local batch_index=0
    local start count tag marker records_name report_name records report
    local process_report process_log
    for ((start = START_SHARD; start < TOTAL_SHARDS; start += BATCH_SHARDS)); do
        if (( batch_index % WORKER_COUNT != worker_index )); then
            batch_index=$((batch_index + 1))
            continue
        fi
        count=$BATCH_SHARDS
        if (( start + count > TOTAL_SHARDS )); then
            count=$((TOTAL_SHARDS - start))
        fi
        tag=$(printf '%03d-%03d' "$start" "$((start + count - 1))")
        marker="${SOURCE_ROOT}/markers/${tag}.json"
        if [[ -s "$marker" ]]; then
            batch_index=$((batch_index + 1))
            continue
        fi
        records_name="batches/unlabeled-${tag}.jsonl"
        report_name="batches/collect-${tag}.json"
        "$PYTHON" "${ROOT}/scripts/collect_hf_unlabeled.py" \
            --registry "$REGISTRY" --source-id voa_ukr_user_grant \
            --output-root "${DATA_ROOT}/sources" \
            --shard-start "$start" --shard-count "$count" \
            --records-name "$records_name" --report-name "$report_name"
        records="${SOURCE_ROOT}/${records_name}"
        report="${SOURCE_ROOT}/${report_name}"
        process_report="${SOURCE_ROOT}/batches/process-${tag}.json"
        process_log="${LOG_ROOT}/process-${tag}.log"
        if ! CUDA_VISIBLE_DEVICES="$gpu_id" \
            "${ROOT}/.venv-nemo/bin/python" "${ROOT}/scripts/process_unlabeled.py" \
            --registry "$REGISTRY" --records "$records" \
            --output-root "${PSEUDO_ROOT}/audio" \
            --output-records "$worker_records" --report "$process_report" \
            --model-cache "$MODEL_CACHE" --append --delete-source-audio \
            >"$process_log" 2>&1; then
            tail -n 120 "$process_log"
            exit 1
        fi
        sed -n '1,120p' "$process_report"
        "$PYTHON" "${ROOT}/scripts/cleanup_unlabeled_cache.py" \
            --records "$records" --collect-report "$report" --marker "$marker"
        "$PYTHON" "${ROOT}/scripts/source_policy.py" \
            --registry "$REGISTRY" --workspace "$ROOT"
        batch_index=$((batch_index + 1))
    done
}

terminate_tree() {
    local parent_pid=$1
    local child_pid
    while read -r child_pid; do
        [[ -n "$child_pid" ]] && terminate_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" || true)
    kill -TERM "$parent_pid" 2>/dev/null || true
}

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    run_worker "$worker_index" &
    pids+=("$!")
done
worker_status=0
for ((finished = 0; finished < ${#pids[@]}; finished++)); do
    if ! wait -n; then
        worker_status=1
        break
    fi
done
if (( worker_status != 0 )); then
    for pid in "${pids[@]}"; do
        terminate_tree "$pid"
    done
    wait || true
    exit "$worker_status"
fi

shopt -s nullglob
record_inputs=("$PSEUDO_ROOT"/records*.jsonl)
shopt -u nullglob
if (( ${#record_inputs[@]} == 0 )); then
    echo "No pseudo-label record files exist." >&2
    exit 1
fi
"$PYTHON" "${ROOT}/scripts/merge_jsonl_records.py" \
    --inputs "${record_inputs[@]}" --output "$OUTPUT_RECORDS"
