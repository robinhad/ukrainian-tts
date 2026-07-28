#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-"${ROOT}/data/expanded_v3"}
VOA_PIPELINE_VERSION=${VOA_PIPELINE_VERSION:-v4-c050-d20-g050-defer-long}
GPU_IDS_CSV=${VOA_GPU_IDS:-0,1}
PSEUDO_ROOT="${DATA_ROOT}/sources/pseudo_uk_${VOA_PIPELINE_VERSION}"
DEFERRED_RECORDS="${PSEUDO_ROOT}/deferred_too_long/records.jsonl"
RECOVERED_ROOT="${PSEUDO_ROOT}/recovered_long"
MODEL_CACHE="${ROOT}/vendor/nemo-cache"
PYTHON="${ROOT}/.venv/bin/python"
NEMO_PYTHON="${ROOT}/.venv-nemo/bin/python"
REGISTRY="${ROOT}/conf/expanded_v3_sources.yaml"
export PYTHONPATH="$(cd "${ROOT}/.." && pwd)${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "$DEFERRED_RECORDS" ]]; then
    echo "The merged deferred-long manifest does not exist: ${DEFERRED_RECORDS}" >&2
    exit 1
fi
mkdir -p "$RECOVERED_ROOT/audio" "$RECOVERED_ROOT/logs"
IFS=, read -r -a GPU_IDS <<<"$GPU_IDS_CSV"
NUM_SHARDS=${#GPU_IDS[@]}
if (( NUM_SHARDS < 1 )); then
    echo "VOA_GPU_IDS must contain at least one GPU identifier." >&2
    exit 2
fi

pids=()
for shard in "${!GPU_IDS[@]}"; do
    gpu=${GPU_IDS[$shard]}
    CUDA_VISIBLE_DEVICES="$gpu" \
        "$NEMO_PYTHON" "${ROOT}/scripts/process_deferred_long.py" \
        --registry "$REGISTRY" \
        --records "$DEFERRED_RECORDS" \
        --output-root "$RECOVERED_ROOT/audio" \
        --output-records "$RECOVERED_ROOT/records-gpu${gpu}.jsonl" \
        --progress "$RECOVERED_ROOT/progress-gpu${gpu}.jsonl" \
        --report "$RECOVERED_ROOT/report-gpu${gpu}.json" \
        --model-cache "$MODEL_CACHE" \
        --shard-index "$shard" --num-shards "$NUM_SHARDS" \
        >"$RECOVERED_ROOT/logs/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo "At least one deferred-long worker failed." >&2
    exit 1
fi

shopt -s nullglob
recovered_inputs=("$RECOVERED_ROOT"/records-gpu*.jsonl)
shopt -u nullglob
if (( ${#recovered_inputs[@]} == 0 )); then
    echo "No recovered-long record files exist." >&2
    exit 1
fi
"$PYTHON" "${ROOT}/scripts/merge_jsonl_records.py" \
    --inputs "${recovered_inputs[@]}" \
    --output "$RECOVERED_ROOT/records.jsonl"
"$PYTHON" "${ROOT}/scripts/merge_jsonl_records.py" \
    --inputs "$PSEUDO_ROOT/records.jsonl" "$RECOVERED_ROOT/records.jsonl" \
    --output "$PSEUDO_ROOT/records_with_recovered_long.jsonl"
