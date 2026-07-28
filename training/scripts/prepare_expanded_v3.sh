#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${1:-smoke}
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
    echo "Usage: prepare_expanded_v3.sh [smoke|full]" >&2
    exit 2
fi
REGISTRY="${ROOT}/conf/expanded_v3_sources.yaml"
DATA_ROOT="${ROOT}/data/expanded_v3${MODE/smoke/_smoke}"
if [[ "$MODE" == full ]]; then
    DATA_ROOT="${ROOT}/data/expanded_v3"
fi
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
MINIMUM_SEGMENT_SECONDS=2
MAXIMUM_SEGMENT_SECONDS=20
VOA_PIPELINE_VERSION=${VOA_PIPELINE_VERSION:-v4-c050-d20-g050-defer-long}
SOURCE_LIMIT=()
LADA_LIMIT=()
if [[ "$MODE" == smoke ]]; then
    SOURCE_LIMIT=(--limit 120 --maximum-shards 1)
    LADA_LIMIT=(--limit 120)
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
source "${ROOT}/activate.sh"
python "${ROOT}/scripts/source_policy.py" \
    --registry "$REGISTRY" --workspace "$ROOT" \
    --report "${ROOT}/reports/expanded_v3_source_policy.json"
python "${ROOT}/scripts/check_resources.py" \
    --mode "$([[ "$MODE" == smoke ]] && echo smoke || echo full)" \
    --require-torch --gpu-uuids "$GPU_UUIDS" \
    --workspace "$ROOT" --output "${ROOT}/reports/resource_usage_expanded_v3.jsonl"

mkdir -p "${DATA_ROOT}/sources" "${DATA_ROOT}/records" "${DATA_ROOT}/logs"
if [[ "$MODE" == smoke ]]; then
    python "${ROOT}/scripts/export_existing_lada.py" \
        --manifest "${ROOT}/data/full_trimmed/manifests/all.parquet" \
        --output "${DATA_ROOT}/sources/opentts_lada/records.jsonl" \
        "${LADA_LIMIT[@]}"
else
    python "${ROOT}/scripts/export_available_multispeaker_sources.py" \
        --source-records "${ROOT}/data/multispeaker_full/source_records.jsonl" \
        --output-root "${DATA_ROOT}/sources"
fi

LABELED_SOURCES=(opentts_tetiana opentts_mykyta tg_voices_uk ukr_dialects)
if [[ "$MODE" == full ]]; then
    LABELED_SOURCES+=(fleurs_uk)
fi
for source_id in "${LABELED_SOURCES[@]}"; do
    python "${ROOT}/scripts/ingest_hf_parquet.py" \
        --registry "$REGISTRY" --source-id "$source_id" \
        --output-root "${DATA_ROOT}/sources" --resume \
        "${SOURCE_LIMIT[@]}"
done

if [[ "$MODE" == full ]]; then
    python "${ROOT}/scripts/collect_ua_ser.py" \
        --registry "$REGISTRY" --output-root "${DATA_ROOT}/sources"
    DATA_ROOT="$DATA_ROOT" VOA_PIPELINE_VERSION="$VOA_PIPELINE_VERSION" \
        "${ROOT}/scripts/process_voa_streaming.sh"
fi

mapfile -t SOURCE_RECORDS < <(
    find "${DATA_ROOT}/sources" -name records.jsonl -type f \
        ! -path "${DATA_ROOT}/sources/pseudo_uk*/records.jsonl" | sort
)
if [[ "$MODE" == full ]]; then
    SOURCE_RECORDS+=(
        "${DATA_ROOT}/sources/pseudo_uk_${VOA_PIPELINE_VERSION}/records.jsonl"
    )
fi
MERGE_ARGS=()
if [[ "$MODE" == smoke ]]; then
    MERGE_ARGS=(--smoke --smoke-maximum 360)
fi
python "${ROOT}/scripts/merge_expanded_records.py" \
    --registry "$REGISTRY" --records "${SOURCE_RECORDS[@]}" \
    --output "${DATA_ROOT}/records/source_records.jsonl" \
    --report "${ROOT}/reports/expanded_v3_${MODE}_sources.json" \
    "${MERGE_ARGS[@]}"

SOURCE_MANIFEST_DIR="${DATA_ROOT}/source_manifests"
python "${ROOT}/scripts/prepare_audio.py" \
    --records "${DATA_ROOT}/records/source_records.jsonl" \
    --output-root "${DATA_ROOT}/canonical_trimmed_24k" \
    --output-records "${DATA_ROOT}/records/canonical_records.jsonl" \
    --trim-silence --workers 16 \
    --minimum-duration "$MINIMUM_SEGMENT_SECONDS" \
    --maximum-duration "$MAXIMUM_SEGMENT_SECONDS"
python "${ROOT}/scripts/build_manifest.py" \
    --records "${DATA_ROOT}/records/canonical_records.jsonl" \
    --output-dir "$SOURCE_MANIFEST_DIR" \
    --cache "${DATA_ROOT}/frontend_cache.sqlite" --workers 16

IFS=, read -r -a GPUS <<< "$GPU_UUIDS"
NUM_SHARDS=$((WORKERS_PER_GPU * ${#GPUS[@]}))
mkdir -p "${DATA_ROOT}/enhanced_records" "${DATA_ROOT}/processed_24k"
if [[ "$MODE" == full ]]; then
    python "${ROOT}/scripts/seed_reused_enhanced_records.py" \
        --source-manifest "${SOURCE_MANIFEST_DIR}/all.parquet" \
        --clean-manifest \
            "${ROOT}/data/multispeaker_enhanced_v2/manifests/all.parquet" \
        --output-records-dir "${DATA_ROOT}/enhanced_records" \
        --num-shards "$NUM_SHARDS"
fi
run_shard() {
    local shard=$1
    local gpu=$2
    while true; do
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 \
            python "${ROOT}/scripts/preprocess_enhanced_audio.py" \
            --manifest "${SOURCE_MANIFEST_DIR}/all.parquet" \
            --output-root "${DATA_ROOT}/processed_24k" \
            --output-records "${DATA_ROOT}/enhanced_records/records-${shard}.jsonl" \
            --model-cache "${ROOT}/vendor/deepfilternet-cache" \
            --shard-index "$shard" --num-shards "$NUM_SHARDS" --resume \
            --allow-failures --max-new-records 300
        status=$?
        set -e
        if (( status == 75 )); then
            continue
        fi
        return "$status"
    done
}

pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    run_shard "$shard" "${GPUS[$((shard % ${#GPUS[@]}))]}" \
        > "${DATA_ROOT}/logs/preprocess-${shard}.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if (( failed )); then
    echo "At least one enhanced-audio worker failed." >&2
    exit 1
fi

python "${ROOT}/scripts/merge_enhanced_manifests.py" \
    --source-manifest "${SOURCE_MANIFEST_DIR}/all.parquet" \
    --records-dir "${DATA_ROOT}/enhanced_records" \
    --output-dir "${DATA_ROOT}/enhanced_intermediate" \
    --output-records "${DATA_ROOT}/records/enhanced_records.jsonl" \
    --report "${ROOT}/reports/expanded_v3_${MODE}_enhancement.json" \
    --minimum-duration "$MINIMUM_SEGMENT_SECONDS" \
    --maximum-duration "$MAXIMUM_SEGMENT_SECONDS"
python "${ROOT}/scripts/build_manifest.py" \
    --records "${DATA_ROOT}/records/enhanced_records.jsonl" \
    --output-dir "${DATA_ROOT}/manifests" \
    --cache "${DATA_ROOT}/frontend_cache.sqlite" --workers 16
python "${ROOT}/scripts/validate_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --report "${ROOT}/reports/expanded_v3_${MODE}_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" \
    --manifest "${DATA_ROOT}/manifests/all.parquet" \
    --output "${ROOT}/reports/expanded_v3_${MODE}_analysis.json"

if [[ "$MODE" == smoke ]]; then
    TRAIN_SET=expanded_v3_smoke_train
    VALID_SET=expanded_v3_smoke_dev
    TEST_SETS=expanded_v3_smoke_eval
    DUMP_DIR="${ROOT}/dump_expanded_v3_smoke"
    EXP_DIR="${ROOT}/exp_expanded_v3_smoke"
else
    TRAIN_SET=expanded_v3_train
    VALID_SET=expanded_v3_dev
    TEST_SETS=expanded_v3_eval
    DUMP_DIR="${ROOT}/dump_expanded_v3"
    EXP_DIR="${ROOT}/exp_expanded_v3"
fi
export MANIFEST_DIR="${DATA_ROOT}/manifests"
export TRAIN_SET VALID_SET TEST_SETS DUMP_DIR EXP_DIR
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export GPU_COUNT=${#GPUS[@]}
"${ROOT}/espnet_recipe/run_expanded_v3.sh" --stage 1 --stop_stage 2

MANIFEST="${DATA_ROOT}/manifests/all.parquet" \
RAW_MANIFEST="${SOURCE_MANIFEST_DIR}/all.parquet" \
OUTPUT_MANIFEST="${DATA_ROOT}/hybrid_manifest/all.parquet" \
KALDI_ROOT="${DATA_ROOT}/hybrid_embedding_data" \
DUMP_DIR="$DUMP_DIR" \
REPORT="${ROOT}/reports/expanded_v3_${MODE}_hybrid_embeddings.json" \
GPU_UUIDS="$GPU_UUIDS" \
    "${ROOT}/scripts/extract_hybrid_embeddings.sh"

"${ROOT}/espnet_recipe/run_expanded_v3.sh" --stage 4 --stop_stage 6
set +e
python "${ROOT}/scripts/audit_expanded_v3_readiness.py" \
    --registry "$REGISTRY" --workspace "$ROOT" --data-root "$DATA_ROOT" \
    --reports-root "${ROOT}/reports" --mode "$MODE" \
    --output "${ROOT}/reports/expanded_v3_${MODE}_scale_readiness.json"
readiness_status=$?
set -e
if [[ "$MODE" == full && $readiness_status -ne 0 ]]; then
    echo "The full expanded-v3 readiness gate is not PASS." >&2
    exit "$readiness_status"
fi
echo "The expanded-v3 ${MODE} data preparation is complete."
