#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT="${ROOT}/data/expanded_v6_sidon_deess_novoa"
DUMP_DIR="${ROOT}/dump_expanded_v6_sidon_deess_novoa"
EXP_DIR="${ROOT}/exp_expanded_v6_sidon_deess_novoa"
MANIFEST_DIR="${DATA_ROOT}/manifests"
RAW_MANIFEST="${DATA_ROOT}/pre_enhancement_embedding_manifest.parquet"
KALDI_ROOT="${DATA_ROOT}/hybrid_embedding_data"
HYBRID_MANIFEST="${DATA_ROOT}/hybrid_manifest/all.parquet"
CLEAN_KALDI_ROOT="${DATA_ROOT}/clean_embedding_data"
CLEAN_DUMP="${ROOT}/dump_expanded_v6_sidon_clean_only"
REPORTS="${ROOT}/reports"
PREFIX=expanded_v6_sidon_deess_novoa_full
GPU_UUIDS=${GPU_UUIDS:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)}

source "${ROOT}/activate.sh"
python "${ROOT}/scripts/check_resources.py" --mode full --require-torch \
    --gpu-uuids "$GPU_UUIDS" --workspace "$ROOT" \
    --output "${REPORTS}/resource_usage_expanded_v6_sidon_deess_novoa.jsonl"
python "${ROOT}/scripts/build_expanded_v6_sidon_deess_novoa.py" \
    --base-manifest "${ROOT}/data/expanded_v5_enhanced_novoa/manifests/all.parquet" \
    --input-audio-manifest "${ROOT}/data/expanded_v5_enhanced_novoa/pre_enhancement_embedding_manifest.parquet" \
    --processing-results "${REPORTS}/expanded_v6_sidon_deess_novoa_processing/shard-0.jsonl" "${REPORTS}/expanded_v6_sidon_deess_novoa_processing/shard-1.jsonl" \
    --output-dir "$MANIFEST_DIR" --raw-manifest "$RAW_MANIFEST" \
    --report "${REPORTS}/${PREFIX}_dataset.json"
python "${ROOT}/scripts/validate_dataset.py" --manifest "${MANIFEST_DIR}/all.parquet" \
    --report "${REPORTS}/${PREFIX}_validation.json"
python "${ROOT}/scripts/analyze_dataset.py" --manifest "${MANIFEST_DIR}/all.parquet" \
    --output "${REPORTS}/${PREFIX}_analysis.json"

export MANIFEST_DIR DUMP_DIR EXP_DIR
export TRAIN_SET=expanded_v6_sidon_deess_novoa_train
export VALID_SET=expanded_v6_sidon_deess_novoa_dev
export TEST_SETS=expanded_v6_sidon_deess_novoa_eval
export DATA_SETS="${TRAIN_SET} ${VALID_SET} ${TEST_SETS}"
export GPU_COUNT=2 CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 1 --stop_stage 2

python "${ROOT}/scripts/prepare_hybrid_embeddings.py" \
    --manifest "${MANIFEST_DIR}/all.parquet" --raw-manifest "$RAW_MANIFEST" \
    --output-manifest "$HYBRID_MANIFEST" --kaldi-root "$KALDI_ROOT" \
    --report "${REPORTS}/${PREFIX}_hybrid_assignment.json"
python "${ROOT}/scripts/prepare_v6_partial_embeddings.py" prepare-clean \
    --hybrid-manifest "$HYBRID_MANIFEST" \
    --v5-hybrid-manifest "${ROOT}/data/expanded_v5_enhanced_novoa/hybrid_manifest/all.parquet" \
    --clean-kaldi-root "$CLEAN_KALDI_ROOT" \
    --report "${REPORTS}/${PREFIX}_clean_embedding_input.json"
MANIFEST="$HYBRID_MANIFEST" RAW_MANIFEST="$RAW_MANIFEST" KALDI_ROOT="$CLEAN_KALDI_ROOT" \
    DUMP_DIR="$CLEAN_DUMP" REPORT="${REPORTS}/${PREFIX}_clean_embedding_input.json" \
    OUTPUT_MANIFEST="$HYBRID_MANIFEST" GPU_UUIDS="$GPU_UUIDS" SKIP_HYBRID_PREPARE=true \
    "${ROOT}/scripts/extract_hybrid_embeddings.sh"
python "${ROOT}/scripts/prepare_v6_partial_embeddings.py" merge \
    --hybrid-manifest "$HYBRID_MANIFEST" \
    --v5-hybrid-manifest "${ROOT}/data/expanded_v5_enhanced_novoa/hybrid_manifest/all.parquet" \
    --v5-xvector-root "${ROOT}/dump_expanded_v5_enhanced_novoa/xvector" \
    --clean-xvector-root "${CLEAN_DUMP}/xvector" --output-root "${DUMP_DIR}/xvector" \
    --report "${REPORTS}/${PREFIX}_hybrid_embeddings.json"

"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 4 --stop_stage 4 --nj 10
TOKEN_DIR="${DUMP_DIR}/token_list/phn_espeak_ng_ukrainian"
mkdir -p "$TOKEN_DIR"
cp "${ROOT}/dump_expanded_v3/token_list/phn_espeak_ng_ukrainian/tokens.txt" "$TOKEN_DIR/tokens.txt"
"${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh" --stage 6 --stop_stage 6 --nj "${STATS_NJ:-20}"
python "${ROOT}/scripts/audit_expanded_v6_sidon_deess_novoa_readiness.py" \
    --workspace "$ROOT" --data-root "$DATA_ROOT" --dump-dir "$DUMP_DIR" \
    --exp-dir "$EXP_DIR" --reports-root "$REPORTS" \
    --output "${REPORTS}/${PREFIX}_scale_readiness.json"
echo "The v6 Sidon and de-essing preparation is complete."
