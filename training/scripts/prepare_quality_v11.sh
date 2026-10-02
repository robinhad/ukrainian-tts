#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
source "${ROOT}/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m training.scripts.verify_local_frontend
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
python -m training.scripts.prepare_ecapa_for_training
export GPU_COUNT=1
export TRAIN_SET=quality_v11_train VALID_SET=quality_v11_dev TEST_SETS=quality_v11_eval
export DATA_SETS="$TRAIN_SET $VALID_SET $TEST_SETS"
export MANIFEST_DIR="${ROOT}/data/quality_v11/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v11" EXP_DIR="${ROOT}/exp_quality_v11"
python - <<'PY'
from pathlib import Path
from training.quality.common import read_rows
from training.scripts.select_quality_v11 import write_rows
root = Path('training/data/quality_v11')
selected = read_rows(root / 'records.jsonl')
# Preserve the fixed evaluation population; quality filtering applies to train/dev.
records = [r for r in selected if not r['split'].endswith('_eval')]
for row in read_rows('training/data/quality_v10_raw/all.parquet'):
    if row['split'] == 'quality_v10_eval':
        records.append({**row, 'split': 'quality_v11_eval'})
write_rows(root / 'training_records.jsonl', records)
PY
python -m training.scripts.build_manifest --records "${ROOT}/data/quality_v11/training_records.jsonl" \
  --output-dir "$MANIFEST_DIR" --cache "${ROOT}/data/quality_v11/frontend.sqlite" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
python -m training.scripts.validate_dataset --manifest "$MANIFEST_DIR/all.parquet" \
  --report "${ROOT}/quality_runs/v11/dataset_validation.json"
RECIPE="${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh"
"$RECIPE" --stage 1 --stop_stage 1
python -m training.scripts.prepare_direct_pcm24_raw_data --manifest-dir "$MANIFEST_DIR" \
  --kaldi-data-root "${ROOT}/espnet_recipe/data" --dump-dir "$DUMP_DIR" \
  --train-set "$TRAIN_SET" --valid-set "$VALID_SET" --test-set "$TEST_SETS"
MANIFEST="$MANIFEST_DIR/all.parquet" RAW_MANIFEST="${ROOT}/data/quality_v10_raw/all.parquet" \
  KALDI_ROOT="${ROOT}/data/quality_v11/hybrid" GPU_UUIDS="${CUDA_VISIBLE_DEVICES:-0}" \
  REPORT="${ROOT}/quality_runs/v11/embeddings.json" \
  OUTPUT_MANIFEST="${ROOT}/data/quality_v11/hybrid_manifest/all.parquet" \
  "${ROOT}/scripts/extract_hybrid_embeddings.sh"
MAX_DURATION=$(python -c 'import pandas as pd; print(pd.read_parquet("training/data/quality_v11/manifests/all.parquet").duration.max() + 0.1)')
"$RECIPE" --stage 4 --stop_stage 6 --nj "${SLURM_CPUS_PER_TASK:-8}" \
  --max_wav_duration "$MAX_DURATION" --write_collected_feats true
