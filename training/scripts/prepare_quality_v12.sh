#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
source "${ROOT}/activate.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
export GPU_COUNT=1
export TRAIN_SET=quality_v12_train VALID_SET=quality_v12_dev TEST_SETS=quality_v12_eval
export DATA_SETS="$TRAIN_SET $VALID_SET $TEST_SETS"
export MANIFEST_DIR="${ROOT}/data/quality_v12/manifests"
export DUMP_DIR="${ROOT}/dump_quality_v12" EXP_DIR="${ROOT}/exp_quality_v12"
python -m training.scripts.audit_quality_v12_training --stage inputs
if [[ -f "${ROOT}/quality_runs/v12/prepared.json" ]]; then
  python -m training.scripts.audit_quality_v12_training --stage prepared
  exit 0
fi
python -m training.scripts.verify_local_frontend
python -m training.scripts.prepare_ecapa_for_training
python -m training.scripts.build_manifest --records "${ROOT}/data/quality_v12/records.jsonl" \
  --output-dir "$MANIFEST_DIR" --cache "${ROOT}/data/quality_v12/frontend.sqlite" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
python -m training.scripts.validate_dataset --manifest "$MANIFEST_DIR/all.parquet" \
  --report "${ROOT}/quality_runs/v12/dataset_validation.json"
RECIPE="${ROOT}/espnet_recipe/run_expanded_v6_sidon_deess_novoa.sh"
"$RECIPE" --stage 1 --stop_stage 1
python -m training.scripts.prepare_direct_pcm24_raw_data --manifest-dir "$MANIFEST_DIR" \
  --kaldi-data-root "${ROOT}/espnet_recipe/data" --dump-dir "$DUMP_DIR" \
  --train-set "$TRAIN_SET" --valid-set "$VALID_SET" --test-set "$TEST_SETS"
# Isolate these three sets. Every embedding uses the exact processed waveform
# supplied to training, including the unfiltered held-out population.
python - <<'PY'
from pathlib import Path
import shutil
from training.quality.common import write_json
root = Path('training/data/quality_v12/embedding_inputs')
for split in ['train', 'dev', 'eval']:
    name = 'quality_v12_' + split
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ['wav.scp', 'utt2spk', 'spk2utt']:
        shutil.copyfile(Path('training/espnet_recipe/data') / name / filename, destination / filename)
write_json('training/quality_runs/v12/embeddings.json', {'policy': 'processed_audio_only'})
PY
MANIFEST="$MANIFEST_DIR/all.parquet" RAW_MANIFEST="$MANIFEST_DIR/all.parquet" \
  SKIP_HYBRID_PREPARE=true KALDI_ROOT="${ROOT}/data/quality_v12/embedding_inputs" \
  GPU_UUIDS="${CUDA_VISIBLE_DEVICES:-0}" REPORT="${ROOT}/quality_runs/v12/embeddings.json" \
  "${ROOT}/scripts/extract_hybrid_embeddings.sh"
MAX_DURATION=$(python -c 'import pandas as pd; print(pd.read_parquet("training/data/quality_v12/manifests/all.parquet").duration.max() + 0.1)')
"$RECIPE" --stage 4 --stop_stage 6 --nj "${SLURM_CPUS_PER_TASK:-8}" \
  --max_wav_duration "$MAX_DURATION" --write_collected_feats true
python -m training.scripts.audit_quality_v12_training --stage seal
