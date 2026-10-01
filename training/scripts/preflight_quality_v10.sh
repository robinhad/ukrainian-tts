#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
source "${ROOT}/activate.sh"
python -m training.scripts.verify_local_frontend
export UKTTS_ESPEAK_DATA_HASH_FILE="${ROOT}/quality_runs/v10/frontend/ESPEAK_NG_DATA_HASH"
python -m training.scripts.build_manifest \
  --records training/data/quality_v10_raw/canonical_records.jsonl \
  --output-dir training/data/quality_v10_raw/frontend \
  --cache training/data/quality_v10_raw/frontend.sqlite \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
python -m training.scripts.validate_dataset \
  --manifest training/data/quality_v10_raw/frontend/all.parquet \
  --report training/quality_runs/v10/raw_frontend_validation.json
