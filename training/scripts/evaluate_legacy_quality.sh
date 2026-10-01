#!/usr/bin/env bash
# Reproduce the historical processing recipe on the fixed quality panel.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside a SLURM GPU allocation}"
source training/activate.sh
ROOT=training/quality_runs/v10
PANEL=${PANEL:-"$ROOT/refinement/processing.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/refinement/trials/legacy_v8_exact"}
QUALITY_CONFIG=${QUALITY_CONFIG:-"$ROOT/quality_config.yaml"}
ORIGINAL_REPORT=${ORIGINAL_REPORT:-"$ROOT/refinement/references/original"}
CONTROL_REPORT=${CONTROL_REPORT:-"$ROOT/refinement/references/identity_wet1"}
PROCESS_WORKERS=${PROCESS_WORKERS:-2}
if [[ ! "$PROCESS_WORKERS" =~ ^[1-8]$ ]]; then
  echo 'PROCESS_WORKERS must be between 1 and 8.' >&2
  exit 2
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$OUTPUT_DIR/wav"
python - "$PANEL" "$OUTPUT_DIR" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from training.quality.common import file_hash, read_rows, write_json
from training.scripts.run_fair_enhancement_comparison import RNNOISE_MODEL_SHA256

panel, output = map(Path, sys.argv[1:])
rows = read_rows(panel)
counts = Counter(r['source_id'] for r in rows)
if (not counts or len(set(counts.values())) != 1 or max(counts.values()) > 100
        or any('voa' in name.lower() for name in counts)):
    raise ValueError('Use a balanced non-VOA panel with at most 100 files per source')
if len({r['utterance_id'] for r in rows}) != len(rows):
    raise ValueError('Duplicate panel IDs')
inputs = []
for row in rows:
    source = row['processing_audio_path']
    if (file_hash(source) != row['processing_input_sha256']
            or file_hash(row['audio_path']) != row['reference_sha256']):
        raise ValueError('Frozen panel audio changed')
    inputs.append({'utterance_id': row['utterance_id'], 'audio_path': source})
archive = Path('training/vendor/rnnoise') / f'rnnoise_data-{RNNOISE_MODEL_SHA256}.tar.gz'
if file_hash(archive) != RNNOISE_MODEL_SHA256:
    raise ValueError('RNNoise model archive checksum mismatch')
paths = ['training/scripts/preprocess_training_cascade_audio.py',
         'training/scripts/run_fair_enhancement_comparison.py',
         'training/scripts/run_enhancement_review_backend.py',
         'training/audio_enhancement/fair_comparison.py',
         'training/audio_enhancement/postprocess.py',
         'training/vendor/rnnoise/examples/rnnoise_demo']
provenance = {'panel_sha256': file_hash(panel), 'files': len(rows),
              'source_counts': dict(counts), 'rnnoise_model_sha256': RNNOISE_MODEL_SHA256,
              'implementation_sha256': {path: file_hash(path) for path in paths},
              'recipe': 'historical cascade including production subgate fallbacks',
              'device_policy': 'GPU-capable neural stages on GPU; RNNoise on CPU'}
prior = output / 'historical_provenance.json'
if prior.exists() and json.loads(prior.read_text()) != provenance:
    raise ValueError('Historical run provenance changed; use a new output directory')
write_json(prior, provenance)
pd.DataFrame(inputs).to_parquet(output / 'inputs.parquet', index=False)
PY
pids=()
for ((shard=0; shard<PROCESS_WORKERS; shard++)); do
  python -m training.scripts.preprocess_training_cascade_audio \
    --manifest "$OUTPUT_DIR/inputs.parquet" --input-audio-manifest "$OUTPUT_DIR/inputs.parquet" \
    --output-root "$OUTPUT_DIR/wav" --output-results "$OUTPUT_DIR/processing_shard$shard.jsonl" \
    --model-cache training/vendor/enhancement-models \
    --rnnoise-binary training/vendor/rnnoise/examples/rnnoise_demo \
    --device cuda:0 --num-shards "$PROCESS_WORKERS" --shard-index "$shard" --resume \
    > "$OUTPUT_DIR/processing_shard$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
python - "$PANEL" "$OUTPUT_DIR" "$PROCESS_WORKERS" <<'PY_AUDIT'
import json
import sys
from pathlib import Path
from training.quality.common import file_hash, read_rows, write_json
from training.scripts.preprocess_training_cascade_audio import PROFILE_HASH

panel, output, workers = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
expected = {r['utterance_id']: r for r in read_rows(panel)}
results = [r for i in range(workers) for r in read_rows(output / f'processing_shard{i}.jsonl')]
if len(results) != len(expected) or {r['utterance_id'] for r in results} != expected.keys():
    raise ValueError('Historical processing results do not cover the fixed panel')
fallbacks = []
for result in results:
    identifier = result['utterance_id']
    wav = output / 'wav' / f'{identifier}.wav'
    metadata = json.loads(wav.with_suffix('.wav.json').read_text())
    if (result['processing_status'] != 'ok' or metadata['status'] != 'PASS'
            or metadata['profile_hash'] != PROFILE_HASH
            or metadata['source_sha256'] != expected[identifier]['processing_input_sha256']
            or metadata['output_sha256'] != file_hash(wav)):
        raise ValueError(f'Invalid historical output: {identifier}')
    if metadata['degenerate_output_fallback']:
        fallbacks.append({'utterance_id': identifier, 'path': metadata['fallback_path']})
write_json(output / 'processing_audit.json', {'status': 'PASS', 'files': len(results),
                                            'fallbacks': fallbacks})
PY_AUDIT
python -m training.quality evaluate --panel "$PANEL" --wav-dir "$OUTPUT_DIR/wav" \
  --label legacy_v8_exact --output "$OUTPUT_DIR/quality" --config "$QUALITY_CONFIG" --resume
python -m training.quality compare --candidate "$OUTPUT_DIR/quality" \
  --original "$ORIGINAL_REPORT" --previous "$CONTROL_REPORT" --output "$OUTPUT_DIR/quality"
