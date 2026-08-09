#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREPROCESS_PID=${1:?Set the preprocessing PID.}
LOG="${ROOT}/reports/expanded_v6_sidon_deess_novoa_preprocessing_launcher.log"
STATUS="${ROOT}/reports/expanded_v6_pipeline_continuation.log"
if ! [[ "$PREPROCESS_PID" =~ ^[0-9]+$ ]] || (( PREPROCESS_PID <= 1 )); then
    echo "The preprocessing PID is not safe." >&2
    exit 2
fi
while kill -0 "$PREPROCESS_PID" 2>/dev/null; do
    TZ=Europe/Kyiv date '+%Y-%m-%dT%H:%M:%S%:z Wait for Sidon preprocessing.' >> "$STATUS"
    sleep 60
done
if ! grep -q 'Sidon processing has 74,156 PASS files.' "$LOG"; then
    echo "Sidon preprocessing did not finish with 74,156 PASS files." >> "$STATUS"
    exit 1
fi
TZ=Europe/Kyiv date '+%Y-%m-%dT%H:%M:%S%:z Start manifests, embeddings, and statistics.' >> "$STATUS"
"${ROOT}/scripts/prepare_expanded_v6_sidon_deess_novoa.sh" >> "$STATUS" 2>&1
TZ=Europe/Kyiv date '+%Y-%m-%dT%H:%M:%S%:z Start the 100-step smoke gate.' >> "$STATUS"
"${ROOT}/scripts/run_expanded_v6_sidon_deess_novoa_smoke.sh" >> "$STATUS" 2>&1
TZ=Europe/Kyiv date '+%Y-%m-%dT%H:%M:%S%:z Start 100,000 new training iterations.' >> "$STATUS"
"${ROOT}/scripts/launch_expanded_v6_sidon_deess_novoa_training.sh" 100000 >> "$STATUS" 2>&1
