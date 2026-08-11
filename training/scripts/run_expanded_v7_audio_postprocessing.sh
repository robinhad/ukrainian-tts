#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${ROOT}/.venv/bin/python"
SOURCE="${ROOT}/data/expanded_v6_sidon_deess_novoa/audio_24k"
OUTPUT="${ROOT}/data/expanded_v7_sidon_deess_declick_limit_novoa/audio_24k"
REPORT="${ROOT}/reports/expanded_v7_sidon_deess_declick_limit_novoa_postprocessing.json"
CONFIG="${ROOT}/conf/audio_postprocess.yaml"
JOBS=${JOBS:-16}

if (( JOBS < 1 )); then
    echo "JOBS must be at least 1." >&2
    exit 2
fi
free_bytes=$(df -B1 --output=avail "$ROOT" | tail -1)
if (( free_bytes < 30 * 1024 * 1024 * 1024 )); then
    echo "Free disk space is below 30 GiB." >&2
    exit 2
fi
source_count=$(find "$SOURCE" -type f -name '*.wav' -printf '.' | wc -c)
if (( source_count != 74156 )); then
    echo "The v6 source directory has ${source_count} WAV files, not 74,156." >&2
    exit 2
fi
mkdir -p "$OUTPUT" "$(dirname "$REPORT")"
output_count=$(find "$OUTPUT" -type f -name '*.wav' -printf '.' | wc -c)
if (( output_count == 74156 )) && [[ -s "$REPORT" ]]; then
    if "$PYTHON" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(
    0
    if report.get("status") == "PASS"
    and report.get("completed_files") == 74_156
    and report.get("failed_files") == 0
    else 1
)
PY
    then
        echo "Reuse 74,156 existing v7 final-stage PASS files."
        exit 0
    fi
fi
"$PYTHON" "${ROOT}/scripts/declick_and_limit_audio.py" batch \
    --config "$CONFIG" --input-dir "$SOURCE" --output-dir "$OUTPUT" \
    --recursive --jobs "$JOBS" --overwrite --summary-only --log-level WARNING \
    --report "$REPORT"
"$PYTHON" - "$REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("status") != "PASS" or report.get("completed_files") != 74_156:
    raise SystemExit("The v7 final-stage report is not complete and PASS.")
print("The v7 final stage has 74,156 PASS files.")
PY
