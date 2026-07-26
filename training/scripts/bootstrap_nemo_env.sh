#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${ROOT}/.venv-nemo"
NEMO_COMMIT=2639d4bef8d1450782263a8f616242acfb6fecb9

command -v uv >/dev/null
if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python /usr/bin/python3 "$VENV"
fi
uv pip install --python "${VENV}/bin/python" \
    torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${VENV}/bin/python" \
    -r "${ROOT}/requirements-nemo.in"
"${VENV}/bin/python" -c "import nemo; print(nemo.__version__)"
printf '%s\n' "$NEMO_COMMIT" >"${ROOT}/vendor/NEMO_COMMIT"
echo "The pinned NeMo environment is ready."
