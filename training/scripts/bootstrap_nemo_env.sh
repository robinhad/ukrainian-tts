#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${ROOT}/.venv-nemo"

command -v uv >/dev/null
if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python /usr/bin/python3 "$VENV"
fi
uv pip install --python "${VENV}/bin/python" \
    torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${VENV}/bin/python" \
    -r "${ROOT}/requirements-nemo.in"
"${VENV}/bin/python" -c \
    "import nemo; assert nemo.__version__ == '2.4.1', nemo.__version__"
echo "The NeMo 2.4.1 environment is ready."
