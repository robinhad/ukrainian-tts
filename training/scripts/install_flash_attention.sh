#!/usr/bin/env bash
# Install a pinned matching wheel; never replace the existing Torch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
if [[ ${FLASH_ATTN_BUILD_SOURCE:-0} == 1 ]]; then
    export MAX_JOBS=${MAX_JOBS:-4}
    export NVCC_THREADS=${NVCC_THREADS:-2}
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    # Optional FLASH_ATTN_CUDA_ARCHS, e.g. 120 for consumer Blackwell/GB10.
    uv pip install --python "$PYTHON" --no-deps --no-build-isolation 'flash-attn==2.8.3.post2'
else
    "$PYTHON" - <<'PY'
import platform
import sys
import torch
compatible = (
    platform.system() == 'Linux' and platform.machine() == 'aarch64'
    and sys.version_info[:2] == (3, 12)
    and torch.__version__.split('.')[:2] == ['2', '9']
    and torch.version.cuda == '13.0'
    and torch._C._GLIBCXX_USE_CXX11_ABI
)
if not compatible:
    raise SystemExit('Pinned wheel requires Linux ARM64, Python 3.12, Torch 2.9, CUDA 13, CXX11 ABI. '
                     'Select a matching wheel or explicitly set FLASH_ATTN_BUILD_SOURCE=1.')
PY
    WHEEL_URL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3%2Bcu13torch2.9cxx11abiTRUE-cp312-cp312-linux_aarch64.whl'
    WHEEL_SHA256='6cfb6cc0b224355363a060d1e34288d14f25311ca0a8e0e9003347722ee43b5b'
    uv pip install --python "$PYTHON" --no-deps "${WHEEL_URL}#sha256=${WHEEL_SHA256}"
fi
"$PYTHON" -c 'import torch, flash_attn; print("Torch", torch.__version__, "FlashAttention", flash_attn.__version__)'
