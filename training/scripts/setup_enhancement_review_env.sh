#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"${ROOT}/.venv-audio-review"}
UV=${UV:-"${HOME}/.local/bin/uv"}
RESEMBLE_COMMIT=8e978149bfe8abab3eb77d965d579a111afdb0ff
CLEARVOICE_COMMIT=6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61

if [[ ! -x "$UV" ]]; then
    echo "uv is not available: ${UV}" >&2
    exit 2
fi
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    "$UV" venv "$ENV_DIR" --python /usr/bin/python3
fi

"$UV" pip install --python "${ENV_DIR}/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 torchaudio==2.9.1 torchvision==0.24.1 \
    numpy==1.26.4 soundfile==0.12.1 scipy==1.14.1 librosa==0.10.2.post1 \
    imageio-ffmpeg==0.6.0 \
    huggingface-hub==0.34.3 transformers==4.57.1 sentencepiece==0.2.1 \
    omegaconf==2.3.0 pandas==2.3.1 matplotlib==3.10.5 tqdm==4.67.1 \
    rich==14.2.0 resampy==0.4.3 tabulate==0.9.0 celluloid==0.2.0 \
    ptflops==0.7.5 deepspeed==0.18.7 setuptools==80.9.0
"$UV" pip install --python "${ENV_DIR}/bin/python" --no-deps \
    "git+https://github.com/resemble-ai/resemble-enhance.git@${RESEMBLE_COMMIT}"
"$UV" pip install --python "${ENV_DIR}/bin/python" \
    --index-strategy unsafe-best-match \
    "git+https://github.com/modelscope/ClearerVoice-Studio.git@${CLEARVOICE_COMMIT}#subdirectory=clearvoice"

"${ENV_DIR}/bin/python" - <<'PY'
import clearvoice
import resemble_enhance
import torch
import torchaudio
import transformers

assert torch.cuda.is_available()
print(
    {
        "clearvoice": getattr(clearvoice, "__version__", "0.1.2"),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
    }
)
PY
