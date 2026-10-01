#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
PYTHON=${QUALITY_PYTHON:-"${ROOT}/.venv/bin/python"}
command -v cargo >/dev/null || { echo 'Cargo/Rust is required to build DeepFilterNet on ARM.' >&2; exit 2; }
uv pip install --python "$PYTHON" torchvision==0.24.1 \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
uv pip install --python "$PYTHON" \
  --constraint <(printf '%s\n' 'torch==2.9.1' 'torchaudio==2.9.1' 'numpy==1.26.4') \
  deepfilternet==0.5.6 transformers==4.57.1 huggingface-hub==0.34.3 \
  gdown==6.4.1 opencv-python==4.10.0.84 pydub==0.25.1 \
  python-speech-features==0.6 rotary-embedding-torch==0.8.3 \
  scenedetect==0.6.6 torchinfo==1.8.0 yamlargparse==1.31.1
# The pinned ClearVoice implementation works with the newer shared audio stack.
# Its metadata otherwise downgrades librosa/soundfile required by quality models.
uv pip install --python "$PYTHON" --no-deps \
  'git+https://github.com/modelscope/ClearerVoice-Studio.git@6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61#subdirectory=clearvoice'
"$PYTHON" -c 'from training.audio_enhancement.deepfilternet_compat import install; install(); import clearvoice, df, transformers; print("Processing dependencies imported successfully")'
