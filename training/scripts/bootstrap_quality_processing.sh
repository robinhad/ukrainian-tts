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
RNNOISE="${ROOT}/vendor/rnnoise"
RNNOISE_COMMIT=70f1d256acd4b34a572f999a05c87bf00b67730d
if [[ ! -d "$RNNOISE/.git" ]]; then
  git clone https://github.com/xiph/rnnoise.git "$RNNOISE"
  git -C "$RNNOISE" checkout "$RNNOISE_COMMIT"
fi
[[ "$(git -C "$RNNOISE" rev-parse HEAD)" == "$RNNOISE_COMMIT" ]] || {
  echo 'Existing RNNoise checkout has a different revision.' >&2; exit 2;
}
cd "$RNNOISE"
bash download_model.sh
cc -O3 -Iinclude -Isrc examples/rnnoise_demo.c src/denoise.c src/rnn.c \
  src/pitch.c src/kiss_fft.c src/celt_lpc.c src/nnet.c src/nnet_default.c \
  src/parse_lpcnet_weights.c src/rnnoise_data.c src/rnnoise_tables.c \
  -lm -o examples/rnnoise_demo
cd "${ROOT}/.."
"$PYTHON" -c 'from training.audio_enhancement.deepfilternet_compat import install; install(); import clearvoice, df, transformers; print("Processing dependencies imported successfully")'
