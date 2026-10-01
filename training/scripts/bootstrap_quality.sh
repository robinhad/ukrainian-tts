#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${ROOT}/.."
PYTHON=${QUALITY_PYTHON:-"${ROOT}/.venv/bin/python"}
uv pip install --python "$PYTHON" -r "${ROOT}/requirements-quality.in" \
  --constraint <(printf '%s\n' 'torch==2.9.1' 'torchaudio==2.9.1')
"$PYTHON" - <<'PY'
from pathlib import Path
import hashlib
import json
import urllib.request
import imageio_ffmpeg
import sys

root = Path('training/vendor/quality-models')
sigmos = root / 'SIG-Challenge/ICASSP2024/sigmos'
sigmos.mkdir(parents=True, exist_ok=True)
commit = 'bf4525153b6ed998f19d9e79ff1fd00f55dec42b'
base = f'https://raw.githubusercontent.com/microsoft/SIG-Challenge/{commit}/ICASSP2024/sigmos/'
urls = {sigmos / name: base + name for name in
        ['sigmos.py', 'model-sigmos_1697718653_41d092e8-epo-200.onnx', 'Transparency_FAQ.md']}
onnx = sigmos / 'model-sigmos_1697718653_41d092e8-epo-200.onnx'
urls[onnx] = f'https://media.githubusercontent.com/media/microsoft/SIG-Challenge/{commit}/ICASSP2024/sigmos/{onnx.name}'
urls[root / 'audiobox.pt'] = 'https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt'
for target, url in urls.items():
    if not target.is_file() or (target == onnx and target.stat().st_size < 1000):
        temporary = target.with_suffix(target.suffix + '.part')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(target)
manifest = {}
for path in urls:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    manifest[str(path.relative_to(root))] = {'sha256': digest.hexdigest(), 'url': urls[path]}
assert manifest[str(onnx.relative_to(root))]['sha256'] == 'f939dcc1945055a435565b4369e27dafd0f87df3cea4e2ff6eb81225e52cc53b'
(root / 'downloads.json').write_text(json.dumps(manifest, indent=2) + '\n')
ffmpeg = Path(sys.executable).parent / 'ffmpeg'
if not ffmpeg.exists():
    ffmpeg.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
