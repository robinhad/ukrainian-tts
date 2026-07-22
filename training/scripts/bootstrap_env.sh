#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${ROOT}/.venv"
ESPNET_SRC="${ROOT}/vendor/espnet-src"
ESPEAK_SRC="${ROOT}/vendor/espeak-ng-src"
ESPEAK_INSTALL="${ROOT}/vendor/espeak-ng-install"
ESPNET_COMMIT=$(tr -d '[:space:]' < "${ROOT}/vendor/ESPNET_COMMIT")
ESPEAK_COMMIT=$(tr -d '[:space:]' < "${ROOT}/vendor/ESPEAK_NG_COMMIT")

command -v uv >/dev/null
command -v git >/dev/null
command -v cmake >/dev/null

if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python /usr/bin/python3 "${VENV}"
fi

uv pip install --python "${VENV}/bin/python" \
    torch==2.9.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${VENV}/bin/python" -r "${ROOT}/requirements-train.in"

if [[ ! -d "${ESPNET_SRC}/.git" ]]; then
    git clone https://github.com/espnet/espnet.git "${ESPNET_SRC}"
fi
git -C "${ESPNET_SRC}" fetch --tags origin
git -C "${ESPNET_SRC}" checkout --detach "${ESPNET_COMMIT}"
if ! git -C "${ESPNET_SRC}" apply --reverse --check "${ROOT}/patches/espnet-espeak-ng-ukrainian.patch" >/dev/null 2>&1; then
    git -C "${ESPNET_SRC}" apply --check "${ROOT}/patches/espnet-espeak-ng-ukrainian.patch"
    git -C "${ESPNET_SRC}" apply "${ROOT}/patches/espnet-espeak-ng-ukrainian.patch"
fi
uv pip install --python "${VENV}/bin/python" --editable "${ESPNET_SRC}"

if [[ ! -d "${ESPEAK_SRC}/.git" ]]; then
    git clone https://github.com/espeak-ng/espeak-ng.git "${ESPEAK_SRC}"
fi
git -C "${ESPEAK_SRC}" fetch --tags origin
git -C "${ESPEAK_SRC}" checkout --detach "${ESPEAK_COMMIT}"
cmake -S "${ESPEAK_SRC}" -B "${ESPEAK_SRC}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ESPEAK_INSTALL}" \
    -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF
cmake --build "${ESPEAK_SRC}/build" --parallel 8
cmake --install "${ESPEAK_SRC}/build"

"${ROOT}/scripts/write_env.sh"
"${ROOT}/scripts/link_espnet_recipe.sh"
echo "Bootstrap complete. Run: source ${ROOT}/activate.sh"
