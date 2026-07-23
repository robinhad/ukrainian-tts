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
    mkdir -p "${ESPNET_SRC}"
    git -C "${ESPNET_SRC}" init
    git -C "${ESPNET_SRC}" remote add origin https://github.com/espnet/espnet.git
fi
git -C "${ESPNET_SRC}" fetch --depth 1 origin tag v.202604-patch1
git -C "${ESPNET_SRC}" checkout --detach FETCH_HEAD
[[ $(git -C "${ESPNET_SRC}" rev-parse HEAD) == ${ESPNET_COMMIT}* ]]
for patch in \
    "${ROOT}/patches/espnet-espeak-ng-ukrainian.patch" \
    "${ROOT}/patches/espnet-speechbrain-variable-batch.patch"
do
    if ! git -C "${ESPNET_SRC}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        git -C "${ESPNET_SRC}" apply --check "${patch}"
        git -C "${ESPNET_SRC}" apply "${patch}"
    fi
done
uv pip install --python "${VENV}/bin/python" --editable "${ESPNET_SRC}"

if [[ ! -d "${ESPEAK_SRC}/.git" ]]; then
    mkdir -p "${ESPEAK_SRC}"
    git -C "${ESPEAK_SRC}" init
    git -C "${ESPEAK_SRC}" remote add origin https://github.com/espeak-ng/espeak-ng.git
fi
git -C "${ESPEAK_SRC}" fetch --depth 1 origin tag 1.52.0
git -C "${ESPEAK_SRC}" checkout --detach FETCH_HEAD
[[ $(git -C "${ESPEAK_SRC}" rev-parse HEAD) == ${ESPEAK_COMMIT}* ]]
cmake -S "${ESPEAK_SRC}" -B "${ESPEAK_SRC}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${ESPEAK_INSTALL}" \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_ASYNC=OFF -DUSE_MBROLA=OFF
cmake --build "${ESPEAK_SRC}/build" --parallel 8
cmake --install "${ESPEAK_SRC}/build"

"${ROOT}/scripts/write_env.sh"
"${ROOT}/scripts/link_espnet_recipe.sh"
echo "Bootstrap complete. Run: source ${ROOT}/activate.sh"
