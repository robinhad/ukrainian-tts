#!/usr/bin/env bash
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"
export PATH="${ESPNET_ROOT}/utils:${PATH}"
export PATH="${ROOT}/espnet_recipe/utils:${PATH}"
export PYTHONPATH="${ESPNET_ROOT}:${ROOT}/..:${PYTHONPATH:-}"
