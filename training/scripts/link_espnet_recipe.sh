#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ROOT}/activate.sh"
RECIPE="${ROOT}/espnet_recipe"
ln -sfn "${ESPNET_ROOT}/egs2/TEMPLATE/asr1/utils" "${RECIPE}/utils"
ln -sfn "${ESPNET_ROOT}/egs2/TEMPLATE/asr1/scripts" "${RECIPE}/scripts"
ln -sfn "${ESPNET_ROOT}/egs2/TEMPLATE/asr1/pyscripts" "${RECIPE}/pyscripts"
