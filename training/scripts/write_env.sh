#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LIB=$(find "${ROOT}/vendor/espeak-ng-install" \( -type f -o -type l \) -name 'libespeak-ng.so*' | sort | head -n 1)
DATA=$(find "${ROOT}/vendor/espeak-ng-install" -type d -name espeak-ng-data | sort | head -n 1)
if [[ -z "${LIB}" || -z "${DATA}" ]]; then
    echo "Local eSpeak-ng installation is incomplete" >&2
    exit 1
fi

sed \
    -e "s|@ROOT@|${ROOT}|g" \
    -e "s|@LIB@|${LIB}|g" \
    -e "s|@DATA@|${DATA}|g" \
    "${ROOT}/activate.sh.in" > "${ROOT}/activate.sh"
chmod +x "${ROOT}/activate.sh"
