#!/usr/bin/env bash
set -euo pipefail

TOKEN_FILE=${TTS_GITHUB_TOKEN_FILE:?Set TTS_GITHUB_TOKEN_FILE.}
if [[ "${1:-}" == *Username* ]]; then
    printf '%s\n' x-access-token
else
    tr -d '\r\n' < "$TOKEN_FILE"
fi
