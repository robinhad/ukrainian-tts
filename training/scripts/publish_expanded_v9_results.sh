#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY=$(cd "${ROOT}/.." && pwd)
NAME=expanded_v9_cascade_with_voa
TOKEN_FILE=${TTS_GITHUB_TOKEN_FILE:-/home/ballvan/Projects/tts-token.txt}

if [[ "$(git -C "$REPOSITORY" branch --show-current)" != autotrain ]]; then
    echo "The active branch is not autotrain." >&2
    exit 2
fi
if ! git -C "$REPOSITORY" diff --cached --quiet; then
    echo "The Git index has unrelated staged changes." >&2
    exit 2
fi

paths=(
    training/reports/final_status.md
    training/reports/${NAME}_full_dataset.json
    training/reports/${NAME}_full_validation.json
    training/reports/${NAME}_full_hybrid_embeddings.json
    training/reports/${NAME}_full_scale_readiness.json
    training/reports/${NAME}_smoke_checkpoint.json
    training/reports/${NAME}_500k_checkpoint.json
    training/reports/${NAME}_tensorboard_metrics.json
    training/reports/${NAME}_training_monitor_summary.json
    training/reports/five_voice_${NAME}_500k.json
    training/reports/${NAME}_tests.log
)
for path in "${paths[@]}"; do
    if [[ ! -s "${REPOSITORY}/${path}" ]]; then
        echo "A publication file does not exist: ${path}" >&2
        exit 2
    fi
done

git -C "$REPOSITORY" add -f -- "${paths[@]}"
if ! git -C "$REPOSITORY" diff --cached --quiet; then
    GIT_AUTHOR_NAME=codex GIT_AUTHOR_EMAIL=codex@openai.com \
    GIT_COMMITTER_NAME=codex GIT_COMMITTER_EMAIL=codex@openai.com \
        git -C "$REPOSITORY" commit -m \
        "Publish expanded v9 scratch 500k results"
fi
if [[ ! -s "$TOKEN_FILE" ]]; then
    echo "The GitHub token file does not exist: ${TOKEN_FILE}" >&2
    exit 2
fi
TTS_GITHUB_TOKEN_FILE="$TOKEN_FILE" \
GIT_ASKPASS="${ROOT}/scripts/git_askpass_from_token.sh" \
GIT_TERMINAL_PROMPT=0 \
    git -C "$REPOSITORY" push origin autotrain
