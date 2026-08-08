#!/usr/bin/env bash
set -euo pipefail

TRAIN_PID=${1:?Usage: preserve_expanded_v5_milestones.sh TRAIN_PID TARGET_ITERATIONS TTS_EXP}
TARGET_ITERATIONS=${2:-100000}
TTS_EXP=${3:?Set the experiment path.}
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
MILESTONE_DIR="${TTS_EXP}/milestones"
MILESTONES=(25 50 75 100)
mkdir -p "$MILESTONE_DIR"

preserve() {
    local epoch=$1
    local source="${TTS_EXP}/${epoch}epoch.pth"
    local target="${MILESTONE_DIR}/${epoch}k.pth"
    if [[ -e "$target" || ! -s "$source" ]]; then
        return
    fi
    local first_size
    first_size=$(stat -c %s "$source")
    sleep 5
    if [[ -s "$source" && "$first_size" == "$(stat -c %s "$source")" ]]; then
        if ! ln "$source" "${target}.part" 2>/dev/null; then
            cp "$source" "${target}.part"
        fi
        cmp --silent "$source" "${target}.part"
        mv "${target}.part" "$target"
    fi
}

preserve_available() {
    local epoch
    for epoch in "${MILESTONES[@]}"; do
        if (( epoch <= MAX_EPOCH )); then
            preserve "$epoch"
        fi
    done
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    preserve_available
    sleep 30
done
preserve_available
