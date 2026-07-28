#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: preserve_expanded_v3_milestones.sh TRAIN_PID TARGET_ITERATIONS TTS_EXP}
TARGET_ITERATIONS=${2:-25000}
TARGET_LABEL="$((TARGET_ITERATIONS / 1000))k"
TTS_EXP=${3:-"${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_${TARGET_LABEL}"}
MAX_EPOCH=$((TARGET_ITERATIONS / 1000))
MILESTONE_DIR="${TTS_EXP}/milestones"
MILESTONES=(1 5 15 25 50 100 200 300 400 500)
mkdir -p "$MILESTONE_DIR"

preserve() {
    local epoch=$1
    local label=$2
    local source="${TTS_EXP}/${epoch}epoch.pth"
    local target="${MILESTONE_DIR}/${label}.pth"
    if [[ -e "$target" || ! -s "$source" ]]; then
        return
    fi
    local first_size
    first_size=$(stat -c %s "$source")
    sleep 5
    if [[ -s "$source" && "$first_size" == "$(stat -c %s "$source")" ]]; then
        cp --reflink=auto "$source" "${target}.part"
        cmp --silent "$source" "${target}.part"
        mv "${target}.part" "$target"
    fi
}

preserve_available() {
    local epoch
    for epoch in "${MILESTONES[@]}"; do
        if (( epoch <= MAX_EPOCH )); then
            preserve "$epoch" "${epoch}k"
        fi
    done
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    preserve_available
    sleep 30
done
preserve_available
