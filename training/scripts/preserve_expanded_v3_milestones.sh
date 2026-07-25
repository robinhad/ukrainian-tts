#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: preserve_expanded_v3_milestones.sh TRAIN_PID}
TTS_EXP="${ROOT}/exp_expanded_v3/tts_jets_uk_24k_expanded_v3_25k"
MILESTONE_DIR="${TTS_EXP}/milestones"
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
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    preserve 1 1k
    preserve 5 5k
    preserve 15 15k
    preserve 25 25k
    sleep 30
done
preserve 1 1k
preserve 5 5k
preserve 15 15k
preserve 25 25k
