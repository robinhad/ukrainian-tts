#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN_PID=${1:?Usage: preserve_multispeaker_milestones.sh TRAIN_PID [INTERVAL_SECONDS]}
INTERVAL_SECONDS=${2:-30}
TTS_EXP="${ROOT}/exp_multispeaker_full/tts_jets_uk_24k_multispeaker"
MILESTONE_DIR="${ROOT}/exp_multispeaker_full/milestones"
MILESTONE_SPECS=${MILESTONE_SPECS:-"1:1k 5:5k 25:25k"}

if ! [[ "$TRAIN_PID" =~ ^[0-9]+$ ]]; then
    echo "TRAIN_PID must be a positive integer." >&2
    exit 2
fi
if (( INTERVAL_SECONDS < 5 || INTERVAL_SECONDS > 60 )); then
    echo "INTERVAL_SECONDS must be in the range 5--60." >&2
    exit 2
fi
mkdir -p "$MILESTONE_DIR"

preserve_one() {
    local epoch=$1
    local label=$2
    local source="${TTS_EXP}/${epoch}epoch.pth"
    local target="${MILESTONE_DIR}/${label}.pth"
    local temporary="${target}.part.$$"
    local first_size
    local second_size

    if [[ -e "$target" || ! -s "$source" ]]; then
        return
    fi
    first_size=$(stat -c %s "$source")
    sleep 5
    if [[ ! -s "$source" ]]; then
        return
    fi
    second_size=$(stat -c %s "$source")
    if [[ "$first_size" != "$second_size" ]]; then
        return
    fi
    cp --reflink=auto "$source" "$temporary"
    cmp --silent "$source" "$temporary"
    mv "$temporary" "$target"
    TZ=Europe/Kyiv date \
        "+Preserved ${label}.pth at %Y-%m-%dT%H:%M:%S%:z"
}

preserve_available() {
    local spec
    local epoch
    local label
    for spec in $MILESTONE_SPECS; do
        IFS=: read -r epoch label <<< "$spec"
        if ! [[ "$epoch" =~ ^[0-9]+$ && "$label" =~ ^[A-Za-z0-9_-]+$ ]]; then
            echo "Invalid milestone specification: $spec" >&2
            exit 2
        fi
        preserve_one "$epoch" "$label"
    done
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    preserve_available
    sleep "$INTERVAL_SECONDS"
done
preserve_available
