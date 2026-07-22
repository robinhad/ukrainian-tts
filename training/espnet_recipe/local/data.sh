#!/usr/bin/env bash
set -euo pipefail

stage=1
stop_stage=1
. ./utils/parse_options.sh

if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
    python local/prepare_data.py \
        --manifest-dir "${MANIFEST_DIR:?MANIFEST_DIR must be set}" \
        --output-root data
    for set_name in smoke_train smoke_dev smoke_eval; do
        ./utils/fix_data_dir.sh "data/${set_name}"
        ./utils/validate_data_dir.sh --no-feats "data/${set_name}"
    done
fi
